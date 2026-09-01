# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bounded transition replay for Tabero tactile DSRL."""

import ctypes
import gc
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Optional

import torch

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.utils.dsrl_observation import DSRL_OBSERVATION_SEMANTICS
from rlinf.utils.dsrl_replay import (
    DSRL_REPLAY_FORMAT,
    DSRL_REPLAY_FORMAT_VERSION,
    DSRL_REPLAY_IMAGE_SIZE,
    DSRL_REPLAY_SEMANTICS,
    dsrl_replay_bytes_per_transition,
    get_dsrl_replay_contract,
    get_dsrl_replay_field_specs,
)
from rlinf.utils.dsrl_reward import (
    DSRL_REWARD_AUDIT_FIELDS,
    DSRL_REWARD_SEMANTICS,
    combine_dsrl_reward_audits,
    empty_dsrl_reward_audit,
    summarize_dsrl_chunk_rewards,
)
from rlinf.utils.dsrl_transition import DSRL_TRANSITION_BOUNDARY_SEMANTICS


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _field_manifest(field_specs) -> dict[str, dict[str, object]]:
    return {
        name: {"shape": list(shape), "dtype": _dtype_name(dtype)}
        for name, (shape, dtype) in field_specs.items()
    }


def _release_checkpoint_staging_memory() -> None:
    """Return freed CPU shard buffers to the OS when libc supports it."""

    gc.collect()
    try:
        malloc_trim = ctypes.CDLL(None).malloc_trim
    except (AttributeError, OSError):
        return
    malloc_trim.argtypes = [ctypes.c_size_t]
    malloc_trim.restype = ctypes.c_int
    malloc_trim(0)


class CompactDSRLReplayBuffer:
    """Fixed-capacity CPU tensor ring containing only SAC transition fields.

    ``size`` intentionally retains the historical RLinf meaning used by SAC
    warmup: number of trajectory insertions. ``total_samples`` is the number of
    currently resident transitions and is capped by ``capacity_transitions``.
    """

    def __init__(
        self,
        *,
        seed: Optional[int] = 1234,
        capacity_transitions: int,
        checkpoint_shard_transitions: int = 4096,
        max_resident_gib: float = 12.0,
        replay_semantics: str = DSRL_REPLAY_SEMANTICS,
        observation_semantics: str = DSRL_OBSERVATION_SEMANTICS,
        transition_boundary_semantics: str = DSRL_TRANSITION_BOUNDARY_SEMANTICS,
    ) -> None:
        if capacity_transitions <= 0:
            raise ValueError("capacity_transitions must be greater than zero.")
        if checkpoint_shard_transitions <= 0:
            raise ValueError("checkpoint_shard_transitions must be greater than zero.")
        if max_resident_gib <= 0:
            raise ValueError("max_resident_gib must be greater than zero.")

        replay_contract = get_dsrl_replay_contract(replay_semantics)
        expected_observation_semantics = replay_contract["observation_semantics"]
        if observation_semantics != expected_observation_semantics:
            raise ValueError(
                "Tabero DSRL compact replay observation semantics mismatch: "
                f"replay {replay_semantics!r} requires "
                f"{expected_observation_semantics!r}, got {observation_semantics!r}."
            )
        expected_transition_semantics = replay_contract["transition_boundary_semantics"]
        if transition_boundary_semantics != expected_transition_semantics:
            raise ValueError(
                "Tabero DSRL compact replay transition boundary semantics mismatch: "
                f"replay {replay_semantics!r} requires "
                f"{expected_transition_semantics!r}, got "
                f"{transition_boundary_semantics!r}."
            )

        self.replay_semantics = replay_semantics
        self.observation_semantics = observation_semantics
        self.transition_boundary_semantics = transition_boundary_semantics
        self.field_specs = get_dsrl_replay_field_specs(replay_semantics)
        self.view_order = tuple(replay_contract["view_order"])
        self.num_images = int(replay_contract["num_images"])
        self.observation_keys = tuple(
            name.removeprefix("curr_obs.")
            for name in self.field_specs
            if name.startswith("curr_obs.")
        )

        self.capacity_transitions = int(capacity_transitions)
        self.checkpoint_shard_transitions = int(checkpoint_shard_transitions)
        self.max_resident_bytes = int(float(max_resident_gib) * (2**30))
        self.bytes_per_transition = dsrl_replay_bytes_per_transition(replay_semantics)
        self.capacity_bytes = self.bytes_per_transition * self.capacity_transitions
        if self.capacity_bytes > self.max_resident_bytes:
            raise ValueError(
                "Tabero DSRL compact replay capacity exceeds its resident-memory "
                f"budget: capacity={self.capacity_bytes} bytes, "
                f"budget={self.max_resident_bytes} bytes."
            )

        self.seed = int(seed if seed is not None else 1234)
        self.random_generator = torch.Generator(device="cpu")
        self.random_generator.manual_seed(self.seed)

        self._storage: dict[str, torch.Tensor] | None = None
        self._write_pos = 0
        self._valid_samples = 0
        self._total_inserted_samples = 0
        self._last_insert_reward_audit = empty_dsrl_reward_audit()
        self.size = 0

    def _ensure_storage(self) -> dict[str, torch.Tensor]:
        if self._storage is None:
            self._storage = {
                name: torch.empty(
                    (self.capacity_transitions, *shape),
                    dtype=dtype,
                    device="cpu",
                )
                for name, (shape, dtype) in self.field_specs.items()
            }
        return self._storage

    def _flatten_trajectory(self, trajectory: Trajectory) -> dict[str, torch.Tensor]:
        if trajectory.forward_inputs:
            raise ValueError(
                "Tabero DSRL compact replay forbids trajectory.forward_inputs; "
                f"got fields={sorted(trajectory.forward_inputs)}."
            )
        if trajectory.rewards is None or trajectory.rewards.ndim != 3:
            shape = (
                tuple(trajectory.rewards.shape)
                if torch.is_tensor(trajectory.rewards)
                else None
            )
            raise ValueError(
                f"Tabero DSRL compact replay rewards must be [T,B,10]; got {shape}."
            )
        traj_len, batch_size = map(int, trajectory.rewards.shape[:2])
        num_samples = traj_len * batch_size

        def flatten_obs(
            obs: Mapping[str, object], prefix: str
        ) -> dict[str, torch.Tensor]:
            expected_keys = set(self.observation_keys)
            if set(obs) != expected_keys:
                raise ValueError(
                    f"Tabero DSRL compact replay {prefix} fields mismatch: expected "
                    f"{sorted(expected_keys)}, got {sorted(obs)}."
                )
            result = {}
            for key in sorted(expected_keys):
                value = obs[key]
                shape, dtype = self.field_specs[f"{prefix}.{key}"]
                expected_shape = (traj_len, batch_size, *shape)
                if not torch.is_tensor(value) or tuple(value.shape) != expected_shape:
                    actual = tuple(value.shape) if hasattr(value, "shape") else None
                    raise ValueError(
                        f"Tabero DSRL compact replay {prefix}.{key} expected shape "
                        f"{expected_shape}; got {actual}."
                    )
                if value.dtype != dtype:
                    raise ValueError(
                        f"Tabero DSRL compact replay {prefix}.{key} expected dtype "
                        f"{dtype}; got {value.dtype}."
                    )
                result[f"{prefix}.{key}"] = (
                    value.reshape(num_samples, *shape).cpu().contiguous()
                )
            return result

        flat = {}
        flat.update(flatten_obs(trajectory.curr_obs, "curr_obs"))
        flat.update(flatten_obs(trajectory.next_obs, "next_obs"))

        for name in ("actions", "rewards", "terminations", "truncations"):
            value = getattr(trajectory, name)
            shape, dtype = self.field_specs[name]
            expected_shape = (traj_len, batch_size, *shape)
            if not torch.is_tensor(value) or tuple(value.shape) != expected_shape:
                actual = tuple(value.shape) if hasattr(value, "shape") else None
                raise ValueError(
                    f"Tabero DSRL compact replay {name} expected shape "
                    f"{expected_shape}; got {actual}."
                )
            if value.dtype != dtype:
                raise ValueError(
                    f"Tabero DSRL compact replay {name} expected dtype {dtype}; "
                    f"got {value.dtype}."
                )
            flat[name] = value.reshape(num_samples, *shape).cpu().contiguous()

        if set(flat) != set(self.field_specs):
            raise AssertionError("Internal compact replay field projection mismatch.")
        return flat

    def _append_flat(self, flat: Mapping[str, torch.Tensor]) -> None:
        first = next(iter(flat.values()))
        original_count = int(first.shape[0])
        if original_count == 0:
            return
        if any(int(value.shape[0]) != original_count for value in flat.values()):
            raise ValueError("Compact replay fields have inconsistent batch lengths.")

        if original_count > self.capacity_transitions:
            start = original_count - self.capacity_transitions
            flat = {name: value[start:] for name, value in flat.items()}
        count = min(original_count, self.capacity_transitions)
        storage = self._ensure_storage()

        first_count = min(count, self.capacity_transitions - self._write_pos)
        second_count = count - first_count
        for name in self.field_specs:
            value = flat[name]
            storage[name][self._write_pos : self._write_pos + first_count].copy_(
                value[:first_count]
            )
            if second_count:
                storage[name][:second_count].copy_(value[first_count:])

        self._write_pos = (self._write_pos + count) % self.capacity_transitions
        self._valid_samples = min(
            self.capacity_transitions, self._valid_samples + count
        )
        self._total_inserted_samples += original_count

    def add_trajectories(self, trajectories: list[Trajectory]) -> None:
        self._last_insert_reward_audit = empty_dsrl_reward_audit()
        if not trajectories:
            return
        insertion_audits = []
        for trajectory in trajectories:
            flat = self._flatten_trajectory(trajectory)
            insertion_audits.append(
                summarize_dsrl_chunk_rewards(
                    flat["rewards"],
                    flat["terminations"],
                    flat["truncations"],
                )
            )
            self._append_flat(flat)
            self.size += 1
        self._last_insert_reward_audit = combine_dsrl_reward_audits(insertion_audits)

    def _nested_batch(self, flat: Mapping[str, torch.Tensor]) -> dict[str, object]:
        return {
            "curr_obs": {key: flat[f"curr_obs.{key}"] for key in self.observation_keys},
            "next_obs": {key: flat[f"next_obs.{key}"] for key in self.observation_keys},
            "actions": flat["actions"],
            "rewards": flat["rewards"],
            "terminations": flat["terminations"],
            "truncations": flat["truncations"],
        }

    def sample(self, num_chunks: int = 0) -> dict[str, object]:
        if num_chunks <= 0:
            raise ValueError("num_chunks must be greater than zero.")
        if self._valid_samples == 0 or self._storage is None:
            raise RuntimeError("Cannot sample from an empty buffer.")
        count = min(int(num_chunks), self._valid_samples)
        indices = torch.randint(
            low=0,
            high=self._valid_samples,
            size=(count,),
            generator=self.random_generator,
        )
        flat = {
            name: tensor.index_select(0, indices)
            for name, tensor in self._storage.items()
        }
        return self._nested_batch(flat)

    def sample_chunks(self, num_chunks: int) -> dict[str, object]:
        return self.sample(num_chunks)

    def is_ready(self, min_size: int) -> bool:
        return self.size >= int(min_size)

    async def is_ready_async(self, min_size: int) -> bool:
        return self.is_ready(min_size)

    def __len__(self) -> int:
        return self.size

    @property
    def total_samples(self) -> int:
        return self._valid_samples

    def get_stats(self) -> dict[str, float]:
        stats = {
            "num_trajectories": float(self.size),
            "total_samples": float(self._valid_samples),
            "total_inserted_samples": float(self._total_inserted_samples),
            "resident_tensor_bytes": float(
                self._valid_samples * self.bytes_per_transition
            ),
            "capacity_bytes": float(self.capacity_bytes),
        }
        stats.update(
            {
                f"last_insert_{field}": self._last_insert_reward_audit[field]
                for field in DSRL_REWARD_AUDIT_FIELDS
            }
        )
        return stats

    def clear(self) -> None:
        self._write_pos = 0
        self._valid_samples = 0
        self._total_inserted_samples = 0
        self._last_insert_reward_audit = empty_dsrl_reward_audit()
        self.size = 0

    def close(self, wait: bool = True) -> None:
        del wait

    def _chronological_indices(self) -> torch.Tensor:
        if self._valid_samples < self.capacity_transitions:
            return torch.arange(self._valid_samples, dtype=torch.long)
        return torch.cat(
            (
                torch.arange(
                    self._write_pos, self.capacity_transitions, dtype=torch.long
                ),
                torch.arange(0, self._write_pos, dtype=torch.long),
            )
        )

    def _physical_indices(self) -> torch.Tensor:
        """Return resident slots in the exact order used by random sampling."""

        return torch.arange(self._valid_samples, dtype=torch.long)

    def _metadata(
        self, *, shard_manifest: list[dict[str, object]]
    ) -> dict[str, object]:
        return {
            "format": DSRL_REPLAY_FORMAT,
            "format_version": DSRL_REPLAY_FORMAT_VERSION,
            "replay_semantics": self.replay_semantics,
            "reward_semantics": DSRL_REWARD_SEMANTICS,
            "observation_semantics": self.observation_semantics,
            "transition_boundary_semantics": self.transition_boundary_semantics,
            "view_order": list(self.view_order),
            "num_images": self.num_images,
            "image_size": DSRL_REPLAY_IMAGE_SIZE,
            "capacity_transitions": self.capacity_transitions,
            "checkpoint_shard_transitions": self.checkpoint_shard_transitions,
            "valid_samples": self._valid_samples,
            "write_pos": self._write_pos,
            "total_inserted_samples": self._total_inserted_samples,
            "insertion_count": self.size,
            "bytes_per_transition": self.bytes_per_transition,
            "num_shards": len(shard_manifest),
            "shards": shard_manifest,
            "seed": self.seed,
            "rng_state": self.random_generator.get_state().tolist(),
            "field_manifest": _field_manifest(self.field_specs),
        }

    def save_checkpoint(self, save_path: str) -> None:
        path = Path(save_path)
        path.mkdir(parents=True, exist_ok=True)
        # Preserve physical ring layout so restoring RNG state also restores
        # the exact sample sequence, including after wraparound.
        indices = self._physical_indices()
        num_shards = (
            self._valid_samples + self.checkpoint_shard_transitions - 1
        ) // self.checkpoint_shard_transitions

        if self._valid_samples and self._storage is None:
            raise AssertionError("Compact replay storage is missing resident samples.")
        shard_manifest = []
        for shard_id in range(num_shards):
            start = shard_id * self.checkpoint_shard_transitions
            end = min(start + self.checkpoint_shard_transitions, self._valid_samples)
            shard_indices = indices[start:end]
            shard = {
                name: tensor.index_select(0, shard_indices).contiguous()
                for name, tensor in self._storage.items()
            }
            target = path / f"shard_{shard_id:05d}.pt"
            temporary = path / f".{target.name}.tmp"
            torch.save(shard, temporary)
            os.replace(temporary, target)
            shard_manifest.append(
                {
                    "name": target.name,
                    "num_samples": end - start,
                    "size_bytes": target.stat().st_size,
                }
            )
            del shard
            _release_checkpoint_staging_memory()

        metadata = self._metadata(shard_manifest=shard_manifest)
        metadata_path = path / "metadata.json"
        metadata_tmp = path / ".metadata.json.tmp"
        with metadata_tmp.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, sort_keys=True)
        os.replace(metadata_tmp, metadata_path)

    @classmethod
    def validate_checkpoint_metadata(
        cls,
        load_path: str,
        *,
        expected_capacity: int | None = None,
        expected_replay_semantics: str = DSRL_REPLAY_SEMANTICS,
        expected_observation_semantics: str | None = None,
        expected_transition_boundary_semantics: str | None = None,
    ) -> dict[str, object]:
        path = Path(load_path)
        metadata_path = path / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Tabero DSRL compact replay metadata not found: {metadata_path}."
            )
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        replay_contract = get_dsrl_replay_contract(expected_replay_semantics)
        if expected_observation_semantics is None:
            expected_observation_semantics = replay_contract["observation_semantics"]
        if expected_transition_boundary_semantics is None:
            expected_transition_boundary_semantics = replay_contract[
                "transition_boundary_semantics"
            ]
        field_specs = get_dsrl_replay_field_specs(expected_replay_semantics)
        expected_values = {
            "format": DSRL_REPLAY_FORMAT,
            "format_version": DSRL_REPLAY_FORMAT_VERSION,
            "replay_semantics": expected_replay_semantics,
            "reward_semantics": DSRL_REWARD_SEMANTICS,
            "observation_semantics": expected_observation_semantics,
            "transition_boundary_semantics": (expected_transition_boundary_semantics),
            "view_order": list(replay_contract["view_order"]),
            "num_images": replay_contract["num_images"],
            "image_size": DSRL_REPLAY_IMAGE_SIZE,
            "bytes_per_transition": dsrl_replay_bytes_per_transition(
                expected_replay_semantics
            ),
            "field_manifest": _field_manifest(field_specs),
        }
        for key, expected in expected_values.items():
            actual = metadata.get(key)
            if actual != expected:
                raise ValueError(
                    f"Tabero DSRL compact replay metadata {key!r} mismatch: "
                    f"expected {expected!r}, got {actual!r}."
                )
        if expected_capacity is not None and metadata.get(
            "capacity_transitions"
        ) != int(expected_capacity):
            raise ValueError(
                "Tabero DSRL compact replay capacity mismatch: expected "
                f"{int(expected_capacity)}, got "
                f"{metadata.get('capacity_transitions')!r}."
            )

        capacity = metadata.get("capacity_transitions")
        valid_samples = metadata.get("valid_samples")
        write_pos = metadata.get("write_pos")
        shard_size = metadata.get("checkpoint_shard_transitions")
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(
                f"Tabero DSRL compact replay capacity is invalid: {capacity!r}."
            )
        if (
            not isinstance(valid_samples, int)
            or valid_samples < 0
            or valid_samples > capacity
        ):
            raise ValueError(
                "Tabero DSRL compact replay valid_samples is invalid: "
                f"{valid_samples!r}."
            )
        if not isinstance(write_pos, int) or write_pos < 0 or write_pos >= capacity:
            raise ValueError(
                f"Tabero DSRL compact replay write_pos is invalid: {write_pos!r}."
            )
        if not isinstance(shard_size, int) or shard_size <= 0:
            raise ValueError(
                "Tabero DSRL compact replay checkpoint_shard_transitions is "
                f"invalid: {shard_size!r}."
            )
        for key in ("total_inserted_samples", "insertion_count"):
            value = metadata.get(key)
            if not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"Tabero DSRL compact replay {key} is invalid: {value!r}."
                )
        rng_state = metadata.get("rng_state")
        if not isinstance(rng_state, list) or not rng_state:
            raise ValueError("Tabero DSRL compact replay RNG state is invalid.")

        num_shards = metadata.get("num_shards")
        if not isinstance(num_shards, int) or num_shards < 0:
            raise ValueError(
                f"Tabero DSRL compact replay num_shards is invalid: {num_shards!r}."
            )
        expected_num_shards = (valid_samples + shard_size - 1) // shard_size
        if num_shards != expected_num_shards:
            raise ValueError(
                "Tabero DSRL compact replay num_shards mismatch: expected "
                f"{expected_num_shards}, got {num_shards}."
            )
        shards = metadata.get("shards")
        if not isinstance(shards, list) or len(shards) != num_shards:
            raise ValueError("Tabero DSRL compact replay shard manifest is invalid.")
        manifest_samples = 0
        for shard_id, shard_info in enumerate(shards):
            expected_name = f"shard_{shard_id:05d}.pt"
            if (
                not isinstance(shard_info, Mapping)
                or shard_info.get("name") != expected_name
            ):
                raise ValueError(
                    "Tabero DSRL compact replay shard manifest entry is invalid: "
                    f"{shard_info!r}."
                )
            shard_samples = shard_info.get("num_samples")
            size_bytes = shard_info.get("size_bytes")
            if not isinstance(shard_samples, int) or shard_samples <= 0:
                raise ValueError(
                    "Tabero DSRL compact replay shard sample count is invalid: "
                    f"{shard_samples!r}."
                )
            if not isinstance(size_bytes, int) or size_bytes <= 0:
                raise ValueError(
                    f"Tabero DSRL compact replay shard size is invalid: {size_bytes!r}."
                )
            shard_path = path / expected_name
            if not shard_path.is_file():
                raise FileNotFoundError(
                    f"Tabero DSRL compact replay shard not found: {shard_path}."
                )
            if shard_path.stat().st_size != size_bytes:
                raise ValueError(
                    "Tabero DSRL compact replay shard size mismatch for "
                    f"{shard_path}: expected {size_bytes}, got "
                    f"{shard_path.stat().st_size}."
                )
            manifest_samples += shard_samples
        if manifest_samples != valid_samples:
            raise ValueError(
                "Tabero DSRL compact replay shard manifest sample count mismatch: "
                f"expected {valid_samples}, got {manifest_samples}."
            )
        return metadata

    def load_checkpoint(
        self,
        load_path: str,
        is_distributed: bool = False,
        local_rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if is_distributed:
            raise ValueError(
                "CompactDSRLReplayBuffer checkpoints are already rank-local; "
                "distributed splitting is not supported."
            )
        del local_rank, world_size
        metadata = self.validate_checkpoint_metadata(
            load_path,
            expected_capacity=self.capacity_transitions,
            expected_replay_semantics=self.replay_semantics,
            expected_observation_semantics=self.observation_semantics,
            expected_transition_boundary_semantics=(self.transition_boundary_semantics),
        )
        valid_samples = int(metadata["valid_samples"])
        if valid_samples > self.capacity_transitions:
            raise ValueError(
                "Tabero DSRL compact replay checkpoint contains more samples than "
                f"the configured capacity: {valid_samples} > "
                f"{self.capacity_transitions}."
            )

        self.clear()
        path = Path(load_path)
        loaded_samples = 0
        for shard_id in range(int(metadata["num_shards"])):
            shard = torch.load(
                path / f"shard_{shard_id:05d}.pt",
                map_location="cpu",
                weights_only=True,
            )
            if set(shard) != set(self.field_specs):
                raise ValueError(
                    "Tabero DSRL compact replay shard fields mismatch: expected "
                    f"{sorted(self.field_specs)}, got {sorted(shard)}."
                )
            shard_count = int(next(iter(shard.values())).shape[0])
            for name, (shape, dtype) in self.field_specs.items():
                tensor = shard[name]
                expected_shape = (shard_count, *shape)
                if tuple(tensor.shape) != expected_shape or tensor.dtype != dtype:
                    raise ValueError(
                        f"Tabero DSRL compact replay shard field {name!r} "
                        f"expected {expected_shape}/{dtype}, got "
                        f"{tuple(tensor.shape)}/{tensor.dtype}."
                    )
            self._append_flat(shard)
            loaded_samples += shard_count
            del shard

        if loaded_samples != valid_samples:
            raise ValueError(
                "Tabero DSRL compact replay shard sample count mismatch: expected "
                f"{valid_samples}, loaded {loaded_samples}."
            )
        self._total_inserted_samples = int(metadata["total_inserted_samples"])
        self.size = int(metadata["insertion_count"])
        write_pos = metadata.get("write_pos")
        if (
            not isinstance(write_pos, int)
            or write_pos < 0
            or write_pos >= self.capacity_transitions
        ):
            raise ValueError(
                f"Tabero DSRL compact replay write_pos is invalid: {write_pos!r}."
            )
        self._write_pos = write_pos
        rng_state = torch.tensor(metadata["rng_state"], dtype=torch.uint8)
        self.random_generator.set_state(rng_state)
