# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real/sim mixtures for the upstream OpenPI PyTorch SFT data path.

``data.cotrain`` replaces ``train_data_paths`` and contains exactly ``real`` and
``sim`` specs plus ``sim_weight`` (default 0.5). Each spec has ``dataset_path``
(local LeRobot root or repo id) and optionally either ``episodes`` or
``episode_source``. Sources are resolved from ``meta/episode_sources.jsonl``.
Both domains use the same transforms and offline union normalization stats.
Keep ``data.train_data_paths: []`` as the SFT entrypoint's training switch;
its contents are ignored by the co-training loader.

Sampling is with replacement within each domain's rank shard. Each micro batch
has an exact domain count, so the ordinary batch mean implements Eq. 7 without
another loss weight. Epoch/consumed-batch offset is the only resume cursor;
prefetched batches never advance checkpoint state. No T2-VLA imports are used.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

DOMAIN_KEY = "domain"


@dataclasses.dataclass(frozen=True)
class CoTrainDatasetSpec:
    """One LeRobot domain, optionally selected by episode id or source."""

    dataset_path: str
    episodes: tuple[int, ...] | None = None
    episode_source: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_path, str) or not self.dataset_path:
            raise ValueError("Co-training dataset_path must be a non-empty string.")
        if self.episodes is not None and self.episode_source is not None:
            raise ValueError("episodes and episode_source are mutually exclusive.")
        if self.episodes is not None:
            if not self.episodes or any(
                isinstance(episode, bool) or not isinstance(episode, int) or episode < 0
                for episode in self.episodes
            ):
                raise ValueError("episodes must contain non-negative integer ids.")
            object.__setattr__(self, "episodes", tuple(self.episodes))
        if self.episode_source is not None and (
            not isinstance(self.episode_source, str) or not self.episode_source
        ):
            raise ValueError("episode_source must be a non-empty string.")


@dataclasses.dataclass(frozen=True)
class CoTrainConfig:
    """Exactly one real domain and one simulation domain."""

    real: CoTrainDatasetSpec
    sim: CoTrainDatasetSpec
    sim_weight: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.sim_weight <= 1.0:
            raise ValueError("sim_weight must be finite and in [0, 1].")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CoTrainConfig:
        """Validate the YAML block, including unknown or missing domains."""
        unknown = set(value) - {"real", "sim", "sim_weight"}
        if unknown or value.get("real") is None or value.get("sim") is None:
            raise ValueError(
                "data.cotrain requires real and sim, with optional sim_weight only; "
                f"unknown keys: {sorted(unknown)}."
            )
        return cls(
            real=CoTrainDatasetSpec(**dict(value["real"])),
            sim=CoTrainDatasetSpec(**dict(value["sim"])),
            sim_weight=float(value.get("sim_weight", 0.5)),
        )


def frame_indices_for_episodes(
    episode_indices: Sequence[int], episodes: Sequence[int]
) -> np.ndarray:
    """Find full-dataset frame indices, including interleaved episode rows."""
    selected = set(episodes)
    if not selected:
        raise ValueError("Episode selection is empty.")
    frame_episodes = np.asarray(episode_indices, dtype=np.int64)
    if frame_episodes.ndim != 1:
        raise ValueError("The episode_index column must be one-dimensional.")
    unknown = selected - set(np.unique(frame_episodes))
    if unknown:
        raise ValueError(f"Unknown episode(s) requested: {sorted(unknown)}.")
    return np.flatnonzero(np.isin(frame_episodes, tuple(selected)))


class FrameIndexDataset(Dataset):
    """Read selected frames through the full LeRobot dataset's index space.

    Never pass ``episodes=`` to LeRobot: its temporal action lookup can break
    for non-contiguous episode subsets. Future action chunks still use the
    original full-dataset episode boundaries.
    """

    def __init__(self, dataset: Dataset, frame_indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.frame_indices = np.asarray(frame_indices, dtype=np.int64)
        if self.frame_indices.ndim != 1 or self.frame_indices.size == 0:
            raise ValueError(
                "Frame selection must be a non-empty one-dimensional array."
            )
        if np.any(self.frame_indices < 0) or np.any(self.frame_indices >= len(dataset)):
            raise ValueError("Frame selection contains out-of-range indices.")

    def __len__(self) -> int:
        return len(self.frame_indices)

    def __getitem__(self, index: int) -> dict:
        return self.dataset[int(self.frame_indices[index])]


class MixtureDataset(Dataset):
    """Concatenate transformed samples and add domain (0 real, 1 sim)."""

    def __init__(self, real: Dataset, sim: Dataset) -> None:
        self.real = real
        self.sim = sim

    def __len__(self) -> int:
        return len(self.real) + len(self.sim)

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index < len(self.real):
            sample, domain = self.real[index], 0
        else:
            sample, domain = self.sim[index - len(self.real)], 1
        return {**sample, DOMAIN_KEY: np.int32(domain)}


def _domain_batch_counts(batch_size: int, sim_weight: float) -> tuple[int, int]:
    if batch_size <= 0:
        raise ValueError("micro_batch_size must be positive.")
    if not 0.0 <= sim_weight <= 1.0:
        raise ValueError("sim_weight must be finite and in [0, 1].")
    n_sim_float = sim_weight * batch_size
    n_sim = round(n_sim_float)
    if not math.isclose(n_sim_float, n_sim, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "sim_weight * micro_batch_size must be an integer; "
            f"got {sim_weight} * {batch_size} = {n_sim_float}."
        )
    return batch_size - n_sim, n_sim


class StratifiedBatchSampler(Sampler[list[int]]):
    """Exact domain counts with deterministic, rank-specific random streams.

    For a domain with at least ``world_size`` frames, ranks draw from disjoint
    strided shards. Smaller domains share their frames (overlap is unavoidable).
    Sampling within a shard is with replacement. Every rank has the same epoch
    length. Iteration is stateless: restoring ``set_epoch`` and skipping consumed
    batches reproduces the stream even when a DataLoader prefetches ahead.
    """

    def __init__(
        self,
        real_size: int,
        sim_size: int,
        batch_size: int,
        sim_weight: float = 0.5,
        *,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        num_batches: int | None = None,
    ) -> None:
        if real_size < 0 or sim_size < 0:
            raise ValueError("Dataset sizes must be non-negative.")
        if world_size <= 0 or not 0 <= rank < world_size or seed < 0:
            raise ValueError(
                "Require world_size > 0, 0 <= rank < world_size and seed >= 0."
            )
        self.n_real, self.n_sim = _domain_batch_counts(batch_size, sim_weight)
        if (self.n_real and real_size == 0) or (self.n_sim and sim_size == 0):
            raise ValueError("Each sampled domain must contain at least one frame.")
        self.real_size, self.sim_size = real_size, sim_size
        self.batch_size = batch_size
        self.seed, self.rank, self.world_size = int(seed), rank, world_size
        self.epoch = 0
        # This defines a nominal epoch, not a promise to visit every frame.
        self.num_batches = (
            num_batches
            if num_batches is not None
            else max(
                math.ceil(size / (count * world_size))
                for size, count in ((real_size, self.n_real), (sim_size, self.n_sim))
                if count
            )
        )
        if self.num_batches <= 0:
            raise ValueError("num_batches must be positive.")

    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        """Select an epoch without relying on the producer's prefetch cursor."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")
        self.epoch = int(epoch)

    def resume_signature(self) -> dict[str, int]:
        """Return rank-independent stream settings saved by rank zero."""
        return {
            key: getattr(self, key)
            for key in (
                "seed",
                "world_size",
                "batch_size",
                "real_size",
                "sim_size",
                "n_sim",
                "num_batches",
            )
        }

    def _draw(self, rng: np.random.Generator, size: int, count: int) -> np.ndarray:
        if count == 0:
            return np.empty(0, dtype=np.int64)
        if size < self.world_size:
            return rng.integers(size, size=count)
        shard_size = (size - 1 - self.rank) // self.world_size + 1
        return self.rank + self.world_size * rng.integers(shard_size, size=count)

    def __iter__(self) -> Iterator[list[int]]:
        for batch_number in range(self.num_batches):
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.rank, self.epoch, batch_number])
            )
            indices = np.concatenate(
                (
                    self._draw(rng, self.real_size, self.n_real),
                    self.real_size + self._draw(rng, self.sim_size, self.n_sim),
                )
            )
            rng.shuffle(indices)
            yield indices.tolist()


def episodes_for_source(root: Path, source: str) -> tuple[int, ...]:
    """Read episode source annotations from a local or cached LeRobot root."""
    source_path = root / "meta" / "episode_sources.jsonl"
    episodes = []
    with source_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source") == source:
                episode = row.get("episode_index", row.get("episode"))
                if isinstance(episode, bool) or not isinstance(episode, int):
                    raise ValueError(
                        f"Invalid episode id in {source_path}:{line_number}."
                    )
                episodes.append(episode)
    if not episodes:
        raise ValueError(f"No episodes with source={source!r} in {source_path}.")
    return tuple(sorted(set(episodes)))


def validate_dataset_compatibility(real: Any, sim: Any) -> None:
    """Check FPS, feature keys and all shapes, including markers and actions."""
    real_fps, sim_fps = float(real.fps), float(sim.fps)
    if (
        real_fps <= 0
        or not math.isfinite(real_fps)
        or not math.isclose(real_fps, sim_fps, rel_tol=0.0, abs_tol=1e-6)
    ):
        raise ValueError(f"Co-training fps mismatch: real={real_fps}, sim={sim_fps}.")
    real_features, sim_features = real.features, sim.features
    if set(real_features) != set(sim_features):
        raise ValueError(
            "Co-training datasets must have identical feature keys; "
            f"real-only={sorted(set(real_features) - set(sim_features))}, "
            f"sim-only={sorted(set(sim_features) - set(real_features))}."
        )
    for key in real_features:
        real_shape = tuple(real_features[key]["shape"])
        sim_shape = tuple(sim_features[key]["shape"])
        if real_shape != sim_shape:
            raise ValueError(
                f"Co-training feature {key!r} has incompatible shapes: "
                f"real={real_shape}, sim={sim_shape}."
            )


def create_cotrain_data_loader(
    config: Any,
    cotrain: CoTrainConfig,
    *,
    micro_batch_size: int,
    rank: int,
    world_size: int,
    seed: int,
) -> tuple[DataLoader, Any]:
    """Build both full datasets, select frames, transform, and mix on this rank.

    ``config`` is RLinf's upstream OpenPI TrainConfig. Norm stats must already
    have been supplied through ``actor.model.openpi_data.norm_stats_path``.
    The returned loader yields raw batched dictionaries, before Observation
    conversion, to retain RLinf-local contact and domain fields.
    """
    import openpi.training.data_loader as openpi_loader
    from openpi import transforms

    from rlinf.data.lerobot_paths import resolve_lerobot_dataset_root

    # Reject invalid alpha before opening datasets or loading norm assets.
    _domain_batch_counts(micro_batch_size, cotrain.sim_weight)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.norm_stats is None:
        raise ValueError("Co-training requires offline union norm_stats_path.")
    if data_config.rlds_data_dir is not None:
        raise ValueError("OpenPI co-training supports LeRobot datasets only.")
    lerobot = openpi_loader.lerobot_dataset
    revision = getattr(data_config, "lerobot_revision", None)
    metadata, roots, full_datasets = {}, {}, {}
    for spec in (cotrain.real, cotrain.sim):
        repo_id = spec.dataset_path
        if repo_id in metadata:
            continue
        local = Path(repo_id).expanduser()
        root_kwargs = (
            {"root": local} if (local / "meta" / "info.json").is_file() else {}
        )
        metadata[repo_id] = lerobot.LeRobotDatasetMetadata(
            repo_id, revision=revision, **root_kwargs
        )
        roots[repo_id] = Path(
            getattr(metadata[repo_id], "root", None)
            or resolve_lerobot_dataset_root(repo_id)
        )
    validate_dataset_compatibility(
        metadata[cotrain.real.dataset_path], metadata[cotrain.sim.dataset_path]
    )
    for repo_id, meta in metadata.items():
        # Loading the full index is essential even for non-contiguous subsets.
        full_datasets[repo_id] = lerobot.LeRobotDataset(
            repo_id,
            root=roots[repo_id],
            revision=revision,
            delta_timestamps={
                key: [t / meta.fps for t in range(config.model.action_horizon)]
                for key in data_config.action_sequence_keys
            },
        )
    domains = []
    for spec in (cotrain.real, cotrain.sim):
        dataset = full_datasets[spec.dataset_path]
        episodes = spec.episodes
        if spec.episode_source is not None:
            episodes = episodes_for_source(
                roots[spec.dataset_path], spec.episode_source
            )
        if episodes is not None:
            dataset = FrameIndexDataset(
                dataset,
                frame_indices_for_episodes(
                    dataset.hf_dataset["episode_index"], episodes
                ),
            )
        if data_config.prompt_from_task:
            dataset = openpi_loader.TransformedDataset(
                dataset,
                [transforms.PromptFromLeRobotTask(metadata[spec.dataset_path].tasks)],
            )
        domains.append(openpi_loader.transform_dataset(dataset, data_config))
    dataset = MixtureDataset(*domains)
    sampler = StratifiedBatchSampler(
        len(dataset.real),
        len(dataset.sim),
        micro_batch_size,
        cotrain.sim_weight,
        rank=rank,
        world_size=world_size,
        seed=seed,
    )
    # Match upstream collation (including JAX arrays) but return torch tensors.
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config.num_workers,
        multiprocessing_context="spawn" if config.num_workers else None,
        collate_fn=_collate_openpi,
        worker_init_fn=openpi_loader._worker_init_fn,
        generator=torch.Generator().manual_seed(seed + rank),
    )
    return loader, data_config


def _collate_openpi(items: list[dict]) -> dict:
    import jax
    from openpi.training.data_loader import _collate_fn

    return jax.tree.map(torch.as_tensor, _collate_fn(items))


def domain_loss_metrics(output: Any, domain: Any) -> dict[str, float]:
    """Compute detached per-domain means when the model supplies sample losses."""
    if (
        domain is None
        or not isinstance(output, dict)
        or "per_sample_loss" not in output
    ):
        return {}
    losses = output["per_sample_loss"].detach()
    domain = torch.as_tensor(domain, device=losses.device)
    if losses.ndim != 1 or domain.shape != losses.shape:
        raise ValueError("per_sample_loss and domain must both have shape [B].")
    if not torch.all((domain == 0) | (domain == 1)):
        raise ValueError("domain must be 0 (real) or 1 (sim).")
    return {
        name: losses[domain == value].mean().item()
        for name, value in (("loss_real", 0), ("loss_sim", 1))
        if torch.any(domain == value)
    }
