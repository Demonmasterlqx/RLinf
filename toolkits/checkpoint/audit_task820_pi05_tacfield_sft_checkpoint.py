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

"""Audit an eight-rank Task820 Pi0.5 TacField SFT checkpoint."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from torch.distributed.checkpoint import FileSystemReader

_EXPECTED_METADATA = {
    "method": "sft_full_lora_tacfield",
    "model_family": "pi05",
    "base_model": "models/pi05_base_pytorch",
    "base_model_sha256": (
        "bf5d442a10c488298e3440a9ff3eead7d0e1e43335597588b00760693a39d621"
    ),
    "openpi_config_name": "pi05_lora_tacfield_realworld_replayed_task820",
    "deployment_config_name": "pi05_lora_tacfield_realworld_replayed_task820",
    "action_horizon": 50,
    "effective_action_dim": 13,
    "tactile_input": "tactile_marker_motion",
    "tactile_prefix_dim_in": 7920,
    "tactile_prefix_history": 8,
    "num_images_in_input": 2,
    "future_force_loss_weight": 0.01,
    "frozen_parameter_precision": "bf16",
    "trainable_parameter_precision": "fp32",
    "compute_precision": "bf16_amp",
    "export_precision": "bf16",
    "paligemma_lora_rank": 16,
    "action_expert_lora_rank": 32,
    "freeze_non_lora": True,
    "trainable_siglip": True,
    "siglip_lora": False,
    "tactile_tcn_initialization": "random",
    "model_weight_ema_decay": 0.99,
    "weight_variant": "ema",
    "target_global_step": 30_000,
}

_EXPECTED_PROFILES = {
    "firm": {
        "dataset": "datasets/realworld_replayed_task820_firm",
        "training_config": (
            "realworld_replayed_task820_firm_pi05_tacfield_sft_8gpu_30k"
        ),
    },
    "firm_mixed": {
        "dataset": "datasets/realworld_replay_task820_firm_mixed",
        "training_config": (
            "realworld_replay_task820_firm_mixed_pi05_tacfield_sft_8gpu_30k"
        ),
    },
}

_TRAINABLE_PREFIXES = {
    "vision_tower": (
        "paligemma_with_expert.paligemma.base_model.model.model.vision_tower."
    ),
    "action_in_proj": "action_in_proj.",
    "time_mlp_in": "time_mlp_in.",
    "time_mlp_out": "time_mlp_out.",
    "action_out_proj": "action_out_proj.",
    "tactile_prefix_encoder": "tactile_prefix_encoder.",
}

_DCP_PREFIXES = {
    "model": "fsdp_checkpoint.model.",
    "optimizer": "fsdp_checkpoint.optimizers.",
    "scheduler": "fsdp_checkpoint.lr_schedulers.",
    "rng": "fsdp_checkpoint.rng.",
}


def audit_checkpoint(
    step_dir: Path,
    expected_step: int,
    expected_final: bool,
    expected_target_step: int = 30_000,
    profile: str = "firm",
) -> dict:
    """Validate DCP state, data progress, and trainable-weight sidecar."""
    match = re.fullmatch(r"global_step_(\d+)", step_dir.name)
    if match is None or int(match.group(1)) != expected_step:
        raise ValueError(f"Expected global_step_{expected_step}, got {step_dir}")

    actor_dir = step_dir / "actor"
    dcp_dir = actor_dir / "dcp_checkpoint"
    data_state_path = actor_dir / "data_state.json"
    sidecar_path = actor_dir / "model_state_dict" / "trainable_weights.pt"
    required_files = [dcp_dir / ".metadata", data_state_path, sidecar_path]
    for path in required_files:
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"Required checkpoint file is missing or empty: {path}")

    shards = sorted(dcp_dir.glob("*.distcp"))
    if len(shards) != 8 or any(path.stat().st_size <= 0 for path in shards):
        raise ValueError(f"Expected 8 non-empty DCP shards, got {len(shards)}")

    ema_dir = actor_dir / "model_weight_ema"
    ema_shards = sorted(ema_dir.glob("rank_*.pt"))
    expected_ema_names = [f"rank_{rank:05d}.pt" for rank in range(8)]
    if [path.name for path in ema_shards] != expected_ema_names:
        raise ValueError(
            f"Expected EMA shards {expected_ema_names}, "
            f"got {[path.name for path in ema_shards]}"
        )
    if any(path.stat().st_size <= 0 for path in ema_shards):
        raise ValueError("One or more EMA shards are empty")

    dcp_keys = set(FileSystemReader(dcp_dir).read_metadata().state_dict_metadata)
    if "fsdp_checkpoint.fsdp_version" not in dcp_keys:
        raise ValueError("DCP metadata is missing the FSDP version entry")
    dcp_coverage = {
        group: sum(key.startswith(prefix) for key in dcp_keys)
        for group, prefix in _DCP_PREFIXES.items()
    }
    missing_dcp = sorted(group for group, count in dcp_coverage.items() if count == 0)
    if missing_dcp:
        raise ValueError(f"DCP metadata is missing state groups: {missing_dcp}")

    data_state = json.loads(data_state_path.read_text())
    for key in ("data_epoch", "data_iter_offset"):
        value = data_state.get(key)
        if type(value) is not int or value < 0:
            raise ValueError(f"data state {key} must be a non-negative integer")

    checkpoint = torch.load(sidecar_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model")
    metadata = checkpoint.get("metadata")
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint model state is missing or empty")
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata is missing")

    expected_metadata = dict(_EXPECTED_METADATA)
    expected_metadata.update(_EXPECTED_PROFILES[profile])
    expected_metadata["target_global_step"] = expected_target_step
    for key, expected in expected_metadata.items():
        actual = metadata.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"metadata {key} must be {expected!r}, got {actual!r}")
    for key in ("step", "global_step"):
        if metadata.get(key) != expected_step:
            raise ValueError(f"metadata {key} must be {expected_step}")
    if metadata.get("world_size") != 8:
        raise ValueError("trainable sidecar world_size must be 8")
    if metadata.get("is_final") is not expected_final:
        raise ValueError(f"metadata is_final must be {expected_final}")
    if metadata.get("parameter_count") != len(state):
        raise ValueError("metadata parameter_count does not match model state")

    names = sorted(state)
    coverage = {
        group: sum(name.startswith(prefix) for name in names)
        for group, prefix in _TRAINABLE_PREFIXES.items()
    }
    coverage["paligemma_lora"] = sum(
        name.startswith("paligemma_with_expert.paligemma") and "lora_" in name
        for name in names
    )
    coverage["action_expert_lora"] = sum(
        name.startswith("paligemma_with_expert.gemma_expert.model") and "lora_" in name
        for name in names
    )
    missing_trainable = sorted(group for group, count in coverage.items() if count == 0)
    if missing_trainable:
        raise ValueError(f"Trainable groups are missing: {missing_trainable}")

    dtypes = sorted({str(tensor.dtype) for tensor in state.values()})
    if dtypes != ["torch.float32"]:
        raise ValueError(f"Trainable sidecar must be FP32, got {dtypes}")
    for name, tensor in state.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite tensor in checkpoint: {name}")

    ema_tensor_count = 0
    ema_parameter_numel = 0
    ema_dtypes: set[str] = set()
    for expected_rank, path in enumerate(ema_shards):
        ema_checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if ema_checkpoint.get("rank") != expected_rank:
            raise ValueError(f"EMA shard {path.name} has the wrong rank")
        if ema_checkpoint.get("world_size") != 8:
            raise ValueError(f"EMA shard {path.name} has the wrong world size")
        ema_state = ema_checkpoint.get("state")
        if not isinstance(ema_state, dict):
            raise ValueError(f"EMA shard {path.name} has no state")
        if ema_state.get("decay") != 0.99:
            raise ValueError(f"EMA shard {path.name} has the wrong decay")
        if ema_state.get("num_updates") != expected_step:
            raise ValueError(f"EMA shard {path.name} has the wrong update count")
        parameter_names = ema_state.get("parameter_names")
        shadows = ema_state.get("shadows")
        if (
            not isinstance(parameter_names, list)
            or not isinstance(shadows, list)
            or not parameter_names
            or len(parameter_names) != len(shadows)
        ):
            raise ValueError(f"EMA shard {path.name} has invalid parameter state")
        for name, tensor in zip(parameter_names, shadows, strict=True):
            if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                raise ValueError(f"EMA shard {path.name} has invalid entries")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Non-finite EMA tensor in {path.name}: {name}")
            ema_tensor_count += 1
            ema_parameter_numel += tensor.numel()
            ema_dtypes.add(str(tensor.dtype))
    if ema_parameter_numel != sum(tensor.numel() for tensor in state.values()):
        raise ValueError("EMA shard parameter count does not match sidecar model state")
    if ema_dtypes != {"torch.float32"}:
        raise ValueError(f"EMA shards must be FP32, got {sorted(ema_dtypes)}")

    return {
        "status": "complete_and_audited",
        "step_dir": str(step_dir),
        "expected_step": expected_step,
        "is_final": expected_final,
        "dcp_shard_count": len(shards),
        "dcp_coverage": dcp_coverage,
        "ema_shard_count": len(ema_shards),
        "ema_tensor_count": ema_tensor_count,
        "ema_parameter_numel": ema_parameter_numel,
        "ema_dtypes": sorted(ema_dtypes),
        "data_state": data_state,
        "trainable_tensor_count": len(state),
        "trainable_parameter_numel": sum(tensor.numel() for tensor in state.values()),
        "trainable_dtypes": dtypes,
        "trainable_coverage": coverage,
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-dir", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--expect-final", action="store_true")
    parser.add_argument("--expected-target-step", type=int, default=30_000)
    parser.add_argument("--profile", choices=sorted(_EXPECTED_PROFILES), default="firm")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_checkpoint(
        args.step_dir.resolve(),
        args.expected_step,
        args.expect_final,
        args.expected_target_step,
        args.profile,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
