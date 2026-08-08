#!/usr/bin/env python3
"""Audit one RLinf Tabero Firm selective-trainable SFT sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

_EXPECTED_METADATA = {
    "format": "trainable_weights",
    "method": "sft_full_lora_tacfield",
    "dataset": "datas/tabero_firm",
    "training_precision": "fp32",
    "frozen_parameter_precision": "bf16",
    "trainable_parameter_precision": "fp32",
    "compute_precision": "bf16_amp",
    "export_precision": "bf16",
    "paligemma_lora_rank": 16,
    "action_expert_lora_rank": 32,
    "freeze_non_lora": True,
    "trainable_siglip": True,
    "siglip_lora": False,
    "training_config": "tabero_firm_sft_2gpu_selective_siglip_20k",
    "target_global_step": 20000,
}

_REQUIRED_PREFIXES = {
    "vision_tower": "paligemma_with_expert.paligemma.base_model.model.model.vision_tower.",
    "state_proj": "state_proj.",
    "action_in_proj": "action_in_proj.",
    "action_time_mlp_in": "action_time_mlp_in.",
    "action_time_mlp_out": "action_time_mlp_out.",
    "action_out_proj": "action_out_proj.",
    "tactile_prefix_encoder": "tactile_prefix_encoder.",
}


def audit_checkpoint(
    checkpoint_path: Path,
    expected_step: int,
    expected_final: bool,
    expected_target_step: int = 20000,
) -> dict:
    """Validate metadata, finiteness, dtype, and trainable module coverage."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model")
    metadata = checkpoint.get("metadata")
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint model state is missing or empty.")
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata is missing.")

    expected_metadata = dict(_EXPECTED_METADATA)
    expected_metadata["target_global_step"] = expected_target_step
    for key, expected in expected_metadata.items():
        actual = metadata.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"metadata {key} must be {expected!r}, got {actual!r}")
    for key in ("step", "global_step"):
        if metadata.get(key) != expected_step:
            raise ValueError(
                f"metadata {key} must be {expected_step}, got {metadata.get(key)!r}"
            )
    if metadata.get("is_final") is not expected_final:
        raise ValueError(
            f"metadata is_final must be {expected_final}, "
            f"got {metadata.get('is_final')!r}"
        )
    if metadata.get("parameter_count") != len(state):
        raise ValueError("metadata parameter_count does not match the model state")

    names = sorted(state)
    coverage = {
        group: sum(name.startswith(prefix) for name in names)
        for group, prefix in _REQUIRED_PREFIXES.items()
    }
    coverage["paligemma_lora"] = sum(
        name.startswith("paligemma_with_expert.paligemma") and "lora_" in name
        for name in names
    )
    coverage["action_expert_lora"] = sum(
        name.startswith("paligemma_with_expert.gemma_expert.model") and "lora_" in name
        for name in names
    )
    missing = sorted(group for group, count in coverage.items() if count <= 0)
    if missing:
        raise ValueError(f"Required trainable groups are missing: {missing}")

    dtypes = sorted({str(tensor.dtype) for tensor in state.values()})
    if dtypes != ["torch.float32"]:
        raise ValueError(f"Trainable sidecar must be FP32, got {dtypes}")
    for name, tensor in state.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite tensor in checkpoint: {name}")

    return {
        "checkpoint": str(checkpoint_path),
        "expected_step": expected_step,
        "is_final": expected_final,
        "tensor_count": len(state),
        "parameter_numel": sum(tensor.numel() for tensor in state.values()),
        "dtypes": dtypes,
        "all_finite": True,
        "coverage": coverage,
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--expect-final", action="store_true")
    parser.add_argument("--expected-target-step", type=int, default=20000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_checkpoint(
        args.checkpoint.resolve(),
        args.expected_step,
        args.expect_final,
        args.expected_target_step,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
