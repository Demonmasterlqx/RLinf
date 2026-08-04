#!/usr/bin/env python3
"""Audit the two trainable sidecars from the Tabero Firm SFT smoke run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def _group(name: str) -> str:
    if name.startswith("tactile_prefix_encoder."):
        return "tcn"
    if name.startswith("paligemma_with_expert.paligemma") and "lora_" in name:
        return "vlm_lora"
    if name.startswith("paligemma_with_expert.gemma_expert.model") and "lora_" in name:
        return "action_expert_lora"
    return "unexpected"


def audit(
    step1_path: Path,
    step2_path: Path,
    expected_steps: tuple[int, int] = (1, 2),
) -> dict:
    checkpoints = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in (step1_path, step2_path)
    ]
    states = [checkpoint["model"] for checkpoint in checkpoints]
    if set(states[0]) != set(states[1]):
        raise ValueError("Step 1 and step 2 sidecar keys differ.")

    groups = {}
    for name in sorted(states[0]):
        group = _group(name)
        entry = groups.setdefault(
            group,
            {
                "tensor_count": 0,
                "changed_tensor_count": 0,
                "max_abs_change": 0.0,
            },
        )
        entry["tensor_count"] += 1
        before = states[0][name]
        after = states[1][name]
        if not torch.isfinite(before).all() or not torch.isfinite(after).all():
            raise ValueError(f"Non-finite sidecar tensor: {name}")
        max_change = float((after - before).abs().max())
        if max_change > 0:
            entry["changed_tensor_count"] += 1
        entry["max_abs_change"] = max(entry["max_abs_change"], max_change)

    required = ("vlm_lora", "action_expert_lora", "tcn")
    if "unexpected" in groups:
        raise ValueError(f"Frozen/non-trainable keys leaked into sidecar: {groups}")
    for group in required:
        if groups.get(group, {}).get("changed_tensor_count", 0) <= 0:
            raise ValueError(f"No nonzero step1-to-step2 change in {group}.")

    metadata = [checkpoint["metadata"] for checkpoint in checkpoints]
    if [item["global_step"] for item in metadata] != list(expected_steps):
        raise ValueError(f"Unexpected checkpoint steps: {metadata}")
    if [item["is_final"] for item in metadata] != [False, True]:
        raise ValueError(f"Unexpected finality flags: {metadata}")
    if any(item["parameter_count"] != len(states[0]) for item in metadata):
        raise ValueError("Checkpoint parameter_count does not match sidecar tensors.")

    result = {
        "step1": str(step1_path),
        "step2": str(step2_path),
        "tensor_count": len(states[0]),
        "dtypes": sorted(
            {str(value.dtype) for state in states for value in state.values()}
        ),
        "all_finite": True,
        "groups": groups,
        "metadata": metadata,
    }
    if not all(math.isfinite(item["max_abs_change"]) for item in groups.values()):
        raise ValueError("Non-finite parameter delta audit.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step1", type=Path, required=True)
    parser.add_argument("--step2", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, nargs=2, default=(1, 2))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.step1.resolve(),
        args.step2.resolve(),
        expected_steps=tuple(args.expected_steps),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
