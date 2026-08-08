#!/usr/bin/env python3
"""Audit a complete RLinf Tabero Firm FSDP checkpoint directory."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from torch.distributed.checkpoint import FileSystemReader

try:
    from toolkits.checkpoint.audit_tabero_firm_sft_checkpoint import audit_checkpoint
except ModuleNotFoundError:  # Support direct execution by absolute script path.
    from audit_tabero_firm_sft_checkpoint import audit_checkpoint

_DCP_REQUIRED_PREFIXES = {
    "model": "fsdp_checkpoint.model.",
    "optimizer": "fsdp_checkpoint.optimizers.",
    "scheduler": "fsdp_checkpoint.lr_schedulers.",
    "rng": "fsdp_checkpoint.rng.",
}


def _read_dcp_state_keys(dcp_dir: Path) -> set[str]:
    metadata = FileSystemReader(dcp_dir).read_metadata()
    return set(metadata.state_dict_metadata)


def audit_checkpoint_dir(
    step_dir: Path,
    expected_step: int,
    expected_final: bool,
    expected_target_step: int = 20000,
) -> dict:
    """Validate DCP, data state, and the FP32 trainable sidecar together."""
    match = re.fullmatch(r"global_step_(\d+)", step_dir.name)
    if match is None or int(match.group(1)) != expected_step:
        raise ValueError(
            f"Checkpoint directory must end in global_step_{expected_step}: {step_dir}"
        )

    actor_dir = step_dir / "actor"
    dcp_dir = actor_dir / "dcp_checkpoint"
    metadata_path = dcp_dir / ".metadata"
    data_state_path = actor_dir / "data_state.json"
    sidecar_path = actor_dir / "model_state_dict" / "trainable_weights.pt"
    for path in (metadata_path, data_state_path, sidecar_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"Required checkpoint file is missing or empty: {path}")

    shards = sorted(dcp_dir.glob("*.distcp"))
    if len(shards) != 2 or any(path.stat().st_size <= 0 for path in shards):
        raise ValueError(
            f"Expected two non-empty DCP shards for world size 2, got {len(shards)}"
        )

    dcp_keys = _read_dcp_state_keys(dcp_dir)
    if "fsdp_checkpoint.fsdp_version" not in dcp_keys:
        raise ValueError("DCP metadata is missing the FSDP version entry.")
    dcp_coverage = {
        group: sum(key.startswith(prefix) for key in dcp_keys)
        for group, prefix in _DCP_REQUIRED_PREFIXES.items()
    }
    missing = sorted(group for group, count in dcp_coverage.items() if count <= 0)
    if missing:
        raise ValueError(f"DCP metadata is missing required state groups: {missing}")

    data_state = json.loads(data_state_path.read_text())
    for key in ("data_epoch", "data_iter_offset"):
        value = data_state.get(key)
        if type(value) is not int or value < 0:
            raise ValueError(f"data state {key} must be a non-negative integer")

    sidecar = audit_checkpoint(
        sidecar_path, expected_step, expected_final, expected_target_step
    )
    files = [metadata_path, *shards, data_state_path, sidecar_path]
    return {
        "status": "complete_and_audited",
        "step_dir": str(step_dir),
        "expected_step": expected_step,
        "is_final": expected_final,
        "total_bytes": sum(path.stat().st_size for path in files),
        "dcp": {
            "metadata_entries": len(dcp_keys),
            "shard_count": len(shards),
            "shard_bytes": [path.stat().st_size for path in shards],
            "coverage": dcp_coverage,
        },
        "data_state": data_state,
        "trainable_sidecar": sidecar,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-dir", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--expect-final", action="store_true")
    parser.add_argument("--expected-target-step", type=int, default=20000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_checkpoint_dir(
        args.step_dir.resolve(),
        args.expected_step,
        args.expect_final,
        args.expected_target_step,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
