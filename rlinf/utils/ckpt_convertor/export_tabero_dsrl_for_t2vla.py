# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Export a final Tabero DSRL-SAC actor bundle for T2-VLA inference."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file

from rlinf.utils.dsrl_checkpoint import (
    DSRL_TRAINABLE_MANIFEST_V1,
    DSRL_TRAINABLE_PARAMETER_COUNT,
    DSRL_TRAINABLE_TENSOR_COUNT,
)
from rlinf.utils.dsrl_rollout_sync import (
    DSRL_ROLLOUT_SYNC_MANIFEST_V1,
    DSRL_ROLLOUT_SYNC_MANIFEST_VERSION,
    DSRL_ROLLOUT_SYNC_PARAMETER_COUNT,
    DSRL_ROLLOUT_SYNC_PREFIXES,
    DSRL_ROLLOUT_SYNC_TENSOR_COUNT,
    validate_dsrl_rollout_state_dict,
)

FORMAT = "tabero_dsrl_t2vla"
FORMAT_VERSION = 1
FINAL_GLOBAL_STEP = 50
ACTOR_WEIGHTS_NAME = "dsrl_actor.safetensors"
MANIFEST_NAME = "manifest.json"
AUDIT_NAME = "artifact_audit.json"
PROVENANCE_KEYS = {
    "TABERO_PROVENANCE_VERSION",
    "TABERO_PROVENANCE_MODE",
    "TABERO_CONFIG_SHA256",
    "TABERO_CONFIG_SNAPSHOT_SHA256",
    "TABERO_GIT_COMMIT",
    "TABERO_GIT_DIRTY",
    "TABERO_BASE_MODEL_PATH",
    "TABERO_BASE_MODEL_SHA256",
    "TABERO_SOURCE_CONFIG_SHA256",
}


def checkpoint_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of one checkpoint file."""
    checkpoint = Path(path)
    digest = hashlib.sha256()
    with checkpoint.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, label: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a 64-character SHA-256 digest")
    return normalized


def _require_strict_int(value: Any, expected: int, label: str) -> None:
    if type(value) is not int or value != expected:
        raise ValueError(
            f"final DSRL checkpoint {label} must be {expected}; got {value!r}"
        )


def _load_provenance(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"formal provenance does not exist: {path}")
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or not value:
            raise ValueError(f"invalid formal provenance line: {line!r}")
        if key in values:
            raise ValueError(f"duplicate formal provenance key: {key}")
        values[key] = value
    if set(values) != PROVENANCE_KEYS:
        raise ValueError(
            "formal provenance keyspace mismatch; "
            f"missing={sorted(PROVENANCE_KEYS - set(values))}; "
            f"unexpected={sorted(set(values) - PROVENANCE_KEYS)}"
        )
    if values["TABERO_PROVENANCE_VERSION"] != "1":
        raise ValueError("unsupported formal provenance version")
    if values["TABERO_PROVENANCE_MODE"] not in {"fresh", "legacy_migration"}:
        raise ValueError("unsupported formal provenance mode")
    for key, label in (
        ("TABERO_CONFIG_SHA256", "formal config SHA-256"),
        ("TABERO_CONFIG_SNAPSHOT_SHA256", "formal config snapshot SHA-256"),
        ("TABERO_BASE_MODEL_SHA256", "formal base model SHA-256"),
    ):
        values[key] = _require_sha256(values[key], label)
    source_hash = values["TABERO_SOURCE_CONFIG_SHA256"]
    if values["TABERO_PROVENANCE_MODE"] == "fresh":
        if source_hash != "none":
            raise ValueError(
                "fresh provenance source config SHA-256 must be literal 'none'"
            )
    else:
        values["TABERO_SOURCE_CONFIG_SHA256"] = _require_sha256(
            source_hash,
            "legacy source config SHA-256",
        )
    return values


def _validate_checkpoint_path(checkpoint: Path) -> Path:
    checkpoint = checkpoint.resolve()
    expected_suffix = (
        "global_step_50",
        "actor",
        "model_state_dict",
        "trainable_weights.pt",
    )
    actual_suffix = tuple(checkpoint.parts[-4:])
    if actual_suffix != expected_suffix:
        raise ValueError(
            "final DSRL checkpoint path must end with "
            "global_step_50/actor/model_state_dict/trainable_weights.pt; "
            f"got {checkpoint}"
        )
    if not checkpoint.is_file():
        raise ValueError(f"final DSRL checkpoint does not exist: {checkpoint}")
    return checkpoint


def _validate_metadata(metadata: Any, task_id: int) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ValueError("final DSRL checkpoint metadata must be a mapping")
    expected_config = (
        f"isaaclab_pi0_dsrl_tacfield_tabero_task{task_id}_firm_8gpu_50step"
    )
    if metadata.get("format") != "trainable_weights":
        raise ValueError("final DSRL checkpoint format must be trainable_weights")
    if metadata.get("method") != "dsrl":
        raise ValueError("final DSRL checkpoint method must be dsrl")
    _require_strict_int(metadata.get("task_id"), task_id, "task_id")
    if metadata.get("training_config") != expected_config:
        raise ValueError(
            f"final DSRL checkpoint training_config must be {expected_config!r}"
        )
    _require_strict_int(metadata.get("step"), FINAL_GLOBAL_STEP, "step")
    _require_strict_int(metadata.get("global_step"), FINAL_GLOBAL_STEP, "global_step")
    _require_strict_int(
        metadata.get("target_global_step"),
        FINAL_GLOBAL_STEP,
        "target_global_step",
    )
    if metadata.get("is_final") is not True:
        raise ValueError("final DSRL checkpoint is_final must be true")
    _require_strict_int(metadata.get("rank"), 0, "rank")
    _require_strict_int(metadata.get("world_size"), 4, "world_size")
    _require_strict_int(
        metadata.get("parameter_count"),
        DSRL_TRAINABLE_TENSOR_COUNT,
        "parameter_count",
    )
    _require_strict_int(
        metadata.get("tensor_count"),
        DSRL_TRAINABLE_TENSOR_COUNT,
        "tensor_count",
    )
    _require_strict_int(
        metadata.get("total_parameter_count"),
        DSRL_TRAINABLE_PARAMETER_COUNT,
        "total_parameter_count",
    )
    return dict(metadata)


def _validate_trainable_state(state: Any) -> dict[str, torch.Tensor]:
    if not isinstance(state, Mapping):
        raise ValueError("final DSRL checkpoint model must be a tensor mapping")
    expected_keys = set(DSRL_TRAINABLE_MANIFEST_V1)
    actual_keys = set(state)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    non_tensors = sorted(
        key
        for key in expected_keys & actual_keys
        if not isinstance(state[key], torch.Tensor)
    )
    shape_mismatches = {
        key: {
            "expected": DSRL_TRAINABLE_MANIFEST_V1[key],
            "actual": tuple(state[key].shape),
        }
        for key in expected_keys & actual_keys
        if isinstance(state[key], torch.Tensor)
        and tuple(state[key].shape) != DSRL_TRAINABLE_MANIFEST_V1[key]
    }
    dtype_mismatches = {
        key: str(state[key].dtype)
        for key in expected_keys & actual_keys
        if isinstance(state[key], torch.Tensor) and state[key].dtype != torch.bfloat16
    }
    nonfinite = sorted(
        key
        for key in expected_keys & actual_keys
        if isinstance(state[key], torch.Tensor)
        and state[key].is_floating_point()
        and not torch.isfinite(state[key]).all().item()
    )
    if missing or unexpected or non_tensors or shape_mismatches:
        raise ValueError(
            "final DSRL trainable tensor manifest mismatch; "
            f"missing={missing}; unexpected={unexpected}; non_tensors={non_tensors}; "
            f"shape_mismatches={shape_mismatches}"
        )
    if dtype_mismatches:
        raise ValueError(
            "final DSRL trainable tensor dtype mismatch; expected bfloat16; "
            f"actual={dtype_mismatches}"
        )
    if nonfinite:
        raise ValueError(
            f"final DSRL trainable tensors must be finite; keys={nonfinite}"
        )
    return {key: state[key] for key in DSRL_TRAINABLE_MANIFEST_V1}


def _validate_config_snapshot(
    config_snapshot: Path,
    base_model: Path,
    metadata: Mapping[str, Any],
) -> None:
    try:
        config = OmegaConf.load(config_snapshot)
    except Exception as error:
        raise ValueError(
            f"formal config snapshot could not be loaded: {config_snapshot}"
        ) from error

    def require_value(path: str, expected: Any) -> None:
        actual = OmegaConf.select(config, path, default=None)
        if type(expected) in {bool, int}:
            valid = type(actual) is type(expected) and actual == expected
        else:
            valid = actual == expected
        if not valid:
            raise ValueError(
                f"formal config snapshot {path} must be {expected!r}; got {actual!r}"
            )

    task_id = metadata["task_id"]
    require_value("env.train.init_params.task_id", task_id)
    require_value("runner.max_epochs", FINAL_GLOBAL_STEP)
    require_value("runner.save_interval", 10)
    require_value("env.train.total_num_envs", 84)
    require_value("env.train.rollout_epoch", 2)
    require_value("algorithm.update_epoch", 200)
    require_value("algorithm.gamma", 0.999)
    require_value("algorithm.tau", 0.005)
    require_value("actor.model.openpi.use_dsrl", True)
    require_value("actor.model.openpi.dsrl_use_tactile", True)
    require_value("actor.model.openpi.dsrl_state_dim", 7)
    require_value("actor.model.openpi.dsrl_action_noise_dim", 32)

    logger_backends = OmegaConf.select(
        config,
        "runner.logger.logger_backends",
        default=None,
    )
    if logger_backends is None or list(logger_backends) != ["tensorboard", "wandb"]:
        raise ValueError(
            "formal config snapshot runner.logger.logger_backends must contain "
            "exactly ['tensorboard', 'wandb']"
        )

    rollout_sync_prefixes = OmegaConf.select(
        config,
        "actor.rollout_sync_prefixes",
        default=None,
    )
    if rollout_sync_prefixes is None or (
        tuple(rollout_sync_prefixes) != DSRL_ROLLOUT_SYNC_PREFIXES
    ):
        raise ValueError(
            "formal config snapshot actor.rollout_sync_prefixes must contain exactly "
            f"{list(DSRL_ROLLOUT_SYNC_PREFIXES)} in this order"
        )

    for config_path in ("actor.model.model_path", "rollout.model.model_path"):
        configured_base_model = OmegaConf.select(
            config,
            config_path,
            default=None,
        )
        if not isinstance(configured_base_model, str) or (
            Path(configured_base_model).resolve() != base_model
        ):
            raise ValueError(
                f"formal config snapshot {config_path} must reference base model "
                f"{base_model}; got {configured_base_model!r}"
            )

    trainable_metadata = {
        "method": "dsrl",
        "task_id": task_id,
        "training_config": metadata["training_config"],
        "target_global_step": metadata["target_global_step"],
    }
    for key, expected in trainable_metadata.items():
        require_value(
            f"actor.fsdp_config.trainable_checkpoint_metadata.{key}",
            expected,
        )


def _validate_provenance(
    checkpoint: Path,
    base_model: Path,
    actual_base_hash: str,
    metadata: Mapping[str, Any],
) -> tuple[Path, Path, Path | None, dict[str, str]]:
    output_root = checkpoint.parents[5]
    provenance_path = output_root / "provenance.env"
    config_snapshot = output_root / "config_snapshot.yaml"
    provenance = _load_provenance(provenance_path)
    if not config_snapshot.is_file():
        raise ValueError(f"formal config snapshot does not exist: {config_snapshot}")
    snapshot_hash = checkpoint_sha256(config_snapshot)
    if (
        snapshot_hash != provenance["TABERO_CONFIG_SHA256"]
        or snapshot_hash != (provenance["TABERO_CONFIG_SNAPSHOT_SHA256"])
    ):
        raise ValueError("formal config snapshot SHA-256 does not match provenance")
    if Path(provenance["TABERO_BASE_MODEL_PATH"]).resolve() != base_model:
        raise ValueError("base model path does not match formal provenance")
    if provenance["TABERO_BASE_MODEL_SHA256"] != actual_base_hash:
        raise ValueError("base model SHA-256 does not match formal provenance")
    _validate_config_snapshot(config_snapshot, base_model, metadata)

    legacy_source_config = None
    if provenance["TABERO_PROVENANCE_MODE"] == "legacy_migration":
        legacy_source_config = output_root / "tensorboard" / "config.yaml"
        if not legacy_source_config.is_file():
            raise ValueError(
                f"legacy source config does not exist: {legacy_source_config}"
            )
        if (
            checkpoint_sha256(legacy_source_config)
            != provenance["TABERO_SOURCE_CONFIG_SHA256"]
        ):
            raise ValueError("legacy source config SHA-256 does not match provenance")
    return provenance_path, config_snapshot, legacy_source_config, provenance


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def export_tabero_dsrl_bundle(
    *,
    trainable_checkpoint: str | Path,
    output_dir: str | Path,
    base_model: str | Path,
    expected_base_model_sha256: str,
    task_id: int,
) -> dict[str, Any]:
    """Validate and export one final Task 0/5 DSRL actor bundle."""
    if type(task_id) is not int or task_id not in {0, 5}:
        raise ValueError(f"task_id must be exactly 0 or 5; got {task_id!r}")
    checkpoint = _validate_checkpoint_path(Path(trainable_checkpoint))
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"DSRL bundle output already exists: {output_dir}")
    base_model = Path(base_model).resolve()
    base_weights = base_model / "model.safetensors"
    if not base_weights.is_file():
        raise ValueError(f"base model weights do not exist: {base_weights}")
    expected_base_hash = _require_sha256(
        expected_base_model_sha256,
        "expected base model SHA-256",
    )
    actual_base_hash = checkpoint_sha256(base_weights)
    if actual_base_hash != expected_base_hash:
        raise ValueError(
            "base model SHA-256 does not match expected digest; "
            f"expected={expected_base_hash}, actual={actual_base_hash}"
        )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or set(payload) != {"model", "metadata"}:
        raise ValueError("final DSRL sidecar must contain exactly model and metadata")
    metadata = _validate_metadata(payload["metadata"], task_id)
    trainable_state = _validate_trainable_state(payload["model"])
    actor_state = {
        key: trainable_state[key].detach().cpu().contiguous()
        for key in DSRL_ROLLOUT_SYNC_MANIFEST_V1
    }
    validate_dsrl_rollout_state_dict(actor_state)
    provenance_path, config_snapshot, legacy_source_config, provenance = (
        _validate_provenance(
            checkpoint,
            base_model,
            actual_base_hash,
            metadata,
        )
    )

    checkpoint_hash = checkpoint_sha256(checkpoint)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        actor_path = temporary_dir / ACTOR_WEIGHTS_NAME
        save_file(
            actor_state,
            actor_path,
            metadata={
                "format": FORMAT,
                "format_version": str(FORMAT_VERSION),
                "task_id": str(task_id),
                "global_step": str(FINAL_GLOBAL_STEP),
                "dtype": "bfloat16",
            },
        )
        saved_actor = load_file(actor_path, device="cpu")
        validate_dsrl_rollout_state_dict(saved_actor)
        if {tensor.dtype for tensor in saved_actor.values()} != {torch.bfloat16}:
            raise ValueError("saved DSRL actor dtype audit failed")
        if not all(
            torch.isfinite(tensor).all().item() for tensor in saved_actor.values()
        ):
            raise ValueError("saved DSRL actor finite audit failed")
        actor_hash = checkpoint_sha256(actor_path)

        manifest: dict[str, Any] = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "algorithm": "dsrl-sac",
            "task_id": task_id,
            "global_step": FINAL_GLOBAL_STEP,
            "is_final": True,
            "training_config": metadata["training_config"],
            "base_model": str(base_model),
            "base_model_sha256": actual_base_hash,
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_sha256": checkpoint_hash,
            "source_provenance": str(provenance_path),
            "source_provenance_sha256": checkpoint_sha256(provenance_path),
            "source_config_snapshot": str(config_snapshot),
            "source_config_snapshot_sha256": checkpoint_sha256(config_snapshot),
            "legacy_source_config": (
                str(legacy_source_config) if legacy_source_config is not None else None
            ),
            "legacy_source_config_sha256": (
                provenance["TABERO_SOURCE_CONFIG_SHA256"]
                if legacy_source_config is not None
                else None
            ),
            "source_git_commit": provenance["TABERO_GIT_COMMIT"],
            "actor_weights": ACTOR_WEIGHTS_NAME,
            "actor_weights_sha256": actor_hash,
            "actor_manifest_version": DSRL_ROLLOUT_SYNC_MANIFEST_VERSION,
            "actor_tensor_count": DSRL_ROLLOUT_SYNC_TENSOR_COUNT,
            "actor_parameter_count": DSRL_ROLLOUT_SYNC_PARAMETER_COUNT,
            "actor_dtype": "bfloat16",
            "observation_contract": {
                "image": {
                    "key": "dsrl_raw_image",
                    "shape": [256, 256, 3],
                    "layout": "HWC",
                    "dtype": "uint8",
                    "value_range": [0, 255],
                    "preprocessing": {
                        "resize": [64, 64],
                        "mode": "bilinear",
                        "align_corners": False,
                        "output_layout": "NCHW",
                        "normalization": "uint8_to_minus_one_one",
                    },
                },
                "state": {"key": "state", "shape": [7], "dtype": "float32"},
                "tactile": {
                    "key": "tactile_marker_motion",
                    "shape": [9, 198, 2],
                    "dtype": "float32",
                    "encoder_shape": [9, 396],
                },
            },
            "feature_contract": {
                "order": ["state", "image", "tactile"],
                "dims": [64, 64, 64],
                "total_dim": 192,
            },
            "noise_contract": {
                "dim": 32,
                "horizon": 50,
                "deterministic": "tanh(mean)",
                "broadcast_across_horizon": True,
                "pi0_denoise_steps": 10,
            },
            "architecture": {
                "image_size": 64,
                "state_dim": 7,
                "tactile_shape": [9, 198, 2],
                "hidden_dims": [128, 128, 128],
                "feature_dim": 192,
                "noise_dim": 32,
            },
            "artifact_audit": AUDIT_NAME,
        }
        manifest_path = temporary_dir / MANIFEST_NAME
        _write_json(manifest_path, manifest)
        audit = {
            "format": "tabero_dsrl_artifact_audit",
            "format_version": 1,
            "status": "passed",
            "task_id": task_id,
            "global_step": FINAL_GLOBAL_STEP,
            "source_checkpoint_sha256": checkpoint_hash,
            "base_model_sha256": actual_base_hash,
            "actor_weights_sha256": actor_hash,
            "manifest_sha256": checkpoint_sha256(manifest_path),
            "checks": {
                "final_checkpoint_path": True,
                "source_metadata": True,
                "source_trainable_manifest": True,
                "actor_manifest": True,
                "actor_dtype": True,
                "actor_finite": True,
                "base_model_sha256": True,
                "formal_provenance": True,
                "output_hashes": True,
            },
        }
        _write_json(temporary_dir / AUDIT_NAME, audit)
        if output_dir.exists():
            raise FileExistsError(f"DSRL bundle output already exists: {output_dir}")
        os.rename(temporary_dir, output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainable-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--expected-base-model-sha256", required=True)
    parser.add_argument("--task-id", type=int, choices=(0, 5), required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = export_tabero_dsrl_bundle(
        trainable_checkpoint=args.trainable_checkpoint,
        output_dir=args.output_dir,
        base_model=args.base_model,
        expected_base_model_sha256=args.expected_base_model_sha256,
        task_id=args.task_id,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
