# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""CPU-only validation and receipt helpers for Tabero DSRL official eval."""

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import signal
import socket
import stat
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from rlinf.utils.dsrl_observation import DSRL_OBSERVATION_SEMANTICS
from rlinf.utils.dsrl_reward import DSRL_REWARD_SEMANTICS
from rlinf.utils.dsrl_rollout_sync import (
    DSRL_ROLLOUT_SYNC_MANIFEST_V2,
    DSRL_ROLLOUT_SYNC_MANIFEST_VERSION,
    DSRL_ROLLOUT_SYNC_PARAMETER_COUNT,
    DSRL_ROLLOUT_SYNC_TENSOR_COUNT,
)
from rlinf.utils.tabero_dsrl_profiles import (
    FORMAL_8GPU_50STEP_PROFILE,
    TABERO_DSRL_TRAINING_PROFILE_CHOICES,
    TaberoDSRLTrainingProfile,
    resolve_tabero_dsrl_training_profile,
)

FORMAT = "tabero_dsrl_t2vla"
FORMAT_VERSION = 2
BASE_WEIGHTS_NAME = "model.safetensors"
ACTOR_NAME = "dsrl_actor.safetensors"
MANIFEST_NAME = "manifest.json"
AUDIT_NAME = "artifact_audit.json"
EXPECTED_AUDIT_CHECKS = {
    # The source path must match the checkpoint selected by the explicit profile.
    "final_checkpoint_path",
    "source_metadata",
    "source_trainable_manifest",
    "actor_manifest",
    "actor_dtype",
    "actor_finite",
    "base_model_sha256",
    "formal_provenance",
    "reward_semantics",
    "observation_semantics",
    "output_hashes",
}
EXPECTED_AUDIT_KEYS = {
    "format",
    "format_version",
    "status",
    "task_id",
    "global_step",
    "reward_semantics",
    "observation_semantics",
    "source_checkpoint_sha256",
    "base_model_sha256",
    "actor_weights_sha256",
    "manifest_sha256",
    "checks",
}
EXPECTED_MANIFEST_KEYS = {
    "format",
    "format_version",
    "algorithm",
    "reward_semantics",
    "observation_semantics",
    "task_id",
    "global_step",
    "is_final",
    "training_config",
    "base_model",
    "base_model_sha256",
    "source_checkpoint",
    "source_checkpoint_sha256",
    "source_provenance",
    "source_provenance_sha256",
    "source_config_snapshot",
    "source_config_snapshot_sha256",
    "legacy_source_config",
    "legacy_source_config_sha256",
    "source_git_commit",
    "actor_weights",
    "actor_weights_sha256",
    "actor_manifest_version",
    "actor_tensor_count",
    "actor_parameter_count",
    "actor_dtype",
    "observation_contract",
    "feature_contract",
    "noise_contract",
    "architecture",
    "artifact_audit",
}
EXPECTED_OBSERVATION_CONTRACT = {
    "main_image": {
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
    "wrist_image": {
        "key": "dsrl_raw_wrist_image",
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
}
EXPECTED_FEATURE_CONTRACT = {
    "order": ["state", "main_image", "wrist_image", "tactile"],
    "dims": [64, 64, 64, 64],
    "total_dim": 256,
}
EXPECTED_NOISE_CONTRACT = {
    "dim": 32,
    "horizon": 50,
    "deterministic": "tanh(mean)",
    "broadcast_across_horizon": True,
    "pi0_denoise_steps": 10,
}
EXPECTED_ARCHITECTURE = {
    "image_size": 64,
    "image_views": ["main", "wrist"],
    "shared_image_encoder": True,
    "per_view_image_dim": 64,
    "image_feature_dim": 128,
    "state_dim": 7,
    "tactile_shape": [9, 198, 2],
    "hidden_dims": [128, 128, 128],
    "feature_dim": 256,
    "noise_dim": 32,
}
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


def sha256_file(path: Path) -> str:
    """Hash one file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_exact(value: Any, expected: Any, label: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{label} must be {expected!r}; got {value!r}")


def _require_absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return Path(value).resolve()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not strict JSON: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON root must be an object")
    return payload


def _read_env(path: Path, label: str) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8: {path}") from error
    for line in lines:
        key, separator, value = line.partition("=")
        if not separator or not key or not value:
            raise ValueError(f"invalid {label} line: {line!r}")
        if key in values:
            raise ValueError(f"duplicate {label} key: {key}")
        values[key] = value
    return values


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(f"output already exists: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _validate_actor(
    actor_path: Path,
    task_id: int,
    profile: TaberoDSRLTrainingProfile,
) -> None:
    try:
        with safe_open(actor_path, framework="pt", device="cpu") as actor:
            metadata = actor.metadata()
            expected_metadata = {
                "format": FORMAT,
                "format_version": str(FORMAT_VERSION),
                "task_id": str(task_id),
                "global_step": str(profile.global_step),
                "dtype": "bfloat16",
                "reward_semantics": DSRL_REWARD_SEMANTICS,
                "observation_semantics": DSRL_OBSERVATION_SEMANTICS,
            }
            if metadata != expected_metadata:
                raise ValueError(
                    "actor safetensors metadata mismatch; "
                    f"expected={expected_metadata}, actual={metadata}"
                )
            actual_keys = set(actor.keys())
            expected_keys = set(DSRL_ROLLOUT_SYNC_MANIFEST_V2)
            if actual_keys != expected_keys:
                raise ValueError(
                    "actor tensor manifest keyspace mismatch; "
                    f"missing={sorted(expected_keys - actual_keys)}, "
                    f"unexpected={sorted(actual_keys - expected_keys)}"
                )
            for key, expected_shape in DSRL_ROLLOUT_SYNC_MANIFEST_V2.items():
                tensor = actor.get_slice(key)
                if tuple(tensor.get_shape()) != expected_shape:
                    raise ValueError(
                        f"actor tensor manifest shape mismatch for {key}: "
                        f"expected={expected_shape}, actual={tuple(tensor.get_shape())}"
                    )
                if tensor.get_dtype() != "BF16":
                    raise ValueError(
                        f"actor tensor manifest dtype mismatch for {key}: "
                        f"{tensor.get_dtype()}"
                    )
                if not torch.isfinite(actor.get_tensor(key)).all().item():
                    raise ValueError(f"actor tensor must be finite: {key}")
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("actor"):
            raise
        raise ValueError(f"actor safetensors validation failed: {error}") from error


def _validate_source_artifacts(
    manifest: Mapping[str, Any], base_model_path: Path, base_hash: str
) -> None:
    source_paths: dict[str, Path] = {}
    for path_key, hash_key, label in (
        ("source_checkpoint", "source_checkpoint_sha256", "source checkpoint"),
        ("source_provenance", "source_provenance_sha256", "source provenance"),
        (
            "source_config_snapshot",
            "source_config_snapshot_sha256",
            "source config snapshot",
        ),
    ):
        path = _require_absolute_path(manifest.get(path_key), label)
        if not path.is_file():
            raise ValueError(f"{label} is missing: {path}")
        expected_hash = _require_sha256(manifest.get(hash_key), f"{label} hash")
        if sha256_file(path) != expected_hash:
            raise ValueError(f"{label} hash mismatch")
        source_paths[path_key] = path

    legacy_path_value = manifest.get("legacy_source_config")
    legacy_hash_value = manifest.get("legacy_source_config_sha256")
    legacy_path: Path | None = None
    if legacy_path_value is not None:
        legacy_path = _require_absolute_path(legacy_path_value, "legacy source config")
        if not legacy_path.is_file():
            raise ValueError(f"legacy source config is missing: {legacy_path}")
        legacy_hash = _require_sha256(legacy_hash_value, "legacy source config hash")
        if sha256_file(legacy_path) != legacy_hash:
            raise ValueError("legacy source config hash mismatch")

    provenance = _read_env(source_paths["source_provenance"], "source provenance")
    if set(provenance) != PROVENANCE_KEYS:
        raise ValueError("source provenance keyspace mismatch")
    _require_exact(
        provenance["TABERO_PROVENANCE_VERSION"], "1", "source provenance version"
    )
    mode = provenance["TABERO_PROVENANCE_MODE"]
    if mode not in {"fresh", "legacy_migration"}:
        raise ValueError(f"source provenance mode is invalid: {mode}")
    config_hash = manifest["source_config_snapshot_sha256"]
    for key in ("TABERO_CONFIG_SHA256", "TABERO_CONFIG_SNAPSHOT_SHA256"):
        if _require_sha256(provenance[key], f"source provenance {key}") != config_hash:
            raise ValueError(f"source provenance {key} does not match config snapshot")
    if provenance["TABERO_GIT_COMMIT"] != manifest["source_git_commit"]:
        raise ValueError("source Git commit does not match provenance")
    if provenance["TABERO_GIT_DIRTY"] != "false":
        raise ValueError("source provenance Git dirty state must be false")
    if Path(provenance["TABERO_BASE_MODEL_PATH"]).resolve() != base_model_path:
        raise ValueError("source provenance base model path mismatch")
    if provenance["TABERO_BASE_MODEL_SHA256"] != base_hash:
        raise ValueError("source provenance base model hash mismatch")
    source_config_hash = provenance["TABERO_SOURCE_CONFIG_SHA256"]
    if mode == "fresh":
        if legacy_path is not None or source_config_hash != "none":
            raise ValueError("fresh source provenance must not reference legacy config")
    elif legacy_path is None or source_config_hash != legacy_hash_value:
        raise ValueError("legacy source provenance config hash mismatch")


def validate_bundle(
    bundle: str | Path,
    task_id: int,
    base_model: str | Path,
    *,
    training_profile: str = FORMAL_8GPU_50STEP_PROFILE,
) -> dict[str, Any]:
    """Strictly validate one allowlisted Task 0/5 audited DSRL bundle."""
    if type(task_id) is not int or task_id not in {0, 5}:
        raise ValueError(f"task_id must be exactly 0 or 5; got {task_id!r}")
    profile = resolve_tabero_dsrl_training_profile(training_profile, task_id)
    bundle_path = Path(bundle).resolve()
    if not bundle_path.is_dir():
        raise ValueError(f"DSRL bundle must be a directory: {bundle_path}")
    base_model_path = Path(base_model).resolve()
    base_weights = base_model_path / BASE_WEIGHTS_NAME
    if not base_weights.is_file():
        raise ValueError(f"fixed base model weights are missing: {base_weights}")
    actor_path = bundle_path / ACTOR_NAME
    manifest_path = bundle_path / MANIFEST_NAME
    audit_path = bundle_path / AUDIT_NAME
    if not actor_path.is_file():
        raise ValueError(f"DSRL actor is missing: {actor_path}")

    manifest = _read_json(manifest_path, "bundle manifest")
    audit = _read_json(audit_path, "bundle audit")
    if set(manifest) != EXPECTED_MANIFEST_KEYS:
        raise ValueError(
            "manifest keyspace mismatch; "
            f"missing={sorted(EXPECTED_MANIFEST_KEYS - set(manifest))}, "
            f"unexpected={sorted(set(manifest) - EXPECTED_MANIFEST_KEYS)}"
        )
    _require_exact(manifest.get("format"), FORMAT, "manifest format")
    _require_exact(
        manifest.get("format_version"), FORMAT_VERSION, "manifest format_version"
    )
    _require_exact(manifest.get("algorithm"), "dsrl-sac", "manifest algorithm")
    _require_exact(
        manifest.get("reward_semantics"),
        DSRL_REWARD_SEMANTICS,
        "manifest reward_semantics",
    )
    _require_exact(
        manifest.get("observation_semantics"),
        DSRL_OBSERVATION_SEMANTICS,
        "manifest observation_semantics",
    )
    _require_exact(manifest.get("task_id"), task_id, "manifest task_id")
    _require_exact(
        manifest.get("global_step"), profile.global_step, "manifest global_step"
    )
    _require_exact(
        manifest.get("is_final"),
        profile.is_final,
        "manifest is_final profile flag",
    )
    _require_exact(
        manifest.get("training_config"),
        profile.training_config(task_id),
        "manifest training_config",
    )
    _require_exact(manifest.get("actor_weights"), ACTOR_NAME, "manifest actor_weights")
    _require_exact(
        manifest.get("actor_manifest_version"),
        DSRL_ROLLOUT_SYNC_MANIFEST_VERSION,
        "manifest actor_manifest_version",
    )
    _require_exact(
        manifest.get("actor_tensor_count"),
        DSRL_ROLLOUT_SYNC_TENSOR_COUNT,
        "manifest actor_tensor_count",
    )
    _require_exact(
        manifest.get("actor_parameter_count"),
        DSRL_ROLLOUT_SYNC_PARAMETER_COUNT,
        "manifest actor_parameter_count",
    )
    _require_exact(manifest.get("actor_dtype"), "bfloat16", "manifest actor_dtype")
    _require_exact(manifest.get("artifact_audit"), AUDIT_NAME, "manifest audit name")
    _require_exact(
        manifest.get("observation_contract"),
        EXPECTED_OBSERVATION_CONTRACT,
        "manifest observation contract",
    )
    _require_exact(
        manifest.get("feature_contract"),
        EXPECTED_FEATURE_CONTRACT,
        "manifest feature contract",
    )
    _require_exact(
        manifest.get("noise_contract"),
        EXPECTED_NOISE_CONTRACT,
        "manifest noise contract",
    )
    _require_exact(
        manifest.get("architecture"),
        EXPECTED_ARCHITECTURE,
        "manifest architecture",
    )
    manifest_base = _require_absolute_path(manifest.get("base_model"), "base model")
    if manifest_base != base_model_path:
        raise ValueError(
            f"manifest base model path mismatch: {manifest_base} != {base_model_path}"
        )
    for path_key, label in (
        ("source_checkpoint", "source checkpoint"),
        ("source_provenance", "source provenance"),
        ("source_config_snapshot", "source config snapshot"),
    ):
        _require_absolute_path(manifest.get(path_key), label)
    for hash_key, label in (
        ("source_checkpoint_sha256", "source checkpoint hash"),
        ("source_provenance_sha256", "source provenance hash"),
        ("source_config_snapshot_sha256", "source config snapshot hash"),
    ):
        _require_sha256(manifest.get(hash_key), label)
    legacy_path = manifest.get("legacy_source_config")
    legacy_hash = manifest.get("legacy_source_config_sha256")
    if (legacy_path is None) != (legacy_hash is None):
        raise ValueError("legacy source config path and hash must both be null or set")
    if legacy_path is not None:
        _require_absolute_path(legacy_path, "legacy source config")
        _require_sha256(legacy_hash, "legacy source config hash")
    source_git_commit = manifest.get("source_git_commit")
    if (
        not isinstance(source_git_commit, str)
        or len(source_git_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_git_commit)
    ):
        raise ValueError("source Git commit must be a lowercase 40-character digest")

    actor_hash = sha256_file(actor_path)
    manifest_hash = sha256_file(manifest_path)
    audit_hash = sha256_file(audit_path)
    base_hash = sha256_file(base_weights)
    manifest_actor_hash = _require_sha256(
        manifest.get("actor_weights_sha256"), "manifest actor hash"
    )
    manifest_base_hash = _require_sha256(
        manifest.get("base_model_sha256"), "manifest base model hash"
    )
    if actor_hash != manifest_actor_hash:
        raise ValueError("actor hash does not match manifest")
    if base_hash != manifest_base_hash:
        raise ValueError("fixed base model hash does not match manifest")
    _validate_source_artifacts(manifest, base_model_path, base_hash)

    if set(audit) != EXPECTED_AUDIT_KEYS:
        raise ValueError("audit keyspace mismatch")
    _require_exact(audit.get("format"), "tabero_dsrl_artifact_audit", "audit format")
    _require_exact(audit.get("format_version"), 2, "audit format_version")
    _require_exact(audit.get("status"), "passed", "audit status")
    _require_exact(audit.get("task_id"), task_id, "audit task_id")
    _require_exact(audit.get("global_step"), profile.global_step, "audit global_step")
    _require_exact(
        audit.get("reward_semantics"),
        DSRL_REWARD_SEMANTICS,
        "audit reward_semantics",
    )
    _require_exact(
        audit.get("observation_semantics"),
        DSRL_OBSERVATION_SEMANTICS,
        "audit observation_semantics",
    )
    checks = audit.get("checks")
    if (
        not isinstance(checks, dict)
        or set(checks) != EXPECTED_AUDIT_CHECKS
        or not all(value is True for value in checks.values())
    ):
        raise ValueError("audit checks must contain the exact passed keyspace")
    for key, expected, label in (
        ("actor_weights_sha256", actor_hash, "audit actor hash"),
        ("manifest_sha256", manifest_hash, "audit manifest hash"),
        ("base_model_sha256", base_hash, "audit base model hash"),
    ):
        actual = _require_sha256(audit.get(key), label)
        if actual != expected:
            raise ValueError(f"{label} mismatch")
    audit_source_hash = _require_sha256(
        audit["source_checkpoint_sha256"], "audit source checkpoint hash"
    )
    if audit_source_hash != manifest.get("source_checkpoint_sha256"):
        raise ValueError("audit source checkpoint hash mismatch")

    _validate_actor(actor_path, task_id, profile)
    bundle_digest = hashlib.sha256(
        f"{actor_hash}\n{manifest_hash}\n{audit_hash}\n".encode()
    ).hexdigest()
    return {
        "bundle_path": str(bundle_path),
        "bundle_sha256": bundle_digest,
        "actor_sha256": actor_hash,
        "manifest_sha256": manifest_hash,
        "audit_sha256": audit_hash,
        "base_model_path": str(base_model_path),
        "base_model_sha256": base_hash,
        "training_profile": profile.name,
        "global_step": profile.global_step,
        "is_final": profile.is_final,
    }


def _git_state(repo: Path) -> dict[str, Any]:
    if not repo.is_dir():
        raise ValueError(f"repository is missing: {repo}")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise ValueError(f"not a valid Git repository: {repo}") from error
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError(f"invalid Git commit for repository: {repo}")
    return {
        "path": str(repo.resolve()),
        "commit": commit,
        "dirty": bool(status.strip()),
    }


def capture_repo_provenance(
    repos: Mapping[str, str | Path], *, allow_dirty: bool
) -> dict[str, dict[str, Any]]:
    """Capture commit and dirty state for exactly RLinf, T2-VLA, and Tabero."""
    if set(repos) != {"rlinf", "t2_vla", "tabero"}:
        raise ValueError(
            "repository set must contain exactly RLinf, T2-VLA, and Tabero"
        )
    states = {name: _git_state(Path(path).resolve()) for name, path in repos.items()}
    dirty = [name for name, state in states.items() if state["dirty"]]
    if dirty and not allow_dirty:
        raise ValueError(
            f"formal official evaluation requires clean repositories; dirty={dirty}"
        )
    return states


def _parse_normalized_result(
    raw_dir: Path, bundle: Path, task_id: int
) -> dict[str, Any]:
    candidates = sorted(raw_dir.glob("success_rates_openpi_tactile_*.json"))
    if len(candidates) != 1:
        raise ValueError(
            "raw output must contain exactly one unique success_rates JSON; "
            f"found={len(candidates)}"
        )
    raw_path = candidates[0].resolve()
    payload = _read_json(raw_path, "raw success rates")
    metadata = payload.get("metadata")
    results = payload.get("results")
    if not isinstance(metadata, dict) or not isinstance(results, dict):
        raise ValueError("raw result must contain metadata and results objects")
    expected_metadata = {
        "policy_model": "openpi",
        "control_mode": "tactile",
        "num_total_experiments": 50,
        "num_success_steps": 8,
        "prompt_mode": "adverb_set",
        "prompt_adverb": "",
        "prompt_adverbs": ["firmly", "tightly"],
        "prompt_seed": 0,
    }
    for key, expected in expected_metadata.items():
        _require_exact(metadata.get(key), expected, f"raw metadata {key}")
    max_policy = metadata.get("max_inference_steps_policy")
    if not isinstance(max_policy, dict) or max_policy.get("libero_object") != 30:
        raise ValueError("raw metadata max inference policy must be 30")
    result_key = f"libero_object_task{task_id}"
    if set(results) != {result_key}:
        raise ValueError(
            f"raw result key must be exactly {result_key!r}; got {sorted(results)}"
        )
    result = results[result_key]
    if not isinstance(result, dict):
        raise ValueError("raw task result must be an object")
    for key, expected in (
        ("task_suite", "libero_object"),
        ("task_id", task_id),
        ("total_experiments", 50),
        ("max_inference_steps", 30),
        ("status", "completed"),
    ):
        _require_exact(result.get(key), expected, f"raw result {key}")
    success_count = result.get("successful_experiments")
    success_rate = result.get("success_rate")
    if type(success_count) is not int or not 0 <= success_count <= 50:
        raise ValueError(
            "raw result successful experiments must be an integer in [0, 50]"
        )
    if isinstance(success_rate, bool) or not isinstance(success_rate, (int, float)):
        raise ValueError("raw result success rate must be numeric")
    expected_rate = success_count * 100.0 / 50
    if float(success_rate) != expected_rate:
        raise ValueError(
            f"raw result rate/count mismatch: rate={success_rate}, count={success_count}"
        )
    normalized_rate = float(success_rate)
    manifest = _read_json(bundle / MANIFEST_NAME, "bundle manifest")
    global_step = manifest.get("global_step")
    if type(global_step) is not int or global_step <= 0:
        raise ValueError("bundle manifest global_step must be a positive integer")
    is_final = manifest.get("is_final")
    if type(is_final) is not bool:
        raise ValueError("bundle manifest is_final must be a boolean")
    return {
        "checkpoint": {
            "global_step": global_step,
            "is_final": is_final,
            "path": str(bundle.resolve()),
        },
        "condition": "firm",
        "config": {"name": f"tabero_task{task_id}_firm_dsrl_official_eval"},
        "method": "dsrl",
        "metric_source": {
            "backend": "tabero_official",
            "path": str(raw_path),
            "result_key": result_key,
            "schema": "tabero_success_rates_percent_v1",
        },
        "protocol": {
            "action_horizon": 10,
            "consecutive_success_steps": 8,
            "control_mode": "tactile",
            "hdf5_source": "assembled_hdf5",
            "headless": True,
            "max_inference_steps": 30,
            "mode": "formal",
            "physical_steps_per_episode": 300,
            "prompt_adverbs": ["firmly", "tightly"],
            "prompt_seed": 0,
            "replan_steps": 10,
            "require_hdf5": True,
            "seed": 11,
            "send_dsrl_raw_image": True,
            "task_suite": "libero_object",
        },
        "schema_version": 1,
        "source_metrics": {
            "success_rate": normalized_rate,
            "successful_experiments": success_count,
            "total_experiments": 50,
        },
        "status": "completed",
        "success_count": success_count,
        "success_rate": normalized_rate,
        "task_id": task_id,
        "total_episodes": 50,
    }


def normalize_result(
    raw_dir: str | Path,
    output_path: str | Path,
    bundle: str | Path,
    task_id: int,
) -> dict[str, Any]:
    """Validate the unique raw result and atomically write the prior schema."""
    if task_id not in {0, 5}:
        raise ValueError("task_id must be exactly 0 or 5")
    result = _parse_normalized_result(Path(raw_dir), Path(bundle), task_id)
    _atomic_write_json(Path(output_path), result)
    return result


def write_receipts(
    normalized: Mapping[str, Any],
    output_dir: str | Path,
    run_id: str,
    *,
    bundle_path: str | Path,
    base_model_path: str | Path,
    expected_task_id: int,
    training_profile: str = FORMAL_8GPU_50STEP_PROFILE,
    summary_writer_cls: Any | None = None,
    wandb_module: Any | None = None,
) -> dict[str, Any]:
    """Log official metrics to TensorBoard and W&B and write a URL receipt."""
    bundle = validate_bundle(
        bundle_path,
        expected_task_id,
        base_model_path,
        training_profile=training_profile,
    )
    global_step = bundle["global_step"]
    output = Path(output_dir)
    receipt_path = output / "wandb_eval.json"
    if receipt_path.exists():
        raise FileExistsError(f"output already exists: {receipt_path}")
    if not run_id.startswith("tabero-official-dsrl-task") or not run_id.endswith(
        "-formal"
    ):
        raise ValueError(f"invalid official W&B run ID: {run_id}")
    metrics = {
        "eval/success_count": normalized["success_count"],
        "eval/success_rate": normalized["success_rate"],
        "eval/total_episodes": normalized["total_episodes"],
    }
    if summary_writer_cls is None:
        from torch.utils.tensorboard import SummaryWriter

        summary_writer_cls = SummaryWriter
    writer = summary_writer_cls(log_dir=output / "tensorboard")
    try:
        for name, value in metrics.items():
            writer.add_scalar(name, value, global_step)
        writer.flush()
    finally:
        writer.close()

    if wandb_module is None:
        import wandb

        wandb_module = wandb
    run = None
    try:
        run = wandb_module.init(
            project="tabero-rlinf",
            id=run_id,
            name=run_id,
            resume="never",
            mode="online",
            config={
                "is_final": bundle["is_final"],
                "method": "dsrl",
                "official_eval": True,
                "source_global_step": global_step,
                "training_profile": bundle["training_profile"],
            },
        )
        run.log(metrics, step=global_step, commit=True)
        url = run.url
        if not isinstance(url, str) or not url:
            raise RuntimeError("W&B did not return a run URL")
        run.finish(exit_code=0)
    except Exception:
        if run is not None:
            run.finish(exit_code=1)
        raise
    receipt = {
        "id": run_id,
        "is_final": bundle["is_final"],
        "project": "tabero-rlinf",
        "run_id": run_id,
        "step": global_step,
        "training_profile": bundle["training_profile"],
        "url": url,
    }
    _atomic_write_json(receipt_path, receipt)
    return receipt


def _descendant_pids(root_pid: int) -> set[int]:
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for status_path in Path("/proc").glob("[0-9]*/status"):
            fields: dict[str, str] = {}
            try:
                for line in status_path.read_text().splitlines():
                    key, separator, value = line.partition(":")
                    if separator and key in {"Pid", "PPid"}:
                        fields[key] = value.strip()
                pid = int(fields["Pid"])
                parent_pid = int(fields["PPid"])
            except (KeyError, OSError, ValueError):
                continue
            if parent_pid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return descendants


def _listening_socket_inodes(port: int) -> set[str]:
    inodes: set[str] = set()
    encoded_port = f"{port:04X}"
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            rows = table.read_text().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            fields = row.split()
            if (
                len(fields) >= 10
                and fields[1].rsplit(":", 1)[-1] == encoded_port
                and fields[3] == "0A"
            ):
                inodes.add(fields[9])
    return inodes


def listener_owned_by_process(pid: int, port: int) -> bool:
    """Return whether pid or one of its descendants owns the listening port."""
    if pid <= 0 or not 1 <= port <= 65535:
        return False
    inodes = _listening_socket_inodes(port)
    if not inodes:
        return False
    for process_id in _descendant_pids(pid):
        try:
            descriptors = list(Path(f"/proc/{process_id}/fd").iterdir())
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                target = str(descriptor.readlink())
            except OSError:
                continue
            if target.startswith("socket:[") and target[8:-1] in inodes:
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        return True
                except OSError:
                    return False
    return False


def sample_gpus_once(
    gpu_file: str | Path,
    process_file: str | Path,
    *,
    nvidia_smi: str = "nvidia-smi",
) -> None:
    """Append one GPU/process sample and propagate query or write failures."""
    gpu_result = subprocess.run(
        [
            nvidia_smi,
            "--id=0,1",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,pstate",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    process_result = subprocess.run(
        [
            nvidia_smi,
            "--id=0,1",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    for output_path, rows in (
        (Path(gpu_file), gpu_result.stdout.splitlines()),
        (Path(process_file), process_result.stdout.splitlines()),
    ):
        with output_path.open("a", encoding="utf-8") as file:
            for row in rows:
                if row:
                    file.write(f"{timestamp},{row}\n")
            file.flush()
            os.fsync(file.fileno())


def _pid_start_time(pid: int) -> str | None:
    try:
        contents = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    closing_parenthesis = contents.rfind(")")
    if closing_parenthesis < 0:
        return None
    fields = contents[closing_parenthesis + 2 :].split()
    if fields and fields[0] == "Z":
        return None
    return fields[19] if len(fields) > 19 else None


def _launcher_is_alive(pid: int, start_time: str) -> bool:
    return pid > 1 and _pid_start_time(pid) == start_time


def _set_child_subreaper() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def _set_parent_death_signal(parent_death_signal: int) -> None:
    expected_parent_pid = os.getppid()
    if expected_parent_pid <= 1:
        raise RuntimeError("supervisor exited before parent-death signal was armed")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, parent_death_signal, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    if os.getppid() != expected_parent_pid:
        raise RuntimeError("supervisor exited while parent-death signal was armed")


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _reap_adopted_children() -> None:
    while True:
        try:
            waited_pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if waited_pid == 0:
            return


def _stop_child_process_group(
    child: subprocess.Popen[Any], *, grace_seconds: float = 3.0
) -> None:
    child.poll()
    if _process_group_exists(child.pid):
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        child.poll()
        if child.returncode is not None:
            _reap_adopted_children()
        if not _process_group_exists(child.pid):
            return
        time.sleep(0.05)
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        child.poll()
        if child.returncode is not None:
            _reap_adopted_children()
        if not _process_group_exists(child.pid):
            return
        time.sleep(0.05)
    raise RuntimeError(f"child process group {child.pid} survived SIGKILL")


def _sample_forever(args: argparse.Namespace) -> None:
    stop_requested = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_requested.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    parent_start_time = _pid_start_time(args.parent_pid)
    if parent_start_time is None:
        raise RuntimeError("launcher exited before GPU sampling started")
    while not stop_requested.is_set():
        started = time.monotonic()
        sample_gpus_once(args.gpu_file, args.process_file)
        if not _launcher_is_alive(args.parent_pid, parent_start_time):
            raise RuntimeError("launcher exited while GPU sampler was running")
        stop_requested.wait(max(0.0, args.interval - (time.monotonic() - started)))


def _supervise(args: argparse.Namespace) -> int:
    command = list(args.child_command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise ValueError("supervise requires a child command")
    if not _launcher_is_alive(args.parent_pid, args.parent_start_time):
        raise RuntimeError("launcher exited before child supervision started")
    _set_child_subreaper()
    requested_signal = 0

    def request_shutdown(received_signal: int, _frame: Any) -> None:
        nonlocal requested_signal
        requested_signal = received_signal

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    child = subprocess.Popen(command, start_new_session=True)
    while True:
        child_status = child.poll()
        if child_status is not None:
            if _process_group_exists(child.pid):
                _stop_child_process_group(child)
            return child_status
        if requested_signal:
            _stop_child_process_group(child)
            return 0
        if not _launcher_is_alive(args.parent_pid, args.parent_start_time):
            _stop_child_process_group(child)
            return 128 + signal.SIGTERM
        time.sleep(0.05)


def _validate_lock_directory(lock_dir: Path) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_dir, flags)
    details = os.fstat(descriptor)
    mode = stat.S_IMODE(details.st_mode)
    trusted_private = details.st_uid == os.geteuid() and mode == 0o700
    trusted_shared = details.st_uid == 0 and mode == 0o1777
    if not stat.S_ISDIR(details.st_mode) or not (trusted_private or trusted_shared):
        os.close(descriptor)
        raise ValueError(
            "GPU lock directory must be EUID-owned mode 0700 or root-owned mode 1777"
        )
    return descriptor


def _open_gpu_lock(directory_fd: int, gpu_id: int) -> int:
    name = f"tabero-formal-gpu-{gpu_id}.lock"
    flags = os.O_RDWR | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(
            name, flags | os.O_CREAT | os.O_EXCL, 0o666, dir_fd=directory_fd
        )
    except FileExistsError:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    else:
        os.fchmod(descriptor, 0o666)
    details = os.fstat(descriptor)
    path_details = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
        or stat.S_IMODE(details.st_mode) != 0o666
        or (details.st_dev, details.st_ino)
        != (path_details.st_dev, path_details.st_ino)
    ):
        os.close(descriptor)
        raise ValueError(f"invalid GPU {gpu_id} lock file")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise RuntimeError(
            f"GPU {gpu_id} lease is already held by another formal launcher"
        ) from None
    return descriptor


def _hold_gpu_locks(args: argparse.Namespace) -> None:
    stop_requested = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_requested.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    _set_parent_death_signal(signal.SIGTERM)
    directory_fd = _validate_lock_directory(args.lock_dir)
    lock_descriptors: list[int] = []
    try:
        for gpu_id in (0, 1):
            lock_descriptors.append(_open_gpu_lock(directory_fd, gpu_id))
        _atomic_write(args.ready_file, "ready\n")
        stop_requested.wait()
    finally:
        for descriptor in lock_descriptors:
            os.close(descriptor)
        os.close(directory_fd)


def _metadata_lines(
    bundle: Mapping[str, Any], repos: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    values = {
        "TABERO_DSRL_BUNDLE": bundle["bundle_path"],
        "TABERO_DSRL_BUNDLE_SHA256": bundle["bundle_sha256"],
        "TABERO_DSRL_ACTOR_SHA256": bundle["actor_sha256"],
        "TABERO_DSRL_MANIFEST_SHA256": bundle["manifest_sha256"],
        "TABERO_DSRL_AUDIT_SHA256": bundle["audit_sha256"],
        "TABERO_BASE_MODEL_PATH": bundle["base_model_path"],
        "TABERO_BASE_MODEL_SHA256": bundle["base_model_sha256"],
        "TABERO_DSRL_GLOBAL_STEP": bundle["global_step"],
        "TABERO_DSRL_IS_FINAL": str(bundle["is_final"]).lower(),
        "TABERO_DSRL_TRAINING_PROFILE": bundle["training_profile"],
    }
    for name, state in repos.items():
        prefix = name.upper()
        values[f"TABERO_{prefix}_REPO_PATH"] = state["path"]
        values[f"TABERO_{prefix}_GIT_COMMIT"] = state["commit"]
        values[f"TABERO_{prefix}_GIT_DIRTY"] = str(state["dirty"]).lower()
    for key, value in values.items():
        if "\n" in str(value) or "=" in str(value):
            raise ValueError(f"unsafe metadata value for {key}")
    return [f"{key}={value}" for key, value in values.items()]


def _preflight(args: argparse.Namespace) -> None:
    bundle = validate_bundle(
        args.bundle,
        args.task_id,
        args.base_model,
        training_profile=args.training_profile,
    )
    repos = capture_repo_provenance(
        {"rlinf": args.rlinf_repo, "t2_vla": args.t2_repo, "tabero": args.tabero_repo},
        allow_dirty=args.allow_dirty,
    )
    _atomic_write(args.metadata_out, "\n".join(_metadata_lines(bundle, repos)) + "\n")


def _finalize(args: argparse.Namespace) -> None:
    normalized_path = args.output_dir / "normalized_result.json"
    if normalized_path.exists():
        raise FileExistsError(f"output already exists: {normalized_path}")
    normalized = _parse_normalized_result(args.raw_dir, args.bundle, args.task_id)
    write_receipts(
        normalized,
        args.output_dir / "output",
        args.run_id,
        bundle_path=args.bundle,
        base_model_path=args.base_model,
        expected_task_id=args.task_id,
        training_profile=args.training_profile,
    )
    _atomic_write_json(normalized_path, normalized)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--bundle", type=Path, required=True)
    preflight.add_argument("--task-id", type=int, choices=(0, 5), required=True)
    preflight.add_argument("--base-model", type=Path, required=True)
    preflight.add_argument(
        "--training-profile",
        choices=TABERO_DSRL_TRAINING_PROFILE_CHOICES,
        required=True,
    )
    preflight.add_argument("--rlinf-repo", type=Path, required=True)
    preflight.add_argument("--t2-repo", type=Path, required=True)
    preflight.add_argument("--tabero-repo", type=Path, required=True)
    preflight.add_argument("--metadata-out", type=Path, required=True)
    preflight.add_argument("--allow-dirty", action="store_true")
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--raw-dir", type=Path, required=True)
    finalize.add_argument("--output-dir", type=Path, required=True)
    finalize.add_argument("--bundle", type=Path, required=True)
    finalize.add_argument("--task-id", type=int, choices=(0, 5), required=True)
    finalize.add_argument("--base-model", type=Path, required=True)
    finalize.add_argument(
        "--training-profile",
        choices=TABERO_DSRL_TRAINING_PROFILE_CHOICES,
        required=True,
    )
    finalize.add_argument("--run-id", required=True)
    listener = commands.add_parser("listener-owned")
    listener.add_argument("--pid", type=int, required=True)
    listener.add_argument("--port", type=int, required=True)
    sampler = commands.add_parser("sample-gpus")
    sampler.add_argument("--gpu-file", type=Path, required=True)
    sampler.add_argument("--process-file", type=Path, required=True)
    sampler.add_argument("--interval", type=float, default=5.0)
    sampler.add_argument("--parent-pid", type=int, required=True)
    supervise = commands.add_parser("supervise")
    supervise.add_argument("--parent-pid", type=int, required=True)
    supervise.add_argument("--parent-start-time", required=True)
    supervise.add_argument("child_command", nargs=argparse.REMAINDER)
    guardian = commands.add_parser("hold-gpu-locks")
    guardian.add_argument("--lock-dir", type=Path, required=True)
    guardian.add_argument("--ready-file", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.command == "preflight":
            _preflight(args)
        elif args.command == "finalize":
            _finalize(args)
        elif args.command == "listener-owned":
            if not listener_owned_by_process(args.pid, args.port):
                raise SystemExit(1)
        elif args.command == "sample-gpus":
            _sample_forever(args)
        elif args.command == "hold-gpu-locks":
            _hold_gpu_locks(args)
        else:
            child_status = _supervise(args)
            if child_status:
                raise SystemExit(child_status)
    except (FileExistsError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
