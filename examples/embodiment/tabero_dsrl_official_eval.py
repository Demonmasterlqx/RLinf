# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""CPU-only validation and receipt helpers for Tabero DSRL official eval."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from safetensors import safe_open

from rlinf.utils.dsrl_rollout_sync import (
    DSRL_ROLLOUT_SYNC_MANIFEST_V1,
    DSRL_ROLLOUT_SYNC_MANIFEST_VERSION,
    DSRL_ROLLOUT_SYNC_PARAMETER_COUNT,
    DSRL_ROLLOUT_SYNC_TENSOR_COUNT,
)

FORMAT = "tabero_dsrl_t2vla"
FORMAT_VERSION = 1
FINAL_GLOBAL_STEP = 50
BASE_WEIGHTS_NAME = "model.safetensors"
ACTOR_NAME = "dsrl_actor.safetensors"
MANIFEST_NAME = "manifest.json"
AUDIT_NAME = "artifact_audit.json"
EXPECTED_AUDIT_CHECKS = {
    "final_checkpoint_path",
    "source_metadata",
    "source_trainable_manifest",
    "actor_manifest",
    "actor_dtype",
    "actor_finite",
    "base_model_sha256",
    "formal_provenance",
    "output_hashes",
}
EXPECTED_AUDIT_KEYS = {
    "format",
    "format_version",
    "status",
    "task_id",
    "global_step",
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
}
EXPECTED_FEATURE_CONTRACT = {
    "order": ["state", "image", "tactile"],
    "dims": [64, 64, 64],
    "total_dim": 192,
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
    "state_dim": 7,
    "tactile_shape": [9, 198, 2],
    "hidden_dims": [128, 128, 128],
    "feature_dim": 192,
    "noise_dim": 32,
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


def _validate_actor(actor_path: Path, task_id: int) -> None:
    try:
        with safe_open(actor_path, framework="pt", device="cpu") as actor:
            metadata = actor.metadata()
            expected_metadata = {
                "format": FORMAT,
                "format_version": str(FORMAT_VERSION),
                "task_id": str(task_id),
                "global_step": str(FINAL_GLOBAL_STEP),
                "dtype": "bfloat16",
            }
            if metadata != expected_metadata:
                raise ValueError(
                    "actor safetensors metadata mismatch; "
                    f"expected={expected_metadata}, actual={metadata}"
                )
            actual_keys = set(actor.keys())
            expected_keys = set(DSRL_ROLLOUT_SYNC_MANIFEST_V1)
            if actual_keys != expected_keys:
                raise ValueError(
                    "actor tensor manifest keyspace mismatch; "
                    f"missing={sorted(expected_keys - actual_keys)}, "
                    f"unexpected={sorted(actual_keys - expected_keys)}"
                )
            for key, expected_shape in DSRL_ROLLOUT_SYNC_MANIFEST_V1.items():
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
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("actor"):
            raise
        raise ValueError(f"actor safetensors validation failed: {error}") from error


def validate_bundle(
    bundle: str | Path,
    task_id: int,
    base_model: str | Path,
) -> dict[str, str]:
    """Strictly validate one final Task 0/5 audited DSRL bundle."""
    if type(task_id) is not int or task_id not in {0, 5}:
        raise ValueError(f"task_id must be exactly 0 or 5; got {task_id!r}")
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
    _require_exact(manifest.get("task_id"), task_id, "manifest task_id")
    _require_exact(
        manifest.get("global_step"), FINAL_GLOBAL_STEP, "manifest global_step"
    )
    _require_exact(manifest.get("is_final"), True, "manifest is_final final flag")
    _require_exact(
        manifest.get("training_config"),
        f"isaaclab_pi0_dsrl_tacfield_tabero_task{task_id}_firm_8gpu_50step",
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

    if set(audit) != EXPECTED_AUDIT_KEYS:
        raise ValueError("audit keyspace mismatch")
    _require_exact(audit.get("format"), "tabero_dsrl_artifact_audit", "audit format")
    _require_exact(audit.get("format_version"), 1, "audit format_version")
    _require_exact(audit.get("status"), "passed", "audit status")
    _require_exact(audit.get("task_id"), task_id, "audit task_id")
    _require_exact(audit.get("global_step"), FINAL_GLOBAL_STEP, "audit global_step")
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

    _validate_actor(actor_path, task_id)
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
    if not math.isclose(float(success_rate), expected_rate, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"raw result rate/count mismatch: rate={success_rate}, count={success_count}"
        )
    normalized_rate = float(success_rate)
    return {
        "checkpoint": {"path": str(bundle.resolve())},
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
    summary_writer_cls: Any | None = None,
    wandb_module: Any | None = None,
) -> dict[str, Any]:
    """Log official metrics to TensorBoard and W&B and write a URL receipt."""
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
            writer.add_scalar(name, value, FINAL_GLOBAL_STEP)
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
            config={"method": "dsrl", "official_eval": True},
        )
        run.log(metrics, step=FINAL_GLOBAL_STEP, commit=True)
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
        "project": "tabero-rlinf",
        "run_id": run_id,
        "step": FINAL_GLOBAL_STEP,
        "url": url,
    }
    _atomic_write_json(receipt_path, receipt)
    return receipt


def _metadata_lines(
    bundle: Mapping[str, str], repos: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    values = {
        "TABERO_DSRL_BUNDLE": bundle["bundle_path"],
        "TABERO_DSRL_BUNDLE_SHA256": bundle["bundle_sha256"],
        "TABERO_DSRL_ACTOR_SHA256": bundle["actor_sha256"],
        "TABERO_DSRL_MANIFEST_SHA256": bundle["manifest_sha256"],
        "TABERO_DSRL_AUDIT_SHA256": bundle["audit_sha256"],
        "TABERO_BASE_MODEL_PATH": bundle["base_model_path"],
        "TABERO_BASE_MODEL_SHA256": bundle["base_model_sha256"],
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
    bundle = validate_bundle(args.bundle, args.task_id, args.base_model)
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
    write_receipts(normalized, args.output_dir / "output", args.run_id)
    _atomic_write_json(normalized_path, normalized)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--bundle", type=Path, required=True)
    preflight.add_argument("--task-id", type=int, choices=(0, 5), required=True)
    preflight.add_argument("--base-model", type=Path, required=True)
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
    finalize.add_argument("--run-id", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.command == "preflight":
            _preflight(args)
        else:
            _finalize(args)
    except (FileExistsError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
