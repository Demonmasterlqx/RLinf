# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from fcntl import LOCK_EX, LOCK_NB, flock
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from rlinf.utils.dsrl_rollout_sync import (
    DSRL_ROLLOUT_SYNC_MANIFEST_V1,
    DSRL_ROLLOUT_SYNC_PARAMETER_COUNT,
    DSRL_ROLLOUT_SYNC_TENSOR_COUNT,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = REPO_ROOT / "examples/embodiment/tabero_dsrl_official_eval.py"
LAUNCHER_PATH = REPO_ROOT / "examples/embodiment/run_tabero_firm_official_eval.sh"


def _load_helper():
    spec = importlib.util.spec_from_file_location(
        "tabero_dsrl_official_eval", HELPER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load official-eval helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def bundle_fixture(tmp_path):
    base_model = tmp_path / "base"
    base_model.mkdir()
    (base_model / "model.safetensors").write_bytes(b"fixed-test-base")
    source_checkpoint = tmp_path / "global_step_50/trainable_weights.pt"
    source_checkpoint.parent.mkdir()
    source_checkpoint.write_bytes(b"final trainable checkpoint")
    config_snapshot = tmp_path / "config_snapshot.yaml"
    config_snapshot.write_text("formal: true\n")
    source_provenance = tmp_path / "provenance.env"
    source_provenance.write_text(
        "\n".join(
            (
                "TABERO_PROVENANCE_VERSION=1",
                "TABERO_PROVENANCE_MODE=fresh",
                f"TABERO_CONFIG_SHA256={_sha256(config_snapshot)}",
                f"TABERO_CONFIG_SNAPSHOT_SHA256={_sha256(config_snapshot)}",
                f"TABERO_GIT_COMMIT={'4' * 40}",
                "TABERO_GIT_DIRTY=false",
                f"TABERO_BASE_MODEL_PATH={base_model.resolve()}",
                f"TABERO_BASE_MODEL_SHA256={_sha256(base_model / 'model.safetensors')}",
                "TABERO_SOURCE_CONFIG_SHA256=none",
            )
        )
        + "\n"
    )
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    actor_path = bundle / "dsrl_actor.safetensors"
    save_file(
        {
            key: torch.zeros(shape, dtype=torch.bfloat16)
            for key, shape in DSRL_ROLLOUT_SYNC_MANIFEST_V1.items()
        },
        actor_path,
        metadata={
            "format": "tabero_dsrl_t2vla",
            "format_version": "1",
            "task_id": "0",
            "global_step": "50",
            "dtype": "bfloat16",
        },
    )
    manifest = {
        "format": "tabero_dsrl_t2vla",
        "format_version": 1,
        "algorithm": "dsrl-sac",
        "task_id": 0,
        "global_step": 50,
        "is_final": True,
        "training_config": "isaaclab_pi0_dsrl_tacfield_tabero_task0_firm_8gpu_50step",
        "base_model": str(base_model.resolve()),
        "base_model_sha256": _sha256(base_model / "model.safetensors"),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": _sha256(source_checkpoint),
        "source_provenance": str(source_provenance),
        "source_provenance_sha256": _sha256(source_provenance),
        "source_config_snapshot": str(config_snapshot),
        "source_config_snapshot_sha256": _sha256(config_snapshot),
        "legacy_source_config": None,
        "legacy_source_config_sha256": None,
        "source_git_commit": "4" * 40,
        "actor_weights": "dsrl_actor.safetensors",
        "actor_weights_sha256": _sha256(actor_path),
        "actor_manifest_version": 1,
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
        "artifact_audit": "artifact_audit.json",
    }
    manifest_path = bundle / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    audit = {
        "format": "tabero_dsrl_artifact_audit",
        "format_version": 1,
        "status": "passed",
        "task_id": 0,
        "global_step": 50,
        "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
        "base_model_sha256": manifest["base_model_sha256"],
        "actor_weights_sha256": manifest["actor_weights_sha256"],
        "manifest_sha256": _sha256(manifest_path),
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
    (bundle / "artifact_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    return bundle, base_model, manifest, audit


def _rewrite_bundle_json(bundle, name, payload):
    (bundle / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def test_validate_bundle_accepts_final_task0_and_captures_hashes(bundle_fixture):
    helper = _load_helper()
    bundle, base_model, manifest, _ = bundle_fixture

    result = helper.validate_bundle(bundle, 0, base_model)

    assert result["bundle_path"] == str(bundle.resolve())
    assert result["actor_sha256"] == manifest["actor_weights_sha256"]
    assert result["manifest_sha256"] == _sha256(bundle / "manifest.json")
    assert result["audit_sha256"] == _sha256(bundle / "artifact_audit.json")
    assert result["base_model_sha256"] == manifest["base_model_sha256"]
    assert len(result["bundle_sha256"]) == 64


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda manifest, audit: manifest.update(task_id=5), "task"),
        (lambda manifest, audit: manifest.update(global_step=40), "global_step"),
        (lambda manifest, audit: manifest.update(is_final=False), "final"),
        (lambda manifest, audit: manifest.update(format="wrong"), "format"),
        (lambda manifest, audit: manifest.update(base_model="/wrong"), "base model"),
        (lambda manifest, audit: audit.update(status="failed"), "audit"),
        (lambda manifest, audit: audit["checks"].update(actor_manifest=False), "audit"),
    ],
)
def test_validate_bundle_rejects_wrong_final_contract(
    bundle_fixture, mutation, message
):
    helper = _load_helper()
    bundle, base_model, manifest, audit = bundle_fixture
    mutation(manifest, audit)
    _rewrite_bundle_json(bundle, "manifest.json", manifest)
    audit["manifest_sha256"] = _sha256(bundle / "manifest.json")
    _rewrite_bundle_json(bundle, "artifact_audit.json", audit)

    with pytest.raises(ValueError, match=message):
        helper.validate_bundle(bundle, 0, base_model)


@pytest.mark.parametrize("target", ["actor", "manifest", "audit", "base"])
def test_validate_bundle_rejects_tampering(bundle_fixture, target):
    helper = _load_helper()
    bundle, base_model, _, _ = bundle_fixture
    paths = {
        "actor": bundle / "dsrl_actor.safetensors",
        "manifest": bundle / "manifest.json",
        "audit": bundle / "artifact_audit.json",
        "base": base_model / "model.safetensors",
    }
    paths[target].write_bytes(paths[target].read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="hash|JSON|safetensors|audit"):
        helper.validate_bundle(bundle, 0, base_model)


def test_validate_bundle_rejects_actor_keyspace(bundle_fixture):
    helper = _load_helper()
    bundle, base_model, manifest, audit = bundle_fixture
    actor_path = bundle / "dsrl_actor.safetensors"
    save_file(
        {"wrong": torch.zeros(1, dtype=torch.bfloat16)},
        actor_path,
        metadata={
            "format": "tabero_dsrl_t2vla",
            "format_version": "1",
            "task_id": "0",
            "global_step": "50",
            "dtype": "bfloat16",
        },
    )
    manifest["actor_weights_sha256"] = _sha256(actor_path)
    _rewrite_bundle_json(bundle, "manifest.json", manifest)
    audit["actor_weights_sha256"] = manifest["actor_weights_sha256"]
    audit["manifest_sha256"] = _sha256(bundle / "manifest.json")
    _rewrite_bundle_json(bundle, "artifact_audit.json", audit)

    with pytest.raises(ValueError, match="actor.*manifest|keyspace"):
        helper.validate_bundle(bundle, 0, base_model)


def test_validate_bundle_rejects_coherently_rehashed_nonfinite_actor(
    bundle_fixture,
):
    helper = _load_helper()
    bundle, base_model, manifest, audit = bundle_fixture
    tensors = {
        key: torch.zeros(shape, dtype=torch.bfloat16)
        for key, shape in DSRL_ROLLOUT_SYNC_MANIFEST_V1.items()
    }
    first_key = next(iter(tensors))
    tensors[first_key].flatten()[0] = float("nan")
    actor_path = bundle / "dsrl_actor.safetensors"
    save_file(
        tensors,
        actor_path,
        metadata={
            "format": "tabero_dsrl_t2vla",
            "format_version": "1",
            "task_id": "0",
            "global_step": "50",
            "dtype": "bfloat16",
        },
    )
    manifest["actor_weights_sha256"] = _sha256(actor_path)
    _rewrite_bundle_json(bundle, "manifest.json", manifest)
    audit["actor_weights_sha256"] = manifest["actor_weights_sha256"]
    audit["manifest_sha256"] = _sha256(bundle / "manifest.json")
    _rewrite_bundle_json(bundle, "artifact_audit.json", audit)

    with pytest.raises(ValueError, match="finite"):
        helper.validate_bundle(bundle, 0, base_model)


@pytest.mark.parametrize(
    "manifest_key",
    ["source_checkpoint", "source_provenance", "source_config_snapshot"],
)
def test_validate_bundle_rejects_tampered_source_artifact(bundle_fixture, manifest_key):
    helper = _load_helper()
    bundle, base_model, manifest, _ = bundle_fixture
    source_path = Path(manifest[manifest_key])
    source_path.write_bytes(source_path.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="source|provenance|config|hash"):
        helper.validate_bundle(bundle, 0, base_model)


def test_validate_bundle_binds_source_git_commit_to_provenance(bundle_fixture):
    helper = _load_helper()
    bundle, base_model, manifest, audit = bundle_fixture
    manifest["source_git_commit"] = "5" * 40
    _rewrite_bundle_json(bundle, "manifest.json", manifest)
    audit["manifest_sha256"] = _sha256(bundle / "manifest.json")
    _rewrite_bundle_json(bundle, "artifact_audit.json", audit)

    with pytest.raises(ValueError, match="Git|provenance"):
        helper.validate_bundle(bundle, 0, base_model)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda manifest: manifest["observation_contract"]["image"].update(
                key="wrong"
            ),
            "observation",
        ),
        (
            lambda manifest: manifest["feature_contract"].update(total_dim=191),
            "feature",
        ),
        (
            lambda manifest: manifest["noise_contract"].update(dim=31),
            "noise",
        ),
        (
            lambda manifest: manifest["architecture"].update(state_dim=8),
            "architecture",
        ),
        (
            lambda manifest: manifest.update(source_checkpoint_sha256="bad"),
            "source checkpoint",
        ),
        (
            lambda manifest: manifest.update(source_git_commit="bad"),
            "source Git",
        ),
    ],
)
def test_validate_bundle_rejects_tampered_dsrl_contract(
    bundle_fixture, mutation, message
):
    helper = _load_helper()
    bundle, base_model, manifest, audit = bundle_fixture
    mutation(manifest)
    _rewrite_bundle_json(bundle, "manifest.json", manifest)
    audit["source_checkpoint_sha256"] = manifest["source_checkpoint_sha256"]
    audit["manifest_sha256"] = _sha256(bundle / "manifest.json")
    _rewrite_bundle_json(bundle, "artifact_audit.json", audit)

    with pytest.raises(ValueError, match=message):
        helper.validate_bundle(bundle, 0, base_model)


def _git_repo(path, *, dirty=False):
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=path, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "tracked").write_text("clean\n")
    subprocess.run(["git", "add", "tracked"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True)
    if dirty:
        (path / "tracked").write_text("dirty\n")
    return path


def test_capture_repo_provenance_requires_three_clean_repositories(tmp_path):
    helper = _load_helper()
    repos = {
        "rlinf": _git_repo(tmp_path / "RLinf"),
        "t2_vla": _git_repo(tmp_path / "T2-VLA"),
        "tabero": _git_repo(tmp_path / "Tabero"),
    }
    states = helper.capture_repo_provenance(repos, allow_dirty=False)
    assert set(states) == {"rlinf", "t2_vla", "tabero"}
    assert all(len(state["commit"]) == 40 for state in states.values())
    assert all(state["dirty"] is False for state in states.values())

    (repos["tabero"] / "tracked").write_text("dirty\n")
    with pytest.raises(ValueError, match="Tabero|tabero|clean"):
        helper.capture_repo_provenance(repos, allow_dirty=False)
    assert helper.capture_repo_provenance(repos, allow_dirty=True)["tabero"]["dirty"]


def _raw_payload(task_id=0):
    return {
        "metadata": {
            "policy_model": "openpi",
            "control_mode": "tactile",
            "num_total_experiments": 50,
            "num_success_steps": 8,
            "prompt_mode": "adverb_set",
            "prompt_adverb": "",
            "prompt_adverbs": ["firmly", "tightly"],
            "prompt_seed": 0,
            "max_inference_steps_policy": {
                "libero_10": 50,
                "libero_goal": 30,
                "libero_spatial": 30,
                "libero_object": 30,
                "default": 30,
            },
        },
        "results": {
            f"libero_object_task{task_id}": {
                "task_suite": "libero_object",
                "task_id": task_id,
                "success_rate": 82.0,
                "successful_experiments": 41,
                "total_experiments": 50,
                "max_inference_steps": 30,
                "status": "completed",
            }
        },
    }


def test_normalize_result_matches_prior_schema_and_is_no_clobber(tmp_path):
    helper = _load_helper()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_path = raw_dir / "success_rates_openpi_tactile_1.json"
    raw_path.write_text(json.dumps(_raw_payload()))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    output = tmp_path / "normalized_result.json"

    result = helper.normalize_result(raw_dir, output, bundle, 0)

    assert result["schema_version"] == 1
    assert result["method"] == "dsrl"
    assert result["task_id"] == 0
    assert result["status"] == "completed"
    assert result["success_count"] == 41
    assert result["success_rate"] == 82.0
    assert result["total_episodes"] == 50
    assert result["checkpoint"] == {"path": str(bundle.resolve())}
    assert result["protocol"] == {
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
    }
    with pytest.raises(FileExistsError):
        helper.normalize_result(raw_dir, output, bundle, 0)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload["metadata"].update(policy_model="wrong"), "policy"),
        (lambda payload: payload["metadata"].update(control_mode="joint"), "control"),
        (lambda payload: payload["metadata"].update(num_total_experiments=49), "50"),
        (lambda payload: payload["metadata"].update(num_success_steps=7), "success"),
        (lambda payload: payload["metadata"].update(prompt_seed=1), "prompt"),
        (
            lambda payload: payload["results"]["libero_object_task0"].update(task_id=5),
            "task",
        ),
        (
            lambda payload: payload["results"]["libero_object_task0"].update(
                status="failed"
            ),
            "status",
        ),
        (
            lambda payload: payload["results"]["libero_object_task0"].update(
                total_experiments=49
            ),
            "50",
        ),
        (
            lambda payload: payload["results"]["libero_object_task0"].update(
                successful_experiments=42
            ),
            "rate",
        ),
        (
            lambda payload: payload["results"]["libero_object_task0"].update(
                successful_experiments=50,
                success_rate=100.0000000000005,
            ),
            "rate",
        ),
    ],
)
def test_normalize_result_rejects_protocol_or_metric_mismatch(
    tmp_path, mutation, message
):
    helper = _load_helper()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    payload = _raw_payload()
    mutation(payload)
    (raw_dir / "success_rates_openpi_tactile_1.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        helper.normalize_result(raw_dir, tmp_path / "normalized.json", tmp_path, 0)


def test_normalize_result_requires_unique_raw_json(tmp_path):
    helper = _load_helper()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for suffix in ("1", "2"):
        (raw_dir / f"success_rates_openpi_tactile_{suffix}.json").write_text(
            json.dumps(_raw_payload())
        )
    with pytest.raises(ValueError, match="exactly one|unique"):
        helper.normalize_result(raw_dir, tmp_path / "normalized.json", tmp_path, 0)


def test_write_receipts_logs_tensorboard_and_wandb_at_step50(tmp_path):
    helper = _load_helper()
    logged = []

    class FakeWriter:
        def __init__(self, log_dir):
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True)

        def add_scalar(self, name, value, step):
            logged.append((name, value, step))

        def flush(self):
            (self.log_dir / "events.fake").write_text("ok")

        def close(self):
            pass

    class FakeRun:
        id = "tabero-official-dsrl-task0-20260728-120000-formal"
        url = "https://wandb.example/run"

        def log(self, values, step, commit):
            logged.append((values, step, commit))

        def finish(self, exit_code):
            logged.append(("finish", exit_code))

    class FakeWandb:
        def init(self, **kwargs):
            logged.append(kwargs)
            return FakeRun()

    normalized = {
        "success_count": 41,
        "success_rate": 82.0,
        "total_episodes": 50,
    }
    output = tmp_path / "output"
    run_id = "tabero-official-dsrl-task0-20260728-120000-formal"

    receipt = helper.write_receipts(
        normalized,
        output,
        run_id,
        summary_writer_cls=FakeWriter,
        wandb_module=FakeWandb(),
    )

    assert (output / "tensorboard/events.fake").is_file()
    assert ("eval/success_count", 41, 50) in logged
    assert ("eval/success_rate", 82.0, 50) in logged
    assert ("eval/total_episodes", 50, 50) in logged
    assert {
        "eval/success_count": 41,
        "eval/success_rate": 82.0,
        "eval/total_episodes": 50,
    } in [entry[0] for entry in logged if isinstance(entry, tuple) and entry]
    assert receipt == {
        "id": run_id,
        "project": "tabero-rlinf",
        "run_id": run_id,
        "step": 50,
        "url": "https://wandb.example/run",
    }
    assert json.loads((output / "wandb_eval.json").read_text()) == receipt
    with pytest.raises(FileExistsError):
        helper.write_receipts(
            normalized,
            output,
            run_id,
            summary_writer_cls=FakeWriter,
            wandb_module=FakeWandb(),
        )


def test_write_receipts_propagates_backend_failure_without_receipt(tmp_path):
    helper = _load_helper()

    class BrokenWandb:
        def init(self, **kwargs):
            raise RuntimeError("backend unavailable")

    class FakeWriter:
        def __init__(self, log_dir):
            Path(log_dir).mkdir(parents=True)

        def add_scalar(self, name, value, step):
            pass

        def flush(self):
            pass

        def close(self):
            pass

    with pytest.raises(RuntimeError, match="backend"):
        helper.write_receipts(
            {"success_count": 1, "success_rate": 2.0, "total_episodes": 50},
            tmp_path / "output",
            "tabero-official-dsrl-task0-20260728-120000-formal",
            summary_writer_cls=FakeWriter,
            wandb_module=BrokenWandb(),
        )
    assert not (tmp_path / "output/wandb_eval.json").exists()


def test_launcher_public_contract_is_present():
    launcher = LAUNCHER_PATH.read_text()
    assert "Usage:" in launcher
    assert "dsrl <0|5> formal --dsrl-bundle ABS_PATH [--dry-run]" in launcher
    assert "CUDA_VISIBLE_DEVICES=0" in launcher
    assert "unset CUDA_VISIBLE_DEVICES" in launcher
    assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in launcher
    assert "JAX_PLATFORMS=cuda" in launcher
    assert "XLA_PYTHON_CLIENT_PREALLOCATE=false" in launcher
    assert "--dsrl-bundle" in launcher
    assert "--send-dsrl-raw-image" in launcher
    assert "--sim-device" in launcher and "cuda:1" in launcher
    assert "--sim-kit-args=--/renderer/activeGpu=1" in launcher
    assert "--interval 5" in launcher
    assert "flock -n" in launcher
    assert "close_gpu_lock_fds" in launcher
    assert "kill -KILL" in launcher
    assert "listener-owned" in launcher
    assert "sample-gpus" in launcher
    assert "--parent-pid" in launcher
    assert "cmp --silent" in launcher


def test_launcher_does_not_clear_process_cleanup_trap_after_installing_it():
    launcher = LAUNCHER_PATH.read_text()
    cleanup_install = launcher.index("trap cleanup EXIT")
    assert "\ntrap - EXIT\n" not in launcher[cleanup_install:]


def _write_executable(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(0o755)


def _retask_bundle(bundle, task_id):
    actor_path = bundle / "dsrl_actor.safetensors"
    save_file(
        {
            key: torch.zeros(shape, dtype=torch.bfloat16)
            for key, shape in DSRL_ROLLOUT_SYNC_MANIFEST_V1.items()
        },
        actor_path,
        metadata={
            "format": "tabero_dsrl_t2vla",
            "format_version": "1",
            "task_id": str(task_id),
            "global_step": "50",
            "dtype": "bfloat16",
        },
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["task_id"] = task_id
    manifest["training_config"] = (
        f"isaaclab_pi0_dsrl_tacfield_tabero_task{task_id}_firm_8gpu_50step"
    )
    manifest["actor_weights_sha256"] = _sha256(actor_path)
    _rewrite_bundle_json(bundle, "manifest.json", manifest)
    audit_path = bundle / "artifact_audit.json"
    audit = json.loads(audit_path.read_text())
    audit["task_id"] = task_id
    audit["actor_weights_sha256"] = manifest["actor_weights_sha256"]
    audit["manifest_sha256"] = _sha256(manifest_path)
    _rewrite_bundle_json(bundle, "artifact_audit.json", audit)


def _launcher_environment(tmp_path, bundle, base_model, *, gpu_mode="normal"):
    project_root = tmp_path / "tabero-root"
    project_root.mkdir()
    t2_repo = _git_repo(project_root / "T2-VLA")
    tabero_repo = _git_repo(project_root / "Tabero")
    _write_executable(t2_repo / ".venv/bin/python", "#!/usr/bin/env bash\nexit 0\n")
    (t2_repo / "scripts").mkdir()
    (t2_repo / "scripts/serve_policy.py").write_text("# fake\n")
    client_script = tabero_repo / "scripts/tools/run_task_evaluations.py"
    client_script.parent.mkdir(parents=True)
    client_script.write_text("# fake\n")
    (tabero_repo / "benchmarks/datasets/libero/assembled_hdf5").mkdir(parents=True)
    fake_bin = tmp_path / "fake-bin"
    nvidia_smi = """#!/usr/bin/env bash
set -euo pipefail
case "$*" in
  *"--query-gpu=index"*)
    case "${FAKE_GPU_MODE:-normal}" in
      normal|busy0|busy1) printf '0\\n1\\n' ;;
      duplicate) printf '0\\n0\\n1\\n' ;;
      missing) printf '0\\n' ;;
    esac
    ;;
  *"--id=0"*"--query-compute-apps=pid"*)
    [[ "${FAKE_GPU_MODE:-normal}" == busy0 ]] && printf '4242\\n' || true
    ;;
  *"--id=1"*"--query-compute-apps=pid"*)
    [[ "${FAKE_GPU_MODE:-normal}" == busy1 ]] && printf '5252\\n' || true
    ;;
  *) exit 9 ;;
esac
"""
    _write_executable(fake_bin / "nvidia-smi", nvidia_smi)
    _write_executable(fake_bin / "conda", "#!/usr/bin/env bash\nexit 0\n")
    results = tmp_path / "results"
    results.mkdir()
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_GPU_MODE": gpu_mode,
        "TABERO_TEST_PROJECT_ROOT": str(project_root),
        "TABERO_TEST_BASE_MODEL": str(base_model),
        "TABERO_TEST_RESULTS_ROOT": str(results),
        "TABERO_TEST_GPU_LOCK_DIR": str(tmp_path / "locks"),
        "TABERO_TEST_TIMESTAMP": "20260728_120000_formal",
        "TABERO_TEST_PORT": "45678",
        "TABERO_TEST_ALLOW_DIRTY": "1",
        "TABERO_TEST_FREE_DISK_KIB": "2000000000",
    }
    return env, project_root, results


def _run_launcher(bundle, task_id, env, *, dry_run=True):
    command = [
        "bash",
        str(LAUNCHER_PATH),
        "dsrl",
        str(task_id),
        "formal",
        "--dsrl-bundle",
        str(bundle.resolve()),
    ]
    if dry_run:
        command.append("--dry-run")
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )


@pytest.mark.parametrize("task_id", [0, 5])
def test_launcher_dry_run_writes_exact_commands_and_provenance(
    tmp_path, bundle_fixture, task_id
):
    bundle, base_model, _, _ = bundle_fixture
    if task_id == 5:
        _retask_bundle(bundle, task_id)
    env, project_root, results = _launcher_environment(tmp_path, bundle, base_model)

    result = _run_launcher(bundle, task_id, env)

    assert result.returncode == 0, result.stderr
    output = (
        results
        / f"tabero_task{task_id}_firm_dsrl_official_eval_formal_20260728_120000_formal"
    )
    assert output.is_dir()
    server_command = (output / "server_command.txt").read_text().strip()
    assert server_command == (
        "Command: env CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 "
        f"{project_root}/T2-VLA/.venv/bin/python "
        f"{project_root}/T2-VLA/scripts/serve_policy.py --port 45678 "
        f"--dsrl-bundle {bundle.resolve()} policy:checkpoint "
        "--policy.config=pi0_lora_tacfield_tabero "
        f"--policy.dir={base_model.resolve()}"
    )
    client_command = (output / "client_command.txt").read_text().strip()
    assert client_command == (
        "Command: conda run --no-capture-output -n tabero python -u "
        f"{project_root}/Tabero/scripts/tools/run_task_evaluations.py "
        "--policy-model openpi --control-mode tactile --server-host 127.0.0.1 "
        "--server-port 45678 --num-total-experiments 50 "
        "--num-success-steps 8 --max-inference-steps 30 --replan-steps 10 "
        f"--task-suites libero_object --task-ids {task_id} --hdf5-folder "
        f"{project_root}/Tabero/benchmarks/datasets/libero/assembled_hdf5 "
        f"--require-hdf5 --output-dir {output}/raw --output-format both "
        "--seed 11 --prompt-seed 0 --prompt-adverbs firmly tightly "
        "--send-dsrl-raw-image --sim-device cuda:1 "
        "--sim-kit-args=--/renderer/activeGpu=1 --headless"
    )
    run_env = (output / "run.env").read_text()
    assert "TABERO_METHOD=dsrl\n" in run_env
    assert f"TABERO_TASK_ID={task_id}\n" in run_env
    assert "TABERO_RUN_MODE=formal\n" in run_env
    assert f"TABERO_DSRL_BUNDLE={bundle.resolve()}\n" in run_env
    assert (
        f"WANDB_RUN_ID=tabero-official-dsrl-task{task_id}-20260728-120000-formal\n"
        in run_env
    )
    metadata = (output / "start_metadata.env").read_text()
    for key in (
        "TABERO_DSRL_BUNDLE_SHA256",
        "TABERO_DSRL_ACTOR_SHA256",
        "TABERO_DSRL_MANIFEST_SHA256",
        "TABERO_DSRL_AUDIT_SHA256",
        "TABERO_BASE_MODEL_SHA256",
        "TABERO_RLINF_GIT_COMMIT",
        "TABERO_T2_VLA_GIT_COMMIT",
        "TABERO_TABERO_GIT_COMMIT",
        "TABERO_EPHEMERAL_PORT=45678",
        "TABERO_GPU_IDS=0,1",
    ):
        assert key in metadata
    assert (
        (output / "run_status.env")
        .read_text()
        .startswith("TABERO_RUN_STATUS=dry_run\nTABERO_EXIT_STATUS=0\n")
    )
    assert (output / "gpu_samples.csv").read_text().startswith("timestamp_utc,index")
    assert (
        (output / "gpu_process_samples.csv")
        .read_text()
        .startswith("timestamp_utc,gpu_uuid")
    )
    assert (output / "server.log").is_file()
    assert (output / "client.log").is_file()


def test_launcher_output_directory_is_no_clobber(tmp_path, bundle_fixture):
    bundle, base_model, _, _ = bundle_fixture
    env, _, _ = _launcher_environment(tmp_path, bundle, base_model)
    assert _run_launcher(bundle, 0, env).returncode == 0

    second = _run_launcher(bundle, 0, env)

    assert second.returncode != 0
    assert "already exists" in second.stderr


@pytest.mark.parametrize("gpu_mode", ["busy0", "busy1"])
def test_launcher_rejects_busy_fixed_gpu(tmp_path, bundle_fixture, gpu_mode):
    bundle, base_model, _, _ = bundle_fixture
    env, _, results = _launcher_environment(
        tmp_path, bundle, base_model, gpu_mode=gpu_mode
    )

    result = _run_launcher(bundle, 0, env)

    assert result.returncode != 0
    assert "busy" in result.stderr
    assert not any(results.iterdir())


@pytest.mark.parametrize("gpu_mode", ["duplicate", "missing"])
def test_launcher_requires_unique_installed_gpu0_and_gpu1(
    tmp_path, bundle_fixture, gpu_mode
):
    bundle, base_model, _, _ = bundle_fixture
    env, _, results = _launcher_environment(
        tmp_path, bundle, base_model, gpu_mode=gpu_mode
    )

    result = _run_launcher(bundle, 0, env)

    assert result.returncode != 0
    assert "GPU" in result.stderr
    assert not any(results.iterdir())


def test_launcher_rejects_gpu_lease_already_held(tmp_path, bundle_fixture):
    bundle, base_model, _, _ = bundle_fixture
    env, _, results = _launcher_environment(tmp_path, bundle, base_model)
    lock_dir = Path(env["TABERO_TEST_GPU_LOCK_DIR"])
    lock_dir.mkdir()
    with (lock_dir / "gpu_0.lock").open("w") as lock_file:
        flock(lock_file, LOCK_EX | LOCK_NB)
        result = _run_launcher(bundle, 0, env)

    assert result.returncode != 0
    assert "lease" in result.stderr
    assert not any(results.iterdir())


def test_launcher_enforces_disk_gate_before_output(tmp_path, bundle_fixture):
    bundle, base_model, _, _ = bundle_fixture
    env, _, results = _launcher_environment(tmp_path, bundle, base_model)
    env["TABERO_TEST_FREE_DISK_KIB"] = "1"

    result = _run_launcher(bundle, 0, env)

    assert result.returncode != 0
    assert "disk gate" in result.stderr
    assert not any(results.iterdir())


def test_launcher_rejects_formal_test_override(tmp_path, bundle_fixture):
    bundle, base_model, _, _ = bundle_fixture
    env, _, _ = _launcher_environment(tmp_path, bundle, base_model)

    result = _run_launcher(bundle, 0, env, dry_run=False)

    assert result.returncode != 0
    assert "only allowed with --dry-run" in result.stderr


def test_launcher_rejects_symlink_gpu_lock_directory(tmp_path, bundle_fixture):
    bundle, base_model, _, _ = bundle_fixture
    env, _, results = _launcher_environment(tmp_path, bundle, base_model)
    lock_target = tmp_path / "lock-target"
    lock_target.mkdir()
    Path(env["TABERO_TEST_GPU_LOCK_DIR"]).symlink_to(
        lock_target, target_is_directory=True
    )

    result = _run_launcher(bundle, 0, env)

    assert result.returncode != 0
    assert "lock" in result.stderr.lower() or "symlink" in result.stderr.lower()
    assert not any(results.iterdir())


def test_listener_ownership_is_bound_to_launched_process():
    helper = _load_helper()
    listener = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import socket,time; "
                "s=socket.socket(); s.bind(('127.0.0.1',0)); s.listen(); "
                "print(s.getsockname()[1],flush=True); time.sleep(30)"
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    unrelated = subprocess.Popen(["sleep", "30"])
    try:
        assert listener.stdout is not None
        port = int(listener.stdout.readline())
        assert helper.listener_owned_by_process(listener.pid, port)
        assert not helper.listener_owned_by_process(unrelated.pid, port)
    finally:
        listener.terminate()
        listener.wait(timeout=5)
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_gpu_sampler_propagates_nvidia_smi_failure(tmp_path):
    helper = _load_helper()
    failed_smi = tmp_path / "nvidia-smi"
    _write_executable(failed_smi, "#!/usr/bin/env bash\nexit 7\n")
    with pytest.raises(subprocess.CalledProcessError):
        helper.sample_gpus_once(
            tmp_path / "gpu.csv",
            tmp_path / "process.csv",
            nvidia_smi=str(failed_smi),
        )


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["pirl", "0", "formal", "--dsrl-bundle", "/tmp/x"],
        ["dsrl", "1", "formal", "--dsrl-bundle", "/tmp/x"],
        ["dsrl", "0", "smoke", "--dsrl-bundle", "/tmp/x"],
        ["dsrl", "0", "formal"],
        ["dsrl", "0", "formal", "--dsrl-bundle", "relative"],
    ],
)
def test_launcher_rejects_invalid_public_arguments(args):
    result = subprocess.run(
        ["bash", str(LAUNCHER_PATH), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "error:" in result.stderr or "Usage:" in result.stderr
