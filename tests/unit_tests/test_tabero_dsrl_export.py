# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import hashlib
import json
from math import prod

import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file

from rlinf.utils.ckpt_convertor import export_tabero_dsrl_for_t2vla as exporter
from rlinf.utils.dsrl_checkpoint import (
    DSRL_TRAINABLE_MANIFEST_V1,
    DSRL_TRAINABLE_PARAMETER_COUNT,
    DSRL_TRAINABLE_TENSOR_COUNT,
)
from rlinf.utils.dsrl_rollout_sync import (
    DSRL_ROLLOUT_SYNC_MANIFEST_V1,
    DSRL_ROLLOUT_SYNC_PARAMETER_COUNT,
    DSRL_ROLLOUT_SYNC_TENSOR_COUNT,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_dict():
    return {
        name: torch.zeros(shape, dtype=torch.bfloat16)
        for name, shape in DSRL_TRAINABLE_MANIFEST_V1.items()
    }


def _checkpoint(tmp_path, *, task_id=0, step=50, metadata_overrides=None):
    experiment = f"tabero_firm_matrix_dsrl_task{task_id}_formal_test"
    output_root = tmp_path / experiment
    checkpoint = (
        output_root
        / experiment
        / "checkpoints"
        / f"global_step_{step}"
        / "actor"
        / "model_state_dict"
        / "trainable_weights.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    metadata = {
        "format": "trainable_weights",
        "method": "dsrl",
        "task_id": task_id,
        "training_config": (
            f"isaaclab_pi0_dsrl_tacfield_tabero_task{task_id}_firm_8gpu_50step"
        ),
        "target_global_step": 50,
        "step": step,
        "global_step": step,
        "is_final": step == 50,
        "rank": 0,
        "world_size": 4,
        "parameter_count": DSRL_TRAINABLE_TENSOR_COUNT,
        "tensor_count": DSRL_TRAINABLE_TENSOR_COUNT,
        "total_parameter_count": DSRL_TRAINABLE_PARAMETER_COUNT,
    }
    metadata.update(metadata_overrides or {})
    torch.save({"model": _state_dict(), "metadata": metadata}, checkpoint)
    return checkpoint


def _base_model(tmp_path):
    base_model = tmp_path / "base_model"
    base_model.mkdir()
    (base_model / "model.safetensors").write_bytes(b"fixed pi0 tacfield base")
    (base_model / "config.json").write_text("{}\n")
    return base_model


def _formal_config(task_id, base_model):
    return f"""
runner:
  max_epochs: 50
  save_interval: 10
  logger:
    logger_backends: [tensorboard, wandb]
algorithm:
  update_epoch: 200
  gamma: 0.999
  tau: 0.005
env:
  train:
    total_num_envs: 84
    rollout_epoch: 2
    init_params:
      task_id: {task_id}
      prompt_conditions:
        condition_cycle: [firm]
        firm_adverbs: [firmly, tightly]
rollout:
  model:
    model_path: {base_model.resolve()}
actor:
  rollout_sync_prefixes:
    - dsrl_action_noise_net.
    - actor_image_encoder.
    - actor_state_encoder.
    - actor_tactile_encoder.
  model:
    model_path: {base_model.resolve()}
    openpi:
      use_dsrl: true
      dsrl_use_tactile: true
      dsrl_state_dim: 7
      dsrl_action_noise_dim: 32
  fsdp_config:
    trainable_checkpoint_metadata:
      method: dsrl
      task_id: {task_id}
      training_config: isaaclab_pi0_dsrl_tacfield_tabero_task{task_id}_firm_8gpu_50step
      target_global_step: 50
""".lstrip()


def _write_provenance(
    checkpoint,
    base_model,
    *,
    snapshot_task_id=None,
    mode="fresh",
):
    output_root = checkpoint.parents[5]
    checkpoint_task_id = int(
        checkpoint.parents[5]
        .name.split("_task", maxsplit=1)[1]
        .split("_", maxsplit=1)[0]
    )
    snapshot_task_id = (
        checkpoint_task_id if snapshot_task_id is None else snapshot_task_id
    )
    config_snapshot = output_root / "config_snapshot.yaml"
    config_snapshot.write_text(_formal_config(snapshot_task_id, base_model))
    config_hash = _sha256(config_snapshot)
    base_hash = _sha256(base_model / "model.safetensors")
    source_hash = "none"
    if mode == "legacy_migration":
        legacy_config = output_root / "tensorboard" / "config.yaml"
        legacy_config.parent.mkdir()
        legacy_config.write_text("legacy_source: immutable\n")
        source_hash = _sha256(legacy_config)
    (output_root / "provenance.env").write_text(
        "\n".join(
            [
                "TABERO_PROVENANCE_VERSION=1",
                f"TABERO_PROVENANCE_MODE={mode}",
                f"TABERO_CONFIG_SHA256={config_hash}",
                f"TABERO_CONFIG_SNAPSHOT_SHA256={config_hash}",
                f"TABERO_GIT_COMMIT={'a' * 40}",
                "TABERO_GIT_DIRTY=false",
                f"TABERO_BASE_MODEL_PATH={base_model.resolve()}",
                f"TABERO_BASE_MODEL_SHA256={base_hash}",
                f"TABERO_SOURCE_CONFIG_SHA256={source_hash}",
            ]
        )
        + "\n"
    )
    return base_hash


def _replace_provenance_value(checkpoint, key, value):
    provenance_path = checkpoint.parents[5] / "provenance.env"
    lines = provenance_path.read_text().splitlines()
    provenance_path.write_text(
        "\n".join(
            f"{key}={value}" if line.startswith(f"{key}=") else line for line in lines
        )
        + "\n"
    )


def _rewrite_config_snapshot(checkpoint, path, value):
    config_snapshot = checkpoint.parents[5] / "config_snapshot.yaml"
    config = OmegaConf.load(config_snapshot)
    OmegaConf.update(config, path, value, force_add=True)
    OmegaConf.save(config, config_snapshot)
    config_hash = _sha256(config_snapshot)
    _replace_provenance_value(checkpoint, "TABERO_CONFIG_SHA256", config_hash)
    _replace_provenance_value(
        checkpoint,
        "TABERO_CONFIG_SNAPSHOT_SHA256",
        config_hash,
    )


def _export(tmp_path, *, task_id=0, checkpoint=None, **kwargs):
    checkpoint = checkpoint or _checkpoint(tmp_path, task_id=task_id)
    base_model = _base_model(tmp_path)
    base_hash = _write_provenance(checkpoint, base_model)
    output_dir = tmp_path / "bundle"
    manifest = exporter.export_tabero_dsrl_bundle(
        trainable_checkpoint=checkpoint,
        output_dir=output_dir,
        base_model=base_model,
        expected_base_model_sha256=base_hash,
        task_id=task_id,
        **kwargs,
    )
    return manifest, checkpoint, base_model, output_dir


def test_export_writes_strict_actor_bundle_and_audit(tmp_path):
    manifest, checkpoint, base_model, output_dir = _export(tmp_path, task_id=5)

    actor_path = output_dir / "dsrl_actor.safetensors"
    actor = load_file(actor_path)
    assert set(actor) == set(DSRL_ROLLOUT_SYNC_MANIFEST_V1)
    assert len(actor) == DSRL_ROLLOUT_SYNC_TENSOR_COUNT == 48
    assert sum(tensor.numel() for tensor in actor.values()) == (
        DSRL_ROLLOUT_SYNC_PARAMETER_COUNT
    )
    assert {tensor.dtype for tensor in actor.values()} == {torch.bfloat16}

    assert manifest == json.loads((output_dir / "manifest.json").read_text())
    assert manifest["format"] == "tabero_dsrl_t2vla"
    assert manifest["format_version"] == 1
    assert manifest["algorithm"] == "dsrl-sac"
    assert manifest["task_id"] == 5
    assert manifest["global_step"] == 50
    assert manifest["is_final"] is True
    assert manifest["actor_weights"] == "dsrl_actor.safetensors"
    assert manifest["actor_tensor_count"] == 48
    assert manifest["actor_parameter_count"] == 2_311_648
    assert manifest["actor_dtype"] == "bfloat16"
    assert manifest["base_model"] == str(base_model.resolve())
    assert manifest["base_model_sha256"] == _sha256(base_model / "model.safetensors")
    assert manifest["source_checkpoint"] == str(checkpoint.resolve())
    assert manifest["source_checkpoint_sha256"] == _sha256(checkpoint)
    config_snapshot = checkpoint.parents[5] / "config_snapshot.yaml"
    assert manifest["source_config_snapshot"] == str(config_snapshot)
    assert manifest["source_config_snapshot_sha256"] == _sha256(config_snapshot)
    assert manifest["legacy_source_config"] is None
    assert manifest["legacy_source_config_sha256"] is None
    assert manifest["observation_contract"] == {
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
    assert manifest["feature_contract"] == {
        "order": ["state", "image", "tactile"],
        "dims": [64, 64, 64],
        "total_dim": 192,
    }
    assert manifest["noise_contract"] == {
        "dim": 32,
        "horizon": 50,
        "deterministic": "tanh(mean)",
        "broadcast_across_horizon": True,
        "pi0_denoise_steps": 10,
    }

    audit = json.loads((output_dir / "artifact_audit.json").read_text())
    assert audit["status"] == "passed"
    assert all(audit["checks"].values())
    assert audit["actor_weights_sha256"] == _sha256(actor_path)
    assert audit["manifest_sha256"] == _sha256(output_dir / "manifest.json")
    assert {path.name for path in output_dir.iterdir()} == {
        "dsrl_actor.safetensors",
        "manifest.json",
        "artifact_audit.json",
    }


@pytest.mark.parametrize(
    ("metadata_overrides", "message"),
    [
        ({"method": "pirl"}, "method"),
        ({"task_id": 5}, "task"),
        ({"step": 40}, "step"),
        ({"global_step": 40}, "global_step"),
        ({"target_global_step": 40}, "target_global_step"),
        ({"is_final": False}, "final"),
        ({"training_config": "wrong"}, "training_config"),
        ({"world_size": 8}, "world_size"),
    ],
)
def test_export_rejects_wrong_final_metadata(tmp_path, metadata_overrides, message):
    checkpoint = _checkpoint(tmp_path, metadata_overrides=metadata_overrides)
    base_model = _base_model(tmp_path)
    base_hash = _write_provenance(checkpoint, base_model)

    with pytest.raises(ValueError, match=message):
        exporter.export_tabero_dsrl_bundle(
            trainable_checkpoint=checkpoint,
            output_dir=tmp_path / "bundle",
            base_model=base_model,
            expected_base_model_sha256=base_hash,
            task_id=0,
        )

    assert not (tmp_path / "bundle").exists()


def test_export_rejects_non_final_checkpoint_path(tmp_path):
    checkpoint = _checkpoint(tmp_path, step=40)
    base_model = _base_model(tmp_path)
    base_hash = _write_provenance(checkpoint, base_model)

    with pytest.raises(ValueError, match="global_step_50"):
        exporter.export_tabero_dsrl_bundle(
            trainable_checkpoint=checkpoint,
            output_dir=tmp_path / "bundle",
            base_model=base_model,
            expected_base_model_sha256=base_hash,
            task_id=0,
        )


@pytest.mark.parametrize("corruption", ["missing", "extra", "shape", "dtype", "nan"])
def test_export_rejects_corrupt_trainable_tensor_manifest(tmp_path, corruption):
    checkpoint = _checkpoint(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    first_key = next(iter(DSRL_TRAINABLE_MANIFEST_V1))
    if corruption == "missing":
        payload["model"].pop(first_key)
    elif corruption == "extra":
        payload["model"]["unexpected.weight"] = torch.zeros(1, dtype=torch.bfloat16)
    elif corruption == "shape":
        payload["model"][first_key] = torch.zeros(1, dtype=torch.bfloat16)
    elif corruption == "dtype":
        payload["model"][first_key] = payload["model"][first_key].float()
    else:
        payload["model"][first_key].view(-1)[0] = torch.nan
    torch.save(payload, checkpoint)
    base_model = _base_model(tmp_path)
    base_hash = _write_provenance(checkpoint, base_model)

    with pytest.raises(ValueError, match="manifest|dtype|finite"):
        exporter.export_tabero_dsrl_bundle(
            trainable_checkpoint=checkpoint,
            output_dir=tmp_path / "bundle",
            base_model=base_model,
            expected_base_model_sha256=base_hash,
            task_id=0,
        )

    assert not (tmp_path / "bundle").exists()


def test_export_rejects_wrong_base_hash_and_existing_output(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    base_model = _base_model(tmp_path)
    _write_provenance(checkpoint, base_model)

    with pytest.raises(ValueError, match="base model SHA-256"):
        exporter.export_tabero_dsrl_bundle(
            trainable_checkpoint=checkpoint,
            output_dir=tmp_path / "bundle",
            base_model=base_model,
            expected_base_model_sha256="f" * 64,
            task_id=0,
        )

    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        exporter.export_tabero_dsrl_bundle(
            trainable_checkpoint=checkpoint,
            output_dir=output_dir,
            base_model=base_model,
            expected_base_model_sha256=_sha256(base_model / "model.safetensors"),
            task_id=0,
        )


def test_export_rejects_config_snapshot_for_other_task(tmp_path):
    checkpoint = _checkpoint(tmp_path, task_id=0)
    base_model = _base_model(tmp_path)
    base_hash = _write_provenance(checkpoint, base_model, snapshot_task_id=5)

    with pytest.raises(ValueError, match="config snapshot.*task"):
        exporter.export_tabero_dsrl_bundle(
            trainable_checkpoint=checkpoint,
            output_dir=tmp_path / "bundle",
            base_model=base_model,
            expected_base_model_sha256=base_hash,
            task_id=0,
        )


@pytest.mark.parametrize(
    ("path", "invalid_value"),
    [
        ("runner.max_epochs", 49),
        ("runner.save_interval", 9),
        ("runner.logger.logger_backends", ["tensorboard"]),
        ("env.train.total_num_envs", 83),
        ("env.train.rollout_epoch", 1),
        ("algorithm.update_epoch", 199),
        ("algorithm.gamma", 0.99),
        ("algorithm.tau", 0.01),
        ("actor.rollout_sync_prefixes", ["dsrl_action_noise_net."]),
        ("actor.model.openpi.use_dsrl", False),
        ("actor.model.openpi.dsrl_use_tactile", False),
        ("actor.model.openpi.dsrl_state_dim", 8),
        ("actor.model.openpi.dsrl_action_noise_dim", 31),
        ("actor.model.model_path", "/wrong/actor/base"),
        ("rollout.model.model_path", "/wrong/rollout/base"),
        ("actor.fsdp_config.trainable_checkpoint_metadata.method", "pirl"),
        ("actor.fsdp_config.trainable_checkpoint_metadata.task_id", 5),
        (
            "actor.fsdp_config.trainable_checkpoint_metadata.training_config",
            "wrong",
        ),
        (
            "actor.fsdp_config.trainable_checkpoint_metadata.target_global_step",
            40,
        ),
    ],
)
def test_export_rejects_config_snapshot_contract_mutation(
    tmp_path,
    path,
    invalid_value,
):
    checkpoint = _checkpoint(tmp_path)
    base_model = _base_model(tmp_path)
    base_hash = _write_provenance(checkpoint, base_model)
    _rewrite_config_snapshot(checkpoint, path, invalid_value)

    with pytest.raises(ValueError, match="config snapshot"):
        exporter.export_tabero_dsrl_bundle(
            trainable_checkpoint=checkpoint,
            output_dir=tmp_path / "bundle",
            base_model=base_model,
            expected_base_model_sha256=base_hash,
            task_id=0,
        )


@pytest.mark.parametrize("corruption", ["fresh_source", "base_path", "base_hash"])
def test_export_rejects_corrupt_formal_provenance(tmp_path, corruption):
    checkpoint = _checkpoint(tmp_path)
    base_model = _base_model(tmp_path)
    base_hash = _write_provenance(checkpoint, base_model)
    provenance_path = checkpoint.parents[5] / "provenance.env"
    lines = provenance_path.read_text().splitlines()
    replacements = {
        "fresh_source": ("TABERO_SOURCE_CONFIG_SHA256", "f" * 64),
        "base_path": ("TABERO_BASE_MODEL_PATH", str(tmp_path / "wrong-base")),
        "base_hash": ("TABERO_BASE_MODEL_SHA256", "f" * 64),
    }
    key, value = replacements[corruption]
    provenance_path.write_text(
        "\n".join(
            f"{key}={value}" if line.startswith(f"{key}=") else line for line in lines
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="source config|base model"):
        exporter.export_tabero_dsrl_bundle(
            trainable_checkpoint=checkpoint,
            output_dir=tmp_path / "bundle",
            base_model=base_model,
            expected_base_model_sha256=base_hash,
            task_id=0,
        )


@pytest.mark.parametrize(
    ("key", "invalid_value", "message"),
    [
        ("TABERO_PROVENANCE_VERSION", "2", "version"),
        ("TABERO_PROVENANCE_MODE", "unknown", "mode"),
        ("TABERO_CONFIG_SHA256", "f" * 64, "config snapshot SHA-256"),
        (
            "TABERO_CONFIG_SNAPSHOT_SHA256",
            "f" * 64,
            "config snapshot SHA-256",
        ),
    ],
)
def test_export_rejects_invalid_provenance_contract(
    tmp_path,
    key,
    invalid_value,
    message,
):
    checkpoint = _checkpoint(tmp_path)
    base_model = _base_model(tmp_path)
    base_hash = _write_provenance(checkpoint, base_model)
    _replace_provenance_value(checkpoint, key, invalid_value)

    with pytest.raises(ValueError, match=message):
        exporter.export_tabero_dsrl_bundle(
            trainable_checkpoint=checkpoint,
            output_dir=tmp_path / "bundle",
            base_model=base_model,
            expected_base_model_sha256=base_hash,
            task_id=0,
        )


@pytest.mark.parametrize("corruption", ["missing", "extra"])
def test_export_rejects_invalid_provenance_keyspace(tmp_path, corruption):
    checkpoint = _checkpoint(tmp_path)
    base_model = _base_model(tmp_path)
    base_hash = _write_provenance(checkpoint, base_model)
    provenance_path = checkpoint.parents[5] / "provenance.env"
    lines = provenance_path.read_text().splitlines()
    if corruption == "missing":
        lines = lines[1:]
    else:
        lines.append("TABERO_UNEXPECTED=value")
    provenance_path.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="keyspace"):
        exporter.export_tabero_dsrl_bundle(
            trainable_checkpoint=checkpoint,
            output_dir=tmp_path / "bundle",
            base_model=base_model,
            expected_base_model_sha256=base_hash,
            task_id=0,
        )


@pytest.mark.parametrize("corruption", ["missing", "tampered"])
def test_export_rejects_invalid_config_snapshot_artifact(tmp_path, corruption):
    checkpoint = _checkpoint(tmp_path)
    base_model = _base_model(tmp_path)
    base_hash = _write_provenance(checkpoint, base_model)
    config_snapshot = checkpoint.parents[5] / "config_snapshot.yaml"
    if corruption == "missing":
        config_snapshot.unlink()
    else:
        config_snapshot.write_text(config_snapshot.read_text() + "# tampered\n")

    with pytest.raises(ValueError, match="config snapshot"):
        exporter.export_tabero_dsrl_bundle(
            trainable_checkpoint=checkpoint,
            output_dir=tmp_path / "bundle",
            base_model=base_model,
            expected_base_model_sha256=base_hash,
            task_id=0,
        )


def test_export_rejects_tampered_legacy_source_config(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    base_model = _base_model(tmp_path)
    base_hash = _write_provenance(
        checkpoint,
        base_model,
        mode="legacy_migration",
    )
    legacy_config = checkpoint.parents[5] / "tensorboard" / "config.yaml"
    legacy_config.write_text("legacy_source: tampered\n")

    with pytest.raises(ValueError, match="legacy source config SHA-256"):
        exporter.export_tabero_dsrl_bundle(
            trainable_checkpoint=checkpoint,
            output_dir=tmp_path / "bundle",
            base_model=base_model,
            expected_base_model_sha256=base_hash,
            task_id=0,
        )


def test_export_rejects_missing_legacy_source_config(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    base_model = _base_model(tmp_path)
    base_hash = _write_provenance(
        checkpoint,
        base_model,
        mode="legacy_migration",
    )
    (checkpoint.parents[5] / "tensorboard" / "config.yaml").unlink()

    with pytest.raises(ValueError, match="legacy source config does not exist"):
        exporter.export_tabero_dsrl_bundle(
            trainable_checkpoint=checkpoint,
            output_dir=tmp_path / "bundle",
            base_model=base_model,
            expected_base_model_sha256=base_hash,
            task_id=0,
        )


def test_export_records_exact_legacy_source_config(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    base_model = _base_model(tmp_path)
    base_hash = _write_provenance(
        checkpoint,
        base_model,
        mode="legacy_migration",
    )
    legacy_config = checkpoint.parents[5] / "tensorboard" / "config.yaml"

    manifest = exporter.export_tabero_dsrl_bundle(
        trainable_checkpoint=checkpoint,
        output_dir=tmp_path / "bundle",
        base_model=base_model,
        expected_base_model_sha256=base_hash,
        task_id=0,
    )

    assert manifest["legacy_source_config"] == str(legacy_config)
    assert manifest["legacy_source_config_sha256"] == _sha256(legacy_config)


def test_export_accepts_uppercase_legacy_source_config_sha256(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    base_model = _base_model(tmp_path)
    base_hash = _write_provenance(
        checkpoint,
        base_model,
        mode="legacy_migration",
    )
    legacy_config = checkpoint.parents[5] / "tensorboard" / "config.yaml"
    _replace_provenance_value(
        checkpoint,
        "TABERO_SOURCE_CONFIG_SHA256",
        _sha256(legacy_config).upper(),
    )

    manifest = exporter.export_tabero_dsrl_bundle(
        trainable_checkpoint=checkpoint,
        output_dir=tmp_path / "bundle",
        base_model=base_model,
        expected_base_model_sha256=base_hash,
        task_id=0,
    )

    assert manifest["legacy_source_config_sha256"] == _sha256(legacy_config)


@pytest.mark.parametrize(
    "artifact",
    ["checkpoint", "base_model", "provenance", "config_snapshot", "legacy"],
)
def test_export_rejects_source_artifact_changed_after_validation(
    tmp_path,
    monkeypatch,
    artifact,
):
    checkpoint = _checkpoint(tmp_path)
    base_model = _base_model(tmp_path)
    base_hash = _write_provenance(
        checkpoint,
        base_model,
        mode="legacy_migration",
    )
    output_root = checkpoint.parents[5]
    real_validate = exporter._validate_provenance

    def validate_then_mutate(*args, **kwargs):
        result = real_validate(*args, **kwargs)
        if artifact == "checkpoint":
            checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
        elif artifact == "base_model":
            base_weights = base_model / "model.safetensors"
            base_weights.write_bytes(base_weights.read_bytes() + b"changed")
        elif artifact == "provenance":
            _replace_provenance_value(
                checkpoint,
                "TABERO_GIT_COMMIT",
                "b" * 40,
            )
        elif artifact == "config_snapshot":
            config_snapshot = output_root / "config_snapshot.yaml"
            config_snapshot.write_text(config_snapshot.read_text() + "# changed\n")
        else:
            legacy_config = output_root / "tensorboard" / "config.yaml"
            legacy_config.write_text("legacy_source: changed\n")
        return result

    monkeypatch.setattr(exporter, "_validate_provenance", validate_then_mutate)

    with pytest.raises(ValueError, match="changed during export"):
        exporter.export_tabero_dsrl_bundle(
            trainable_checkpoint=checkpoint,
            output_dir=tmp_path / "bundle",
            base_model=base_model,
            expected_base_model_sha256=base_hash,
            task_id=0,
        )


def test_export_loads_sidecar_in_weights_only_mode(tmp_path, monkeypatch):
    checkpoint = _checkpoint(tmp_path)
    base_model = _base_model(tmp_path)
    base_hash = _write_provenance(checkpoint, base_model)
    real_load = torch.load
    weights_only_calls = []

    def tracked_load(*args, **kwargs):
        weights_only_calls.append(kwargs.get("weights_only"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(exporter.torch, "load", tracked_load)
    exporter.export_tabero_dsrl_bundle(
        trainable_checkpoint=checkpoint,
        output_dir=tmp_path / "bundle",
        base_model=base_model,
        expected_base_model_sha256=base_hash,
        task_id=0,
    )

    assert weights_only_calls == [True]


def test_canonical_manifests_have_expected_sizes():
    assert len(DSRL_TRAINABLE_MANIFEST_V1) == 220
    assert sum(prod(shape) for shape in DSRL_TRAINABLE_MANIFEST_V1.values()) == (
        5_183_754
    )
    assert len(DSRL_ROLLOUT_SYNC_MANIFEST_V1) == 48
    assert sum(prod(shape) for shape in DSRL_ROLLOUT_SYNC_MANIFEST_V1.values()) == (
        2_311_648
    )
