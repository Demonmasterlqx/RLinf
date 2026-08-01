# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import hashlib
import json
import sys
from math import prod
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file

from rlinf.utils.ckpt_convertor import export_tabero_dsrl_for_t2vla as exporter
from rlinf.utils.dsrl_checkpoint import (
    DSRL_TRAINABLE_MANIFEST_V2,
    DSRL_TRAINABLE_PARAMETER_COUNT,
    DSRL_TRAINABLE_TENSOR_COUNT,
)
from rlinf.utils.dsrl_observation import DSRL_OBSERVATION_SEMANTICS
from rlinf.utils.dsrl_reward import DSRL_REWARD_SEMANTICS
from rlinf.utils.dsrl_rollout_sync import (
    DSRL_ROLLOUT_SYNC_MANIFEST_V2,
    DSRL_ROLLOUT_SYNC_PARAMETER_COUNT,
    DSRL_ROLLOUT_SYNC_TENSOR_COUNT,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_dict():
    return {
        name: torch.zeros(shape, dtype=torch.bfloat16)
        for name, shape in DSRL_TRAINABLE_MANIFEST_V2.items()
    }


SMALL4GPU40_CONFIG = "isaaclab_pi0_dsrl_tacfield_tabero_task5_firm_4gpu_40step_small"
SMALL4GPU40_PROFILE = "task5_4gpu_40step_small"
TASK0_SELECTED_STEP10_PROFILE = "task0_selected_step10"
TASK0_FORMAL60_PROFILE = "task0_8gpu_60step"
TASK0_FORMAL60_SELECTED_PROFILE_TEMPLATE = "task0_8gpu_60step_selected_step{step}"


def _checkpoint(
    tmp_path,
    *,
    task_id=0,
    step=50,
    metadata_overrides=None,
    profile="formal",
):
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
    profile_metadata = {
        "formal": {
            "training_config": (
                f"isaaclab_pi0_dsrl_tacfield_tabero_task{task_id}_firm_8gpu_50step"
            ),
            "target_global_step": 50,
            "world_size": 4,
        },
        "small4gpu40": {
            "training_config": SMALL4GPU40_CONFIG,
            "target_global_step": 40,
            "world_size": 2,
        },
        "formal60": {
            "training_config": (
                "isaaclab_pi0_dsrl_tacfield_tabero_task0_firm_8gpu_60step"
            ),
            "target_global_step": 60,
            "world_size": 4,
        },
    }[profile]
    metadata = {
        "format": "trainable_weights",
        "method": "dsrl",
        "reward_semantics": DSRL_REWARD_SEMANTICS,
        "observation_semantics": DSRL_OBSERVATION_SEMANTICS,
        "manifest_version": 2,
        "task_id": task_id,
        "training_config": profile_metadata["training_config"],
        "target_global_step": profile_metadata["target_global_step"],
        "step": step,
        "global_step": step,
        "is_final": step == profile_metadata["target_global_step"],
        "rank": 0,
        "world_size": profile_metadata["world_size"],
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


def _formal_config(task_id, base_model, *, profile="formal"):
    is_custom = profile == "small4gpu40"
    is_formal60 = profile == "formal60"
    training_config = (
        SMALL4GPU40_CONFIG
        if is_custom
        else (
            "isaaclab_pi0_dsrl_tacfield_tabero_task0_firm_8gpu_60step"
            if is_formal60
            else f"isaaclab_pi0_dsrl_tacfield_tabero_task{task_id}_firm_8gpu_50step"
        )
    )
    config = {
        "runner": {
            "max_epochs": 40 if is_custom else (60 if is_formal60 else 50),
            "save_interval": 10,
            "logger": {"logger_backends": ["tensorboard", "wandb"]},
        },
        "algorithm": {
            "update_epoch": 20 if is_custom else 200,
            "gamma": 0.999,
            "dsrl_reward_semantics": DSRL_REWARD_SEMANTICS,
            "dsrl_observation_semantics": DSRL_OBSERVATION_SEMANTICS,
            "tau": 0.005,
        },
        "env": {
            "train": {
                "total_num_envs": 20 if is_custom else 84,
                "rollout_epoch": 1 if is_custom else 2,
                "init_params": {
                    "task_id": task_id,
                    "prompt_conditions": {
                        "condition_cycle": ["firm"],
                        "firm_adverbs": ["firmly", "tightly"],
                    },
                },
            }
        },
        "rollout": {"model": {"model_path": str(base_model.resolve())}},
        "actor": {
            "rollout_sync_prefixes": [
                "dsrl_action_noise_net.",
                "actor_image_encoder.",
                "actor_state_encoder.",
                "actor_tactile_encoder.",
            ],
            "model": {
                "model_path": str(base_model.resolve()),
                "openpi": {
                    "use_dsrl": True,
                    "dsrl_use_tactile": True,
                    "dsrl_num_images": 2,
                    "dsrl_state_dim": 7,
                    "dsrl_action_noise_dim": 32,
                },
            },
            "fsdp_config": {
                "trainable_checkpoint_metadata": {
                    "method": "dsrl",
                    "task_id": task_id,
                    "training_config": training_config,
                    "target_global_step": (
                        40 if is_custom else (60 if is_formal60 else 50)
                    ),
                }
            },
        },
    }
    if profile == "small4gpu40":
        config["cluster"] = {
            "component_placement": {
                "actor": "2-3",
                "rollout": "0-1",
                "env": "0-1",
            }
        }
        config["algorithm"].update(
            train_actor_steps=10,
            replay_buffer={"min_buffer_size": 5},
        )
        config["env"]["train"].update(
            max_steps_per_rollout_epoch=360,
            max_episode_steps=360,
        )
        config["env"]["train"]["init_params"].update(
            max_episode_steps=360,
            marker_history_len=8,
            combined_marker_count=198,
            main_image_key="agentview_rgb",
            wrist_image_key="eye_in_hand_rgb",
            marker_motion_key="gripper_marker_motion",
        )
        config["actor"].update(micro_batch_size=2, global_batch_size=20)
    return OmegaConf.to_yaml(OmegaConf.create(config))


def _write_provenance(
    checkpoint,
    base_model,
    *,
    snapshot_task_id=None,
    mode="fresh",
    profile="formal",
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
    config_snapshot.write_text(
        _formal_config(snapshot_task_id, base_model, profile=profile)
    )
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


def test_export_formal_profile_accepts_original_minimal_snapshot_contract(tmp_path):
    manifest, checkpoint, _, _ = _export(tmp_path, task_id=0)
    config = OmegaConf.load(checkpoint.parents[5] / "config_snapshot.yaml")

    assert manifest["global_step"] == 50
    for path in (
        "cluster.component_placement.actor",
        "env.train.max_steps_per_rollout_epoch",
        "env.train.max_episode_steps",
        "algorithm.replay_buffer.min_buffer_size",
        "algorithm.train_actor_steps",
        "actor.global_batch_size",
        "actor.micro_batch_size",
    ):
        assert OmegaConf.select(config, path, default=None) is None


def test_export_accepts_final_task0_8gpu_60step_profile(tmp_path):
    checkpoint = _checkpoint(
        tmp_path,
        task_id=0,
        step=60,
        profile="formal60",
    )
    base_model = _base_model(tmp_path)
    base_hash = _write_provenance(
        checkpoint,
        base_model,
        profile="formal60",
    )
    output_dir = tmp_path / "bundle"

    manifest = exporter.export_tabero_dsrl_bundle(
        trainable_checkpoint=checkpoint,
        output_dir=output_dir,
        base_model=base_model,
        expected_base_model_sha256=base_hash,
        task_id=0,
        training_profile=TASK0_FORMAL60_PROFILE,
    )

    assert manifest["task_id"] == 0
    assert manifest["global_step"] == 60
    assert manifest["is_final"] is True
    assert manifest["training_config"].endswith("task0_firm_8gpu_60step")


@pytest.mark.parametrize("step", [10, 20, 30, 40, 50])
def test_export_accepts_selected_non_final_task0_8gpu_60step_profiles(
    tmp_path,
    step,
):
    checkpoint = _checkpoint(
        tmp_path,
        task_id=0,
        step=step,
        profile="formal60",
    )
    base_model = _base_model(tmp_path)
    base_hash = _write_provenance(
        checkpoint,
        base_model,
        profile="formal60",
    )
    output_dir = tmp_path / "bundle"

    manifest = exporter.export_tabero_dsrl_bundle(
        trainable_checkpoint=checkpoint,
        output_dir=output_dir,
        base_model=base_model,
        expected_base_model_sha256=base_hash,
        task_id=0,
        training_profile=TASK0_FORMAL60_SELECTED_PROFILE_TEMPLATE.format(step=step),
    )

    assert manifest["task_id"] == 0
    assert manifest["global_step"] == step
    assert manifest["is_final"] is False
    assert manifest["training_config"].endswith("task0_firm_8gpu_60step")


def test_export_writes_strict_actor_bundle_and_audit(tmp_path):
    manifest, checkpoint, base_model, output_dir = _export(tmp_path, task_id=5)

    actor_path = output_dir / "dsrl_actor.safetensors"
    actor = load_file(actor_path)
    assert set(actor) == set(DSRL_ROLLOUT_SYNC_MANIFEST_V2)
    assert len(actor) == DSRL_ROLLOUT_SYNC_TENSOR_COUNT == 48
    assert sum(tensor.numel() for tensor in actor.values()) == (
        DSRL_ROLLOUT_SYNC_PARAMETER_COUNT
    )
    assert {tensor.dtype for tensor in actor.values()} == {torch.bfloat16}

    assert manifest == json.loads((output_dir / "manifest.json").read_text())
    assert manifest["format"] == "tabero_dsrl_t2vla"
    assert manifest["format_version"] == 2
    assert manifest["algorithm"] == "dsrl-sac"
    assert manifest["reward_semantics"] == DSRL_REWARD_SEMANTICS
    assert manifest["observation_semantics"] == DSRL_OBSERVATION_SEMANTICS
    assert manifest["task_id"] == 5
    assert manifest["global_step"] == 50
    assert manifest["is_final"] is True
    assert manifest["actor_weights"] == "dsrl_actor.safetensors"
    assert manifest["actor_tensor_count"] == 48
    assert manifest["actor_parameter_count"] == 2_319_840
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
    assert manifest["actor_manifest_version"] == 2
    image_contract = {
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
    }
    wrist_contract = {
        **image_contract,
        "key": "dsrl_raw_wrist_image",
    }
    assert manifest["observation_contract"] == {
        "main_image": image_contract,
        "wrist_image": wrist_contract,
        "state": {"key": "state", "shape": [7], "dtype": "float32"},
        "tactile": {
            "key": "tactile_marker_motion",
            "shape": [9, 198, 2],
            "dtype": "float32",
            "encoder_shape": [9, 396],
        },
    }
    assert manifest["feature_contract"] == {
        "order": ["state", "main_image", "wrist_image", "tactile"],
        "dims": [64, 64, 64, 64],
        "total_dim": 256,
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
    assert audit["format_version"] == 2
    assert audit["reward_semantics"] == DSRL_REWARD_SEMANTICS
    assert audit["observation_semantics"] == DSRL_OBSERVATION_SEMANTICS
    assert all(audit["checks"].values())
    assert audit["actor_weights_sha256"] == _sha256(actor_path)
    assert audit["manifest_sha256"] == _sha256(output_dir / "manifest.json")
    assert {path.name for path in output_dir.iterdir()} == {
        "dsrl_actor.safetensors",
        "manifest.json",
        "artifact_audit.json",
    }


def test_export_accepts_final_task5_small4gpu40_profile(tmp_path):
    checkpoint = _checkpoint(
        tmp_path,
        task_id=5,
        step=40,
        profile="small4gpu40",
    )
    base_model = _base_model(tmp_path)
    base_hash = _write_provenance(
        checkpoint,
        base_model,
        profile="small4gpu40",
    )

    manifest = exporter.export_tabero_dsrl_bundle(
        trainable_checkpoint=checkpoint,
        output_dir=tmp_path / "bundle",
        base_model=base_model,
        expected_base_model_sha256=base_hash,
        task_id=5,
        training_profile=SMALL4GPU40_PROFILE,
    )

    assert manifest["task_id"] == 5
    assert manifest["global_step"] == 40
    assert manifest["is_final"] is True
    assert manifest["training_config"] == SMALL4GPU40_CONFIG
    audit = json.loads((tmp_path / "bundle/artifact_audit.json").read_text())
    assert audit["task_id"] == 5
    assert audit["global_step"] == 40


@pytest.mark.parametrize(
    ("metadata_overrides", "message"),
    [
        ({"method": "pirl"}, "method"),
        ({"reward_semantics": None}, "reward_semantics"),
        ({"observation_semantics": None}, "observation_semantics"),
        ({"manifest_version": 1}, "manifest_version"),
        ({"task_id": 0}, "task_id"),
        ({"step": 39}, "step"),
        ({"global_step": 39}, "global_step"),
        ({"target_global_step": 50}, "target_global_step"),
        ({"is_final": False}, "final"),
        (
            {
                "training_config": (
                    "isaaclab_pi0_dsrl_tacfield_tabero_task5_firm_8gpu_50step"
                )
            },
            "training_config",
        ),
        ({"world_size": 4}, "world_size"),
    ],
)
def test_export_rejects_wrong_small4gpu40_metadata(
    tmp_path,
    metadata_overrides,
    message,
):
    checkpoint = _checkpoint(
        tmp_path,
        task_id=5,
        step=40,
        profile="small4gpu40",
        metadata_overrides=metadata_overrides,
    )
    base_model = _base_model(tmp_path)
    base_hash = _write_provenance(
        checkpoint,
        base_model,
        profile="small4gpu40",
    )

    with pytest.raises(ValueError, match=message):
        exporter.export_tabero_dsrl_bundle(
            trainable_checkpoint=checkpoint,
            output_dir=tmp_path / "bundle",
            base_model=base_model,
            expected_base_model_sha256=base_hash,
            task_id=5,
            training_profile=SMALL4GPU40_PROFILE,
        )


@pytest.mark.parametrize(
    ("path", "invalid_value"),
    [
        ("runner.max_epochs", 50),
        ("cluster.component_placement.actor", "0-1"),
        ("cluster.component_placement.rollout", "2-3"),
        ("cluster.component_placement.env", "2-3"),
        ("env.train.total_num_envs", 84),
        ("env.train.rollout_epoch", 2),
        ("env.train.max_steps_per_rollout_epoch", 180),
        ("env.train.max_episode_steps", 180),
        ("env.train.init_params.max_episode_steps", 180),
        ("env.train.init_params.marker_history_len", 7),
        ("env.train.init_params.combined_marker_count", 197),
        ("env.train.init_params.main_image_key", "wrong"),
        ("env.train.init_params.wrist_image_key", "wrong"),
        ("env.train.init_params.marker_motion_key", "wrong"),
        ("algorithm.update_epoch", 200),
        ("algorithm.dsrl_reward_semantics", "legacy"),
        ("algorithm.dsrl_observation_semantics", "single_camera_v1"),
        ("actor.model.openpi.dsrl_num_images", 1),
        ("algorithm.replay_buffer.min_buffer_size", 10),
        ("algorithm.train_actor_steps", 9),
        ("actor.global_batch_size", 40),
        ("actor.micro_batch_size", 4),
    ],
)
def test_export_rejects_small4gpu40_config_snapshot_mutation(
    tmp_path,
    path,
    invalid_value,
):
    checkpoint = _checkpoint(
        tmp_path,
        task_id=5,
        step=40,
        profile="small4gpu40",
    )
    base_model = _base_model(tmp_path)
    base_hash = _write_provenance(
        checkpoint,
        base_model,
        profile="small4gpu40",
    )
    _rewrite_config_snapshot(checkpoint, path, invalid_value)

    with pytest.raises(ValueError, match="config snapshot"):
        exporter.export_tabero_dsrl_bundle(
            trainable_checkpoint=checkpoint,
            output_dir=tmp_path / "bundle",
            base_model=base_model,
            expected_base_model_sha256=base_hash,
            task_id=5,
            training_profile=SMALL4GPU40_PROFILE,
        )


def test_export_rejects_task5_small4gpu40_without_explicit_profile(tmp_path):
    checkpoint = _checkpoint(
        tmp_path,
        task_id=5,
        step=40,
        profile="small4gpu40",
    )
    base_model = _base_model(tmp_path)
    base_hash = _write_provenance(
        checkpoint,
        base_model,
        profile="small4gpu40",
    )

    with pytest.raises(ValueError, match="global_step_50|training profile"):
        exporter.export_tabero_dsrl_bundle(
            trainable_checkpoint=checkpoint,
            output_dir=tmp_path / "bundle",
            base_model=base_model,
            expected_base_model_sha256=base_hash,
            task_id=5,
        )


def test_export_rejects_task0_with_small4gpu40_profile(tmp_path):
    checkpoint = _checkpoint(tmp_path, task_id=0, step=40)
    base_model = _base_model(tmp_path)
    base_hash = _write_provenance(checkpoint, base_model)

    with pytest.raises(ValueError, match="Task 5|task_id"):
        exporter.export_tabero_dsrl_bundle(
            trainable_checkpoint=checkpoint,
            output_dir=tmp_path / "bundle",
            base_model=base_model,
            expected_base_model_sha256=base_hash,
            task_id=0,
            training_profile=SMALL4GPU40_PROFILE,
        )


def test_export_cli_defaults_to_formal_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_tabero_dsrl_for_t2vla.py",
            "--trainable-checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--output-dir",
            str(tmp_path / "bundle"),
            "--base-model",
            str(tmp_path / "base"),
            "--expected-base-model-sha256",
            "f" * 64,
            "--task-id",
            "0",
        ],
    )

    args = exporter._parse_args()

    assert args.training_profile == "formal_8gpu_50step"


def test_export_task0_selected_step10_preserves_non_final_source(tmp_path):
    checkpoint = _checkpoint(tmp_path, task_id=0, step=10)
    base_model = _base_model(tmp_path)
    base_hash = _write_provenance(checkpoint, base_model)

    manifest = exporter.export_tabero_dsrl_bundle(
        trainable_checkpoint=checkpoint,
        output_dir=tmp_path / "bundle",
        base_model=base_model,
        expected_base_model_sha256=base_hash,
        task_id=0,
        training_profile=TASK0_SELECTED_STEP10_PROFILE,
    )

    assert manifest["task_id"] == 0
    assert manifest["global_step"] == 10
    assert manifest["is_final"] is False
    assert manifest["training_config"] == (
        "isaaclab_pi0_dsrl_tacfield_tabero_task0_firm_8gpu_50step"
    )
    assert (
        json.loads((tmp_path / "bundle" / "artifact_audit.json").read_text())["status"]
        == "passed"
    )


@pytest.mark.parametrize(
    ("metadata_overrides", "message"),
    [
        ({"target_global_step": 10}, "target_global_step"),
        ({"is_final": True}, "is_final"),
        ({"training_config": "wrong"}, "training_config"),
        ({"world_size": 2}, "world_size"),
    ],
)
def test_export_task0_selected_step10_rejects_wrong_metadata(
    tmp_path,
    metadata_overrides,
    message,
):
    checkpoint = _checkpoint(
        tmp_path,
        task_id=0,
        step=10,
        metadata_overrides=metadata_overrides,
    )
    base_model = _base_model(tmp_path)
    base_hash = _write_provenance(checkpoint, base_model)

    with pytest.raises(ValueError, match=message):
        exporter.export_tabero_dsrl_bundle(
            trainable_checkpoint=checkpoint,
            output_dir=tmp_path / "bundle",
            base_model=base_model,
            expected_base_model_sha256=base_hash,
            task_id=0,
            training_profile=TASK0_SELECTED_STEP10_PROFILE,
        )


@pytest.mark.parametrize(
    ("task_id", "step", "profile", "message"),
    [
        (0, 10, "formal_8gpu_50step", "global_step_50"),
        (0, 20, TASK0_SELECTED_STEP10_PROFILE, "global_step_10"),
        (5, 10, TASK0_SELECTED_STEP10_PROFILE, "Task 0|task_id"),
    ],
)
def test_export_rejects_non_allowlisted_selected_checkpoint_paths(
    tmp_path,
    task_id,
    step,
    profile,
    message,
):
    checkpoint = _checkpoint(tmp_path, task_id=task_id, step=step)
    base_model = _base_model(tmp_path)
    base_hash = _write_provenance(checkpoint, base_model)

    with pytest.raises(ValueError, match=message):
        exporter.export_tabero_dsrl_bundle(
            trainable_checkpoint=checkpoint,
            output_dir=tmp_path / "bundle",
            base_model=base_model,
            expected_base_model_sha256=base_hash,
            task_id=task_id,
            training_profile=profile,
        )


@pytest.mark.parametrize(
    ("metadata_overrides", "message"),
    [
        ({"method": "pirl"}, "method"),
        ({"observation_semantics": "single_camera_v1"}, "observation_semantics"),
        ({"manifest_version": 1}, "manifest_version"),
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
    first_key = next(iter(DSRL_TRAINABLE_MANIFEST_V2))
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
        ("algorithm.dsrl_observation_semantics", "single_camera_v1"),
        ("algorithm.tau", 0.01),
        ("actor.rollout_sync_prefixes", ["dsrl_action_noise_net."]),
        ("actor.model.openpi.use_dsrl", False),
        ("actor.model.openpi.dsrl_use_tactile", False),
        ("actor.model.openpi.dsrl_num_images", 1),
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


def test_export_uses_captured_checkpoint_bytes_during_aba_swap(
    tmp_path,
    monkeypatch,
):
    checkpoint = _checkpoint(tmp_path)
    base_model = _base_model(tmp_path)
    base_hash = _write_provenance(checkpoint, base_model)
    original_bytes = checkpoint.read_bytes()
    replacement_payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    replacement_payload["model"] = {
        key: torch.ones_like(tensor)
        for key, tensor in replacement_payload["model"].items()
    }
    replacement_checkpoint = tmp_path / "replacement.pt"
    torch.save(replacement_payload, replacement_checkpoint)
    replacement_bytes = replacement_checkpoint.read_bytes()
    real_load = exporter.torch.load

    def aba_load(source, *args, **kwargs):
        checkpoint.write_bytes(replacement_bytes)
        try:
            return real_load(source, *args, **kwargs)
        finally:
            checkpoint.write_bytes(original_bytes)

    monkeypatch.setattr(exporter.torch, "load", aba_load)

    manifest = exporter.export_tabero_dsrl_bundle(
        trainable_checkpoint=checkpoint,
        output_dir=tmp_path / "bundle",
        base_model=base_model,
        expected_base_model_sha256=base_hash,
        task_id=0,
    )

    actor = load_file(tmp_path / "bundle" / "dsrl_actor.safetensors")
    assert all(torch.count_nonzero(tensor).item() == 0 for tensor in actor.values())
    assert (
        manifest["source_checkpoint_sha256"]
        == hashlib.sha256(original_bytes).hexdigest()
    )


def test_export_atomic_publish_rejects_racing_empty_target(tmp_path, monkeypatch):
    checkpoint = _checkpoint(tmp_path)
    base_model = _base_model(tmp_path)
    base_hash = _write_provenance(checkpoint, base_model)
    output_dir = (tmp_path / "bundle").resolve()
    real_require_sources_unchanged = exporter._require_sources_unchanged
    real_exists = Path.exists

    def validate_sources_then_create_target(**kwargs):
        real_require_sources_unchanged(**kwargs)
        output_dir.mkdir()

    def stale_exists(path):
        if path == output_dir:
            return False
        return real_exists(path)

    monkeypatch.setattr(
        exporter,
        "_require_sources_unchanged",
        validate_sources_then_create_target,
    )
    monkeypatch.setattr(Path, "exists", stale_exists)

    with pytest.raises(FileExistsError, match="already exists"):
        exporter.export_tabero_dsrl_bundle(
            trainable_checkpoint=checkpoint,
            output_dir=output_dir,
            base_model=base_model,
            expected_base_model_sha256=base_hash,
            task_id=0,
        )

    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []


@pytest.mark.parametrize("value", [None, 123, b"0" * 64])
def test_require_sha256_rejects_non_string_values(value):
    with pytest.raises(TypeError, match="SHA-256.*string"):
        exporter._require_sha256(value, "test SHA-256")


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
    assert len(DSRL_TRAINABLE_MANIFEST_V2) == 220
    assert sum(prod(shape) for shape in DSRL_TRAINABLE_MANIFEST_V2.values()) == (
        5_273_866
    )
    assert len(DSRL_ROLLOUT_SYNC_MANIFEST_V2) == 48
    assert sum(prod(shape) for shape in DSRL_ROLLOUT_SYNC_MANIFEST_V2.values()) == (
        2_319_840
    )
