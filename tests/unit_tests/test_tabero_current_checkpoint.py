# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Current-format tests using synthetic configuration and tiny CPU tensors."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.envs.isaaclab.tasks.tabero_tacfield import (
    validate_tabero_chunk_boundary_mode,
)
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.utils.tabero_ppo_boundary import (
    TABERO_PPO_DEFAULT_RESET_TRANSITION_BOUNDARY_SEMANTICS,
    TABERO_REALWORLD_CHECKPOINT_FORMAT,
    build_realworld_pirl_checkpoint_metadata,
    validate_realworld_pirl_checkpoint_format,
    validate_tabero_ppo_checkpoint_boundary_metadata,
)


@pytest.fixture
def runtime():
    return OmegaConf.create(
        {
            "algorithm": {
                "tabero_ppo_transition_boundary_semantics": TABERO_PPO_DEFAULT_RESET_TRANSITION_BOUNDARY_SEMANTICS
            },
            "runner": {"max_epochs": 300, "max_steps": 100},
            "actor": {
                "fsdp_config": {
                    "save_trainable_model_weights": True,
                    "gradient_checkpointing": False,
                },
                "model": {
                    "deployment_config_name": "synthetic_deployment",
                    "export_norm_asset_id": "synthetic_norm",
                    "num_action_chunks": 10,
                    "openpi": {
                        "action_horizon": 50,
                        "effective_action_dim": 13,
                        "discrete_state_input": False,
                        "tactile_prefix_dim_in": 12,
                        "tactile_prefix_history": 2,
                    },
                },
            },
            "env": {
                "train": {
                    "init_params": {
                        "task_suite": "synthetic",
                        "task_id": 6,
                        "target_object": "cookie",
                        "task_description": "synthetic prompt",
                        "reset_source": "task_config_default_reset",
                    }
                }
            },
        }
    )


@pytest.mark.parametrize("step,is_final", [(99, False), (100, True)])
def test_save_and_resume_current_format_from_runtime(
    runtime, monkeypatch, tmp_path, step, is_final
):
    manager = FSDPModelManager.__new__(FSDPModelManager)
    manager.model = torch.nn.Linear(2, 1)
    manager.cfg = runtime
    manager._cfg = runtime.actor
    manager._logger = SimpleNamespace(info=lambda *args: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    manager._save_trainable_model_weights(str(tmp_path), step=step)
    metadata = validate_tabero_ppo_checkpoint_boundary_metadata(
        tmp_path,
        expected_semantics=TABERO_PPO_DEFAULT_RESET_TRANSITION_BOUNDARY_SEMANTICS,
    )
    assert metadata["format"] == "trainable_weights"
    assert metadata["checkpoint_format"] == TABERO_REALWORLD_CHECKPOINT_FORMAT
    assert metadata["global_step"] == step
    assert metadata["is_final"] is is_final
    assert metadata["target_global_step"] == 100
    assert metadata["target_object"] == "cookie"
    assert metadata["deployment_config_name"] == "synthetic_deployment"
    assert metadata["normalization_asset_id"] == "synthetic_norm"
    assert "training_config" not in metadata
    assert "tabero_ppo_transition_boundary_semantics" not in metadata


def test_metadata_tracks_runtime_changes(runtime):
    runtime.env.train.init_params.target_object = "changed"
    runtime.actor.model.openpi.discrete_state_input = True
    runtime.actor.fsdp_config.gradient_checkpointing = True
    runtime.runner.max_steps = -1
    metadata = build_realworld_pirl_checkpoint_metadata(runtime)
    assert metadata["target_object"] == "changed"
    assert metadata["discrete_state_input"] is True
    assert metadata["gradient_checkpointing"] is True
    assert metadata["target_global_step"] == 300


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"checkpoint_format": "old"},
        {
            "tabero_ppo_transition_boundary_semantics": "terminal_observation_first_done_prefix_logprob_default_reset_gripper_map_v2",
        },
    ],
)
def test_old_checkpoint_formats_are_rejected(metadata, tmp_path):
    with pytest.raises(
        ValueError, match="Unsupported RealWorld PiRL checkpoint format"
    ):
        validate_realworld_pirl_checkpoint_format(metadata)
    destination = tmp_path / "model_state_dict"
    destination.mkdir()
    torch.save(
        {"model": {}, "metadata": metadata}, destination / "trainable_weights.pt"
    )
    with pytest.raises(
        ValueError, match="Unsupported RealWorld PiRL checkpoint format"
    ):
        validate_tabero_ppo_checkpoint_boundary_metadata(
            tmp_path,
            expected_semantics=TABERO_PPO_DEFAULT_RESET_TRANSITION_BOUNDARY_SEMANTICS,
        )


@pytest.mark.parametrize("mode", [None, "legacy", "unknown"])
def test_no_legacy_chunk_fallback(mode):
    with pytest.raises(ValueError, match="Unsupported Tabero chunk_boundary_mode"):
        validate_tabero_chunk_boundary_mode(mode)


def test_independent_hdf5_mode_remains_available():
    assert (
        validate_tabero_chunk_boundary_mode("terminal_safe_hdf5_v1")
        == "terminal_safe_hdf5_v1"
    )


@pytest.mark.parametrize(
    "mode",
    [
        None,
        "terminal_observation_first_done_prefix_logprob_default_reset_gripper_map_v2",
    ],
)
def test_realworld_does_not_fall_back_to_generic_ppo(runtime, mode):
    from rlinf.config import validate_embodied_cfg

    runtime.actor.model.model_type = "openpi"
    runtime.actor.model.openpi.pi05 = True
    runtime.env.train.init_params.id = (
        "Isaac-RealWorld-GentleGrasp-XarmUmi-Hybrid-Tactile-v0"
    )
    runtime.algorithm.tabero_ppo_transition_boundary_semantics = mode
    with pytest.raises(ValueError, match="missing or old boundary modes"):
        validate_embodied_cfg(runtime)


def test_export_accepts_generated_metadata_without_config_filename(
    runtime, monkeypatch
):
    from pathlib import Path

    from rlinf.utils.ckpt_convertor import export_openpi_lora_for_t2vla as exporter

    metadata = build_realworld_pirl_checkpoint_metadata(runtime)
    metadata.update(
        format="trainable_weights", step=100, global_step=100, is_final=True
    )
    monkeypatch.setattr(exporter, "_checkpoint_path", lambda path: Path(path))
    monkeypatch.setattr(exporter, "_sha256", lambda path: "synthetic-not-a-digest")
    monkeypatch.setattr(exporter, "_safetensor_count", lambda path: 1)
    result = exporter._build_export_metadata(
        train_config_path="/synthetic/renamed.yaml",
        ckpt_path="/synthetic/checkpoint.pt",
        source_model_path="/synthetic/base",
        checkpoint_meta=metadata,
        model_path="/synthetic/output",
        lora_target="vlm",
        adapter_dirs=[],
    )
    assert result["global_step"] == 100
    assert (
        result["source_ckpt_metadata"]["checkpoint_format"]
        == TABERO_REALWORLD_CHECKPOINT_FORMAT
    )
    assert "training_config" not in result["source_ckpt_metadata"]


def test_export_rejects_old_format_before_model_loading(runtime, monkeypatch, tmp_path):
    from rlinf.utils.ckpt_convertor import export_openpi_lora_for_t2vla as exporter

    runtime.actor.model.is_lora = True
    checkpoint = tmp_path / "old.pt"
    torch.save({"model": {}, "metadata": {"method": "pirl"}}, checkpoint)
    monkeypatch.setattr(exporter, "_load_model_cfg", lambda _: runtime.actor.model)
    monkeypatch.setattr(
        exporter, "get_model", lambda _: pytest.fail("must reject before loading model")
    )
    with pytest.raises(
        ValueError, match="Unsupported RealWorld PiRL checkpoint format"
    ):
        exporter.export_checkpoint(
            "unused", str(checkpoint), str(tmp_path / "out"), False
        )
