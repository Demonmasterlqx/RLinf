# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError

from rlinf.config import validate_embodied_cfg
from rlinf.utils.tabero_ppo_boundary import (
    TABERO_PPO_TRANSITION_BOUNDARY_SEMANTICS,
)

CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "embodiment"
    / "config"
    / "isaaclab_pi0_peft_lora_tacfield_tabero_task0_no_adverb_mass_friction_2gpu_100step.yaml"
)
MODEL_PATH = (
    "/data/home/sim6g/code/tabero/models/pi0_lora_tacfield_tabero_all_firm_safetensors"
)
PROFILE_DIR = Path(
    "/data/home/sim6g/code/tabero/Tabero/benchmarks/datasets/libero/"
    "config_profiles/firm_damage_fixed_friction_05_from_rlinf_sft_20k"
)
HDF5_PATH = (
    "/data/home/sim6g/code/tabero/Tabero/benchmarks/datasets/libero/"
    "assembled_hdf5/"
    "libero_object_task0_pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
)


def _load(monkeypatch):
    monkeypatch.setenv("TABERO_TASK0_NO_ADVERB_RUN_ID", "20260810_220000_formal")
    return OmegaConf.load(CONFIG_PATH)


def test_config_has_exact_two_gpu_training_contract(monkeypatch):
    cfg = _load(monkeypatch)

    assert cfg.cluster.num_nodes == 1
    assert OmegaConf.to_container(cfg.cluster.component_placement) == {
        "actor": "1",
        "rollout": "0",
        "env": "0",
    }
    assert cfg.env.train.total_num_envs == 21
    assert cfg.env.train.rollout_epoch == 2
    assert cfg.actor.global_batch_size == 42
    assert cfg.actor.micro_batch_size == 2
    assert (
        cfg.env.train.total_num_envs * cfg.env.train.rollout_epoch
        == cfg.actor.global_batch_size
    )
    assert cfg.runner.max_epochs == 100
    assert cfg.runner.max_steps == -1
    assert cfg.runner.save_interval == 10
    assert cfg.runner.val_check_interval == -1
    assert cfg.runner.logger.logger_backends == ["tensorboard", "wandb"]
    assert "task0_no_adverb_mass_friction_action_expert" in (
        cfg.runner.logger.experiment_name
    )


def test_config_uses_plain_task0_prompt_and_mass_friction_profile(monkeypatch):
    cfg = _load(monkeypatch)

    for split_cfg in (cfg.env.train, cfg.env.eval):
        split = split_cfg.init_params
        assert split.task_suite == "libero_object"
        assert split.task_id == 0
        assert OmegaConf.to_container(split.tasks) == [
            {"task_suite": "libero_object", "task_id": 0}
        ]
        assert split.task_description == (
            "pick up the alphabet soup and place it in the basket"
        )
        assert split.prompt_conditions.enabled is False
        assert split.libero_config_dir == str(PROFILE_DIR)
        assert split.hdf5_initial_states_path == HDF5_PATH
        assert split.hdf5_reset_assignment == "cyclic"
        assert split.chunk_boundary_mode == "terminal_safe_hdf5_v1"
        assert split.success.required_consecutive_steps == 8
        assert split_cfg.auto_reset is False
        assert split_cfg.ignore_terminations is False

    profile = json.loads((PROFILE_DIR / "libero_object.json").read_text())
    task0 = next(task for task in profile["tasks"] if task["task_id"] == 0)
    physics = task0["physics"]
    target = physics["objects"]["alphabet_soup_1"]
    assert target["mass_kg"] == {
        "distribution": "uniform",
        "range": [0.8, 1.2],
        "apply_on": "reset",
    }
    assert target["friction"] == {
        "distribution": "uniform",
        "static_range": [0.4, 0.8],
        "dynamic_range": [0.3, 0.6],
        "apply_on": "reset",
        "num_buckets": 64,
    }
    assert target["damage"] == {
        "threshold": {
            "mode": "mass_friction",
            "tolerance_factor": 1.1,
        },
        "consecutive_frames": 4,
    }
    assert physics["gripper"]["friction"] == {
        "static": 0.5,
        "dynamic": 0.5,
    }


def test_config_trains_only_action_expert_lora_and_value_head(monkeypatch):
    cfg = _load(monkeypatch)
    model = cfg.actor.model

    assert cfg.rollout.model.model_path == MODEL_PATH
    assert model.model_path == MODEL_PATH
    assert model.is_lora is True
    assert model.lora_target == "action_expert"
    assert model.lora_rank == 32
    assert model.freeze_non_lora is True
    assert model.openpi.train_expert_only is True
    assert model.add_value_head is True
    assert model.openpi.add_value_head is True
    assert cfg.actor.optim.lr == 1.0e-6
    assert cfg.actor.optim.value_lr == 1.0e-4
    assert cfg.algorithm.adv_type == "gae"
    assert cfg.algorithm.loss_type == "actor_critic"
    assert cfg.actor.fsdp_config.sharding_strategy == "no_shard"
    assert cfg.actor.fsdp_config.save_trainable_model_weights is True
    assert cfg.actor.fsdp_config.save_full_model_weights is False
    metadata = cfg.actor.fsdp_config.trainable_checkpoint_metadata
    assert metadata.method == "pirl"
    assert metadata.task_id == 0
    assert metadata.prompt_condition == "no_adverb"
    assert metadata.training_config == CONFIG_PATH.stem
    assert metadata.target_global_step == 100
    assert (
        metadata.tabero_ppo_transition_boundary_semantics
        == TABERO_PPO_TRANSITION_BOUNDARY_SEMANTICS
    )
    assert (
        cfg.algorithm.tabero_ppo_transition_boundary_semantics
        == TABERO_PPO_TRANSITION_BOUNDARY_SEMANTICS
    )
    assert cfg.weight_syncer.type == "patch"
    assert cfg.weight_syncer.patch.snapshot_device == "cpu"
    assert cfg.weight_syncer.patch.delta_encoding is True
    assert cfg.weight_syncer.patch.compression == "none"
    assert cfg.weight_syncer.patch.init_sync.enabled is True
    assert cfg.weight_syncer.patch.init_sync.prefixes is None
    assert cfg.weight_syncer.patch.init_sync.bucket_size == 134217728


def test_config_requires_collision_free_runtime_run_id(monkeypatch):
    monkeypatch.delenv("TABERO_TASK0_NO_ADVERB_RUN_ID", raising=False)
    cfg = OmegaConf.load(CONFIG_PATH)

    with pytest.raises(
        InterpolationResolutionError, match="TABERO_TASK0_NO_ADVERB_RUN_ID"
    ):
        _ = cfg.runner.logger.log_path


def test_boundary_safe_task0_config_passes_embodied_validation(monkeypatch):
    cfg = _load(monkeypatch)

    assert validate_embodied_cfg(cfg) is cfg


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            "algorithm.tabero_ppo_transition_boundary_semantics",
            "legacy_typo",
            "Unsupported algorithm.tabero_ppo_transition_boundary_semantics",
        ),
        (
            "algorithm.logprob_type",
            "action_level",
            "algorithm.logprob_type",
        ),
        (
            "env.train.init_params.chunk_boundary_mode",
            "legacy",
            "env.train.init_params.chunk_boundary_mode",
        ),
        ("env.eval.auto_reset", True, "env.eval.auto_reset"),
        ("env.train.ignore_terminations", True, "env.train.ignore_terminations"),
        (
            "env.eval.init_params.hdf5_initial_states_path",
            "",
            "env.eval.init_params.hdf5_initial_states_path",
        ),
        (
            "actor.fsdp_config.trainable_checkpoint_metadata.tabero_ppo_transition_boundary_semantics",
            "legacy",
            "trainable_checkpoint_metadata",
        ),
    ],
)
def test_boundary_safe_config_rejects_inconsistent_contract(
    monkeypatch, path, value, message
):
    cfg = _load(monkeypatch)
    OmegaConf.update(cfg, path, value)

    with pytest.raises(ValueError, match=message):
        validate_embodied_cfg(cfg)


def test_legacy_config_without_new_semantics_keeps_legacy_defaults(monkeypatch):
    cfg = _load(monkeypatch)
    del cfg.algorithm.tabero_ppo_transition_boundary_semantics
    del cfg.actor.fsdp_config.trainable_checkpoint_metadata
    cfg.env.train.init_params.chunk_boundary_mode = "legacy"
    cfg.env.eval.init_params.chunk_boundary_mode = "legacy"
    cfg.env.eval.auto_reset = True
    cfg.env.eval.ignore_terminations = True

    assert validate_embodied_cfg(cfg) is cfg
