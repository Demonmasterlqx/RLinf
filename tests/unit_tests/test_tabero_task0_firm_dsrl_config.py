# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError

from rlinf.config import validate_embodied_cfg
from rlinf.utils.dsrl_observation import DSRL_OBSERVATION_SEMANTICS
from rlinf.utils.dsrl_replay import (
    DSRL_REPLAY_BACKEND,
    DSRL_REPLAY_CAPACITY_TRANSITIONS,
    DSRL_REPLAY_CHECKPOINT_SHARD_TRANSITIONS,
    DSRL_REPLAY_MAX_RESIDENT_GIB,
    DSRL_REPLAY_SEMANTICS,
)
from rlinf.utils.dsrl_reward import DSRL_REWARD_SEMANTICS

CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "embodiment"
    / "config"
    / "isaaclab_pi0_dsrl_tacfield_tabero_task0_firm_8gpu_smoke.yaml"
)
MODEL_PATH = "/data/home/sim6g/code/tabero/models/pi0_lora_tacfield_tabero_safetensors"
HDF5_PATH = (
    "/data/home/sim6g/code/tabero/Tabero/benchmarks/datasets/libero/"
    "assembled_hdf5/"
    "libero_object_task0_pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
)


def _load(monkeypatch):
    monkeypatch.setenv("TABERO_TASK0_DSRL_RUN_ID", "20260726_120000_smoke")
    return OmegaConf.load(CONFIG_PATH)


def test_dsrl_smoke_config_has_exact_8_gpu_capacity(monkeypatch):
    cfg = _load(monkeypatch)

    assert cfg.cluster.num_nodes == 1
    assert OmegaConf.to_container(cfg.cluster.component_placement) == {
        "actor": "4-7",
        "rollout": "0-3",
        "env": "0-3",
    }
    assert cfg.env.train.total_num_envs == 84
    assert cfg.env.train.rollout_epoch == 2
    assert cfg.env.train.max_steps_per_rollout_epoch == 360
    assert cfg.env.train.max_episode_steps == 360
    assert cfg.actor.global_batch_size == 168
    assert cfg.actor.micro_batch_size == 2
    assert cfg.env.train.total_num_envs * cfg.env.train.rollout_epoch == 168
    assert cfg.actor.global_batch_size // 4 == 42


def test_dsrl_smoke_config_preserves_task0_firm_tacfield_contract(monkeypatch):
    cfg = _load(monkeypatch)

    for split in (cfg.env.train.init_params, cfg.env.eval.init_params):
        assert split.task_suite == "libero_object"
        assert split.task_id == 0
        assert OmegaConf.to_container(split.tasks) == [
            {"task_suite": "libero_object", "task_id": 0}
        ]
        assert split.require_all_tasks_active is True
        assert split.task_description == (
            "pick up the alphabet soup and place it in the basket"
        )
        assert split.hdf5_initial_states_path == HDF5_PATH
        assert split.hdf5_reset_assignment == "cyclic"
        assert split.success.required_consecutive_steps == 8
        assert split.marker_history_len == 8
        assert split.combined_marker_count == 198
        assert split.main_image_key == "agentview_rgb"
        assert split.wrist_image_key == "eye_in_hand_rgb"
        assert split.marker_motion_key == "gripper_marker_motion"
    prompt = cfg.env.train.init_params.prompt_conditions
    assert prompt.enabled is True
    assert prompt.assignment == "cyclic"
    assert prompt.condition_cycle == ["firm"]
    assert prompt.firm_adverbs == ["firmly", "tightly"]
    assert prompt.prompt_seed == 0
    assert cfg.env.eval.init_params.prompt_conditions.enabled is False
    assert cfg.env.train.use_step_penalty is False


def test_dsrl_smoke_runs_one_complete_sac_update(monkeypatch):
    cfg = _load(monkeypatch)

    assert cfg.runner.max_epochs == 1
    assert cfg.runner.max_steps == -1
    assert cfg.runner.save_interval == 1
    assert cfg.runner.val_check_interval == -1
    assert cfg.runner.resume_dir is None
    assert cfg.runner.ckpt_path is None
    assert cfg.runner.logger.logger_backends == ["tensorboard", "wandb"]
    assert cfg.algorithm.adv_type == "embodied_sac"
    assert cfg.algorithm.loss_type == "embodied_sac"
    assert cfg.algorithm.update_epoch == 1
    assert cfg.algorithm.gamma == 0.999
    assert cfg.algorithm.dsrl_reward_semantics == DSRL_REWARD_SEMANTICS
    assert cfg.algorithm.dsrl_observation_semantics == DSRL_OBSERVATION_SEMANTICS
    assert cfg.algorithm.dsrl_replay_semantics == DSRL_REPLAY_SEMANTICS
    assert cfg.algorithm.replay_buffer.backend == DSRL_REPLAY_BACKEND
    assert (
        cfg.algorithm.replay_buffer.capacity_transitions
        == DSRL_REPLAY_CAPACITY_TRANSITIONS
    )
    assert (
        cfg.algorithm.replay_buffer.checkpoint_shard_transitions
        == DSRL_REPLAY_CHECKPOINT_SHARD_TRANSITIONS
    )
    assert cfg.algorithm.replay_buffer.max_resident_gib == DSRL_REPLAY_MAX_RESIDENT_GIB
    assert cfg.algorithm.tau == 0.005
    assert cfg.algorithm.replay_buffer.min_buffer_size == 1
    assert cfg.algorithm.train_actor_steps == 1
    assert cfg.algorithm.agg_q == "mean"
    assert cfg.algorithm.actor_agg_q == "mean"
    assert cfg.algorithm.entropy_tuning.target_entropy == -16
    assert cfg.algorithm.entropy_tuning.optim.lr == 3.0e-4
    assert cfg.rollout.collect_transitions is True


@pytest.mark.parametrize("semantics", [None, "legacy"])
def test_dsrl_smoke_config_validation_rejects_missing_or_wrong_reward_semantics(
    monkeypatch, semantics
):
    cfg = _load(monkeypatch)
    if semantics is None:
        del cfg.algorithm.dsrl_reward_semantics
    else:
        cfg.algorithm.dsrl_reward_semantics = semantics

    with pytest.raises(ValueError, match="dsrl_reward_semantics"):
        validate_embodied_cfg(cfg)


@pytest.mark.parametrize("semantics", [None, "single_camera_v1"])
def test_dsrl_smoke_config_validation_rejects_wrong_observation_semantics(
    monkeypatch,
    semantics,
):
    cfg = _load(monkeypatch)
    if semantics is None:
        del cfg.algorithm.dsrl_observation_semantics
    else:
        cfg.algorithm.dsrl_observation_semantics = semantics

    with pytest.raises(ValueError, match="dsrl_observation_semantics"):
        validate_embodied_cfg(cfg)


@pytest.mark.parametrize("semantics", [None, "trajectory_v0"])
def test_dsrl_smoke_config_validation_rejects_wrong_replay_semantics(
    monkeypatch,
    semantics,
):
    cfg = _load(monkeypatch)
    if semantics is None:
        del cfg.algorithm.dsrl_replay_semantics
    else:
        cfg.algorithm.dsrl_replay_semantics = semantics

    with pytest.raises(ValueError, match="dsrl_replay_semantics"):
        validate_embodied_cfg(cfg)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend", "trajectory"),
        ("capacity_transitions", 99_999),
        ("checkpoint_shard_transitions", 2048),
        ("max_resident_gib", 11.0),
    ],
)
def test_dsrl_smoke_config_validation_rejects_wrong_compact_replay_contract(
    monkeypatch,
    field,
    value,
):
    cfg = _load(monkeypatch)
    cfg.algorithm.replay_buffer[field] = value

    with pytest.raises(ValueError, match=field):
        validate_embodied_cfg(cfg)


@pytest.mark.parametrize("num_images", [None, 1, 3])
def test_dsrl_smoke_config_validation_requires_two_images(monkeypatch, num_images):
    cfg = _load(monkeypatch)
    if num_images is None:
        del cfg.actor.model.openpi.dsrl_num_images
    else:
        cfg.actor.model.openpi.dsrl_num_images = num_images

    with pytest.raises(ValueError, match="dsrl_num_images=2"):
        validate_embodied_cfg(cfg)


def test_dsrl_smoke_model_and_optimizers_are_tactile_only_steering(monkeypatch):
    cfg = _load(monkeypatch)
    model = cfg.actor.model
    openpi = model.openpi

    assert cfg.rollout.model.model_path == MODEL_PATH
    assert model.model_path == MODEL_PATH
    assert model.is_lora is False
    assert model.add_value_head is False
    assert model.add_q_head is True
    assert model.num_q_heads == 10
    assert openpi.add_value_head is False
    assert openpi.use_dsrl is True
    assert openpi.dsrl_use_tactile is True
    assert openpi.dsrl_num_images == 2
    assert openpi.dsrl_state_dim == 7
    assert openpi.dsrl_action_noise_dim == 32
    assert openpi.dsrl_num_q_heads == 10
    assert openpi.dsrl_agg_q == "mean"
    assert openpi.dsrl_image_latent_dim == 64
    assert openpi.dsrl_state_latent_dim == 64
    assert openpi.dsrl_tactile_latent_dim == 64
    assert cfg.actor.optim.lr == 1.0e-4
    assert cfg.actor.critic_optim.lr == 3.0e-4
    assert cfg.actor.fsdp_config.sharding_strategy == "no_shard"
    assert cfg.actor.fsdp_config.use_orig_params is True
    assert cfg.actor.fsdp_config.checkpoint_format == "local_shard"
    assert cfg.actor.fsdp_config.save_trainable_model_weights is True
    assert cfg.actor.fsdp_config.save_full_model_weights is False
    assert cfg.rollout.enable_offload is True


def test_dsrl_smoke_config_has_exact_rollout_sync_prefixes(monkeypatch):
    cfg = _load(monkeypatch)

    assert OmegaConf.to_container(cfg.actor.rollout_sync_prefixes) == [
        "dsrl_action_noise_net.",
        "actor_image_encoder.",
        "actor_state_encoder.",
        "actor_tactile_encoder.",
    ]


def test_dsrl_smoke_config_requires_collision_free_run_id(monkeypatch):
    monkeypatch.delenv("TABERO_TASK0_DSRL_RUN_ID", raising=False)
    cfg = OmegaConf.load(CONFIG_PATH)

    with pytest.raises(InterpolationResolutionError, match="TABERO_TASK0_DSRL_RUN_ID"):
        _ = cfg.runner.logger.log_path
