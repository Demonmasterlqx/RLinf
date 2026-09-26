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

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from rlinf.config import _validate_tabero_realworld_pi05_dsrl_contract
from rlinf.data.dsrl_replay_buffer import CompactDSRLReplayBuffer
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0ForRLActionPrediction,
)
from rlinf.utils.dsrl_checkpoint import (
    get_dsrl_checkpoint_contract,
    select_compact_target_parameters,
    select_dsrl_trainable_state,
)
from rlinf.utils.dsrl_observation import (
    REALWORLD_TACIMG_DSRL_OBSERVATION_SEMANTICS,
)
from rlinf.utils.dsrl_replay import (
    DSRL_REPLAY_BACKEND,
    REALWORLD_TACIMG_DSRL_REPLAY_SEMANTICS,
    compact_tabero_dsrl_observation,
    dsrl_replay_bytes_per_transition,
    validate_compact_dsrl_observation,
)
from rlinf.utils.dsrl_reward import DSRL_REWARD_SEMANTICS
from rlinf.utils.dsrl_rollout_sync import (
    REALWORLD_TACIMG_DSRL_ROLLOUT_SYNC_PREFIXES,
    select_named_parameters_by_prefix,
    validate_dsrl_rollout_state_dict,
    validate_dsrl_rollout_sync_config,
)
from rlinf.utils.dsrl_transition import (
    REALWORLD_TACIMG_DSRL_CHUNK_BOUNDARY_MODE,
    REALWORLD_TACIMG_DSRL_TRANSITION_BOUNDARY_SEMANTICS,
)


def _dsrl_model() -> OpenPi0ForRLActionPrediction:
    config = SimpleNamespace(
        use_dsrl=True,
        dsrl_use_tactile=False,
        dsrl_num_images=3,
        dsrl_tactile_latent_dim=64,
        dsrl_state_dim=7,
        dsrl_action_noise_dim=32,
        dsrl_num_q_heads=10,
        dsrl_image_latent_dim=64,
        dsrl_state_latent_dim=64,
        dsrl_hidden_dims=(128, 128, 128),
        action_horizon=50,
        config_name="pi05_lora_tacimg_realworld_replayed_task820_force",
    )
    model = OpenPi0ForRLActionPrediction.__new__(OpenPi0ForRLActionPrediction)
    nn.Module.__init__(model)
    model.config = config
    model._init_dsrl_components()
    return model


def test_realworld_tacimg_compact_projection_uses_three_heterogeneous_rgb_views():
    obs = {
        "main_images": torch.randint(0, 256, (2, 480, 640, 3), dtype=torch.uint8),
        "wrist_images": torch.randint(0, 256, (2, 480, 640, 3), dtype=torch.uint8),
        "tactile_images": torch.randint(0, 256, (2, 224, 224, 3), dtype=torch.uint8),
        "states": torch.randn(2, 7),
        "tactile_marker_motion": torch.randn(2, 9, 440, 2),
    }
    source_copies = {key: value.clone() for key, value in obs.items()}

    compact = compact_tabero_dsrl_observation(
        obs, replay_semantics=REALWORLD_TACIMG_DSRL_REPLAY_SEMANTICS
    )

    assert set(compact) == {"dsrl_images", "states"}
    assert compact["dsrl_images"].shape == (2, 3, 3, 64, 64)
    assert compact["dsrl_images"].dtype == torch.bfloat16
    assert compact["states"].shape == (2, 7)
    assert compact["states"].dtype == torch.bfloat16
    validate_compact_dsrl_observation(
        compact,
        batch_size=2,
        prefix="curr_obs",
        replay_semantics=REALWORLD_TACIMG_DSRL_REPLAY_SEMANTICS,
    )
    for key, source in source_copies.items():
        torch.testing.assert_close(obs[key], source)


def test_realworld_tacimg_replay_memory_contract_fits_twelve_gib():
    bytes_per_transition = dsrl_replay_bytes_per_transition(
        REALWORLD_TACIMG_DSRL_REPLAY_SEMANTICS
    )

    assert bytes_per_transition == 147_608
    assert bytes_per_transition * 80_000 <= 12 * 2**30
    assert bytes_per_transition * 100_000 > 12 * 2**30


def test_realworld_tacimg_compact_replay_checkpoint_round_trip(tmp_path):
    def compact_obs(offset: float):
        return {
            "dsrl_images": torch.full(
                (2, 1, 3, 3, 64, 64), offset, dtype=torch.bfloat16
            ),
            "states": torch.full((2, 1, 7), offset, dtype=torch.bfloat16),
        }

    trajectory = Trajectory(
        actions=torch.zeros(2, 1, 32, dtype=torch.bfloat16),
        rewards=torch.zeros(2, 1, 10, dtype=torch.float32),
        terminations=torch.zeros(2, 1, 10, dtype=torch.bool),
        truncations=torch.zeros(2, 1, 10, dtype=torch.bool),
        curr_obs=compact_obs(0.0),
        next_obs=compact_obs(1.0),
    )
    kwargs = {
        "seed": 42,
        "capacity_transitions": 4,
        "checkpoint_shard_transitions": 2,
        "max_resident_gib": 0.001,
        "replay_semantics": REALWORLD_TACIMG_DSRL_REPLAY_SEMANTICS,
        "observation_semantics": REALWORLD_TACIMG_DSRL_OBSERVATION_SEMANTICS,
        "transition_boundary_semantics": (
            REALWORLD_TACIMG_DSRL_TRANSITION_BOUNDARY_SEMANTICS
        ),
    }
    source = CompactDSRLReplayBuffer(**kwargs)
    source.add_trajectories([trajectory])
    source.save_checkpoint(str(tmp_path))
    restored = CompactDSRLReplayBuffer(**kwargs)
    restored.load_checkpoint(str(tmp_path))
    sample = restored.sample_chunks(2)

    assert restored.size == 1
    assert restored.total_samples == 2
    assert set(sample["curr_obs"]) == {"dsrl_images", "states"}
    assert sample["curr_obs"]["dsrl_images"].shape == (2, 3, 3, 64, 64)


def test_realworld_tacimg_dsrl_model_and_checkpoint_contract_exclude_marker_tcn():
    model = _dsrl_model()
    obs = {
        "main_images": torch.zeros(2, 480, 640, 3, dtype=torch.uint8),
        "wrist_images": torch.zeros(2, 480, 640, 3, dtype=torch.uint8),
        "tactile_images": torch.zeros(2, 224, 224, 3, dtype=torch.uint8),
        "states": torch.zeros(2, 7),
    }

    normalized = model._normalize_dsrl_obs(obs)
    images = model._prepare_dsrl_images(normalized)
    selected = select_named_parameters_by_prefix(
        model, REALWORLD_TACIMG_DSRL_ROLLOUT_SYNC_PREFIXES
    )
    validate_dsrl_rollout_state_dict(
        selected,
        observation_semantics=REALWORLD_TACIMG_DSRL_OBSERVATION_SEMANTICS,
    )
    trainable = select_dsrl_trainable_state(
        model,
        observation_semantics=REALWORLD_TACIMG_DSRL_OBSERVATION_SEMANTICS,
    )
    target = select_compact_target_parameters(
        model,
        observation_semantics=REALWORLD_TACIMG_DSRL_OBSERVATION_SEMANTICS,
    )
    contract = get_dsrl_checkpoint_contract(REALWORLD_TACIMG_DSRL_OBSERVATION_SEMANTICS)

    assert images.shape == (2, 3, 3, 64, 64)
    assert not hasattr(model, "actor_tactile_encoder")
    assert not hasattr(model, "critic_tactile_encoder")
    assert len(selected) == contract["trainable_tensor_count"] - len(target)
    assert len(trainable) == contract["trainable_tensor_count"]
    assert len(target) == contract["target_tensor_count"]
    assert not any("tactile_encoder" in name for name in trainable)


def test_realworld_tacimg_dsrl_rollout_sync_requires_three_image_prefixes():
    actor_cfg = OmegaConf.create(
        {
            "training_backend": "fsdp",
            "rollout_sync_prefixes": list(REALWORLD_TACIMG_DSRL_ROLLOUT_SYNC_PREFIXES),
            "model": {
                "openpi": {
                    "use_dsrl": True,
                    "dsrl_use_tactile": False,
                    "dsrl_num_images": 3,
                }
            },
            "fsdp_config": {
                "sharding_strategy": "no_shard",
                "use_orig_params": True,
            },
        }
    )

    assert validate_dsrl_rollout_sync_config(actor_cfg) == (
        REALWORLD_TACIMG_DSRL_ROLLOUT_SYNC_PREFIXES
    )


def test_dsrl_preflight_accepts_tuning_without_checkpoint_audit():
    model = {
        "num_action_chunks": 10,
        "action_dim": 13,
        "is_lora": False,
        "use_proprio": True,
        "add_value_head": False,
        "add_q_head": True,
        "q_head_type": "default",
        "num_q_heads": 4,
        "num_steps": 5,
        "model_path": "/synthetic/base",
        "checkpoint_load_allowed_missing_prefixes": [
            "dsrl_action_noise_net.",
            "actor_image_encoder.",
            "actor_state_encoder.",
            "critic_image_encoder.",
            "critic_state_encoder.",
            "q_head.",
        ],
        "openpi_data": {"norm_stats_path": "/synthetic/stats"},
        "openpi": {
            "config_name": "pi05_lora_tacimg_realworld_replayed_task820_force",
            "pi05": True,
            "action_horizon": 50,
            "discrete_state_input": False,
            "num_images_in_input": 3,
            "action_chunk": 10,
            "train_expert_only": True,
            "action_env_dim": 13,
            "effective_action_dim": 13,
            "add_value_head": False,
            "joint_logprob": False,
            "detach_critic_input": True,
            "tactile_type": "expert_his_c_fut",
            "tactile_dim": 6,
            "tactile_dim_in": 0,
            "use_dsrl": True,
            "dsrl_use_tactile": False,
            "dsrl_num_images": 3,
            "dsrl_state_dim": 7,
            "dsrl_action_noise_dim": 32,
            "dsrl_num_q_heads": 4,
            "dsrl_hidden_dims": [64, 64],
            "num_steps": 5,
        },
    }
    cfg = OmegaConf.create(
        {
            "runner": {"val_check_interval": -1},
            "algorithm": {
                "adv_type": "embodied_sac",
                "loss_type": "embodied_sac",
                "reward_type": "chunk_level",
                "logprob_type": "chunk_level",
                "dsrl_reward_semantics": DSRL_REWARD_SEMANTICS,
                "dsrl_observation_semantics": REALWORLD_TACIMG_DSRL_OBSERVATION_SEMANTICS,
                "dsrl_replay_semantics": REALWORLD_TACIMG_DSRL_REPLAY_SEMANTICS,
                "dsrl_transition_boundary_semantics": REALWORLD_TACIMG_DSRL_TRANSITION_BOUNDARY_SEMANTICS,
                "replay_buffer": {
                    "backend": DSRL_REPLAY_BACKEND,
                    "capacity_transitions": 128,
                },
            },
            "actor": {
                "model": model,
                "global_batch_size": 4,
                "fsdp_config": {
                    "sharding_strategy": "no_shard",
                    "gradient_checkpointing": True,
                    "use_orig_params": True,
                    "checkpoint_format": "local_shard",
                    "save_full_model_weights": False,
                    "save_trainable_model_weights": True,
                },
            },
            "rollout": {
                "collect_transitions": True,
                "model": {"model_path": "/synthetic/base"},
            },
            "env": {
                "train": {
                    "auto_reset": False,
                    "ignore_terminations": False,
                    "total_num_envs": 2,
                    "rollout_epoch": 1,
                    "max_episode_steps": 200,
                    "init_params": {
                        "id": "Isaac-RealWorld-GentleGrasp-XarmUmi-Hybrid-Tactile-v0",
                        "target_object": "target_object_3",
                        "task_description": "pick up the cookie and put it into the basket",
                        "reset_source": "task_config_default_reset",
                        "task_suite": "gentle_grasp",
                        "task_id": 6,
                        "tactile_backend": "taxim_fots",
                        "chunk_boundary_mode": REALWORLD_TACIMG_DSRL_CHUNK_BOUNDARY_MODE,
                        "marker_history_len": 8,
                        "combined_marker_count": 440,
                        "tactile_image_history_len": 8,
                        "success": {"force_bonus": {"enabled": False}},
                    },
                }
            },
        }
    )
    _validate_tabero_realworld_pi05_dsrl_contract(cfg, cfg.actor.model)
    cfg.env.train.auto_reset = True
    with pytest.raises(ValueError, match="auto_reset=false"):
        _validate_tabero_realworld_pi05_dsrl_contract(cfg, cfg.actor.model)
