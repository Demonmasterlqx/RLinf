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

import torch
import torch.nn as nn
from omegaconf import OmegaConf

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
    REALWORLD_TACIMG_DSRL_REPLAY_SEMANTICS,
    compact_tabero_dsrl_observation,
    dsrl_replay_bytes_per_transition,
    validate_compact_dsrl_observation,
)
from rlinf.utils.dsrl_rollout_sync import (
    REALWORLD_TACIMG_DSRL_ROLLOUT_SYNC_PREFIXES,
    select_named_parameters_by_prefix,
    validate_dsrl_rollout_state_dict,
    validate_dsrl_rollout_sync_config,
)
from rlinf.utils.dsrl_transition import (
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
