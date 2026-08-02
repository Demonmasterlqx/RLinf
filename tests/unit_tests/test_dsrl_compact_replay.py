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

import json

import pytest
import torch

from rlinf.data.dsrl_replay_buffer import CompactDSRLReplayBuffer
from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    Trajectory,
)
from rlinf.utils.dsrl_replay import (
    DSRL_REPLAY_FIELD_SPECS,
    DSRL_REPLAY_SEMANTICS,
    compact_tabero_dsrl_observation,
    dsrl_replay_bytes_per_transition,
)
from rlinf.utils.dsrl_transition import DSRL_TRANSITION_BOUNDARY_SEMANTICS
from rlinf.workers.env.env_worker import (
    project_compact_dsrl_rollout_inputs,
    project_compact_dsrl_step_result,
)


def _compact_obs(traj_len: int, batch_size: int, *, offset: float = 0.0):
    return {
        "dsrl_images": torch.full(
            (traj_len, batch_size, 2, 3, 64, 64),
            offset,
            dtype=torch.bfloat16,
        ),
        "states": torch.full((traj_len, batch_size, 7), offset, dtype=torch.bfloat16),
        "tactile_marker_motion": torch.full(
            (traj_len, batch_size, 9, 198, 2),
            offset,
            dtype=torch.bfloat16,
        ),
    }


def _trajectory(
    *,
    traj_len: int = 2,
    batch_size: int = 2,
    offset: float = 0.0,
) -> Trajectory:
    action = torch.arange(traj_len * batch_size, dtype=torch.bfloat16).reshape(
        traj_len, batch_size, 1
    )
    action = action.expand(-1, -1, 32).contiguous() + offset
    return Trajectory(
        actions=action,
        rewards=torch.full((traj_len, batch_size, 10), offset, dtype=torch.float32),
        terminations=torch.zeros(traj_len, batch_size, 10, dtype=torch.bool),
        truncations=torch.zeros(traj_len, batch_size, 10, dtype=torch.bool),
        curr_obs=_compact_obs(traj_len, batch_size, offset=offset),
        next_obs=_compact_obs(traj_len, batch_size, offset=offset + 1),
    )


def _chronological_flat(buffer: CompactDSRLReplayBuffer):
    indices = buffer._chronological_indices()
    return {
        name: tensor.index_select(0, indices)
        for name, tensor in buffer._storage.items()
    }


def test_compact_observation_preserves_main_wrist_order_and_model_input_values():
    raw = {
        "main_images": torch.zeros(2, 256, 256, 3, dtype=torch.uint8),
        "wrist_images": torch.full((2, 256, 256, 3), 255, dtype=torch.uint8),
        "states": torch.randn(2, 7),
        "tactile_marker_motion": torch.randn(2, 9, 198, 2),
        "task_descriptions": ["unused", "unused"],
    }

    compact = compact_tabero_dsrl_observation(raw)

    assert set(compact) == {"dsrl_images", "states", "tactile_marker_motion"}
    assert compact["dsrl_images"].shape == (2, 2, 3, 64, 64)
    assert compact["dsrl_images"].dtype == torch.bfloat16
    assert torch.equal(
        compact["dsrl_images"][:, 0],
        torch.full_like(compact["dsrl_images"][:, 0], -1),
    )
    assert torch.equal(
        compact["dsrl_images"][:, 1],
        torch.full_like(compact["dsrl_images"][:, 1], 1),
    )
    assert compact["states"].dtype == torch.bfloat16
    assert compact["tactile_marker_motion"].dtype == torch.bfloat16
    assert raw["main_images"].shape == (2, 256, 256, 3)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda obs: obs.pop("wrist_images"), "missing required fields"),
        (
            lambda obs: obs.__setitem__("main_images", obs["main_images"].float()),
            "must use uint8",
        ),
        (
            lambda obs: obs.__setitem__(
                "wrist_images", torch.zeros(2, 255, 256, 3, dtype=torch.uint8)
            ),
            "trailing shape",
        ),
        (
            lambda obs: obs.__setitem__(
                "tactile_marker_motion", torch.zeros(2, 8, 198, 2)
            ),
            "expected shape",
        ),
    ],
)
def test_compact_observation_rejects_invalid_raw_contract(mutation, message):
    raw = {
        "main_images": torch.zeros(2, 256, 256, 3, dtype=torch.uint8),
        "wrist_images": torch.zeros(2, 256, 256, 3, dtype=torch.uint8),
        "states": torch.zeros(2, 7),
        "tactile_marker_motion": torch.zeros(2, 9, 198, 2),
    }
    mutation(raw)

    with pytest.raises(ValueError, match=message):
        compact_tabero_dsrl_observation(raw)


def test_compact_replay_exact_byte_budget_matches_formal_trajectory():
    assert dsrl_replay_bytes_per_transition() == 112_712
    assert dsrl_replay_bytes_per_transition() * (72 * 21) == 170_420_544
    assert set(DSRL_REPLAY_FIELD_SPECS) == {
        "curr_obs.dsrl_images",
        "curr_obs.states",
        "curr_obs.tactile_marker_motion",
        "next_obs.dsrl_images",
        "next_obs.states",
        "next_obs.tactile_marker_motion",
        "actions",
        "rewards",
        "terminations",
        "truncations",
    }


def test_compact_replay_samples_only_sac_fields_and_32d_actions():
    replay = CompactDSRLReplayBuffer(
        seed=3,
        capacity_transitions=8,
        checkpoint_shard_transitions=2,
        max_resident_gib=0.01,
    )
    replay.add_trajectories([_trajectory()])

    batch = replay.sample(4)

    assert set(batch) == {
        "curr_obs",
        "next_obs",
        "actions",
        "rewards",
        "terminations",
        "truncations",
    }
    assert set(batch["curr_obs"]) == {
        "dsrl_images",
        "states",
        "tactile_marker_motion",
    }
    assert batch["actions"].shape == (4, 32)
    assert batch["rewards"].dtype == torch.float32
    assert "forward_inputs" not in batch


def test_compact_replay_rejects_forward_inputs_and_wrong_action_horizon():
    replay = CompactDSRLReplayBuffer(
        seed=3,
        capacity_transitions=8,
        checkpoint_shard_transitions=2,
        max_resident_gib=0.01,
    )
    trajectory = _trajectory()
    trajectory.forward_inputs = {"chains": torch.zeros(2, 2, 11, 50, 32)}
    with pytest.raises(ValueError, match="forbids trajectory.forward_inputs"):
        replay.add_trajectories([trajectory])

    trajectory = _trajectory()
    trajectory.actions = torch.zeros(2, 2, 50, 32, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="actions expected shape"):
        replay.add_trajectories([trajectory])


def test_compact_rollout_projection_keeps_only_one_32d_bf16_latent():
    action = torch.randn(3, 32, dtype=torch.bfloat16)

    projected_action, projected_forward_inputs = project_compact_dsrl_rollout_inputs(
        {"action": action}
    )

    assert torch.equal(projected_action, action)
    assert projected_action.dtype == torch.bfloat16
    assert projected_forward_inputs == {}

    with pytest.raises(ValueError, match="only the 32D latent"):
        project_compact_dsrl_rollout_inputs(
            {"action": action, "chains": torch.zeros(3, 11, 50, 32)}
        )
    with pytest.raises(ValueError, match=r"expected \[B,32\]"):
        project_compact_dsrl_rollout_inputs(
            {"action": torch.zeros(3, 50, 32, dtype=torch.bfloat16)}
        )
    projected_float, _ = project_compact_dsrl_rollout_inputs({"action": action.float()})
    assert projected_float.dtype == torch.bfloat16
    with pytest.raises(ValueError, match="must be floating point"):
        project_compact_dsrl_rollout_inputs({"action": torch.zeros(3, 32).long()})


def _append_epoch_step_results(
    collector: EmbodiedRolloutResult,
    *,
    compact: bool,
    epoch: int,
    num_transitions: int,
    batch_size: int,
) -> None:
    for transition in range(num_transitions):
        has_previous_transition = transition > 0
        result = ChunkStepResult(
            actions=torch.full(
                (batch_size, 32), epoch * 100 + transition, dtype=torch.bfloat16
            ),
            rewards=(
                torch.full(
                    (batch_size, 10),
                    epoch * 100 + transition - 1,
                    dtype=torch.float32,
                )
                if has_previous_transition
                else None
            ),
            dones=torch.zeros(batch_size, 10, dtype=torch.bool),
            terminations=torch.zeros(batch_size, 10, dtype=torch.bool),
            truncations=torch.zeros(batch_size, 10, dtype=torch.bool),
        )
        collector.append_step_result(
            project_compact_dsrl_step_result(result) if compact else result
        )

    final_result = ChunkStepResult(
        rewards=torch.full(
            (batch_size, 10),
            epoch * 100 + num_transitions - 1,
            dtype=torch.float32,
        ),
        dones=torch.zeros(batch_size, 10, dtype=torch.bool),
        terminations=torch.zeros(batch_size, 10, dtype=torch.bool),
        truncations=torch.zeros(batch_size, 10, dtype=torch.bool),
    )
    # Prove that the terminal metadata of the epoch's final executed action is
    # retained even though it arrives during the bootstrap-only policy call.
    final_result.dones[0, 7] = True
    final_result.terminations[0, 7] = True
    collector.append_step_result(
        project_compact_dsrl_step_result(final_result) if compact else final_result
    )


def test_compact_step_projection_aligns_two_rollout_epochs_without_clipping():
    compact = EmbodiedRolloutResult()
    legacy = EmbodiedRolloutResult()
    for epoch in range(2):
        _append_epoch_step_results(
            compact,
            compact=True,
            epoch=epoch,
            num_transitions=36,
            batch_size=21,
        )
        _append_epoch_step_results(
            legacy,
            compact=False,
            epoch=epoch,
            num_transitions=36,
            batch_size=21,
        )

    compact_trajectory = compact.to_trajectory()
    legacy_trajectory = legacy.to_trajectory()

    for field in ("actions", "rewards", "dones", "terminations", "truncations"):
        assert getattr(compact_trajectory, field).shape[:2] == (72, 21)
    assert compact_trajectory.terminations[35, 0, 7]
    assert compact_trajectory.terminations[71, 0, 7]

    assert legacy_trajectory.actions.shape[:2] == (72, 21)
    assert legacy_trajectory.rewards.shape[:2] == (72, 21)
    for field in ("dones", "terminations", "truncations"):
        assert getattr(legacy_trajectory, field).shape[:2] == (74, 21)


def test_compact_replay_ring_wrap_and_checkpoint_round_trip(tmp_path):
    replay = CompactDSRLReplayBuffer(
        seed=11,
        capacity_transitions=5,
        checkpoint_shard_transitions=2,
        max_resident_gib=0.01,
    )
    replay.add_trajectories([_trajectory(offset=0), _trajectory(offset=10)])
    assert replay.total_samples == 5
    assert len(replay) == 2
    expected_flat = _chronological_flat(replay)
    replay.save_checkpoint(str(tmp_path))
    expected_sample = replay.sample(5)

    restored = CompactDSRLReplayBuffer(
        seed=99,
        capacity_transitions=5,
        checkpoint_shard_transitions=2,
        max_resident_gib=0.01,
    )
    restored.load_checkpoint(str(tmp_path))
    actual_flat = _chronological_flat(restored)
    actual_sample = restored.sample(5)

    assert restored.total_samples == 5
    assert len(restored) == 2
    for name in expected_flat:
        assert torch.equal(expected_flat[name], actual_flat[name])
    for obs_name in ("curr_obs", "next_obs"):
        for field in expected_sample[obs_name]:
            assert torch.equal(
                expected_sample[obs_name][field], actual_sample[obs_name][field]
            )
    for field in ("actions", "rewards", "terminations", "truncations"):
        assert torch.equal(expected_sample[field], actual_sample[field])

    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["replay_semantics"] == DSRL_REPLAY_SEMANTICS
    assert (
        metadata["transition_boundary_semantics"] == DSRL_TRANSITION_BOUNDARY_SEMANTICS
    )
    assert metadata["num_shards"] == 3
    assert len(list(tmp_path.glob("shard_*.pt"))) == 3


def test_compact_replay_rejects_capacity_over_resident_budget():
    with pytest.raises(ValueError, match="exceeds its resident-memory budget"):
        CompactDSRLReplayBuffer(
            capacity_transitions=100,
            checkpoint_shard_transitions=2,
            max_resident_gib=0.001,
        )


def test_compact_replay_rejects_missing_or_wrong_metadata_before_loading(tmp_path):
    with pytest.raises(FileNotFoundError, match="metadata not found"):
        CompactDSRLReplayBuffer.validate_checkpoint_metadata(str(tmp_path))

    (tmp_path / "metadata.json").write_text(json.dumps({"replay_semantics": "legacy"}))
    with pytest.raises(ValueError, match="metadata 'format' mismatch"):
        CompactDSRLReplayBuffer.validate_checkpoint_metadata(str(tmp_path))


@pytest.mark.parametrize("semantics", [None, "cross_episode_chunk_v0"])
def test_compact_replay_rejects_wrong_transition_boundary_semantics(
    tmp_path,
    semantics,
):
    replay = CompactDSRLReplayBuffer(
        capacity_transitions=5,
        checkpoint_shard_transitions=2,
        max_resident_gib=0.01,
    )
    replay.save_checkpoint(str(tmp_path))
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if semantics is None:
        metadata.pop("transition_boundary_semantics")
    else:
        metadata["transition_boundary_semantics"] = semantics
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="transition_boundary_semantics"):
        CompactDSRLReplayBuffer.validate_checkpoint_metadata(str(tmp_path))
