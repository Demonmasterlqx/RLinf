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

import torch
from omegaconf import OmegaConf

from rlinf.data.embodied_io_struct import EnvOutput
from rlinf.envs import get_env_cls

from rlinf.envs.isaaclab.tasks.tabero_tacfield import (
    IsaaclabTaberoTacFieldEnv,
    TacManipMarkerMotionHistory,
    build_tabero_state,
)


def _marker_motion(num_envs: int, offset: float = 0.0) -> torch.Tensor:
    motion = torch.zeros((num_envs, 2, 2, 99, 2), dtype=torch.float32)
    base = torch.arange(2 * 99 * 2, dtype=torch.float32).reshape(2, 99, 2)
    for env_id in range(num_envs):
        motion[env_id, :, 0] = base + offset + env_id * 1000.0
        motion[env_id, :, 1] = base + offset + env_id * 1000.0 + 100.0
    return motion


def test_marker_motion_history_builds_tabero_prefix_with_front_padding():
    history = TacManipMarkerMotionHistory(num_envs=2, history_len=8)

    first = history.update(_marker_motion(2, offset=0.0))
    second = history.update(_marker_motion(2, offset=10.0))

    assert first.shape == (2, 9, 198, 2)
    assert second.shape == (2, 9, 198, 2)

    # Prefix slot 0 is the per-env reference marker position from the first frame.
    expected_init_env0 = _marker_motion(2, 0.0)[0, :, 0].reshape(198, 2)
    torch.testing.assert_close(second[0, 0], expected_init_env0)

    # History slots are front-padded by repeating the earliest current frame.
    expected_first_current = _marker_motion(2, 0.0)[0, :, 1].reshape(198, 2)
    expected_second_current = _marker_motion(2, 10.0)[0, :, 1].reshape(198, 2)
    torch.testing.assert_close(second[0, 1], expected_first_current)
    torch.testing.assert_close(second[0, 7], expected_first_current)
    torch.testing.assert_close(second[0, 8], expected_second_current)


def test_marker_motion_history_reset_reinitializes_selected_envs():
    history = TacManipMarkerMotionHistory(num_envs=2, history_len=8)
    history.update(_marker_motion(2, offset=0.0))
    history.reset(torch.tensor([1]))

    updated = history.update(_marker_motion(2, offset=20.0))

    original_init_env0 = _marker_motion(2, 0.0)[0, :, 0].reshape(198, 2)
    new_init_env1 = _marker_motion(2, 20.0)[1, :, 0].reshape(198, 2)
    torch.testing.assert_close(updated[0, 0], original_init_env0)
    torch.testing.assert_close(updated[1, 0], new_init_env1)


def test_build_tabero_state_uses_pose_axis_angle_and_gripper_scalar():
    policy_obs = {
        "eef_pose": torch.tensor(
            [
                [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
                [0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "gripper_pos": torch.tensor([[0.01, 0.02], [0.03, 0.04]], dtype=torch.float32),
    }

    state = build_tabero_state(policy_obs)

    assert state.shape == (2, 7)
    torch.testing.assert_close(state[:, :3], policy_obs["eef_pose"][:, :3])
    torch.testing.assert_close(state[:, 3:6], torch.zeros((2, 3)))
    torch.testing.assert_close(state[:, 6], torch.tensor([0.01, 0.03]))


def test_get_env_cls_resolves_tabero_tacfield_id_to_adapter():
    cfg = OmegaConf.create(
        {
            "init_params": {
                "id": "Isaac-Libero-Franka-Hybrid-Tactile-v0",
            }
        }
    )

    assert get_env_cls("isaaclab", cfg) is IsaaclabTaberoTacFieldEnv


def test_env_output_preserves_tabero_tactile_marker_motion_for_rollout_channel():
    tactile = torch.zeros((2, 9, 198, 2), dtype=torch.float32)
    custom_extra = torch.ones((2, 3), dtype=torch.float32)
    env_output = EnvOutput(
        obs={
            "main_images": torch.zeros((2, 4, 4, 3), dtype=torch.uint8),
            "wrist_images": torch.ones((2, 4, 4, 3), dtype=torch.uint8),
            "states": torch.zeros((2, 7), dtype=torch.float32),
            "task_descriptions": ["task a", "task b"],
            "tactile_marker_motion": tactile,
            "custom_extra": custom_extra,
        },
        final_obs={
            "main_images": torch.zeros((2, 4, 4, 3), dtype=torch.uint8),
            "wrist_images": torch.ones((2, 4, 4, 3), dtype=torch.uint8),
            "states": torch.zeros((2, 7), dtype=torch.float32),
            "task_descriptions": ["task a", "task b"],
            "tactile_marker_motion": tactile + 1,
            "custom_extra": custom_extra + 1,
        },
    )

    output_dict = env_output.to_dict()

    assert "tactile_marker_motion" in output_dict["obs"]
    assert "custom_extra" in output_dict["obs"]
    assert "tactile_marker_motion" in output_dict["final_obs"]
    assert "custom_extra" in output_dict["final_obs"]
    torch.testing.assert_close(output_dict["obs"]["tactile_marker_motion"], tactile)
    torch.testing.assert_close(output_dict["obs"]["custom_extra"], custom_extra)
    torch.testing.assert_close(
        output_dict["final_obs"]["tactile_marker_motion"], tactile + 1
    )
    torch.testing.assert_close(
        output_dict["final_obs"]["custom_extra"], custom_extra + 1
    )
