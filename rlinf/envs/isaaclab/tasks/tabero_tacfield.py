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

from __future__ import annotations

import os
import sys
from typing import Any

import gymnasium as gym
import torch

from rlinf.envs.isaaclab.utils import quat2axisangle_torch

from ..isaaclab_env import IsaaclabBaseEnv


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    return getattr(cfg, name, default) if cfg is not None else default


def _camera_rgb_observation(env, camera_name: str) -> torch.Tensor:
    camera = env.scene[camera_name]
    return camera.data.output["rgb"]


def _success_terminal_reward(
    env,
    success_func: Any,
    success_params: dict[str, Any] | None = None,
) -> torch.Tensor:
    success = success_func(env, **(success_params or {}))
    return success.to(dtype=torch.float32) / float(env.step_dt)


def _install_success_reward(
    isaac_env_cfg: Any,
    reward_term_cls: Any,
    reward_coef: float,
) -> None:
    terminations_cfg = getattr(isaac_env_cfg, "terminations", None)
    success_term = getattr(terminations_cfg, "success", None)
    success_func = getattr(success_term, "func", None)
    if success_func is None:
        return

    success_reward = reward_term_cls(
        func=_success_terminal_reward,
        weight=float(reward_coef),
        params={
            "success_func": success_func,
            "success_params": dict(getattr(success_term, "params", {}) or {}),
        },
    )
    rewards_cfg = getattr(isaac_env_cfg, "rewards", None)
    if rewards_cfg is None:
        isaac_env_cfg.rewards = {"success": success_reward}
    elif isinstance(rewards_cfg, dict):
        rewards_cfg["success"] = success_reward
    else:
        setattr(rewards_cfg, "success", success_reward)


def _set_camera_resolution(scene, camera_name: str, camera_cfg: Any) -> None:
    if camera_cfg is None or not hasattr(scene, camera_name):
        return
    camera = getattr(scene, camera_name)
    height = _cfg_get(camera_cfg, "height")
    width = _cfg_get(camera_cfg, "width")
    if height is not None:
        camera.height = int(height)
    if width is not None:
        camera.width = int(width)


def _prepend_python_path(path: str | None) -> None:
    if not path:
        return
    path = os.path.abspath(os.path.expanduser(str(path)))
    if path not in sys.path:
        sys.path.insert(0, path)


def _first_gripper_scalar(gripper_pos: torch.Tensor) -> torch.Tensor:
    if gripper_pos.ndim == 1:
        return gripper_pos[:, None]
    return gripper_pos[:, :1]


def build_tabero_state(policy_obs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Build Tabero's 7D state: xyz + axis-angle + gripper scalar."""

    if "eef_pose" in policy_obs:
        eef_pose = policy_obs["eef_pose"]
        pos = eef_pose[:, :3]
        # TacManip's ee_frame_pose_in_base_frame emits quaternion as wxyz.
        quat_xyzw = eef_pose[:, 3:7][:, [1, 2, 3, 0]]
    elif "eef_pos" in policy_obs and "eef_quat" in policy_obs:
        pos = policy_obs["eef_pos"]
        quat = policy_obs["eef_quat"]
        quat_xyzw = quat[:, [1, 2, 3, 0]]
    else:
        raise KeyError("Tabero TacField env expects 'eef_pose' or 'eef_pos'/'eef_quat'.")

    if "gripper_pos" not in policy_obs:
        raise KeyError("Tabero TacField env expects 'gripper_pos'.")

    axis_angle = quat2axisangle_torch(quat_xyzw)
    gripper = _first_gripper_scalar(policy_obs["gripper_pos"])
    return torch.cat([pos, axis_angle, gripper], dim=1).to(dtype=torch.float32)


class TacManipMarkerMotionHistory:
    """Convert TacManip per-step marker motion into Tabero TacField prefix input."""

    def __init__(
        self,
        num_envs: int,
        history_len: int = 8,
        expected_combined_markers: int = 198,
    ) -> None:
        if history_len <= 0:
            raise ValueError("history_len must be positive.")
        self.num_envs = int(num_envs)
        self.history_len = int(history_len)
        self.expected_combined_markers = int(expected_combined_markers)
        self._reference: torch.Tensor | None = None
        self._history: torch.Tensor | None = None
        self._initialized: torch.Tensor | None = None

    def _ensure_storage(self, current: torch.Tensor) -> None:
        shape = (
            self.num_envs,
            self.history_len,
            current.shape[-2],
            current.shape[-1],
        )
        if (
            self._history is not None
            and self._history.shape == shape
            and self._history.device == current.device
            and self._history.dtype == current.dtype
        ):
            return

        self._reference = torch.zeros(
            (self.num_envs, current.shape[-2], current.shape[-1]),
            device=current.device,
            dtype=current.dtype,
        )
        self._history = torch.zeros(shape, device=current.device, dtype=current.dtype)
        self._initialized = torch.zeros(
            self.num_envs, device=current.device, dtype=torch.bool
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._initialized is None:
            return
        if env_ids is None:
            self._initialized.zero_()
            if self._reference is not None:
                self._reference.zero_()
            if self._history is not None:
                self._history.zero_()
            return

        env_ids = env_ids.to(device=self._initialized.device, dtype=torch.long)
        self._initialized[env_ids] = False
        if self._reference is not None:
            self._reference[env_ids] = 0
        if self._history is not None:
            self._history[env_ids] = 0

    def update(self, marker_motion: torch.Tensor) -> torch.Tensor:
        """Update and return shape ``(N, 1 + history_len, 2 * markers, 2)``."""

        if marker_motion.ndim != 5:
            raise ValueError(
                "marker_motion must have shape (N, sensors, init/current, markers, xy); "
                f"got {tuple(marker_motion.shape)}."
            )
        num_envs, sensors, time_dim, markers, xy = marker_motion.shape
        if num_envs != self.num_envs:
            raise ValueError(
                f"marker_motion batch size {num_envs} does not match num_envs={self.num_envs}."
            )
        if time_dim != 2 or xy != 2:
            raise ValueError(
                "marker_motion must include init/current and xy dimensions; "
                f"got time_dim={time_dim}, xy={xy}."
            )
        combined_markers = sensors * markers
        if combined_markers != self.expected_combined_markers:
            raise ValueError(
                "TacField checkpoint expects "
                f"{self.expected_combined_markers} combined markers, got {combined_markers}."
            )

        marker_motion = marker_motion.to(dtype=torch.float32)
        init_pos = marker_motion[:, :, 0].reshape(num_envs, combined_markers, xy)
        current_pos = marker_motion[:, :, 1].reshape(num_envs, combined_markers, xy)

        self._ensure_storage(current_pos)
        assert self._reference is not None
        assert self._history is not None
        assert self._initialized is not None

        new_envs = ~self._initialized
        if new_envs.any():
            self._reference[new_envs] = init_pos[new_envs]
            self._history[new_envs] = current_pos[new_envs, None].expand(
                -1, self.history_len, -1, -1
            )
            self._initialized[new_envs] = True

        existing_envs = ~new_envs
        if existing_envs.any():
            self._history[existing_envs] = torch.roll(
                self._history[existing_envs], shifts=-1, dims=1
            )
            self._history[existing_envs, -1] = current_pos[existing_envs]

        return torch.cat([self._reference[:, None], self._history], dim=1)


class IsaaclabTaberoTacFieldEnv(IsaaclabBaseEnv):
    """RLinf adapter for Tabero TacManip IsaacLab tactile environments."""

    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
    ):
        init_params = cfg.init_params
        self._main_image_key = _cfg_get(init_params, "main_image_key", "agentview_rgb")
        self._wrist_image_key = _cfg_get(
            init_params, "wrist_image_key", "eye_in_hand_rgb"
        )
        self._marker_motion_key = _cfg_get(
            init_params, "marker_motion_key", "gripper_marker_motion"
        )
        self._force_key = _cfg_get(init_params, "force_key", "gripper_net_force")
        history_len = int(_cfg_get(init_params, "marker_history_len", 8))
        expected_markers = int(_cfg_get(init_params, "combined_marker_count", 198))
        self._marker_history = TacManipMarkerMotionHistory(
            num_envs=num_envs,
            history_len=history_len,
            expected_combined_markers=expected_markers,
        )

        super().__init__(
            cfg,
            num_envs,
            seed_offset,
            total_num_processes,
            worker_info,
        )

    def _make_env_function(self):
        def make_env_isaaclab():
            os.environ.pop("DISPLAY", None)

            init_params = self.cfg.init_params
            _prepend_python_path(_cfg_get(init_params, "extension_path"))

            task_suite = str(_cfg_get(init_params, "task_suite", "libero_10"))
            task_id = str(_cfg_get(init_params, "task_id", "0"))
            os.environ["TASK_SUITE"] = task_suite
            os.environ["TASK_ID"] = task_id
            os.environ["USE_RELATIVE_MODE"] = str(
                _cfg_get(init_params, "use_relative_mode", False)
            )

            libero_config_dir = _cfg_get(init_params, "libero_config_dir")
            libero_assets_dir = _cfg_get(init_params, "libero_assets_dir")
            if libero_config_dir is not None:
                os.environ["LIBERO_CONFIG_DIR"] = str(libero_config_dir)
            if libero_assets_dir is not None:
                os.environ["LIBERO_ASSETS_DATA_DIR"] = str(libero_assets_dir)

            from isaaclab.app import AppLauncher

            sim_app = AppLauncher(headless=True, enable_cameras=True).app

            import tac_manip  # noqa: F401
            from isaaclab.managers import ObservationTermCfg as ObsTerm
            from isaaclab.managers import RewardTermCfg as RewTerm
            from isaaclab_tasks.utils import load_cfg_from_registry

            isaac_env_cfg = load_cfg_from_registry(
                self.isaaclab_env_id, "env_cfg_entry_point"
            )
            isaac_env_cfg.seed = self.seed
            isaac_env_cfg.scene.num_envs = self.cfg.init_params.num_envs

            _set_camera_resolution(
                isaac_env_cfg.scene,
                "agentview_cam",
                _cfg_get(init_params, "agentview_cam", _cfg_get(init_params, "table_cam")),
            )
            _set_camera_resolution(
                isaac_env_cfg.scene,
                "eye_in_hand_cam",
                _cfg_get(init_params, "eye_in_hand_cam", _cfg_get(init_params, "wrist_cam")),
            )

            isaac_env_cfg.observations.policy.agentview_rgb = ObsTerm(
                func=_camera_rgb_observation,
                params={"camera_name": "agentview_cam"},
            )
            isaac_env_cfg.observations.policy.eye_in_hand_rgb = ObsTerm(
                func=_camera_rgb_observation,
                params={"camera_name": "eye_in_hand_cam"},
            )
            _install_success_reward(
                isaac_env_cfg,
                RewTerm,
                float(self.cfg.reward_coef),
            )

            env = gym.make(
                self.isaaclab_env_id, cfg=isaac_env_cfg, render_mode="rgb_array"
            ).unwrapped
            return env, sim_app

        return make_env_isaaclab

    def reset(self, seed=None, env_ids: torch.Tensor | None = None):
        self._marker_history.reset(env_ids)
        return super().reset(seed=seed, env_ids=env_ids)

    def _wrap_obs(self, obs):
        policy_obs = obs["policy"]
        state = build_tabero_state(policy_obs)
        tactile_marker_motion = self._marker_history.update(
            policy_obs[self._marker_motion_key]
        )

        env_obs = {
            "main_images": policy_obs[self._main_image_key],
            "wrist_images": policy_obs[self._wrist_image_key],
            "states": state,
            "task_descriptions": [self.task_description] * self.num_envs,
            "tactile_marker_motion": tactile_marker_motion,
        }
        if self._force_key in policy_obs:
            env_obs["tactile_gripper_force"] = policy_obs[self._force_key]
        return env_obs
