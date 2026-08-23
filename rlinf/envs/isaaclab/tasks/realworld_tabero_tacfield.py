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

"""RLinf adapter for the Tabero_X XArm UMI RealWorld tactile environment.

This module is intentionally independent from ``tabero_tacfield.py``. It owns
the fixed Task-6 contract, success reward, terminal-safe chunk boundary,
observation conversion, and Hybrid action validation.
"""

from __future__ import annotations

import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
from omegaconf import open_dict

from rlinf.envs.isaaclab.utils import quat2axisangle_torch

from ..isaaclab_env import IsaaclabBaseEnv

REALWORLD_ENV_ID = "Isaac-RealWorld-GentleGrasp-XarmUmi-Hybrid-Tactile-v0"
REALWORLD_TASK_SUITE = "gentle_grasp"
REALWORLD_TASK_ID = 6
REALWORLD_TACTILE_BACKEND = "taxim_fots"

REALWORLD_ACTION_DIM = 13
REALWORLD_ACTION_HORIZON = 10
REALWORLD_MARKER_HISTORY_LEN = 8
REALWORLD_COMBINED_MARKERS = 440
REALWORLD_CHUNK_BOUNDARY_MODE = "terminal_safe_v1"

_TERMINAL_RAW_OBSERVATION_KEY = "_realworld_terminal_raw_observation"
_TERMINAL_OBSERVATION_MASK_KEY = "_realworld_terminal_observation_mask"
_EXECUTED_CHUNK_ACTIONS_KEY = "_tabero_executed_chunk_actions"
_RAW_CHUNK_ACTIONS_KEY = "_tabero_raw_chunk_actions"

REALWORLD_TARGET_PROMPTS = {
    "target_object_1": "pick up the Vitasoy and put it into the basket",
    "target_object_2": "Pick up the Coca-Cola and put it into the basket",
    "target_object_3": "pick up the cookie and put it into the basket",
}
REALWORLD_BASKET_OBJECT = "target_object_4"
REALWORLD_CAMERA_SOURCE_HW = (480, 640)
REALWORLD_CAMERA_TARGET_HW = (224, 224)


def _clone_nested(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_nested(item) for item in value)
    return value


def _replace_batch_rows(base: Any, replacement: Any, mask: torch.Tensor) -> Any:
    if isinstance(base, torch.Tensor) and isinstance(replacement, torch.Tensor):
        if base.ndim == 0 or replacement.ndim == 0 or base.shape[0] != mask.numel():
            return base.clone()
        result = base.clone()
        dst_mask = mask.to(device=result.device, dtype=torch.bool)
        src_mask = mask.to(device=replacement.device, dtype=torch.bool)
        result[dst_mask] = replacement[src_mask].to(device=result.device)
        return result
    if isinstance(base, dict) and isinstance(replacement, dict):
        return {
            key: (
                _replace_batch_rows(item, replacement[key], mask)
                if key in replacement
                else _clone_nested(item)
            )
            for key, item in base.items()
        }
    if isinstance(base, list) and isinstance(replacement, list):
        result = list(base)
        selected = mask.detach().to(device="cpu", dtype=torch.bool).tolist()
        if len(result) == len(selected) and len(replacement) == len(selected):
            for index, should_replace in enumerate(selected):
                if should_replace:
                    result[index] = replacement[index]
        return result
    return _clone_nested(base)


class _SuccessStreak:
    def __init__(self, num_envs: int, required_steps: int, device: Any) -> None:
        self.required_steps = int(required_steps)
        self._streak = torch.zeros(
            int(num_envs), dtype=torch.int64, device=torch.device(device)
        )

    def reset(self, env_ids: Any = None) -> None:
        if env_ids is None:
            self._streak.zero_()
        elif isinstance(env_ids, slice):
            self._streak[env_ids] = 0
        else:
            indices = torch.as_tensor(
                env_ids, device=self._streak.device, dtype=torch.long
            )
            self._streak[indices] = 0

    def update(self, raw_success: torch.Tensor) -> torch.Tensor:
        raw_success = torch.as_tensor(
            raw_success, device=self._streak.device, dtype=torch.bool
        )
        if raw_success.shape != self._streak.shape:
            raise ValueError(
                "RealWorld success term returned shape "
                f"{tuple(raw_success.shape)}, expected {tuple(self._streak.shape)}."
            )
        self._streak = torch.where(raw_success, self._streak + 1, 0)
        return self._streak >= self.required_steps


def _make_consecutive_success_term(manager_term_base_cls: type) -> type:
    class RealWorldConsecutiveSuccessTerm(manager_term_base_cls):
        def __init__(self, cfg: Any, env: Any) -> None:
            super().__init__(cfg, env)
            self._tracker = _SuccessStreak(
                num_envs=env.num_envs,
                required_steps=int(cfg.params["required_steps"]),
                device=env.device,
            )

        def reset(self, env_ids: Any = None) -> None:
            self._tracker.reset(env_ids)

        def __call__(
            self,
            env: Any,
            success_func: Any,
            success_params: dict[str, Any],
            required_steps: int,
        ) -> torch.Tensor:
            if int(required_steps) != self._tracker.required_steps:
                raise RuntimeError(
                    "RealWorld required success streak changed at runtime."
                )
            return self._tracker.update(success_func(env, **success_params))

    RealWorldConsecutiveSuccessTerm.__name__ = "RealWorldConsecutiveSuccessTerm"
    return RealWorldConsecutiveSuccessTerm


def _terminal_success_reward(
    env: Any,
    success_term_name: str = "success",
) -> torch.Tensor:
    success = env.termination_manager.get_term(success_term_name).to(dtype=torch.bool)
    success &= ~env.termination_manager.time_outs.to(dtype=torch.bool)
    return success.to(dtype=torch.float32) / float(env.step_dt)


class _TerminalObservationCapture:
    """Capture policy observations immediately before IsaacLab's internal reset."""

    def __init__(self, env: Any) -> None:
        self._env = env
        self._capture_enabled = False
        self._captured_observation: Any = None
        self._captured_mask = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )
        original_reset_idx = env._reset_idx

        def capture_then_reset(env_ids: Any) -> None:
            indices = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
            if self._capture_enabled and indices.numel() > 0:
                terminal_observation = env.observation_manager.compute(
                    update_history=False
                )
                capture_mask = torch.zeros_like(self._captured_mask)
                capture_mask[indices] = True
                if self._captured_observation is None:
                    self._captured_observation = _clone_nested(terminal_observation)
                else:
                    self._captured_observation = _replace_batch_rows(
                        self._captured_observation,
                        terminal_observation,
                        capture_mask,
                    )
                self._captured_mask |= capture_mask
            original_reset_idx(indices)

        env._reset_idx = capture_then_reset

    @property
    def device(self):
        return self._env.device

    def reset(self, seed=None, env_ids=None):
        self._captured_observation = None
        self._captured_mask.zero_()
        return self._env.reset(seed=seed, env_ids=env_ids)

    def step(self, action: torch.Tensor):
        self._captured_observation = None
        self._captured_mask.zero_()
        self._capture_enabled = True
        try:
            obs, reward, terminated, truncated, infos = self._env.step(action)
        finally:
            self._capture_enabled = False
        infos = dict(infos or {})
        infos[_TERMINAL_RAW_OBSERVATION_KEY] = self._captured_observation
        infos[_TERMINAL_OBSERVATION_MASK_KEY] = self._captured_mask.clone()
        return obs, reward, terminated, truncated, infos

    def close(self) -> None:
        self._env.close()


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    return getattr(cfg, name, default) if cfg is not None else default


def _required_directory(init_params: Any, name: str) -> Path:
    raw_path = _cfg_get(init_params, name)
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"RealWorld init_params.{name} is required.")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"RealWorld {name} directory not found: {path}")
    return path


def _validate_extension_path(extension_path: Path) -> None:
    expected_suffix = ("Tabero_X", "source", "tac_manip")
    if tuple(extension_path.parts[-3:]) != expected_suffix:
        raise ValueError(
            "RealWorld extension_path must end in "
            f"{'/'.join(expected_suffix)}; got {extension_path}."
        )


def _validate_extension_import(
    module_file: str | os.PathLike[str] | None,
    extension_path: Path,
) -> None:
    if module_file is None:
        raise RuntimeError("Imported tac_manip package has no __file__.")
    imported_path = Path(module_file).expanduser().resolve()
    try:
        imported_path.relative_to(extension_path)
    except ValueError as error:
        raise RuntimeError(
            "tac_manip was imported from the wrong checkout: "
            f"expected under {extension_path}, got {imported_path}."
        ) from error


def _load_task_contract(
    config_dir: Path,
    *,
    target_object: str,
    task_description: str,
) -> tuple[str, list[dict[str, Any]]]:
    config_path = config_dir / f"{REALWORLD_TASK_SUITE}.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"RealWorld task config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    tasks = config.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError(f"RealWorld task config has no task list: {config_path}")

    task_entry = next(
        (
            task
            for task in tasks
            if isinstance(task, dict)
            and int(task.get("task_id", -1)) == REALWORLD_TASK_ID
        ),
        None,
    )
    if task_entry is None:
        raise ValueError(
            f"RealWorld {REALWORLD_TASK_SUITE} Task {REALWORLD_TASK_ID} is missing "
            f"from {config_path}."
        )
    configured_description = str(task_description).strip()
    expected_description = REALWORLD_TARGET_PROMPTS.get(target_object)
    if expected_description is None:
        raise ValueError(
            "RealWorld Task 6 target_object must be one of "
            f"{sorted(REALWORLD_TARGET_PROMPTS)}; got {target_object!r}."
        )
    if configured_description != expected_description:
        raise ValueError(
            "RealWorld Task 6 target prompt mismatch: "
            f"target={target_object!r}, expected={expected_description!r}, "
            f"got={configured_description!r}."
        )
    goals = task_entry.get("goals")
    if not isinstance(goals, list) or not goals:
        raise ValueError(
            f"RealWorld {REALWORLD_TASK_SUITE} Task {REALWORLD_TASK_ID} must "
            "define non-empty goals."
        )
    matching_branches = []
    for goal in goals:
        if not isinstance(goal, dict):
            continue
        any_of = goal.get("any_of")
        if not isinstance(any_of, list):
            continue
        matching_branches.extend(
            branch
            for branch in any_of
            if isinstance(branch, dict)
            and branch.get("ref_obj") == target_object
            and branch.get("target") == REALWORLD_BASKET_OBJECT
        )
    if len(matching_branches) != 1:
        raise ValueError(
            "RealWorld Task 6 must contain exactly one success branch for "
            f"{target_object!r} -> {REALWORLD_BASKET_OBJECT!r}; "
            f"got {len(matching_branches)}."
        )
    return configured_description, [{"any_of": [matching_branches[0]]}]


def _prepend_python_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _validate_camera_source_cfg(scene: Any, camera_name: str) -> None:
    camera = getattr(scene, camera_name, None)
    if camera is None:
        raise ValueError(f"RealWorld scene is missing camera {camera_name!r}.")
    actual = (int(camera.height), int(camera.width))
    if actual != REALWORLD_CAMERA_SOURCE_HW:
        raise ValueError(
            f"RealWorld {camera_name} must preserve native source resolution "
            f"{REALWORLD_CAMERA_SOURCE_HW}; got {actual}."
        )


def _camera_rgb_observation(env: Any, camera_name: str) -> torch.Tensor:
    return env.scene[camera_name].data.output["rgb"]


def _validate_hybrid_action_cfg(
    isaac_env_cfg: Any,
    *,
    require_policy_gripper_sign_bridge: bool,
) -> None:
    actions_cfg = getattr(isaac_env_cfg, "actions", None)
    arm_action = getattr(actions_cfg, "arm_action", None)
    gripper_action = getattr(actions_cfg, "gripper_action", None)
    action_class = getattr(arm_action, "class_type", None)
    action_mro_names = {
        getattr(base_class, "__name__", "")
        for base_class in getattr(action_class, "__mro__", ())
    }
    if "ForcePositionAction" not in action_mro_names:
        raise ValueError(
            "RealWorld Task 6 requires the 13D ForcePositionAction controller."
        )
    if require_policy_gripper_sign_bridge and (
        getattr(action_class, "__name__", None)
        != "XarmPolicyGripperSignBridgeForcePositionAction"
    ):
        raise ValueError(
            "RealWorld legacy Tabero normalization requires the XArm policy "
            "gripper sign bridge action term."
        )
    if gripper_action is not None:
        raise ValueError(
            "RealWorld Hybrid control must not expose a separate binary gripper action."
        )
    ik_cfg = getattr(arm_action, "ik_cfg", None)
    controller_cfg = getattr(ik_cfg, "controller", None)
    if bool(getattr(controller_cfg, "use_relative_mode", True)):
        raise ValueError("RealWorld Hybrid control requires absolute EEF poses.")
    if require_policy_gripper_sign_bridge:
        observations_cfg = getattr(isaac_env_cfg, "observations", None)
        policy_cfg = getattr(observations_cfg, "policy", None)
        gripper_pos_cfg = getattr(policy_cfg, "gripper_pos", None)
        gripper_pos_func = getattr(gripper_pos_cfg, "func", None)
        if getattr(gripper_pos_func, "__name__", None) != "policy_gripper_pos":
            raise ValueError(
                "RealWorld legacy Tabero normalization requires the policy "
                "gripper sign bridge observation term."
            )


def _build_state(policy_obs: dict[str, torch.Tensor]) -> torch.Tensor:
    eef_pose = policy_obs.get("eef_pose")
    gripper_pos = policy_obs.get("gripper_pos")
    if eef_pose is None or tuple(eef_pose.shape[1:]) != (7,):
        shape = None if eef_pose is None else tuple(eef_pose.shape)
        raise ValueError(f"RealWorld eef_pose must have shape (N, 7); got {shape}.")
    if gripper_pos is None or gripper_pos.ndim not in (1, 2):
        shape = None if gripper_pos is None else tuple(gripper_pos.shape)
        raise ValueError(
            f"RealWorld gripper_pos must have shape (N,) or (N, J); got {shape}."
        )
    if gripper_pos.shape[0] != eef_pose.shape[0]:
        raise ValueError("RealWorld eef_pose and gripper_pos batches do not match.")

    position = eef_pose[:, :3]
    quaternion_xyzw = eef_pose[:, 3:7][:, [1, 2, 3, 0]]
    axis_angle = quat2axisangle_torch(quaternion_xyzw)
    gripper_scalar = (
        gripper_pos[:, None] if gripper_pos.ndim == 1 else gripper_pos[:, :1]
    )
    state = torch.cat([position, axis_angle, gripper_scalar], dim=1).to(
        dtype=torch.float32
    )
    if tuple(state.shape[1:]) != (7,) or not torch.isfinite(state).all():
        raise ValueError("RealWorld state must be finite with shape (N, 7).")
    return state


class _RealWorldActionChunkFilter:
    """Vectorized XArm absolute-pose filter matching the Tabero_X evaluator."""

    def __init__(
        self,
        num_envs: int,
        *,
        transition_steps: int,
        max_position_step_m: float,
        max_position_delta_change_m: float,
        max_orientation_step_deg: float,
        max_orientation_delta_change_deg: float,
    ) -> None:
        if transition_steps <= 0:
            raise ValueError(
                "RealWorld action filter transition_steps must be positive."
            )
        positive_limits = {
            "max_position_step_m": max_position_step_m,
            "max_position_delta_change_m": max_position_delta_change_m,
            "max_orientation_step_deg": max_orientation_step_deg,
            "max_orientation_delta_change_deg": max_orientation_delta_change_deg,
        }
        for name, value in positive_limits.items():
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(
                    f"RealWorld action filter {name} must be finite and positive."
                )
        self.num_envs = int(num_envs)
        self.transition_steps = int(transition_steps)
        self.max_position_step_m = float(max_position_step_m)
        self.max_position_delta_change_m = float(max_position_delta_change_m)
        self.max_orientation_step_rad = math.radians(float(max_orientation_step_deg))
        self.max_orientation_delta_change_rad = math.radians(
            float(max_orientation_delta_change_deg)
        )
        self._last_action: torch.Tensor | None = None
        self._last_position_delta: torch.Tensor | None = None
        self._last_orientation_delta: torch.Tensor | None = None
        self._initialized: torch.Tensor | None = None

    @staticmethod
    def _limit_norm(vector: torch.Tensor, limit: float) -> torch.Tensor:
        norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
        scale = torch.clamp(float(limit) / torch.clamp(norm, min=1e-12), max=1.0)
        return vector * scale

    @staticmethod
    def _rotvec_to_quat(rotation_vector: torch.Tensor) -> torch.Tensor:
        angle = torch.linalg.vector_norm(rotation_vector, dim=-1, keepdim=True)
        half_angle = 0.5 * angle
        xyz = torch.where(
            angle > 1e-12,
            rotation_vector / torch.clamp(angle, min=1e-12) * torch.sin(half_angle),
            torch.zeros_like(rotation_vector),
        )
        return torch.cat([torch.cos(half_angle), xyz], dim=-1)

    @staticmethod
    def _quat_multiply(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        scalar = first[..., :1] * second[..., :1] - torch.sum(
            first[..., 1:] * second[..., 1:], dim=-1, keepdim=True
        )
        vector = (
            first[..., :1] * second[..., 1:]
            + second[..., :1] * first[..., 1:]
            + torch.linalg.cross(first[..., 1:], second[..., 1:], dim=-1)
        )
        return torch.cat([scalar, vector], dim=-1)

    @staticmethod
    def _quat_inverse(quaternion: torch.Tensor) -> torch.Tensor:
        conjugate = torch.cat([quaternion[..., :1], -quaternion[..., 1:]], dim=-1)
        return conjugate / torch.sum(quaternion * quaternion, dim=-1, keepdim=True)

    @staticmethod
    def _quat_to_rotvec(quaternion: torch.Tensor) -> torch.Tensor:
        quaternion = quaternion / torch.linalg.vector_norm(
            quaternion, dim=-1, keepdim=True
        )
        quaternion = torch.where(
            quaternion[..., :1] < 0,
            -quaternion,
            quaternion,
        )
        sin_half_angle = torch.linalg.vector_norm(
            quaternion[..., 1:], dim=-1, keepdim=True
        )
        angle = 2.0 * torch.atan2(sin_half_angle, quaternion[..., :1])
        return torch.where(
            sin_half_angle > 1e-12,
            quaternion[..., 1:] / torch.clamp(sin_half_angle, min=1e-12) * angle,
            torch.zeros_like(quaternion[..., 1:]),
        )

    @classmethod
    def _quat_to_nearest_rotvec(
        cls,
        quaternion: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        canonical = cls._quat_to_rotvec(quaternion)
        angle = torch.linalg.vector_norm(canonical, dim=-1, keepdim=True)
        axis = canonical / torch.clamp(angle, min=1e-12)
        reference_turn = (torch.sum(reference * axis, dim=-1, keepdim=True) - angle) / (
            2.0 * math.pi
        )
        center = torch.round(reference_turn)
        offsets = torch.tensor(
            [-1.0, 0.0, 1.0],
            device=canonical.device,
            dtype=canonical.dtype,
        ).view(1, 3, 1)
        candidates = (
            canonical[:, None]
            + (center[:, None] + offsets) * (2.0 * math.pi) * axis[:, None]
        )
        distances = torch.linalg.vector_norm(candidates - reference[:, None], dim=-1)
        selected = distances.argmin(dim=1)
        nearest = candidates[
            torch.arange(candidates.shape[0], device=candidates.device), selected
        ]
        return torch.where(angle > 1e-12, nearest, canonical)

    @staticmethod
    def _shortest_slerp(
        first: torch.Tensor,
        second: torch.Tensor,
        fraction: float,
    ) -> torch.Tensor:
        dot = torch.sum(first * second, dim=-1, keepdim=True)
        second = torch.where(dot < 0, -second, second)
        dot = torch.clamp(torch.abs(dot), -1.0, 1.0)
        linear = first + float(fraction) * (second - first)
        linear = linear / torch.linalg.vector_norm(linear, dim=-1, keepdim=True)
        angle = torch.acos(dot)
        sin_angle = torch.sin(angle)
        spherical = (
            torch.sin((1.0 - float(fraction)) * angle)
            / torch.clamp(sin_angle, min=1e-12)
            * first
            + torch.sin(float(fraction) * angle)
            / torch.clamp(sin_angle, min=1e-12)
            * second
        )
        return torch.where(dot > 0.9995, linear, spherical)

    def reset(
        self,
        anchor_state: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        if tuple(anchor_state.shape) != (self.num_envs, 7):
            raise ValueError(
                "RealWorld action filter anchor must have shape "
                f"({self.num_envs}, 7); got {tuple(anchor_state.shape)}."
            )
        if self._last_action is None or (
            self._last_action.device != anchor_state.device
            or self._last_action.dtype != anchor_state.dtype
        ):
            self._last_action = torch.zeros(
                self.num_envs,
                REALWORLD_ACTION_DIM,
                device=anchor_state.device,
                dtype=anchor_state.dtype,
            )
            self._last_position_delta = torch.zeros(
                self.num_envs, 3, device=anchor_state.device, dtype=torch.float64
            )
            self._last_orientation_delta = torch.zeros_like(self._last_position_delta)
            self._initialized = torch.zeros(
                self.num_envs, device=anchor_state.device, dtype=torch.bool
            )
        assert self._last_position_delta is not None
        assert self._last_orientation_delta is not None
        assert self._initialized is not None
        indices = (
            torch.arange(self.num_envs, device=anchor_state.device)
            if env_ids is None
            else torch.as_tensor(env_ids, device=anchor_state.device, dtype=torch.long)
        )
        self._last_action[indices] = 0
        self._last_action[indices, :7] = anchor_state[indices]
        self._last_position_delta[indices] = 0
        self._last_orientation_delta[indices] = 0
        self._initialized[indices] = True

    def filter(self, actions: torch.Tensor) -> torch.Tensor:
        expected_shape = (
            self.num_envs,
            REALWORLD_ACTION_HORIZON,
            REALWORLD_ACTION_DIM,
        )
        if tuple(actions.shape) != expected_shape:
            raise ValueError(
                f"RealWorld action filter expects {expected_shape}; "
                f"got {tuple(actions.shape)}."
            )
        if self._initialized is None or not bool(self._initialized.all()):
            raise RuntimeError(
                "RealWorld action filter must be anchored from reset observations "
                "before filtering a policy chunk."
            )
        assert self._last_action is not None
        assert self._last_position_delta is not None
        assert self._last_orientation_delta is not None
        assert self._initialized is not None

        if self._last_action.device != actions.device:
            self._last_action = self._last_action.to(device=actions.device)
            self._last_position_delta = self._last_position_delta.to(
                device=actions.device
            )
            self._last_orientation_delta = self._last_orientation_delta.to(
                device=actions.device
            )
            self._initialized = self._initialized.to(device=actions.device)

        filtered = actions.clone()
        work_dtype = torch.float64
        chunk_anchor = self._last_action.to(work_dtype)
        previous = chunk_anchor.clone()
        previous_position_delta = self._last_position_delta.clone()
        previous_orientation_delta = self._last_orientation_delta.clone()

        for action_index in range(REALWORLD_ACTION_HORIZON):
            candidate = filtered[:, action_index].to(work_dtype)
            if action_index < self.transition_steps:
                fraction = (action_index + 1) / self.transition_steps
                candidate[:, :3] = chunk_anchor[:, :3] + fraction * (
                    candidate[:, :3] - chunk_anchor[:, :3]
                )
                anchor_quat = self._rotvec_to_quat(chunk_anchor[:, 3:6])
                candidate_quat = self._rotvec_to_quat(candidate[:, 3:6])
                candidate[:, 3:6] = self._quat_to_rotvec(
                    self._shortest_slerp(anchor_quat, candidate_quat, fraction)
                )

            position_delta = self._limit_norm(
                candidate[:, :3] - previous[:, :3],
                self.max_position_step_m,
            )
            position_delta_change = self._limit_norm(
                position_delta - previous_position_delta,
                self.max_position_delta_change_m,
            )
            position_delta = self._limit_norm(
                previous_position_delta + position_delta_change,
                self.max_position_step_m,
            )
            candidate[:, :3] = previous[:, :3] + position_delta

            previous_quat = self._rotvec_to_quat(previous[:, 3:6])
            candidate_quat = self._rotvec_to_quat(candidate[:, 3:6])
            orientation_delta = self._quat_to_rotvec(
                self._quat_multiply(
                    candidate_quat,
                    self._quat_inverse(previous_quat),
                )
            )
            orientation_delta = self._limit_norm(
                orientation_delta,
                self.max_orientation_step_rad,
            )
            orientation_delta_change = self._limit_norm(
                orientation_delta - previous_orientation_delta,
                self.max_orientation_delta_change_rad,
            )
            orientation_delta = self._limit_norm(
                previous_orientation_delta + orientation_delta_change,
                self.max_orientation_step_rad,
            )
            candidate_quat = self._quat_multiply(
                self._rotvec_to_quat(orientation_delta),
                previous_quat,
            )
            candidate[:, 3:6] = self._quat_to_nearest_rotvec(
                candidate_quat,
                previous[:, 3:6],
            )

            filtered[:, action_index, :6] = candidate[:, :6].to(actions.dtype)
            previous_position_delta = candidate[:, :3] - previous[:, :3]
            previous_orientation_delta = orientation_delta
            previous = filtered[:, action_index].to(work_dtype)

        self._last_action = filtered[:, -1].clone()
        self._last_position_delta = previous_position_delta
        self._last_orientation_delta = previous_orientation_delta
        return filtered


class _RealWorldMarkerHistory:
    """Build one reference frame plus eight current marker-motion frames."""

    def __init__(self, num_envs: int) -> None:
        self.num_envs = int(num_envs)
        self._reference: torch.Tensor | None = None
        self._history: torch.Tensor | None = None
        self._initialized: torch.Tensor | None = None

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
        indices = torch.as_tensor(
            env_ids, device=self._initialized.device, dtype=torch.long
        )
        self._initialized[indices] = False
        if self._reference is not None:
            self._reference[indices] = 0
        if self._history is not None:
            self._history[indices] = 0

    def _allocate(self, current: torch.Tensor) -> None:
        expected_history_shape = (
            self.num_envs,
            REALWORLD_MARKER_HISTORY_LEN,
            REALWORLD_COMBINED_MARKERS,
            2,
        )
        if (
            self._history is not None
            and tuple(self._history.shape) == expected_history_shape
            and self._history.device == current.device
            and self._history.dtype == current.dtype
        ):
            return
        self._reference = torch.zeros_like(current)
        self._history = torch.zeros(
            expected_history_shape, device=current.device, dtype=current.dtype
        )
        self._initialized = torch.zeros(
            self.num_envs, device=current.device, dtype=torch.bool
        )

    def update(
        self,
        marker_motion: torch.Tensor,
        update_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expected_prefix = (self.num_envs, 2, 2)
        if (
            marker_motion.ndim != 5
            or tuple(marker_motion.shape[:3]) != expected_prefix
            or marker_motion.shape[-1] != 2
        ):
            raise ValueError(
                "RealWorld gripper_marker_motion must have shape "
                f"(N, 2, 2, M, 2); got {tuple(marker_motion.shape)}."
            )
        combined_markers = int(marker_motion.shape[1] * marker_motion.shape[3])
        if combined_markers != REALWORLD_COMBINED_MARKERS:
            raise ValueError(
                f"RealWorld tactile input requires {REALWORLD_COMBINED_MARKERS} "
                f"combined markers; got {combined_markers}."
            )

        marker_motion = marker_motion.to(dtype=torch.float32)
        reference = marker_motion[:, :, 0].reshape(
            self.num_envs, REALWORLD_COMBINED_MARKERS, 2
        )
        current = marker_motion[:, :, 1].reshape(
            self.num_envs, REALWORLD_COMBINED_MARKERS, 2
        )
        if not torch.isfinite(reference).all() or not torch.isfinite(current).all():
            raise ValueError("RealWorld tactile marker motion contains NaN or Inf.")

        self._allocate(current)
        assert self._reference is not None
        assert self._history is not None
        assert self._initialized is not None

        if update_mask is None:
            update_mask = torch.ones(
                self.num_envs, device=current.device, dtype=torch.bool
            )
        else:
            update_mask = torch.as_tensor(
                update_mask, device=current.device, dtype=torch.bool
            )
            if update_mask.shape != (self.num_envs,):
                raise ValueError(
                    "RealWorld marker update mask must have shape "
                    f"({self.num_envs},); got {tuple(update_mask.shape)}."
                )

        new_envs = ~self._initialized & update_mask
        if new_envs.any():
            self._reference[new_envs] = reference[new_envs]
            self._history[new_envs] = current[new_envs, None].expand(
                -1, REALWORLD_MARKER_HISTORY_LEN, -1, -1
            )
            self._initialized[new_envs] = True

        existing_envs = self._initialized & update_mask & ~new_envs
        if existing_envs.any():
            self._history[existing_envs] = torch.roll(
                self._history[existing_envs], shifts=-1, dims=1
            )
            self._history[existing_envs, -1] = current[existing_envs]

        result = torch.cat([self._reference[:, None], self._history], dim=1)
        expected_shape = (
            self.num_envs,
            REALWORLD_MARKER_HISTORY_LEN + 1,
            REALWORLD_COMBINED_MARKERS,
            2,
        )
        if tuple(result.shape) != expected_shape:
            raise RuntimeError(
                f"RealWorld tactile history has shape {tuple(result.shape)}, "
                f"expected {expected_shape}."
            )
        return result


class IsaaclabRealWorldTaberoTacFieldEnv(IsaaclabBaseEnv):
    """Independent RLinf wrapper for XArm UMI GentleGrasp Task 6."""

    def __init__(
        self,
        cfg: Any,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any,
    ) -> None:
        init_params = cfg.init_params
        env_id = str(_cfg_get(init_params, "id", ""))
        if env_id != REALWORLD_ENV_ID:
            raise ValueError(
                f"RealWorld wrapper supports only {REALWORLD_ENV_ID!r}; got {env_id!r}."
            )

        task_suite = str(_cfg_get(init_params, "task_suite", REALWORLD_TASK_SUITE))
        task_id = int(_cfg_get(init_params, "task_id", REALWORLD_TASK_ID))
        if task_suite != REALWORLD_TASK_SUITE or task_id != REALWORLD_TASK_ID:
            raise ValueError(
                "RealWorld wrapper v1 is fixed to gentle_grasp Task 6; "
                f"got suite={task_suite!r}, task_id={task_id}."
            )

        tactile_backend = (
            str(_cfg_get(init_params, "tactile_backend", REALWORLD_TACTILE_BACKEND))
            .strip()
            .lower()
        )
        if tactile_backend != REALWORLD_TACTILE_BACKEND:
            raise ValueError(
                "RealWorld wrapper requires tactile_backend='taxim_fots'; "
                f"got {tactile_backend!r}."
            )
        history_len = int(
            _cfg_get(
                init_params,
                "marker_history_len",
                REALWORLD_MARKER_HISTORY_LEN,
            )
        )
        marker_count = int(
            _cfg_get(
                init_params,
                "combined_marker_count",
                REALWORLD_COMBINED_MARKERS,
            )
        )
        if (
            history_len != REALWORLD_MARKER_HISTORY_LEN
            or marker_count != REALWORLD_COMBINED_MARKERS
        ):
            raise ValueError(
                "RealWorld wrapper requires marker_history_len=8 and "
                f"combined_marker_count=440; got {history_len} and {marker_count}."
            )

        chunk_boundary_mode = str(
            _cfg_get(
                init_params,
                "chunk_boundary_mode",
                REALWORLD_CHUNK_BOUNDARY_MODE,
            )
        )
        if chunk_boundary_mode != REALWORLD_CHUNK_BOUNDARY_MODE:
            raise ValueError(
                "RealWorld wrapper requires chunk_boundary_mode="
                f"{REALWORLD_CHUNK_BOUNDARY_MODE!r}; got {chunk_boundary_mode!r}."
            )
        if cfg.auto_reset is not False or cfg.ignore_terminations is not False:
            raise ValueError(
                "RealWorld terminal-safe chunks require auto_reset=false and "
                "ignore_terminations=false."
            )

        max_episode_steps = cfg.max_episode_steps
        if (
            isinstance(max_episode_steps, bool)
            or not isinstance(max_episode_steps, int)
            or max_episode_steps <= 0
            or max_episode_steps % REALWORLD_ACTION_HORIZON != 0
        ):
            raise ValueError(
                "RealWorld max_episode_steps must be a positive multiple of "
                f"{REALWORLD_ACTION_HORIZON}; got {max_episode_steps!r}."
            )

        success_cfg = _cfg_get(init_params, "success")
        required_success_steps = int(
            _cfg_get(success_cfg, "required_consecutive_steps", 8)
        )
        terminal_reward = float(_cfg_get(success_cfg, "terminal_reward", 1.0))
        if required_success_steps <= 0:
            raise ValueError("RealWorld required_consecutive_steps must be positive.")
        if not torch.isfinite(torch.tensor(terminal_reward)) or terminal_reward <= 0:
            raise ValueError("RealWorld terminal_reward must be finite and positive.")

        policy_gripper_sign_bridge = _cfg_get(
            init_params, "policy_gripper_sign_bridge", False
        )
        if not isinstance(policy_gripper_sign_bridge, bool):
            raise ValueError(
                "RealWorld init_params.policy_gripper_sign_bridge must be boolean."
            )
        if policy_gripper_sign_bridge:
            raise ValueError(
                "RealWorld XArm-gripper PiRL requires "
                "policy_gripper_sign_bridge=false; use a dedicated legacy preset "
                "for negative-open checkpoints."
            )

        target_object = str(_cfg_get(init_params, "target_object", "")).strip()
        configured_task_description = str(
            _cfg_get(init_params, "task_description", "")
        ).strip()
        reset_source = str(_cfg_get(init_params, "reset_source", "")).strip()
        if reset_source != "task_config_default_reset":
            raise ValueError(
                "RealWorld Task 6 PiRL requires reset_source="
                "'task_config_default_reset'."
            )

        action_filter_cfg = _cfg_get(init_params, "action_filter")
        if (
            action_filter_cfg is None
            or _cfg_get(action_filter_cfg, "enabled") is not True
        ):
            raise ValueError(
                "RealWorld Task 6 PiRL requires action_filter.enabled=true."
            )
        self._action_filter = _RealWorldActionChunkFilter(
            num_envs,
            transition_steps=int(_cfg_get(action_filter_cfg, "transition_steps", 0)),
            max_position_step_m=float(
                _cfg_get(action_filter_cfg, "max_position_step_m", 0.0)
            ),
            max_position_delta_change_m=float(
                _cfg_get(action_filter_cfg, "max_position_delta_change_m", 0.0)
            ),
            max_orientation_step_deg=float(
                _cfg_get(action_filter_cfg, "max_orientation_step_deg", 0.0)
            ),
            max_orientation_delta_change_deg=float(
                _cfg_get(action_filter_cfg, "max_orientation_delta_change_deg", 0.0)
            ),
        )

        camera_preprocess_cfg = _cfg_get(init_params, "camera_preprocess")
        actual_camera_contract = {
            "source_height": int(_cfg_get(camera_preprocess_cfg, "source_height", -1)),
            "source_width": int(_cfg_get(camera_preprocess_cfg, "source_width", -1)),
            "target_height": int(_cfg_get(camera_preprocess_cfg, "target_height", -1)),
            "target_width": int(_cfg_get(camera_preprocess_cfg, "target_width", -1)),
            "mode": str(_cfg_get(camera_preprocess_cfg, "mode", "")),
            "interpolation": str(_cfg_get(camera_preprocess_cfg, "interpolation", "")),
        }
        expected_camera_contract = {
            "source_height": REALWORLD_CAMERA_SOURCE_HW[0],
            "source_width": REALWORLD_CAMERA_SOURCE_HW[1],
            "target_height": REALWORLD_CAMERA_TARGET_HW[0],
            "target_width": REALWORLD_CAMERA_TARGET_HW[1],
            "mode": "stretch",
            "interpolation": "INTER_AREA",
        }
        if actual_camera_contract != expected_camera_contract:
            raise ValueError(
                "RealWorld camera preprocessing contract mismatch: "
                f"expected={expected_camera_contract}, actual={actual_camera_contract}."
            )

        self._extension_path = _required_directory(init_params, "extension_path")
        _validate_extension_path(self._extension_path)
        self._config_dir = _required_directory(init_params, "realworld_config_dir")
        self._assets_dir = _required_directory(init_params, "realworld_assets_dir")
        self._tactile_backend = tactile_backend
        self._policy_gripper_sign_bridge = policy_gripper_sign_bridge
        self._target_object = target_object
        self._reset_source = reset_source
        self._required_success_steps = required_success_steps
        self._terminal_reward = terminal_reward
        self._task_description, self._success_goals = _load_task_contract(
            self._config_dir,
            target_object=target_object,
            task_description=configured_task_description,
        )
        self._marker_history = _RealWorldMarkerHistory(num_envs)

        with open_dict(cfg):
            cfg.init_params.task_description = self._task_description

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
            os.environ["REALWORLD_TASK_SUITE"] = REALWORLD_TASK_SUITE
            os.environ["REALWORLD_TASK_ID"] = str(REALWORLD_TASK_ID)
            os.environ["TASK_SUITE"] = REALWORLD_TASK_SUITE
            os.environ["TASK_ID"] = str(REALWORLD_TASK_ID)
            os.environ["REALWORLD_CONFIG_DIR"] = str(self._config_dir)
            os.environ["REALWORLD_ASSETS_DATA_DIR"] = str(self._assets_dir)
            os.environ["TACTILE_BACKEND"] = self._tactile_backend
            _prepend_python_path(self._extension_path)

            from isaaclab.app import AppLauncher

            sim_app = AppLauncher(headless=True, enable_cameras=True).app
            try:
                import tac_manip
                from isaaclab.managers import ManagerTermBase
                from isaaclab.managers import ObservationTermCfg as ObsTerm
                from isaaclab.managers import RewardTermCfg as RewTerm
                from isaaclab_tasks.utils import load_cfg_from_registry

                _validate_extension_import(
                    tac_manip.__file__,
                    self._extension_path,
                )
                isaac_env_cfg = load_cfg_from_registry(
                    self.isaaclab_env_id, "env_cfg_entry_point"
                )
                if not hasattr(isaac_env_cfg, "policy_gripper_sign_bridge"):
                    raise ValueError(
                        "Tabero_X RealWorld EnvCfg does not expose "
                        "policy_gripper_sign_bridge."
                    )
                isaac_env_cfg.policy_gripper_sign_bridge = (
                    self._policy_gripper_sign_bridge
                )
                _validate_hybrid_action_cfg(
                    isaac_env_cfg,
                    require_policy_gripper_sign_bridge=(
                        self._policy_gripper_sign_bridge
                    ),
                )
                success_term_cfg = getattr(
                    getattr(isaac_env_cfg, "terminations", None),
                    "success",
                    None,
                )
                if success_term_cfg is None:
                    raise ValueError(
                        "RealWorld Task 6 environment has no success termination term."
                    )
                raw_success_func = success_term_cfg.func
                raw_success_params = dict(success_term_cfg.params)
                if (
                    getattr(raw_success_func, "__name__", None)
                    != "realworld_goals_reached"
                ):
                    raise ValueError(
                        "RealWorld Task 6 success must use realworld_goals_reached."
                    )
                if not raw_success_params.get("goals"):
                    raise ValueError(
                        "RealWorld Task 6 success must contain non-empty task goals."
                    )
                raw_success_params["goals"] = _clone_nested(self._success_goals)
                success_term_cfg.func = _make_consecutive_success_term(ManagerTermBase)
                success_term_cfg.params = {
                    "success_func": raw_success_func,
                    "success_params": raw_success_params,
                    "required_steps": self._required_success_steps,
                }
                isaac_env_cfg.rewards = {
                    "success": RewTerm(
                        func=_terminal_success_reward,
                        weight=float(self.cfg.reward_coef) * self._terminal_reward,
                        params={"success_term_name": "success"},
                    )
                }
                isaac_env_cfg.seed = self.seed
                isaac_env_cfg.scene.num_envs = self.cfg.init_params.num_envs
                isaac_env_cfg.episode_length_s = (
                    int(self.cfg.max_episode_steps)
                    * float(isaac_env_cfg.sim.dt)
                    * int(isaac_env_cfg.decimation)
                )

                _validate_camera_source_cfg(isaac_env_cfg.scene, "agentview_cam")
                _validate_camera_source_cfg(isaac_env_cfg.scene, "eye_in_hand_cam")
                isaac_env_cfg.observations.policy.agentview_rgb = ObsTerm(
                    func=_camera_rgb_observation,
                    params={"camera_name": "agentview_cam"},
                )
                isaac_env_cfg.observations.policy.eye_in_hand_rgb = ObsTerm(
                    func=_camera_rgb_observation,
                    params={"camera_name": "eye_in_hand_cam"},
                )

                env = gym.make(
                    self.isaaclab_env_id,
                    cfg=isaac_env_cfg,
                    render_mode="rgb_array",
                ).unwrapped
                action_dim = int(env.action_manager.total_action_dim)
                if action_dim != REALWORLD_ACTION_DIM:
                    env.close()
                    raise ValueError(
                        f"RealWorld Hybrid action dimension must be 13; got {action_dim}."
                    )
                actual_episode_steps = int(env.max_episode_length)
                expected_episode_steps = int(self.cfg.max_episode_steps)
                if actual_episode_steps != expected_episode_steps:
                    env.close()
                    raise RuntimeError(
                        "RealWorld IsaacLab episode horizon mismatch: "
                        f"expected {expected_episode_steps}, got {actual_episode_steps}."
                    )
                return _TerminalObservationCapture(env), sim_app
            except BaseException:
                traceback.print_exc()
                sys.stderr.flush()
                sim_app.close()
                raise

        return make_env_isaaclab

    def reset(self, seed=None, env_ids: torch.Tensor | None = None):
        self._marker_history.reset(env_ids)
        update_mask = None
        target_env_ids = None
        if env_ids is not None:
            target_env_ids = torch.as_tensor(
                env_ids, device=self.device, dtype=torch.long
            )
            update_mask = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.bool
            )
            update_mask[target_env_ids] = True

        if target_env_ids is None:
            raw_obs, _ = self.env.reset(seed=seed)
        else:
            raw_obs, _ = self.env.reset(seed=seed, env_ids=target_env_ids)
        obs = self._wrap_obs(raw_obs, marker_update_mask=update_mask)
        self._action_filter.reset(obs["states"], env_ids=target_env_ids)
        self._reset_metrics(target_env_ids)
        return obs, {}

    def step(self, actions=None, auto_reset=True):
        actions = self._validate_actions(actions, expected_rank=2)
        active_mask = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        obs, reward, terminations, truncations, infos = self._terminal_safe_step(
            actions, active_mask=active_mask
        )
        dones = terminations | truncations
        if dones.any():
            env_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
            reset_obs, _ = self.reset(env_ids=env_ids)
            obs = _replace_batch_rows(obs, reset_obs, dones)
        return obs, reward, terminations, truncations, infos

    def chunk_step(self, chunk_actions: torch.Tensor):
        raw_chunk_actions = self._validate_actions(
            chunk_actions, expected_rank=3
        ).clone()
        chunk_actions = self._action_filter.filter(raw_chunk_actions)
        active_mask = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        first_done_step = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )
        latest_hold_state: torch.Tensor | None = None
        post_done_hold_steps = 0
        terminal_observation_captures = 0
        obs_list = []
        infos_list = []
        rewards = []
        terminations = []
        truncations = []
        executed_actions = []

        for step_index in range(REALWORLD_ACTION_HORIZON):
            actions = chunk_actions[:, step_index].clone()
            inactive_mask = ~active_mask
            if inactive_mask.any():
                if latest_hold_state is None:
                    raise RuntimeError(
                        "RealWorld terminal-safe chunk has no reset state for hold."
                    )
                hold_actions = self._build_hold_actions(actions, latest_hold_state)
                action_mask = inactive_mask.to(device=actions.device)
                actions[action_mask] = hold_actions[action_mask]
                post_done_hold_steps += int(inactive_mask.sum().item())

            obs, reward, step_terminations, step_truncations, infos = (
                self._terminal_safe_step(actions, active_mask=active_mask)
            )
            latest_hold_state = infos.pop("_realworld_hold_state")
            terminal_capture = infos.pop("_realworld_terminal_capture")
            newly_done = (step_terminations | step_truncations) & active_mask
            if newly_done.any():
                first_done_step[newly_done] = step_index
                terminal_observation_captures += int(
                    terminal_capture[newly_done].sum().item()
                )
                active_mask &= ~newly_done

            obs_list.append(obs)
            infos_list.append(infos)
            rewards.append(reward)
            terminations.append(step_terminations)
            truncations.append(step_truncations)
            executed_actions.append(actions)

        chunk_rewards = torch.stack(rewards, dim=1)
        chunk_terminations = torch.stack(terminations, dim=1)
        chunk_truncations = torch.stack(truncations, dim=1)
        past_dones = first_done_step >= 0
        standard_reset_envs = int(past_dones.sum().item())
        if past_dones.any():
            env_ids = torch.nonzero(past_dones, as_tuple=False).squeeze(-1)
            reset_obs, _ = self.reset(env_ids=env_ids)
            obs_list[-1] = _replace_batch_rows(obs_list[-1], reset_obs, past_dones)

        early_done_envs = int(
            ((first_done_step >= 0) & (first_done_step < REALWORLD_ACTION_HORIZON - 1))
            .sum()
            .item()
        )
        infos_list[-1]["chunk_boundary_metrics"] = {
            "done_envs": torch.tensor([standard_reset_envs], device=self.device),
            "early_done_envs": torch.tensor([early_done_envs], device=self.device),
            "post_done_hold_steps": torch.tensor(
                [post_done_hold_steps], device=self.device
            ),
            "post_done_policy_actions": torch.zeros(
                1, device=self.device, dtype=torch.long
            ),
            "terminal_observation_captures": torch.tensor(
                [terminal_observation_captures], device=self.device
            ),
            "standard_reset_envs": torch.tensor(
                [standard_reset_envs], device=self.device
            ),
        }
        infos_list[-1][_EXECUTED_CHUNK_ACTIONS_KEY] = torch.stack(
            executed_actions, dim=1
        )
        infos_list[-1][_RAW_CHUNK_ACTIONS_KEY] = raw_chunk_actions
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _terminal_safe_step(
        self,
        actions: torch.Tensor,
        *,
        active_mask: torch.Tensor,
    ):
        active_mask = torch.as_tensor(active_mask, device=self.device, dtype=torch.bool)
        if active_mask.shape != (self.num_envs,):
            raise ValueError(
                "RealWorld active mask must have shape "
                f"({self.num_envs},); got {tuple(active_mask.shape)}."
            )

        raw_obs, reward, raw_terminations, raw_truncations, raw_infos = self.env.step(
            actions
        )
        reward = reward.clone()
        raw_terminations = raw_terminations.clone().to(dtype=torch.bool)
        raw_truncations = raw_truncations.clone().to(dtype=torch.bool)
        raw_infos = dict(raw_infos or {})
        captured_raw_obs = raw_infos.pop(_TERMINAL_RAW_OBSERVATION_KEY, None)
        captured_mask = torch.as_tensor(
            raw_infos.pop(
                _TERMINAL_OBSERVATION_MASK_KEY,
                torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
            ),
            device=self.device,
            dtype=torch.bool,
        )
        if captured_mask.shape != (self.num_envs,):
            raise RuntimeError(
                "RealWorld terminal observation mask has invalid shape "
                f"{tuple(captured_mask.shape)}."
            )

        internal_done = active_mask & (raw_terminations | raw_truncations)
        if internal_done.any() and (
            captured_raw_obs is None or not torch.all(captured_mask[internal_done])
        ):
            raise RuntimeError(
                "RealWorld terminal-safe step is missing the pre-reset IsaacLab frame."
            )

        terminal_source = raw_obs
        terminal_capture_mask = captured_mask & active_mask
        if captured_raw_obs is not None and terminal_capture_mask.any():
            terminal_source = _replace_batch_rows(
                raw_obs, captured_raw_obs, terminal_capture_mask
            )
        obs = self._wrap_obs(
            terminal_source,
            marker_update_mask=active_mask,
        )

        self._elapsed_steps[active_mask] += 1
        expected_time_outs = active_mask & (
            self._elapsed_steps >= int(self.cfg.max_episode_steps)
        )
        if (expected_time_outs & ~raw_truncations).any():
            raise RuntimeError(
                "RealWorld wrapper and IsaacLab episode horizons are misaligned."
            )

        terminations = raw_terminations & active_mask
        truncations = raw_truncations & active_mask
        newly_done = terminations | truncations
        reward = torch.where(active_mask, reward, 0.0)
        infos = self._record_metrics(reward, terminations, {})
        if newly_done.any():
            infos["final_observation"] = _clone_nested(obs)
            infos["final_info"] = {"episode": _clone_nested(infos["episode"])}
            infos["_final_info"] = newly_done.clone()
            infos["_final_observation"] = newly_done.clone()
            infos["_elapsed_steps"] = newly_done.clone()

        infos["_realworld_hold_state"] = _build_state(raw_obs["policy"])
        infos["_realworld_terminal_capture"] = terminal_capture_mask.clone()
        return obs, reward, terminations, truncations, infos

    @staticmethod
    def _build_hold_actions(
        action_template: torch.Tensor,
        hold_state: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(action_template.shape[1:]) != (REALWORLD_ACTION_DIM,):
            raise ValueError(
                "RealWorld hold action template must have shape (N, 13); "
                f"got {tuple(action_template.shape)}."
            )
        if tuple(hold_state.shape[1:]) != (7,):
            raise ValueError(
                "RealWorld hold state must have shape (N, 7); "
                f"got {tuple(hold_state.shape)}."
            )
        hold_actions = torch.zeros_like(action_template)
        hold_actions[:, :7] = hold_state.to(
            device=action_template.device,
            dtype=action_template.dtype,
        )
        return hold_actions

    def _validate_actions(
        self,
        actions: Any,
        *,
        expected_rank: int,
    ) -> torch.Tensor:
        if actions is None:
            raise ValueError("RealWorld Hybrid control requires an action tensor.")
        actions = torch.as_tensor(actions)
        expected_shape = (
            (self.num_envs, REALWORLD_ACTION_DIM)
            if expected_rank == 2
            else (
                self.num_envs,
                REALWORLD_ACTION_HORIZON,
                REALWORLD_ACTION_DIM,
            )
        )
        if tuple(actions.shape) != expected_shape:
            raise ValueError(
                f"RealWorld Hybrid action must have shape {expected_shape}; "
                f"got {tuple(actions.shape)}."
            )
        if not actions.is_floating_point() or not torch.isfinite(actions).all():
            raise ValueError("RealWorld Hybrid actions must be finite floating point.")
        return actions

    def _wrap_obs(
        self,
        obs: dict[str, Any],
        marker_update_mask: torch.Tensor | None = None,
    ):
        if "policy" not in obs or not isinstance(obs["policy"], dict):
            raise KeyError("RealWorld IsaacLab observation has no policy group.")
        policy_obs = obs["policy"]
        required_keys = {
            "agentview_rgb",
            "eye_in_hand_rgb",
            "eef_pose",
            "gripper_pos",
            "gripper_marker_motion",
        }
        missing = sorted(required_keys.difference(policy_obs))
        if missing:
            raise KeyError(f"RealWorld policy observation is missing keys: {missing}.")

        main_image = policy_obs["agentview_rgb"]
        wrist_image = policy_obs["eye_in_hand_rgb"]
        for name, image in (
            ("agentview_rgb", main_image),
            ("eye_in_hand_rgb", wrist_image),
        ):
            if (
                image.ndim != 4
                or image.shape[0] != self.num_envs
                or tuple(image.shape[1:3]) != REALWORLD_CAMERA_SOURCE_HW
                or image.shape[-1] != 3
            ):
                raise ValueError(
                    "RealWorld camera parity requires "
                    f"{name} shape (N, {REALWORLD_CAMERA_SOURCE_HW[0]}, "
                    f"{REALWORLD_CAMERA_SOURCE_HW[1]}, 3); "
                    f"got {tuple(image.shape)}."
                )

        env_obs = {
            "main_images": main_image,
            "wrist_images": wrist_image,
            "states": _build_state(policy_obs),
            "task_descriptions": [self._task_description] * self.num_envs,
            "tactile_marker_motion": self._marker_history.update(
                policy_obs["gripper_marker_motion"],
                update_mask=marker_update_mask,
            ),
        }
        force = policy_obs.get("gripper_net_force")
        if force is not None:
            expected_force_shape = (self.num_envs, 1, 2, 3)
            if (
                tuple(force.shape) != expected_force_shape
                or not torch.isfinite(force).all()
            ):
                raise ValueError(
                    "RealWorld gripper_net_force must be finite with shape "
                    f"{expected_force_shape}; got {tuple(force.shape)}."
                )
            env_obs["tactile_gripper_force"] = force
        return env_obs
