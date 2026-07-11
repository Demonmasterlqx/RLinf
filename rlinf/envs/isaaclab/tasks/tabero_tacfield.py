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

import hashlib
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
from omegaconf import open_dict

from rlinf.envs.isaaclab.utils import quat2axisangle_torch

from ..isaaclab_env import IsaaclabBaseEnv

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    return getattr(cfg, name, default) if cfg is not None else default


@dataclass(frozen=True)
class TaberoTaskSpec:
    task_suite: str
    task_id: int
    task_description: str


def _plain_container(value: Any) -> Any:
    if hasattr(value, "items"):
        return {key: _plain_container(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_container(item) for item in value]
    return value


def _load_libero_instruction(
    libero_config_dir: str | None,
    task_suite: str,
    task_id: int,
) -> str | None:
    if not libero_config_dir:
        return None

    config_path = Path(str(libero_config_dir)).expanduser() / f"{task_suite}.json"
    if not config_path.exists():
        return None

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    for task in config.get("tasks", []):
        if int(task.get("task_id", -1)) == int(task_id):
            instruction = task.get("language_instruction")
            return str(instruction) if instruction is not None else None
    return None


def _build_tabero_task(
    raw_task: dict[str, Any],
    libero_config_dir: str | None,
    fallback_description: str | None = None,
) -> TaberoTaskSpec:
    task_suite = str(raw_task.get("task_suite", raw_task.get("suite", "libero_10")))
    task_id = int(raw_task.get("task_id", raw_task.get("id", 0)))
    task_description = raw_task.get("task_description")
    if task_description is None:
        task_description = raw_task.get("language_instruction")
    if task_description is None:
        task_description = _load_libero_instruction(
            libero_config_dir, task_suite, task_id
        )
    if task_description is None:
        task_description = fallback_description
    if task_description is None:
        task_description = f"Task {task_id}"

    return TaberoTaskSpec(
        task_suite=task_suite,
        task_id=task_id,
        task_description=str(task_description),
    )


def _load_tabero_subset_tasks(
    subset_path: str | os.PathLike[str],
) -> list[dict[str, Any]]:
    path = Path(str(subset_path)).expanduser()
    with path.open("r", encoding="utf-8") as file:
        subset = json.load(file)

    raw_tasks: list[dict[str, Any]] = []
    if not isinstance(subset, dict):
        raise ValueError(f"Tabero task subset must be a JSON object: {path}")
    for task_suite, task_ids in subset.items():
        if task_ids is None:
            continue
        for task_id in task_ids:
            raw_tasks.append({"task_suite": str(task_suite), "task_id": int(task_id)})
    return raw_tasks


def resolve_tabero_tasks(init_params: Any) -> list[TaberoTaskSpec]:
    """Resolve single-task or multi-task Tabero config into concrete task specs."""

    libero_config_dir = _cfg_get(init_params, "libero_config_dir")
    fallback_description = _cfg_get(init_params, "task_description")
    explicit_tasks = _cfg_get(init_params, "tasks", None)
    subset_path = _cfg_get(init_params, "tabero_task_subset_path", None)

    if explicit_tasks is not None:
        raw_tasks = _plain_container(explicit_tasks)
        if len(raw_tasks) == 0:
            raise ValueError("Tabero multitask config must contain at least one task.")
    elif subset_path is not None:
        raw_tasks = _load_tabero_subset_tasks(subset_path)
        if len(raw_tasks) == 0:
            raise ValueError("Tabero task subset must contain at least one task.")
    else:
        raw_tasks = [
            {
                "task_suite": _cfg_get(init_params, "task_suite", "libero_10"),
                "task_id": _cfg_get(init_params, "task_id", 0),
                "task_description": fallback_description,
            }
        ]

    return [
        _build_tabero_task(
            raw_task,
            libero_config_dir=str(libero_config_dir) if libero_config_dir else None,
            fallback_description=str(fallback_description)
            if fallback_description is not None
            else None,
        )
        for raw_task in raw_tasks
    ]


def assign_tabero_task(tasks: list[TaberoTaskSpec], seed_offset: int) -> TaberoTaskSpec:
    if len(tasks) == 0:
        raise ValueError("Tabero multitask config must contain at least one task.")
    return tasks[int(seed_offset) % len(tasks)]


def ensure_tabero_root_task_description(cfg: Any, task: TaberoTaskSpec) -> None:
    if _cfg_get(cfg.init_params, "task_description", None) is not None:
        return

    try:
        with open_dict(cfg):
            cfg.init_params.task_description = task.task_description
    except Exception:
        cfg.init_params.task_description = task.task_description


def validate_tabero_task_assignment(
    tasks: list[TaberoTaskSpec],
    total_num_processes: int,
    require_all_tasks_active: bool = False,
) -> None:
    if len(tasks) <= int(total_num_processes):
        return

    message = (
        f"Tabero multitask config has {len(tasks)} tasks but only "
        f"{total_num_processes} logical env processes; only the first "
        f"{total_num_processes} task(s) are active in this run."
    )
    if require_all_tasks_active:
        raise ValueError(message)
    warnings.warn(message, stacklevel=2)


def _camera_rgb_observation(env, camera_name: str) -> torch.Tensor:
    camera = env.scene[camera_name]
    return camera.data.output["rgb"]


class ConsecutiveSuccessTracker:
    """Track independent consecutive-success streaks for vector environments."""

    def __init__(
        self,
        num_envs: int,
        required_steps: int = 1,
        device: torch.device | str = "cpu",
    ) -> None:
        if required_steps <= 0:
            raise ValueError("required_steps must be positive.")
        self.required_steps = int(required_steps)
        self.streak = torch.zeros(
            int(num_envs), dtype=torch.int64, device=torch.device(device)
        )

    def update(self, raw_success: torch.Tensor) -> torch.Tensor:
        raw_success = raw_success.to(device=self.streak.device, dtype=torch.bool)
        if raw_success.shape != self.streak.shape:
            raise ValueError(
                "raw_success must have shape "
                f"{tuple(self.streak.shape)}, got {tuple(raw_success.shape)}."
            )
        self.streak = torch.where(raw_success, self.streak + 1, 0)
        return self.streak >= self.required_steps

    def reset(self, env_ids: torch.Tensor | list[int] | slice | None = None) -> None:
        if env_ids is None:
            self.streak.zero_()
            return
        if isinstance(env_ids, slice):
            self.streak[env_ids] = 0
            return
        env_ids = torch.as_tensor(env_ids, device=self.streak.device, dtype=torch.long)
        self.streak[env_ids] = 0


def _make_consecutive_success_term(manager_term_base_cls: type) -> type:
    class ConsecutiveSuccessTerm(manager_term_base_cls):
        def __init__(self, cfg, env) -> None:
            super().__init__(cfg, env)
            required_steps = int(cfg.params.get("required_steps", 1))
            self._tracker = ConsecutiveSuccessTracker(
                num_envs=env.num_envs,
                required_steps=required_steps,
                device=env.device,
            )

        def reset(self, env_ids=None) -> None:
            self._tracker.reset(env_ids)

        def __call__(
            self,
            env,
            success_func: Any,
            success_params: dict[str, Any] | None = None,
            required_steps: int = 1,
        ) -> torch.Tensor:
            if int(required_steps) != self._tracker.required_steps:
                raise ValueError(
                    "Consecutive success required_steps changed after initialization."
                )
            raw_success = success_func(env, **(success_params or {}))
            return self._tracker.update(raw_success)

    ConsecutiveSuccessTerm.__name__ = "ConsecutiveSuccessTerm"
    return ConsecutiveSuccessTerm


def _stable_success_terminal_reward(
    env,
    success_term_name: str = "success",
    failure_term_names: tuple[str, ...] = (),
    env_reward_multipliers: tuple[float, ...] | None = None,
) -> torch.Tensor:
    # IsaacLab computes this before its internal auto-reset, then resets both
    # manager terms for the finished env ids, so this emits once per episode.
    success = env.termination_manager.get_term(success_term_name).to(dtype=torch.bool)
    invalid = env.termination_manager.time_outs.to(dtype=torch.bool).clone()
    for term_name in failure_term_names:
        invalid |= env.termination_manager.get_term(term_name).to(dtype=torch.bool)
    reward = (success & ~invalid).to(dtype=torch.float32) / float(env.step_dt)
    if env_reward_multipliers is not None:
        multipliers = torch.as_tensor(
            env_reward_multipliers,
            dtype=reward.dtype,
            device=reward.device,
        )
        if multipliers.shape != reward.shape:
            raise ValueError(
                "env_reward_multipliers must match the vector env reward shape; "
                f"got {tuple(multipliers.shape)} for {tuple(reward.shape)}."
            )
        reward = reward * multipliers
    return reward


def _termination_term_items(terminations_cfg: Any) -> list[tuple[str, Any]]:
    terms: list[tuple[str, Any]] = []
    for name in dir(terminations_cfg):
        if name.startswith("_"):
            continue
        term = getattr(terminations_cfg, name, None)
        if term is not None and hasattr(term, "func"):
            terms.append((name, term))
    return terms


def _install_success_reward(
    isaac_env_cfg: Any,
    reward_term_cls: Any,
    reward_coef: float,
    manager_term_base_cls: type,
    required_steps: int = 1,
    env_reward_multipliers: tuple[float, ...] | None = None,
) -> None:
    terminations_cfg = getattr(isaac_env_cfg, "terminations", None)
    success_term = getattr(terminations_cfg, "success", None)
    success_func = getattr(success_term, "func", None)
    if success_func is None:
        return

    success_params = dict(getattr(success_term, "params", {}) or {})
    success_term.func = _make_consecutive_success_term(manager_term_base_cls)
    success_term.params = {
        "success_func": success_func,
        "success_params": success_params,
        "required_steps": int(required_steps),
    }
    failure_term_names = tuple(
        name
        for name, term in _termination_term_items(terminations_cfg)
        if name != "success" and not bool(getattr(term, "time_out", False))
    )
    reward_params = {
        "success_term_name": "success",
        "failure_term_names": failure_term_names,
    }
    if env_reward_multipliers is not None:
        reward_params["env_reward_multipliers"] = tuple(env_reward_multipliers)
    success_reward = reward_term_cls(
        func=_stable_success_terminal_reward,
        weight=float(reward_coef),
        params=reward_params,
    )
    rewards_cfg = getattr(isaac_env_cfg, "rewards", None)
    if rewards_cfg is None:
        isaac_env_cfg.rewards = {"success": success_reward}
    elif isinstance(rewards_cfg, dict):
        rewards_cfg["success"] = success_reward
    else:
        setattr(rewards_cfg, "success", success_reward)


def _choose_prompt_option(seed: int, key: str, options: list[str]) -> str:
    if not options:
        raise ValueError("Prompt adverb lists must not be empty.")
    digest = hashlib.blake2b(
        f"{int(seed)}:{key}".encode("utf-8"), digest_size=8
    ).digest()
    base_index = int.from_bytes(digest, "big") % len(options)
    return str(options[base_index])


def _rewrite_tabero_instruction(
    instruction: str,
    adverb: str,
    seed: int,
    key: str,
) -> str:
    instruction = instruction.strip()
    adverb = adverb.strip()
    if not adverb:
        return instruction
    if not instruction:
        return adverb

    lower = instruction.lower()
    if lower.startswith(f"{adverb.lower()} ") or lower.endswith(f" {adverb.lower()}"):
        return instruction
    style = _choose_prompt_option(seed, f"{key}:style", ["prefix", "suffix"])
    if style == "suffix":
        return f"{instruction} {adverb}"
    return f"{adverb} {instruction}"


def build_tabero_conditioned_prompts(
    instruction: str,
    task_suite: str,
    task_id: int,
    num_envs: int,
    prompt_cfg: Any,
    rollout_round: int,
) -> tuple[list[str], list[int]]:
    if not bool(_cfg_get(prompt_cfg, "enabled", False)):
        return [instruction] * int(num_envs), []
    assignment = str(_cfg_get(prompt_cfg, "assignment", "paired"))
    if assignment != "paired":
        raise ValueError(
            f"Unsupported Tabero prompt assignment {assignment!r}; expected 'paired'."
        )
    if int(num_envs) % 2 != 0:
        raise ValueError(
            "Paired firm/gentle prompts require an even number of vector envs."
        )

    firm_adverbs = [str(x) for x in _cfg_get(prompt_cfg, "firm_adverbs", [])]
    gentle_adverbs = [str(x) for x in _cfg_get(prompt_cfg, "gentle_adverbs", [])]
    prompt_seed = int(_cfg_get(prompt_cfg, "prompt_seed", 0))
    condition_ids = [env_id % 2 for env_id in range(int(num_envs))]
    prompts: list[str] = []
    for env_id, condition_id in enumerate(condition_ids):
        condition = "firm" if condition_id == 0 else "gentle"
        adverbs = firm_adverbs if condition_id == 0 else gentle_adverbs
        base_key = f"{task_suite}:{int(task_id)}:{env_id}:{condition}"
        hashed_adverb = _choose_prompt_option(prompt_seed, base_key, adverbs)
        base_index = adverbs.index(hashed_adverb)
        adverb = adverbs[(base_index + int(rollout_round)) % len(adverbs)]
        prompts.append(
            _rewrite_tabero_instruction(
                instruction,
                adverb,
                seed=prompt_seed,
                key=f"{base_key}:{int(rollout_round)}",
            )
        )
    return prompts, condition_ids


def build_condition_reward_multipliers(
    condition_ids: list[int] | tuple[int, ...],
    success_cfg: Any,
) -> tuple[float, ...] | None:
    multiplier_cfg = _cfg_get(success_cfg, "condition_reward_multipliers", None)
    if not condition_ids or multiplier_cfg is None:
        return None

    firm = float(_cfg_get(multiplier_cfg, "firm", 1.0))
    gentle = float(_cfg_get(multiplier_cfg, "gentle", 1.0))
    if firm <= 0 or gentle <= 0:
        raise ValueError("Condition reward multipliers must be positive.")
    return tuple(firm if int(condition_id) == 0 else gentle for condition_id in condition_ids)


def compute_tabero_predicted_squeeze(actions: torch.Tensor) -> torch.Tensor:
    if actions.ndim != 2 or actions.shape[-1] < 13:
        raise ValueError(
            "Tabero predicted squeeze expects actions with shape (N, >=13); "
            f"got {tuple(actions.shape)}."
        )
    return 2.0 * torch.minimum(actions[:, 9].abs(), actions[:, 12].abs())


def _broadcast_condition_mean(
    values: torch.Tensor,
    condition_mask: torch.Tensor,
    output_like: torch.Tensor,
) -> torch.Tensor:
    if not condition_mask.any():
        return torch.full_like(output_like, float("nan"), dtype=torch.float32)
    mean = values[condition_mask].to(dtype=torch.float32).mean()
    return torch.full_like(output_like, mean.item(), dtype=torch.float32)


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
        raise KeyError(
            "Tabero TacField env expects 'eef_pose' or 'eef_pos'/'eef_quat'."
        )

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
        self._success_cfg = _cfg_get(init_params, "success", None)
        self._prompt_cfg = _cfg_get(init_params, "prompt_conditions", None)
        self._tabero_tasks = resolve_tabero_tasks(init_params)
        validate_tabero_task_assignment(
            self._tabero_tasks,
            total_num_processes=total_num_processes,
            require_all_tasks_active=bool(
                _cfg_get(init_params, "require_all_tasks_active", False)
            ),
        )
        self._tabero_task = assign_tabero_task(self._tabero_tasks, seed_offset)
        self._tabero_task_shard_id = int(seed_offset)
        ensure_tabero_root_task_description(cfg, self._tabero_task)
        history_len = int(_cfg_get(init_params, "marker_history_len", 8))
        expected_markers = int(_cfg_get(init_params, "combined_marker_count", 198))
        self._marker_history = TacManipMarkerMotionHistory(
            num_envs=num_envs,
            history_len=history_len,
            expected_combined_markers=expected_markers,
        )
        self._prompt_rollout_round = -1
        self._conditioned_prompts, condition_ids = build_tabero_conditioned_prompts(
            instruction=self._tabero_task.task_description,
            task_suite=self._tabero_task.task_suite,
            task_id=self._tabero_task.task_id,
            num_envs=num_envs,
            prompt_cfg=self._prompt_cfg,
            rollout_round=0,
        )
        self._prompt_condition_ids = tuple(condition_ids)
        logger.info(
            "Assigned Tabero task shard=%d/%d suite=%s task_id=%d num_envs=%d",
            self._tabero_task_shard_id,
            int(total_num_processes),
            self._tabero_task.task_suite,
            self._tabero_task.task_id,
            int(num_envs),
        )

        super().__init__(
            cfg,
            num_envs,
            seed_offset,
            total_num_processes,
            worker_info,
        )
        self.task_description = self._tabero_task.task_description
        self._condition_squeeze_sum = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._condition_squeeze_count = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

    def _make_env_function(self):
        def make_env_isaaclab():
            os.environ.pop("DISPLAY", None)

            init_params = self.cfg.init_params
            _prepend_python_path(_cfg_get(init_params, "extension_path"))

            task_suite = self._tabero_task.task_suite
            task_id = str(self._tabero_task.task_id)
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
            from isaaclab.managers import ManagerTermBase
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
                _cfg_get(
                    init_params, "agentview_cam", _cfg_get(init_params, "table_cam")
                ),
            )
            _set_camera_resolution(
                isaac_env_cfg.scene,
                "eye_in_hand_cam",
                _cfg_get(
                    init_params, "eye_in_hand_cam", _cfg_get(init_params, "wrist_cam")
                ),
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
                float(self.cfg.reward_coef)
                * float(_cfg_get(self._success_cfg, "terminal_reward", 1.0)),
                ManagerTermBase,
                required_steps=int(
                    _cfg_get(self._success_cfg, "required_consecutive_steps", 1)
                ),
                env_reward_multipliers=build_condition_reward_multipliers(
                    self._prompt_condition_ids,
                    self._success_cfg,
                ),
            )

            env = gym.make(
                self.isaaclab_env_id, cfg=isaac_env_cfg, render_mode="rgb_array"
            ).unwrapped
            return env, sim_app

        return make_env_isaaclab

    def reset(self, seed=None, env_ids: torch.Tensor | None = None):
        self._marker_history.reset(env_ids)
        if getattr(self, "_prompt_condition_ids", ()):
            self._prompt_rollout_round += 1
            self._conditioned_prompts, condition_ids = build_tabero_conditioned_prompts(
                instruction=self._tabero_task.task_description,
                task_suite=self._tabero_task.task_suite,
                task_id=self._tabero_task.task_id,
                num_envs=self.num_envs,
                prompt_cfg=self._prompt_cfg,
                rollout_round=self._prompt_rollout_round,
            )
            self._prompt_condition_ids = tuple(condition_ids)
            logger.info(
                "Tabero prompts shard=%d round=%d prompts=%s",
                self._tabero_task_shard_id,
                self._prompt_rollout_round,
                self._conditioned_prompts,
            )
        if hasattr(self, "_condition_squeeze_sum"):
            if env_ids is None:
                self._condition_squeeze_sum.zero_()
                self._condition_squeeze_count.zero_()
            else:
                env_ids = env_ids.to(device=self.device, dtype=torch.long)
                self._condition_squeeze_sum[env_ids] = 0
                self._condition_squeeze_count[env_ids] = 0
        return super().reset(seed=seed, env_ids=env_ids)

    def step(self, actions=None, auto_reset=True):
        if getattr(self, "_prompt_condition_ids", ()) and actions is not None:
            actions_tensor = torch.as_tensor(actions, device=self.device)
            squeeze = compute_tabero_predicted_squeeze(actions_tensor)
            self._condition_squeeze_sum += squeeze.to(dtype=torch.float32)
            self._condition_squeeze_count += 1
        return super().step(actions=actions, auto_reset=auto_reset)

    def _record_metrics(self, step_reward, terminations, infos):
        infos = super()._record_metrics(step_reward, terminations, infos)
        episode_info = infos["episode"]
        episode_info["task_id"] = torch.full(
            (self.num_envs,),
            float(self._tabero_task.task_id),
            dtype=torch.float32,
            device=step_reward.device,
        )
        episode_info["task_shard_id"] = torch.full(
            (self.num_envs,),
            float(self._tabero_task_shard_id),
            dtype=torch.float32,
            device=step_reward.device,
        )
        if getattr(self, "_prompt_condition_ids", ()):
            condition_ids = torch.tensor(
                self._prompt_condition_ids,
                dtype=torch.int64,
                device=step_reward.device,
            )
            firm_mask = condition_ids == 0
            gentle_mask = condition_ids == 1
            squeeze_mean = (
                self._condition_squeeze_sum / self._condition_squeeze_count.clamp(min=1)
            )
            episode_info["firm_success_once"] = _broadcast_condition_mean(
                self.success_once.to(dtype=torch.float32), firm_mask, step_reward
            )
            episode_info["gentle_success_once"] = _broadcast_condition_mean(
                self.success_once.to(dtype=torch.float32), gentle_mask, step_reward
            )
            episode_info["firm_return"] = _broadcast_condition_mean(
                self.returns, firm_mask, step_reward
            )
            episode_info["gentle_return"] = _broadcast_condition_mean(
                self.returns, gentle_mask, step_reward
            )
            episode_info["firm_squeeze_pred_mean"] = _broadcast_condition_mean(
                squeeze_mean, firm_mask, step_reward
            )
            episode_info["gentle_squeeze_pred_mean"] = _broadcast_condition_mean(
                squeeze_mean, gentle_mask, step_reward
            )
        return infos

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
            "task_descriptions": (
                list(self._conditioned_prompts)
                if getattr(self, "_prompt_condition_ids", ())
                else [self.task_description] * self.num_envs
            ),
            "tactile_marker_motion": tactile_marker_motion,
        }
        if self._force_key is not None and self._force_key in policy_obs:
            env_obs["tactile_gripper_force"] = policy_obs[self._force_key]
        return env_obs
