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

_TERMINAL_RAW_OBSERVATION_KEY = "tabero_terminal_raw_observation"
_TERMINAL_OBSERVATION_MASK_KEY = "tabero_terminal_observation_mask"
_LEGACY_CHUNK_BOUNDARY_MODE = "legacy"
_TERMINAL_SAFE_HDF5_MODE = "terminal_safe_hdf5_v1"
_VALID_CHUNK_BOUNDARY_MODES = frozenset(
    {_LEGACY_CHUNK_BOUNDARY_MODE, _TERMINAL_SAFE_HDF5_MODE}
)
_CHUNK_EPISODE_RECORDS_KEY = "_tabero_chunk_episode_records"
_EPISODE_CONDITION_ID_KEY = "_tabero_condition_id"
_EPISODE_SQUEEZE_PRED_MEAN_KEY = "_tabero_squeeze_pred_mean"


def validate_tabero_firm_prompts(
    prompts: list[str] | tuple[str, ...],
    condition_ids: list[int] | tuple[int, ...],
) -> None:
    """Require the runtime prompt batch used by formal Tabero DSRL runs."""

    if not prompts or len(prompts) != len(condition_ids):
        raise ValueError(
            "Tabero terminal-safe Firm prompts and condition ids must be non-empty "
            "and batch aligned."
        )
    invalid_condition_ids = [
        index
        for index, condition_id in enumerate(condition_ids)
        if int(condition_id) != 0
    ]
    if invalid_condition_ids:
        raise ValueError(
            "Tabero terminal-safe DSRL accepts only Firm condition id 0; invalid "
            f"prompt rows={invalid_condition_ids}."
        )
    invalid_prompts = [
        index
        for index, prompt in enumerate(prompts)
        if not any(
            adverb in str(prompt).lower().split() for adverb in ("firmly", "tightly")
        )
    ]
    if invalid_prompts:
        raise ValueError(
            "Tabero terminal-safe DSRL prompts must contain 'firmly' or 'tightly'; "
            f"invalid prompt rows={invalid_prompts}."
        )


def validate_tabero_chunk_boundary_mode(mode: Any) -> str:
    """Validate the adapter boundary mode without silently selecting legacy."""

    normalized = str(mode)
    if normalized not in _VALID_CHUNK_BOUNDARY_MODES:
        raise ValueError(
            f"Unsupported Tabero chunk_boundary_mode {normalized!r}; expected one of "
            f"{sorted(_VALID_CHUNK_BOUNDARY_MODES)}."
        )
    return normalized


def _clone_nested_tensors(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_nested_tensors(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_nested_tensors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_nested_tensors(item) for item in value)
    return value


def _replace_batch_rows(base: Any, replacement: Any, mask: torch.Tensor) -> Any:
    """Clone ``base`` and replace batch-aligned rows selected by ``mask``."""

    if isinstance(base, torch.Tensor) and isinstance(replacement, torch.Tensor):
        result = base.clone()
        dst_mask = mask.to(device=result.device, dtype=torch.bool)
        src_mask = mask.to(device=replacement.device, dtype=torch.bool)
        result[dst_mask] = replacement[src_mask].to(device=result.device)
        return result
    if isinstance(base, dict) and isinstance(replacement, dict):
        return {
            key: _replace_batch_rows(base[key], replacement[key], mask)
            if key in replacement
            else _clone_nested_tensors(base[key])
            for key in base
        }
    if isinstance(base, list) and isinstance(replacement, list):
        result = list(base)
        selected = mask.detach().to(device="cpu", dtype=torch.bool).tolist()
        if len(result) == len(selected) and len(replacement) == len(selected):
            for index, should_replace in enumerate(selected):
                if should_replace:
                    result[index] = replacement[index]
        return result
    return _clone_nested_tensors(base)


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    return getattr(cfg, name, default) if cfg is not None else default


def configure_tabero_episode_horizon(init_params: Any, isaac_env_cfg: Any) -> int:
    max_episode_steps = _cfg_get(init_params, "max_episode_steps", None)
    if (
        isinstance(max_episode_steps, bool)
        or not isinstance(max_episode_steps, int)
        or max_episode_steps <= 0
    ):
        raise ValueError(
            "Tabero init_params.max_episode_steps must be a positive integer."
        )
    isaac_env_cfg.episode_length_s = (
        max_episode_steps * float(isaac_env_cfg.sim.dt) * int(isaac_env_cfg.decimation)
    )
    return max_episode_steps


def validate_tabero_episode_horizon(env: Any, expected_steps: int) -> None:
    actual_steps = getattr(env, "max_episode_length", None)
    if actual_steps != expected_steps:
        raise ValueError(
            "Tabero IsaacLab episode horizon mismatch: "
            f"expected {expected_steps} steps, got {actual_steps}."
        )


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


def assign_tabero_reset_episode_names(
    episode_names: list[str],
    *,
    num_envs: int,
    shard_id: int,
    total_shards: int,
    rollout_round: int,
    env_ids: torch.Tensor | list[int] | None = None,
) -> list[str]:
    """Assign deterministic cyclic dataset episodes across env workers."""

    if not episode_names:
        raise ValueError("Tabero HDF5 reset requires at least one episode.")
    if num_envs <= 0 or total_shards <= 0:
        raise ValueError("num_envs and total_shards must be positive.")

    if env_ids is None:
        local_env_ids = list(range(int(num_envs)))
    else:
        local_env_ids = torch.as_tensor(env_ids, dtype=torch.long).cpu().tolist()

    global_round_size = int(num_envs) * int(total_shards)
    global_start = int(rollout_round) * global_round_size + int(shard_id) * int(
        num_envs
    )
    return [
        episode_names[(global_start + int(env_id)) % len(episode_names)]
        for env_id in local_env_ids
    ]


def stack_tabero_initial_states(states: list[Any]) -> Any:
    """Concatenate IsaacLab nested initial-state dictionaries by env batch."""

    if not states:
        raise ValueError("Cannot stack an empty list of Tabero initial states.")
    first = states[0]
    if isinstance(first, dict):
        expected_keys = set(first)
        if any(set(state) != expected_keys for state in states):
            raise ValueError("Tabero initial states must have matching keys.")
        return {
            key: stack_tabero_initial_states([state[key] for state in states])
            for key in first
        }
    if isinstance(first, torch.Tensor):
        return torch.cat(states, dim=0)
    raise TypeError(f"Unsupported Tabero initial-state value: {type(first)!r}.")


class TaberoHdf5ResetWrapper:
    """Apply HDF5 demo initial states after the simulator bootstrap reset."""

    def __init__(
        self,
        env: Any,
        *,
        dataset_handler: Any,
        episode_names: list[str],
        shard_id: int,
        total_shards: int,
        capture_terminal_observation: bool = False,
    ) -> None:
        self._env = env
        self._dataset_handler = dataset_handler
        self._episode_names = list(episode_names)
        self._shard_id = int(shard_id)
        self._total_shards = int(total_shards)
        self._bootstrap_reset_done = False
        self._rollout_round = 0
        self._capture_terminal_observation = bool(capture_terminal_observation)
        self._capture_terminal_on_reset = False
        self._step_terminal_observation: Any | None = None
        self._step_terminal_mask = torch.zeros(
            int(self._env.num_envs), dtype=torch.bool, device=self._env.device
        )
        if self._capture_terminal_observation:
            self._install_terminal_observation_capture()

    def _install_terminal_observation_capture(self) -> None:
        if not hasattr(self._env, "_reset_idx") or not hasattr(
            self._env, "observation_manager"
        ):
            raise ValueError(
                "Tabero terminal-safe HDF5 mode requires IsaacLab _reset_idx "
                "and observation_manager support."
            )

        original_reset_idx = self._env._reset_idx

        def reset_idx_with_terminal_capture(env_ids) -> None:
            if self._capture_terminal_on_reset:
                terminal_obs = self._env.observation_manager.compute(
                    update_history=False
                )
                env_ids_tensor = torch.as_tensor(
                    env_ids, device=self._env.device, dtype=torch.long
                )
                capture_mask = torch.zeros_like(self._step_terminal_mask)
                capture_mask[env_ids_tensor] = True
                if self._step_terminal_observation is None:
                    self._step_terminal_observation = _clone_nested_tensors(
                        terminal_obs
                    )
                else:
                    self._step_terminal_observation = _replace_batch_rows(
                        self._step_terminal_observation,
                        terminal_obs,
                        capture_mask,
                    )
                self._step_terminal_mask[env_ids_tensor] = True
            original_reset_idx(env_ids)

        self._env._reset_idx = reset_idx_with_terminal_capture

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def reset(
        self,
        seed: int | None = None,
        env_ids: torch.Tensor | None = None,
    ):
        reset_result = self._env.reset(seed=seed, env_ids=env_ids)
        if not self._bootstrap_reset_done:
            self._bootstrap_reset_done = True
            return reset_result

        target_env_ids = (
            torch.arange(self._env.num_envs, device=self._env.device)
            if env_ids is None
            else torch.as_tensor(env_ids, device=self._env.device, dtype=torch.long)
        )
        assigned_names = assign_tabero_reset_episode_names(
            self._episode_names,
            num_envs=self._env.num_envs,
            shard_id=self._shard_id,
            total_shards=self._total_shards,
            rollout_round=self._rollout_round,
            env_ids=target_env_ids,
        )
        initial_states = []
        for episode_name in assigned_names:
            episode = self._dataset_handler.load_episode(episode_name, self._env.device)
            if episode is None or "initial_state" not in episode.data:
                raise ValueError(
                    f"Tabero HDF5 episode {episode_name!r} has no initial_state."
                )
            initial_states.append(episode.get_initial_state())

        state = stack_tabero_initial_states(initial_states)
        reset_result = self._env.reset_to(
            state,
            target_env_ids,
            is_relative=True,
        )
        print(
            "Tabero HDF5 reset "
            f"shard={self._shard_id} round={self._rollout_round} "
            f"episodes={assigned_names}",
            flush=True,
        )
        self._rollout_round += 1
        return reset_result

    def step(self, action: torch.Tensor):
        if not self._capture_terminal_observation:
            return self._env.step(action)

        self._step_terminal_observation = None
        self._step_terminal_mask.zero_()
        self._capture_terminal_on_reset = True
        try:
            obs, reward, terminations, truncations, extras = self._env.step(action)
        finally:
            self._capture_terminal_on_reset = False

        dones = torch.logical_or(terminations, truncations).to(dtype=torch.bool)
        if dones.any():
            if self._step_terminal_observation is None or not torch.all(
                self._step_terminal_mask[dones]
            ):
                raise RuntimeError(
                    "IsaacLab reset did not provide every Tabero terminal observation."
                )
            extras = dict(extras or {})
            extras[_TERMINAL_RAW_OBSERVATION_KEY] = self._step_terminal_observation
            extras[_TERMINAL_OBSERVATION_MASK_KEY] = self._step_terminal_mask.clone()
        return obs, reward, terminations, truncations, extras

    def close(self) -> None:
        try:
            self._dataset_handler.close()
        finally:
            self._env.close()


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
    if assignment == "paired":
        condition_cycle = ["firm", "gentle"]
    elif assignment == "cyclic":
        condition_cycle = [
            str(condition).lower()
            for condition in _cfg_get(prompt_cfg, "condition_cycle", [])
        ]
        if not condition_cycle or any(
            condition not in {"firm", "gentle"} for condition in condition_cycle
        ):
            raise ValueError(
                "Cyclic prompt assignment requires a non-empty condition_cycle "
                "containing only 'firm' and 'gentle'."
            )
    else:
        raise ValueError(
            f"Unsupported Tabero prompt assignment {assignment!r}; "
            "expected 'paired' or 'cyclic'."
        )
    if int(num_envs) % len(condition_cycle) != 0:
        if assignment == "paired":
            raise ValueError(
                "Paired firm/gentle prompts require an even number of vector envs."
            )
        raise ValueError(
            f"The number of vector envs ({int(num_envs)}) must be divisible by "
            f"condition cycle length ({len(condition_cycle)})."
        )

    firm_adverbs = [str(x) for x in _cfg_get(prompt_cfg, "firm_adverbs", [])]
    gentle_adverbs = [str(x) for x in _cfg_get(prompt_cfg, "gentle_adverbs", [])]
    prompt_seed = int(_cfg_get(prompt_cfg, "prompt_seed", 0))
    cycle_ids = [0 if condition == "firm" else 1 for condition in condition_cycle]
    condition_ids = [
        cycle_ids[env_id % len(cycle_ids)] for env_id in range(int(num_envs))
    ]
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
    return tuple(
        firm if int(condition_id) == 0 else gentle for condition_id in condition_ids
    )


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

    def update(
        self,
        marker_motion: torch.Tensor,
        update_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
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

        if update_mask is None:
            update_mask = torch.ones(
                self.num_envs, device=current_pos.device, dtype=torch.bool
            )
        else:
            update_mask = torch.as_tensor(
                update_mask, device=current_pos.device, dtype=torch.bool
            )
            if update_mask.shape != (self.num_envs,):
                raise ValueError(
                    "marker history update_mask must have shape "
                    f"({self.num_envs},), got {tuple(update_mask.shape)}."
                )

        new_envs = ~self._initialized & update_mask
        if new_envs.any():
            self._reference[new_envs] = init_pos[new_envs]
            self._history[new_envs] = current_pos[new_envs, None].expand(
                -1, self.history_len, -1, -1
            )
            self._initialized[new_envs] = True

        existing_envs = self._initialized & update_mask & ~new_envs
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
        self._hdf5_initial_states_path = _cfg_get(
            init_params, "hdf5_initial_states_path", None
        )
        self._hdf5_reset_assignment = str(
            _cfg_get(init_params, "hdf5_reset_assignment", "cyclic")
        )
        self._chunk_boundary_mode = validate_tabero_chunk_boundary_mode(
            _cfg_get(init_params, "chunk_boundary_mode", _LEGACY_CHUNK_BOUNDARY_MODE)
        )
        if self._hdf5_initial_states_path is not None:
            hdf5_path = Path(str(self._hdf5_initial_states_path)).expanduser()
            if not hdf5_path.is_file():
                raise FileNotFoundError(
                    f"Tabero HDF5 initial-state file not found: {hdf5_path}"
                )
            if self._hdf5_reset_assignment != "cyclic":
                raise ValueError(
                    "Tabero hdf5_reset_assignment currently supports only 'cyclic'."
                )
            self._hdf5_initial_states_path = str(hdf5_path)
        if (
            self._chunk_boundary_mode == _TERMINAL_SAFE_HDF5_MODE
            and self._hdf5_initial_states_path is None
        ):
            raise ValueError(
                "Tabero terminal-safe HDF5 mode requires hdf5_initial_states_path."
            )
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
            expected_episode_steps = configure_tabero_episode_horizon(
                init_params, isaac_env_cfg
            )

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
            validate_tabero_episode_horizon(env, expected_episode_steps)
            if self._hdf5_initial_states_path is not None:
                from isaaclab.utils.datasets import HDF5DatasetFileHandler

                dataset_handler = HDF5DatasetFileHandler()
                dataset_handler.open(self._hdf5_initial_states_path)

                def episode_sort_key(name: str) -> tuple[int, str]:
                    suffix = str(name).removeprefix("demo_")
                    return (
                        (int(suffix), str(name))
                        if suffix.isdigit()
                        else (sys.maxsize, str(name))
                    )

                episode_names = sorted(
                    (str(name) for name in dataset_handler.get_episode_names()),
                    key=episode_sort_key,
                )
                env = TaberoHdf5ResetWrapper(
                    env,
                    dataset_handler=dataset_handler,
                    episode_names=episode_names,
                    shard_id=self._tabero_task_shard_id,
                    total_shards=self.total_num_processes,
                    capture_terminal_observation=(
                        self._chunk_boundary_mode == _TERMINAL_SAFE_HDF5_MODE
                    ),
                )
            return env, sim_app

        return make_env_isaaclab

    def reset(self, seed=None, env_ids: torch.Tensor | None = None):
        self._marker_history.reset(env_ids)
        target_mask = None
        target_env_ids = None
        if env_ids is not None:
            target_env_ids = torch.as_tensor(
                env_ids, device=self.device, dtype=torch.long
            )
            target_mask = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.bool
            )
            target_mask[target_env_ids] = True
        if getattr(self, "_prompt_condition_ids", ()):
            self._prompt_rollout_round += 1
            conditioned_prompts, condition_ids = build_tabero_conditioned_prompts(
                instruction=self._tabero_task.task_description,
                task_suite=self._tabero_task.task_suite,
                task_id=self._tabero_task.task_id,
                num_envs=self.num_envs,
                prompt_cfg=self._prompt_cfg,
                rollout_round=self._prompt_rollout_round,
            )
            if target_env_ids is None:
                self._conditioned_prompts = conditioned_prompts
                self._prompt_condition_ids = tuple(condition_ids)
            else:
                current_prompts = list(self._conditioned_prompts)
                current_condition_ids = list(self._prompt_condition_ids)
                for env_id in target_env_ids.detach().cpu().tolist():
                    current_prompts[env_id] = conditioned_prompts[env_id]
                    current_condition_ids[env_id] = condition_ids[env_id]
                self._conditioned_prompts = current_prompts
                self._prompt_condition_ids = tuple(current_condition_ids)
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
                self._condition_squeeze_sum[target_env_ids] = 0
                self._condition_squeeze_count[target_env_ids] = 0

        if target_env_ids is None:
            raw_obs, _ = self.env.reset(seed=seed)
        else:
            raw_obs, _ = self.env.reset(seed=seed, env_ids=target_env_ids)
        obs = self._wrap_obs(raw_obs, marker_update_mask=target_mask)
        self._reset_metrics(target_env_ids)
        return obs, {}

    def step(self, actions=None, auto_reset=True):
        if self._chunk_boundary_mode == _TERMINAL_SAFE_HDF5_MODE:
            active_mask = torch.ones(
                self.num_envs, device=self.device, dtype=torch.bool
            )
            return self._terminal_safe_step(actions, active_mask=active_mask)
        if getattr(self, "_prompt_condition_ids", ()) and actions is not None:
            actions_tensor = torch.as_tensor(actions, device=self.device)
            squeeze = compute_tabero_predicted_squeeze(actions_tensor)
            self._condition_squeeze_sum += squeeze.to(dtype=torch.float32)
            self._condition_squeeze_count += 1
        return super().step(actions=actions, auto_reset=auto_reset)

    def _terminal_safe_step(
        self,
        actions: torch.Tensor,
        *,
        active_mask: torch.Tensor,
    ):
        active_mask = torch.as_tensor(active_mask, device=self.device, dtype=torch.bool)
        if active_mask.shape != (self.num_envs,):
            raise ValueError(
                "terminal-safe active_mask must have shape "
                f"({self.num_envs},), got {tuple(active_mask.shape)}."
            )

        actions_tensor = torch.as_tensor(actions, device=self.device)
        if getattr(self, "_prompt_condition_ids", ()):
            squeeze = compute_tabero_predicted_squeeze(actions_tensor)
            self._condition_squeeze_sum[active_mask] += squeeze[active_mask].to(
                dtype=torch.float32
            )
            self._condition_squeeze_count[active_mask] += 1

        raw_obs, step_reward, raw_terminations, raw_truncations, raw_infos = (
            self.env.step(actions_tensor)
        )
        step_reward = step_reward.clone()
        raw_terminations = raw_terminations.clone().to(dtype=torch.bool)
        raw_truncations = raw_truncations.clone().to(dtype=torch.bool)
        raw_infos = dict(raw_infos or {})

        self._elapsed_steps[active_mask] += 1
        horizon_truncations = active_mask & (
            self.elapsed_steps >= self.cfg.max_episode_steps
        )
        terminations = raw_terminations & active_mask
        truncations = (raw_truncations | horizon_truncations) & active_mask
        newly_done = terminations | truncations

        captured_mask = torch.as_tensor(
            raw_infos.get(
                _TERMINAL_OBSERVATION_MASK_KEY,
                torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
            ),
            device=self.device,
            dtype=torch.bool,
        )
        captured_raw_obs = raw_infos.get(_TERMINAL_RAW_OBSERVATION_KEY)
        internal_done = active_mask & (raw_terminations | raw_truncations)
        if internal_done.any() and (
            captured_raw_obs is None or not torch.all(captured_mask[internal_done])
        ):
            raise RuntimeError(
                "Tabero terminal-safe step is missing an IsaacLab terminal frame."
            )

        wrapped_source = raw_obs
        if captured_raw_obs is not None and captured_mask.any():
            wrapped_source = _replace_batch_rows(
                raw_obs, captured_raw_obs, captured_mask & active_mask
            )
        obs = self._wrap_obs(wrapped_source, marker_update_mask=active_mask)

        step_reward = torch.where(active_mask, step_reward, 0.0)
        infos = self._record_metrics(step_reward, terminations, {})
        final_info = {"episode": _clone_nested_tensors(infos["episode"])}
        returned_terminations = terminations.clone()
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = terminations.clone()
            returned_terminations.zero_()

        if newly_done.any():
            infos["final_observation"] = _clone_nested_tensors(obs)
            infos["final_info"] = final_info
            infos["_final_info"] = newly_done.clone()
            infos["_final_observation"] = newly_done.clone()
            infos["_elapsed_steps"] = newly_done.clone()

        infos["_tabero_hold_state"] = build_tabero_state(raw_obs["policy"])
        infos["_tabero_boundary_done"] = newly_done
        infos["_tabero_boundary_termination"] = terminations.clone()
        infos["_tabero_boundary_truncation"] = truncations.clone()
        infos["_tabero_terminal_capture"] = newly_done.clone()
        return (
            obs,
            step_reward,
            returned_terminations,
            truncations,
            infos,
        )

    @staticmethod
    def _build_hold_actions(
        action_template: torch.Tensor, hold_state: torch.Tensor
    ) -> torch.Tensor:
        if action_template.ndim != 2 or action_template.shape[-1] != 13:
            raise ValueError(
                "Tabero terminal-safe hold expects actions with shape (N, 13); "
                f"got {tuple(action_template.shape)}."
            )
        hold_actions = torch.zeros_like(action_template)
        hold_actions[:, :7] = hold_state.to(
            device=action_template.device, dtype=action_template.dtype
        )
        return hold_actions

    def chunk_step(self, chunk_actions: torch.Tensor):
        if self._chunk_boundary_mode != _TERMINAL_SAFE_HDF5_MODE:
            return super().chunk_step(chunk_actions)
        if chunk_actions.ndim != 3 or chunk_actions.shape[-1] != 13:
            raise ValueError(
                "Tabero terminal-safe chunk expects shape (N, chunk, 13); "
                f"got {tuple(chunk_actions.shape)}."
            )

        chunk_size = int(chunk_actions.shape[1])
        if chunk_size <= 0:
            raise ValueError("Tabero terminal-safe chunk must contain an action.")
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
        episode_record_shards: dict[str, list[torch.Tensor]] = {}

        for step_index in range(chunk_size):
            actions = chunk_actions[:, step_index].clone()
            inactive_mask = ~active_mask
            if inactive_mask.any():
                if latest_hold_state is None:
                    raise RuntimeError(
                        "Tabero terminal-safe chunk has no state for a hold action."
                    )
                hold_actions = self._build_hold_actions(actions, latest_hold_state)
                # Env-worker action preparation intentionally returns CPU tensors,
                # while IsaacLab lifecycle masks live on the simulator device.
                # Keep executed-action bookkeeping on the input device and move
                # only the indexing mask for this replacement.
                action_inactive_mask = inactive_mask.to(device=actions.device)
                actions[action_inactive_mask] = hold_actions[action_inactive_mask]
                post_done_hold_steps += int(inactive_mask.sum().item())

            obs, reward, step_terminations, step_truncations, infos = (
                self._terminal_safe_step(actions, active_mask=active_mask)
            )
            boundary_done = infos.pop("_tabero_boundary_done")
            boundary_terminations = infos.pop("_tabero_boundary_termination")
            boundary_truncations = infos.pop("_tabero_boundary_truncation")
            terminal_capture = infos.pop("_tabero_terminal_capture")
            latest_hold_state = infos.pop("_tabero_hold_state")
            newly_done = boundary_done & active_mask
            if newly_done.any():
                episode = infos["final_info"]["episode"]
                env_indices = torch.nonzero(newly_done, as_tuple=False).squeeze(-1)
                selected_episode_fields = {
                    "env_index": env_indices,
                    "primitive_step_index": torch.full_like(env_indices, step_index),
                    "condition_id": episode[_EPISODE_CONDITION_ID_KEY][newly_done],
                    "termination": boundary_terminations[newly_done],
                    "truncation": boundary_truncations[newly_done],
                    "success_once": episode["success_once"][newly_done].to(
                        torch.float32
                    ),
                    "return": episode["return"][newly_done].to(torch.float32),
                    "episode_len": episode["episode_len"][newly_done].to(torch.float32),
                    "reward": episode["reward"][newly_done].to(torch.float32),
                    "reward_sum": episode["return"][newly_done].to(torch.float32),
                    "terminal_step_reward": reward[newly_done].to(torch.float32),
                    "squeeze_pred_mean": episode[_EPISODE_SQUEEZE_PRED_MEAN_KEY][
                        newly_done
                    ].to(torch.float32),
                    "task_id": episode["task_id"][newly_done].to(torch.float32),
                    "task_shard_id": episode["task_shard_id"][newly_done].to(
                        torch.float32
                    ),
                }
                for field, values in selected_episode_fields.items():
                    episode_record_shards.setdefault(field, []).append(values.clone())
                first_done_step[newly_done] = step_index
                terminal_observation_captures += int(
                    terminal_capture[newly_done].sum().item()
                )
                active_mask = active_mask & ~newly_done

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
        hdf5_reset_envs = int(past_dones.sum().item())
        episode_records = {
            field: torch.cat(shards, dim=0)
            for field, shards in episode_record_shards.items()
        }
        if episode_records:
            record_count = int(episode_records["env_index"].numel())
            if record_count != hdf5_reset_envs:
                raise RuntimeError(
                    "Tabero terminal-safe episode-record count mismatch: "
                    f"records={record_count}, done_envs={hdf5_reset_envs}."
                )
            if torch.unique(episode_records["env_index"]).numel() != record_count:
                raise RuntimeError(
                    "Tabero terminal-safe chunk recorded one environment more than once."
                )
        if past_dones.any():
            env_ids = torch.nonzero(past_dones, as_tuple=False).squeeze(-1)
            reset_obs, _ = self.reset(env_ids=env_ids)
            obs_list[-1] = _replace_batch_rows(obs_list[-1], reset_obs, past_dones)

        early_done_envs = int(
            ((first_done_step >= 0) & (first_done_step < chunk_size - 1)).sum().item()
        )
        infos_list[-1]["chunk_boundary_metrics"] = {
            "done_envs": torch.tensor([hdf5_reset_envs], device=self.device),
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
            "hdf5_reset_envs": torch.tensor([hdf5_reset_envs], device=self.device),
        }
        infos_list[-1][_CHUNK_EPISODE_RECORDS_KEY] = episode_records
        infos_list[-1]["_tabero_executed_chunk_actions"] = torch.stack(
            executed_actions, dim=1
        )
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

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
            episode_info[_EPISODE_CONDITION_ID_KEY] = condition_ids.clone()
            episode_info[_EPISODE_SQUEEZE_PRED_MEAN_KEY] = squeeze_mean.clone()
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
        else:
            episode_info[_EPISODE_CONDITION_ID_KEY] = torch.full(
                (self.num_envs,),
                -1,
                dtype=torch.int64,
                device=step_reward.device,
            )
            episode_info[_EPISODE_SQUEEZE_PRED_MEAN_KEY] = torch.full_like(
                step_reward, float("nan"), dtype=torch.float32
            )
        return infos

    def _wrap_obs(
        self,
        obs,
        marker_update_mask: torch.Tensor | None = None,
    ):
        policy_obs = obs["policy"]
        state = build_tabero_state(policy_obs)
        tactile_marker_motion = self._marker_history.update(
            policy_obs[self._marker_motion_key], update_mask=marker_update_mask
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
