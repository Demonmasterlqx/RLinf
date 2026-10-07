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

import math
from typing import Any

import torch

TERMINAL_SUCCESS_KEY = "_tabero_terminal_valid_success"


def valid_success_mask(env: Any) -> torch.Tensor:
    """Read actual terminal success before IsaacLab resets its managers."""
    manager = env.termination_manager
    success = manager.get_term("success").to(dtype=torch.bool).clone()
    invalid = manager.time_outs.to(dtype=torch.bool).clone()
    for name in manager.active_terms:
        if name != "success":
            invalid |= manager.get_term(name).to(dtype=torch.bool)
    return success & ~invalid


def record_success_metrics(env: Any, step_reward: torch.Tensor, infos: dict) -> dict:
    """Keep Tabero success independent of signed rewards and return scaling."""
    success = infos.pop(TERMINAL_SUCCESS_KEY)
    if success.shape != step_reward.shape or success.dtype != torch.bool:
        raise ValueError("Tabero terminal success must be a reward-shaped bool tensor.")
    env.returns += step_reward
    env.success_once |= success
    infos["episode"] = {
        "success_once": env.success_once.clone(),
        "return": env.returns.clone(),
        "episode_len": env.elapsed_steps.clone(),
        "reward": env.returns / env.elapsed_steps.clamp(min=1),
    }
    return infos


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def validate_force_bonus_cfg(
    force_bonus_cfg: Any, terminal_reward: float
) -> dict[str, Any]:
    enabled = bool(_cfg_get(force_bonus_cfg, "enabled", False))
    normalized = {
        "enabled": enabled,
        "coefficient": float(_cfg_get(force_bonus_cfg, "coefficient", 0.0)),
        "epsilon": float(_cfg_get(force_bonus_cfg, "epsilon", 0.1)),
        "max_bonus": float(_cfg_get(force_bonus_cfg, "max_bonus", 0.0)),
        "min_valid_samples": int(_cfg_get(force_bonus_cfg, "min_valid_samples", 1)),
        "contact_epsilon": float(_cfg_get(force_bonus_cfg, "contact_epsilon", 1.0e-4)),
    }
    if not enabled:
        return normalized

    numeric_values = {
        "terminal_reward": float(terminal_reward),
        "coefficient": normalized["coefficient"],
        "epsilon": normalized["epsilon"],
        "max_bonus": normalized["max_bonus"],
        "contact_epsilon": normalized["contact_epsilon"],
    }
    non_finite = [
        name
        for name, value in numeric_values.items()
        if not torch.isfinite(torch.tensor(value)).item()
    ]
    if non_finite:
        raise ValueError(
            "Tabero success.force_bonus values must be finite; invalid fields: "
            f"{non_finite}."
        )
    if terminal_reward <= 0:
        raise ValueError(
            "Tabero success.force_bonus requires terminal_reward to be finite and positive."
        )
    if normalized["coefficient"] < 0:
        raise ValueError("Tabero success.force_bonus.coefficient must be non-negative.")
    if normalized["epsilon"] <= 0:
        raise ValueError("Tabero success.force_bonus.epsilon must be positive.")
    if normalized["max_bonus"] <= 0:
        raise ValueError("Tabero success.force_bonus.max_bonus must be positive.")
    if normalized["min_valid_samples"] < 1:
        raise ValueError(
            "Tabero success.force_bonus.min_valid_samples must be at least one."
        )
    if normalized["contact_epsilon"] < 0:
        raise ValueError(
            "Tabero success.force_bonus.contact_epsilon must be non-negative."
        )
    return normalized


def validate_force_reward_cfg(
    success_cfg: Any, terminal_reward: float
) -> dict[str, Any]:
    """Resolve one task's optional force reward, preserving legacy defaults."""
    old_cfg = _cfg_get(success_cfg, "force_bonus")
    new_cfg = _cfg_get(success_cfg, "normalize_effort_reward")
    old_enabled = _cfg_get(old_cfg, "enabled", False)
    new_enabled = _cfg_get(new_cfg, "enabled", False)
    if not isinstance(old_enabled, bool) or not isinstance(new_enabled, bool):
        raise ValueError("Tabero force reward enabled flags must be boolean.")
    if old_enabled and new_enabled:
        raise ValueError(
            "Tabero force_bonus and normalize_effort_reward are mutually exclusive."
        )
    if not new_enabled:
        if not old_enabled:
            return {"enabled": False}
        return validate_force_bonus_cfg(old_cfg, terminal_reward)

    def finite_number(name: str, default: Any = None) -> float:
        value = _cfg_get(new_cfg, name, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
        ):
            raise ValueError(
                f"Tabero normalize_effort_reward.{name} must be a finite number."
            )
        return float(value)

    minimum = finite_number("min_effort")
    midpoint = finite_number("mid_effort")
    maximum = midpoint + (midpoint - minimum)
    if minimum < 0 or midpoint <= minimum or not math.isfinite(maximum):
        raise ValueError(
            "Tabero normalize_effort_reward requires 0 <= min_effort < mid_effort "
            "and finite max_effort."
        )
    if not math.isfinite(terminal_reward) or terminal_reward <= 0:
        raise ValueError("Tabero terminal_reward must be finite and positive.")
    samples = _cfg_get(new_cfg, "min_valid_samples", 1)
    if type(samples) is not int or samples < 1:
        raise ValueError(
            "Tabero normalize_effort_reward.min_valid_samples must be a positive integer."
        )
    contact_epsilon = finite_number("contact_epsilon", 1.0e-4)
    if contact_epsilon < 0:
        raise ValueError(
            "Tabero normalize_effort_reward.contact_epsilon must be non-negative."
        )
    if _cfg_get(new_cfg, "max_effort") is not None:
        raise ValueError(
            "Tabero normalize_effort_reward.max_effort is derived, not configurable."
        )
    return {
        "enabled": True,
        "normalize_effort_reward": {"min_effort": minimum, "mid_effort": midpoint},
        "min_valid_samples": samples,
        "contact_epsilon": contact_epsilon,
    }


def make_trajectory_force_success_reward_term(manager_term_base_cls: type) -> type:
    class TrajectoryForceSuccessRewardTerm(manager_term_base_cls):
        def __init__(self, cfg: Any, env: Any) -> None:
            super().__init__(cfg, env)
            params = cfg.params
            self._force_sources = tuple(params.get("force_sources", ()))
            self._force_reader = params.get("force_reader")
            self._direct_force_reader = params.get("direct_force_reader")
            self._direct_force_params = dict(params.get("direct_force_params", {}))
            if self._direct_force_reader is None and (
                not self._force_sources or self._force_reader is None
            ):
                raise ValueError(
                    "Tabero force bonus requires either a direct measured-force reader "
                    "or at least one grasp-gated force source."
                )
            self._terminal_reward = float(params["terminal_reward"])
            self._normalize_effort_reward = params.get("normalize_effort_reward")
            self._coefficient = float(params.get("coefficient", 0.0))
            self._epsilon = float(params.get("epsilon", 0.1))
            self._max_bonus = float(params.get("max_bonus", 0.0))
            self._min_valid_samples = int(params["min_valid_samples"])
            self._contact_epsilon = float(params["contact_epsilon"])
            source_count = (
                1 if self._direct_force_reader is not None else len(self._force_sources)
            )
            self._grasp_started = torch.zeros(
                (env.num_envs, source_count), dtype=torch.bool, device=env.device
            )
            self._force_sum = torch.zeros(
                env.num_envs, dtype=torch.float32, device=env.device
            )
            self._force_count = torch.zeros(
                env.num_envs, dtype=torch.int64, device=env.device
            )

        @property
        def trajectory_mean_force(self) -> torch.Tensor:
            return self._force_sum / self._force_count.clamp(min=1)

        @property
        def valid_sample_count(self) -> torch.Tensor:
            return self._force_count

        @property
        def current_force_bonus(self) -> torch.Tensor:
            """Selected force component; the legacy name also permits signed values."""
            if self._normalize_effort_reward is not None:
                minimum = self._normalize_effort_reward["min_effort"]
                midpoint = self._normalize_effort_reward["mid_effort"]
                # Equivalent to 2 * (mid - effort) / (max - min).
                # Float64 avoids overflow/underflow of otherwise valid bounds.
                bonus = (
                    (
                        (midpoint - self.trajectory_mean_force.double())
                        / (midpoint - minimum)
                    )
                    .clamp(-1.0, 1.0)
                    .to(self._force_sum.dtype)
                )
            else:
                bonus = self._coefficient / self.trajectory_mean_force.clamp(
                    min=self._epsilon
                )
                bonus = bonus.clamp(min=0.0, max=self._max_bonus)
            return torch.where(
                self._force_count >= self._min_valid_samples,
                bonus,
                torch.zeros_like(bonus),
            )

        def reset(self, env_ids: Any = None) -> None:
            if env_ids is None:
                env_ids = slice(None)
            elif not isinstance(env_ids, slice):
                env_ids = torch.as_tensor(
                    env_ids, device=self._force_sum.device, dtype=torch.long
                )
            self._grasp_started[env_ids] = False
            self._force_sum[env_ids] = 0.0
            self._force_count[env_ids] = 0

        def _validate_force(self, force: Any, env: Any) -> torch.Tensor:
            force = torch.as_tensor(
                force, device=self._force_sum.device, dtype=torch.float32
            )
            if force.shape == (env.num_envs, 1, 2, 3):
                force = force[:, 0]
            if force.shape != (env.num_envs, 2, 3):
                raise RuntimeError(
                    "Tabero force bonus expected measured force shape "
                    f"({env.num_envs}, 2, 3), got {tuple(force.shape)}."
                )
            if not torch.isfinite(force).all():
                raise RuntimeError(
                    "Tabero force bonus received non-finite measured contact force."
                )
            return force

        def _contact_measurement(
            self, force_lr: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            finger_norms = torch.linalg.vector_norm(force_lr, dim=-1)
            both_fingers_contact = torch.all(
                finger_norms > self._contact_epsilon, dim=1
            )
            squeeze = 2.0 * torch.minimum(
                force_lr[:, 0, 2].abs(), force_lr[:, 1, 2].abs()
            )
            valid = both_fingers_contact & (squeeze > self._contact_epsilon)
            return squeeze, valid

        def _update_direct_force_history(self, env: Any) -> None:
            force_lr = self._validate_force(
                self._direct_force_reader(env, **self._direct_force_params), env
            )
            squeeze, contact_valid = self._contact_measurement(force_lr)
            self._grasp_started[:, 0] |= contact_valid
            valid_step = self._grasp_started[:, 0] & contact_valid
            self._force_sum[valid_step] += squeeze[valid_step]
            self._force_count[valid_step] += 1

        def _update_grasp_gated_force_history(self, env: Any) -> None:
            grasp_observations = env.observation_manager.compute_group(
                "subtask_terms", update_history=False
            )
            if not isinstance(grasp_observations, dict):
                raise RuntimeError(
                    "Tabero force bonus requires non-concatenated subtask observations."
                )

            per_source_squeeze = []
            per_source_valid = []
            for source_index, (grasp_term_name, force_source) in enumerate(
                self._force_sources
            ):
                if grasp_term_name not in grasp_observations:
                    raise RuntimeError(
                        "Tabero force bonus could not find grasp observation "
                        f"{grasp_term_name!r}."
                    )
                grasped = torch.as_tensor(
                    grasp_observations[grasp_term_name],
                    device=self._force_sum.device,
                    dtype=torch.bool,
                ).reshape(env.num_envs)
                self._grasp_started[:, source_index] |= grasped
                force_lr = self._validate_force(
                    self._force_reader(env, contact_sensor_name=force_source), env
                )
                squeeze, contact_valid = self._contact_measurement(force_lr)
                per_source_squeeze.append(squeeze)
                per_source_valid.append(
                    self._grasp_started[:, source_index] & contact_valid
                )

            squeeze_by_source = torch.stack(per_source_squeeze, dim=1)
            valid_by_source = torch.stack(per_source_valid, dim=1)
            valid_step = valid_by_source.any(dim=1)
            measured_squeeze = torch.where(
                valid_by_source,
                squeeze_by_source,
                torch.zeros_like(squeeze_by_source),
            ).amax(dim=1)
            self._force_sum[valid_step] += measured_squeeze[valid_step]
            self._force_count[valid_step] += 1

        def __call__(
            self,
            env: Any,
            success_term_name: str,
            failure_term_names: tuple[str, ...],
            terminal_reward: float,
            coefficient: float = 0.0,
            epsilon: float = 0.1,
            max_bonus: float = 0.0,
            min_valid_samples: int = 1,
            contact_epsilon: float = 1.0e-4,
            force_sources: tuple[tuple[str, Any], ...] = (),
            force_reader: Any = None,
            direct_force_reader: Any = None,
            direct_force_params: dict[str, Any] | None = None,
            env_reward_multipliers: tuple[float, ...] | None = None,
            normalize_effort_reward: dict[str, float] | None = None,
        ) -> torch.Tensor:
            del (
                force_sources,
                force_reader,
                direct_force_reader,
                direct_force_params,
                terminal_reward,
                coefficient,
                epsilon,
                max_bonus,
                min_valid_samples,
                contact_epsilon,
                normalize_effort_reward,
            )
            if self._direct_force_reader is not None:
                self._update_direct_force_history(env)
            else:
                self._update_grasp_gated_force_history(env)

            success = env.termination_manager.get_term(success_term_name).to(
                dtype=torch.bool
            )
            invalid = env.termination_manager.time_outs.to(dtype=torch.bool).clone()
            for term_name in failure_term_names:
                invalid |= env.termination_manager.get_term(term_name).to(
                    dtype=torch.bool
                )
            reward = (success & ~invalid).to(dtype=torch.float32) * (
                self._terminal_reward + self.current_force_bonus
            )
            reward = reward / float(env.step_dt)
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

    TrajectoryForceSuccessRewardTerm.__name__ = "TrajectoryForceSuccessRewardTerm"
    return TrajectoryForceSuccessRewardTerm
