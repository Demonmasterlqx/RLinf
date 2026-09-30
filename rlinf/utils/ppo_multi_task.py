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

"""Optional synchronous PPO task assignment, success weighting and persistence.

This module does not import workers, simulators or models. Task weights are
training metadata; they never change rewards, advantages or policy inputs.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
from pathlib import Path

import torch
from omegaconf import OmegaConf, open_dict

STATE_FILE = "multi_task_state.json"
COUNT_KEY = "_ppo_multi_task_counts"
DEFAULTS = {
    "ema_decay": 0.9,
    "weight_min": 0.5,
    "weight_max": 2.0,
    "sigmoid_scale": 10.0,
}
SUPPORTED_ENVS = {
    "Isaac-Libero-Franka-Hybrid-Tactile-v0",
    "Isaac-RealWorld-GentleGrasp-XarmUmi-Hybrid-Tactile-v0",
}


def enabled(cfg) -> bool:
    """Return False without inserting defaults into legacy configurations."""
    value = OmegaConf.select(cfg, "algorithm.multi_task.enabled", default=False)
    if type(value) is not bool:
        raise ValueError("algorithm.multi_task.enabled must be boolean.")
    return value


def options(cfg) -> dict:
    """Resolve and validate numerical controller settings."""
    result = {
        k: OmegaConf.select(cfg, f"algorithm.multi_task.{k}", default=v)
        for k, v in DEFAULTS.items()
    }
    if any(
        isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
        for v in result.values()
    ):
        raise ValueError("PPO multi_task settings must be finite numbers.")
    if not 0 <= result["ema_decay"] < 1:
        raise ValueError("PPO multi_task ema_decay must be in [0, 1).")
    if not 0 < result["weight_min"] <= result["weight_max"]:
        raise ValueError("PPO multi_task requires 0 < weight_min <= weight_max.")
    if result["sigmoid_scale"] < 0:
        raise ValueError("PPO multi_task sigmoid_scale must be nonnegative.")
    return result


def task_list(env_cfg) -> list[dict]:
    """Read explicit tasks; names are also safe metric path components."""
    tasks = OmegaConf.select(env_cfg, "multi_task.tasks")
    if tasks is None or len(tasks) == 0:
        raise ValueError("PPO multi_task requires env.<split>.multi_task.tasks.")
    tasks = OmegaConf.to_container(tasks, resolve=True)
    names = []
    for task in tasks:
        if not isinstance(task, dict) or set(task) != {"name", "init_params"}:
            raise ValueError("Each PPO task requires exactly name and init_params.")
        name = task["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("PPO task names must contain letters, digits, _ or -.")
        if not isinstance(task["init_params"], dict):
            raise ValueError("Task init_params must be a mapping.")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("PPO task names must be unique.")
    return tasks


def task_env_cfg(env_cfg, index: int):
    """Build an isolated single-task view for existing environment adapters."""
    task = task_list(env_cfg)[index]
    result = copy.deepcopy(env_cfg)
    with open_dict(result):
        result.init_params = OmegaConf.merge(result.init_params, task["init_params"])
        del result.multi_task
        if result.init_params.id == "Isaac-Libero-Franka-Hybrid-Tactile-v0":
            # Prevent the legacy task subset from reassigning this shard.
            result.init_params.tasks = [
                {
                    "task_suite": result.init_params.task_suite,
                    "task_id": result.init_params.task_id,
                    "task_description": result.init_params.get("task_description"),
                }
            ]
            result.init_params.tabero_task_subset_path = None
    return result


def validate_config(cfg) -> None:
    """Gate v1 to the existing terminal-safe OpenPI synchronous PPO contract."""
    options(cfg)
    required = {
        "runner.task_type": "embodied",
        "actor.training_backend": "fsdp",
        "actor.model.model_type": "openpi",
        "algorithm.adv_type": "gae",
        "algorithm.loss_type": "actor_critic",
        "algorithm.reward_type": "chunk_level",
        "algorithm.logprob_type": "chunk_level",
    }
    for key, expected in required.items():
        if OmegaConf.select(cfg, key) != expected:
            raise ValueError(f"PPO multi_task requires {key}={expected!r}.")
    for key in (
        "runner.use_training_pipeline",
        "runner.only_eval",
        "actor.model.openpi.use_dsrl",
        "actor.enable_sft_co_train",
        "reward.use_reward_model",
    ):
        if OmegaConf.select(cfg, key, default=False):
            raise ValueError(f"PPO multi_task does not support {key}.")
    if (
        OmegaConf.select(cfg, "algorithm.tabero_ppo_transition_boundary_semantics")
        is None
    ):
        raise ValueError(
            "PPO multi_task requires an explicit terminal-safe PPO boundary."
        )
    if OmegaConf.select(cfg, "rollout.collect_prev_infos", default=True) is not True:
        raise ValueError("PPO multi_task requires rollout.collect_prev_infos=true.")
    train_tasks = task_list(cfg.env.train)
    splits = ["train"]
    if cfg.runner.get("val_check_interval", -1) > 0:
        splits.append("eval")
    for split in splits:
        env = cfg.env[split]
        if [t["name"] for t in task_list(env)] != [t["name"] for t in train_tasks]:
            raise ValueError("Train and eval PPO task names/order must agree.")
        if env.env_type != "isaaclab" or env.init_params.id not in SUPPORTED_ENVS:
            raise ValueError(
                "PPO multi_task has no completed-episode adapter for this env."
            )
        if env.get("auto_reset", True) or env.get("ignore_terminations", True):
            raise ValueError(
                "PPO multi_task v1 requires terminal-safe non-auto-reset envs."
            )
        if env.max_steps_per_rollout_epoch < env.max_episode_steps:
            raise ValueError(
                "PPO multi_task rollout must cover complete episodes before reset."
            )
        stage_count = cfg.rollout.pipeline_stage_num
        if type(stage_count) is not int or stage_count < 1:
            raise ValueError(
                "PPO multi_task requires positive rollout.pipeline_stage_num."
            )
        # Interface-changing overrides cannot be combined under one policy.
        interfaces = []
        for i in range(len(train_tasks)):
            effective = task_env_cfg(env, i)
            params = OmegaConf.to_container(effective.init_params, resolve=True)
            if params.get("id") != env.init_params.id:
                raise ValueError(
                    "PPO tasks must use the common environment adapter id."
                )
            if "num_envs" in task_list(env)[i]["init_params"]:
                raise ValueError(
                    "Task overrides cannot change num_envs; use total_num_envs."
                )
            varying = {
                "task_suite",
                "task_id",
                "task_description",
                "target_object",
                "realworld_config_dir",
                "libero_config_dir",
                "hdf5_initial_states_path",
                "tasks",
                "tabero_task_subset_path",
                "prompt_conditions",
            }
            interfaces.append({k: v for k, v in params.items() if k not in varying})
        if any(value != interfaces[0] for value in interfaces[1:]):
            raise ValueError(
                "PPO tasks must share observation, action, reward and reset interfaces."
            )


def manifest(cfg) -> dict:
    """Record task and optimization settings for provenance, not resume gating."""
    splits = ["train"] + (
        ["eval"] if cfg.runner.get("val_check_interval", -1) > 0 else []
    )
    task_contracts = {}
    for split in splits:
        task_contracts[split] = []
        for i, task in enumerate(task_list(cfg.env[split])):
            env = OmegaConf.to_container(task_env_cfg(cfg.env[split], i), resolve=True)
            env.pop("video_cfg", None)
            env["init_params"].pop("gripper_diagnostics_path", None)
            task_contracts[split].append({"name": task["name"], "env": env})
    return {
        "options": options(cfg),
        "algorithm": OmegaConf.to_container(cfg.algorithm, resolve=True),
        "model": OmegaConf.to_container(cfg.actor.model, resolve=True),
        "batch_size": cfg.actor.global_batch_size,
        "seed": cfg.actor.seed,
        "optimizer": OmegaConf.to_container(cfg.actor.optim, resolve=True),
        "placement": OmegaConf.to_container(
            cfg.cluster.component_placement, resolve=True
        ),
        "stages": cfg.rollout.pipeline_stage_num,
        "tasks": task_contracts,
    }


def checkpoint_metadata(metadata: dict, cfg) -> dict:
    """Describe every task in exported trainable weights, without a false target."""
    result = dict(metadata)
    fields = ("task_suite", "task_id", "target_object", "task_description")
    for key in fields:
        result.pop(key, None)
    result["multi_task_format"] = "ppo_multitask_v1"
    result["tasks"] = []
    for i, task in enumerate(task_list(cfg.env.train)):
        params = task_env_cfg(cfg.env.train, i).init_params
        result["tasks"].append(
            {"name": task["name"], **{k: params.get(k) for k in fields}}
        )
    return result


class SuccessWeightController:
    """One controller per runner; one update per completed rollout round."""

    def __init__(self, names: list[str], settings: dict, contract: dict):
        self.names = names
        self.settings = settings
        self.contract = contract
        self.ema = torch.zeros(len(names), dtype=torch.float64)
        self.initialized = torch.zeros(len(names), dtype=torch.bool)
        self.totals = torch.zeros((2, len(names)), dtype=torch.int64)
        self.version = 0

    @property
    def weights(self) -> torch.Tensor:
        if not self.initialized.all():
            return torch.ones(len(self.names), dtype=torch.float32)
        cfg = self.settings
        return (
            cfg["weight_min"]
            + (cfg["weight_max"] - cfg["weight_min"])
            * torch.sigmoid(cfg["sigmoid_scale"] * (self.ema.mean() - self.ema))
        ).float()

    def update(self, counts: torch.Tensor, version: int) -> dict[str, float]:
        if version != self.version + 1:
            raise ValueError("PPO task weights must advance exactly once per rollout.")
        counts = torch.as_tensor(counts, device="cpu")
        if (
            counts.shape != self.totals.shape
            or not torch.isfinite(counts).all()
            or (counts < 0).any()
            or (counts != counts.round()).any()
            or (counts[0] > counts[1]).any()
        ):
            raise ValueError("Invalid PPO task success/episode counts.")
        counts = counts.to(torch.int64)
        observed = counts[1] > 0
        rates = counts[0].double() / counts[1].clamp_min(1)
        first = observed & ~self.initialized
        continuing = observed & self.initialized
        self.ema[first] = rates[first]
        decay = self.settings["ema_decay"]
        self.ema[continuing] = (
            decay * self.ema[continuing] + (1 - decay) * rates[continuing]
        )
        self.initialized |= observed
        self.totals += counts
        self.version = version
        metrics = {"multi_task/weight_version": version}
        for i, name in enumerate(self.names):
            prefix = f"multi_task/{name}"
            metrics.update(
                {
                    f"{prefix}/successes": counts[0, i].item(),
                    f"{prefix}/episodes": counts[1, i].item(),
                    f"{prefix}/initialized": self.initialized[i].item(),
                    f"{prefix}/success_ema": self.ema[i].item(),
                    f"{prefix}/raw_weight": self.weights[i].item(),
                }
            )
            if observed[i]:
                metrics[f"{prefix}/success_rate"] = rates[i].item()
        return metrics

    def save(self, directory: str, step: int) -> None:
        if step != self.version:
            raise ValueError("PPO task controller and actor checkpoint steps disagree.")
        payload = {
            "format": "ppo_multitask_v1",
            "step": step,
            "names": self.names,
            "contract": self.contract,
            "ema": self.ema.tolist(),
            "initialized": self.initialized.tolist(),
            "totals": self.totals.tolist(),
        }
        path = Path(directory) / STATE_FILE
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            json.dump(payload, stream, allow_nan=False, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def restore(self, directory: str, step: int) -> None:
        path = Path(directory) / STATE_FILE
        if not path.is_file():
            raise ValueError(f"Multi-task continuation requires {path}.")
        payload = json.loads(path.read_text())
        if payload.get("format") != "ppo_multitask_v1" or payload.get("step") != step:
            raise ValueError("PPO multi-task checkpoint format or step mismatch.")
        saved_names = payload.get("names")
        if not isinstance(saved_names, list) or not all(
            isinstance(name, str) for name in saved_names
        ):
            raise ValueError("Invalid PPO multi-task checkpoint task names.")
        duplicates = sorted(
            {name for name in saved_names if saved_names.count(name) > 1}
        )
        if duplicates:
            raise ValueError(f"Duplicate PPO checkpoint task names: {duplicates}.")
        if set(saved_names) != set(self.names):
            missing = sorted(set(self.names) - set(saved_names))
            unexpected = sorted(set(saved_names) - set(self.names))
            raise ValueError(
                "PPO multi-task checkpoint task names mismatch: "
                f"missing={missing}, unexpected={unexpected}."
            )
        ema = torch.tensor(payload["ema"], dtype=torch.float64)
        initialized = torch.tensor(payload["initialized"], dtype=torch.bool)
        totals = torch.tensor(payload["totals"])
        if (
            ema.shape != self.ema.shape
            or initialized.shape != self.initialized.shape
            or totals.shape != self.totals.shape
            or not torch.isfinite(ema).all()
            or not torch.isfinite(totals).all()
            or ((ema < 0) | (ema > 1)).any()
            or (totals < 0).any()
            or (totals != totals.round()).any()
            or (totals[0] > totals[1]).any()
            or not torch.equal(initialized, totals[1] > 0)
        ):
            raise ValueError("Invalid PPO multi-task checkpoint state.")
        # Task identity is its name; old configuration records are informational.
        order = [saved_names.index(name) for name in self.names]
        self.ema, self.initialized, self.totals = (
            ema[order],
            initialized[order],
            totals[:, order].to(torch.int64),
        )
        self.version = step


def batch_weight_stats(weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Additive statistics for normalization across ranks and accumulation."""
    weights, mask = weights.reshape(-1), mask.reshape(-1).bool()
    if (
        weights.shape != mask.shape
        or not torch.isfinite(weights).all()
        or (weights <= 0).any()
    ):
        raise ValueError("PPO sample weights must be positive finite aligned scalars.")
    return torch.stack([(weights.double() * mask).sum(), mask.double().sum()])


def weighted_reduce(
    values: torch.Tensor,
    mask: torch.Tensor | None,
    ratio: torch.Tensor | None = None,
    *,
    sample_weights: torch.Tensor,
    reduction_scale: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Apply weights after PPO clipping, preserving legacy length correction.

    When provided, reduction_scale is the local numerator multiplier accounting
    for the complete optimizer batch, DDP averaging and gradient accumulation.
    """
    weights = sample_weights.detach().to(values).reshape(-1)
    if values.numel() != weights.numel():
        raise ValueError("Multi-task PPO expects one chunk loss per sample.")
    values = values.reshape(-1)
    valid = (
        torch.ones_like(values, dtype=torch.bool)
        if mask is None
        else mask.reshape(-1).bool()
    )
    values = torch.where(valid, values, torch.zeros_like(values))
    if ratio is not None:
        denominator = torch.where(
            valid, ratio.reshape(-1).to(values), torch.ones_like(values)
        )
        values = values / denominator
    numerator = (values * weights).sum()
    if reduction_scale is not None:
        return numerator * reduction_scale
    denominator = values.numel() if ratio is not None else valid.sum().clamp_min(1)
    return numerator / denominator
