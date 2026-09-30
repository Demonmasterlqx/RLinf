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

"""PPO-specific validation and reductions; shared task helpers remain exported."""

import torch
from omegaconf import OmegaConf

from rlinf.utils.multi_task import (  # noqa: F401 - compatibility exports
    COUNT_KEY,
    DEFAULTS,
    STATE_FILE,
    SUPPORTED_ENVS,
    SuccessWeightController,
    batch_weight_stats,
    checkpoint_metadata,
    enabled,
    manifest,
    options,
    task_env_cfg,
    task_list,
    validate_environments,
)


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
    validate_environments(cfg)


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
