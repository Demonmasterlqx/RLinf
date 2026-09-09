# Copyright 2025 The RLinf Authors.
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

import math
import os
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed

from rlinf.utils.dsrl_reward import (
    DSRL_REWARD_AUDIT_FIELDS,
    combine_dsrl_reward_audits,
)

if TYPE_CHECKING:
    from rlinf.data.embodied_io_struct import Trajectory


def mean_bool_tensor_rate(
    tensors: Sequence[torch.Tensor | None],
    *,
    sum_key: str,
    count_key: str,
    reducer: Callable[[dict[str, float]], dict[str, float]] | None = None,
) -> float | None:
    """Mean of flattened bool-like tensors, optionally reduced across ranks."""
    shards = [
        tensor.detach().reshape(-1).to(torch.float32)
        for tensor in tensors
        if isinstance(tensor, torch.Tensor) and tensor.numel() > 0
    ]
    if not shards:
        return None

    local_values = torch.cat(shards, dim=0)
    reduced = {
        sum_key: float(local_values.sum().item()),
        count_key: float(local_values.numel()),
    }
    if reducer is not None:
        reduced = reducer(reduced)
    if reduced[count_key] <= 0:
        return 0.0
    return reduced[sum_key] / reduced[count_key]


def mean_bool_tensor_rate_from_trajectories(
    trajectories: Sequence["Trajectory"],
    tensor_getter: Callable[["Trajectory"], torch.Tensor | None],
    *,
    sum_key: str,
    count_key: str,
    reducer: Callable[[dict[str, float]], dict[str, float]] | None = None,
) -> float | None:
    return mean_bool_tensor_rate(
        [tensor_getter(trajectory) for trajectory in trajectories],
        sum_key=sum_key,
        count_key=count_key,
        reducer=reducer,
    )


def trajectory_forward_input_tensor(
    trajectory: "Trajectory", key: str
) -> torch.Tensor | None:
    forward_inputs = trajectory.forward_inputs
    if not isinstance(forward_inputs, dict):
        return None
    value = forward_inputs.get(key)
    return value if isinstance(value, torch.Tensor) else None


def trajectory_has_bool_tensor(tensor: torch.Tensor | None) -> bool:
    return bool(
        isinstance(tensor, torch.Tensor) and tensor.detach().to(torch.bool).any()
    )


def collect_trajectory_replay_metrics(
    trajectories: Sequence["Trajectory"],
    *,
    reducer: Callable[[dict[str, float]], dict[str, float]] | None = None,
) -> dict[str, float]:
    """Replay-route diagnostics aggregated from received trajectories."""
    metrics: dict[str, float] = {}
    rate_specs = (
        (
            "replay/record_transition_rate",
            lambda trajectory: trajectory_forward_input_tensor(
                trajectory, "record_transition"
            ),
            "record_transition_sum",
            "record_transition_count",
        ),
        (
            "replay/actor_switch_rate",
            lambda trajectory: trajectory_forward_input_tensor(
                trajectory, "actor_switch"
            ),
            "actor_switch_sum",
            "actor_switch_count",
        ),
        (
            "replay/intervention_requested_rate",
            lambda trajectory: trajectory_forward_input_tensor(
                trajectory, "intervention_requested"
            ),
            "intervention_requested_sum",
            "intervention_requested_count",
        ),
        (
            "replay/intervention_rate",
            lambda trajectory: trajectory.intervene_flags,
            "intervention_sum",
            "intervention_count",
        ),
    )
    for metric_key, tensor_getter, sum_key, count_key in rate_specs:
        rate = mean_bool_tensor_rate_from_trajectories(
            trajectories,
            tensor_getter,
            sum_key=sum_key,
            count_key=count_key,
            reducer=reducer,
        )
        if rate is not None:
            metrics[metric_key] = rate
    return metrics


METRIC_SUM_PREFIX = "__sum__/"

CRITIC_EXPLAINED_VARIANCE_KEY = "critic/explained_variance"
CRITIC_EXPLAINED_VARIANCE_STATS_PREFIX = (
    f"{METRIC_SUM_PREFIX}_critic_explained_variance/"
)
CRITIC_EXPLAINED_VARIANCE_COUNT_KEY = f"{CRITIC_EXPLAINED_VARIANCE_STATS_PREFIX}count"
CRITIC_EXPLAINED_VARIANCE_RETURNS_SUM_KEY = (
    f"{CRITIC_EXPLAINED_VARIANCE_STATS_PREFIX}returns_sum"
)
CRITIC_EXPLAINED_VARIANCE_RETURNS_SQ_SUM_KEY = (
    f"{CRITIC_EXPLAINED_VARIANCE_STATS_PREFIX}returns_sq_sum"
)
CRITIC_EXPLAINED_VARIANCE_ERRORS_SUM_KEY = (
    f"{CRITIC_EXPLAINED_VARIANCE_STATS_PREFIX}errors_sum"
)
CRITIC_EXPLAINED_VARIANCE_ERRORS_SQ_SUM_KEY = (
    f"{CRITIC_EXPLAINED_VARIANCE_STATS_PREFIX}errors_sq_sum"
)
CRITIC_EXPLAINED_VARIANCE_STAT_KEYS = (
    CRITIC_EXPLAINED_VARIANCE_COUNT_KEY,
    CRITIC_EXPLAINED_VARIANCE_RETURNS_SUM_KEY,
    CRITIC_EXPLAINED_VARIANCE_RETURNS_SQ_SUM_KEY,
    CRITIC_EXPLAINED_VARIANCE_ERRORS_SUM_KEY,
    CRITIC_EXPLAINED_VARIANCE_ERRORS_SQ_SUM_KEY,
)


def compute_split_num(num, split_num):
    return math.lcm(num, split_num) // split_num


def compute_critic_explained_variance_stats(
    returns: torch.Tensor,
    values: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute sufficient statistics for critic explained variance."""
    returns = returns.detach().float()
    values = values.detach().float()
    if loss_mask is not None:
        mask = loss_mask.to(device=returns.device, dtype=torch.bool)
        if mask.shape != returns.shape:
            mask = torch.broadcast_to(mask, returns.shape)
        returns = returns[mask]
        values = values[mask]
    else:
        returns = returns.reshape(-1)
        values = values.reshape(-1)

    errors = returns - values
    count = torch.tensor(float(returns.numel()), device=returns.device)
    return {
        CRITIC_EXPLAINED_VARIANCE_COUNT_KEY: count,
        CRITIC_EXPLAINED_VARIANCE_RETURNS_SUM_KEY: returns.sum(),
        CRITIC_EXPLAINED_VARIANCE_RETURNS_SQ_SUM_KEY: (returns * returns).sum(),
        CRITIC_EXPLAINED_VARIANCE_ERRORS_SUM_KEY: errors.sum(),
        CRITIC_EXPLAINED_VARIANCE_ERRORS_SQ_SUM_KEY: (errors * errors).sum(),
    }


def compute_critic_explained_variance_from_stats(
    stats: dict[str, float | torch.Tensor],
) -> torch.Tensor:
    """Compute critic explained variance from summed sufficient statistics."""
    tensor_value = next(
        (v for v in stats.values() if isinstance(v, torch.Tensor)), None
    )
    device = tensor_value.device if tensor_value is not None else torch.device("cpu")

    def as_tensor(key: str) -> torch.Tensor:
        return torch.as_tensor(stats[key], dtype=torch.float32, device=device)

    count = as_tensor(CRITIC_EXPLAINED_VARIANCE_COUNT_KEY)
    returns_sum = as_tensor(CRITIC_EXPLAINED_VARIANCE_RETURNS_SUM_KEY)
    returns_sq_sum = as_tensor(CRITIC_EXPLAINED_VARIANCE_RETURNS_SQ_SUM_KEY)
    errors_sum = as_tensor(CRITIC_EXPLAINED_VARIANCE_ERRORS_SUM_KEY)
    errors_sq_sum = as_tensor(CRITIC_EXPLAINED_VARIANCE_ERRORS_SQ_SUM_KEY)

    zero = torch.tensor(0.0, device=device)
    nan = torch.tensor(float("nan"), device=device)
    sufficient_stats = torch.stack(
        (
            count,
            returns_sum,
            returns_sq_sum,
            errors_sum,
            errors_sq_sum,
        )
    )
    if not torch.isfinite(sufficient_stats).all():
        return nan
    if count < 2:
        return zero

    returns_centered_sq_sum = returns_sq_sum - returns_sum * returns_sum / count
    if not torch.isfinite(returns_centered_sq_sum):
        return nan
    if returns_centered_sq_sum == 0:
        return zero

    errors_centered_sq_sum = errors_sq_sum - errors_sum * errors_sum / count
    if not torch.isfinite(errors_centered_sq_sum):
        return nan
    return 1 - errors_centered_sq_sum / returns_centered_sq_sum


def pop_critic_explained_variance_stats(
    metrics: dict[str, object],
) -> dict[str, torch.Tensor]:
    """Pop hidden critic explained-variance stats and sum list values."""

    def sum_metric_values(value: object) -> torch.Tensor:
        if isinstance(value, list):
            if not value:
                return torch.tensor(0.0)
            tensors = [
                item.detach()
                if isinstance(item, torch.Tensor)
                else torch.as_tensor(item)
                for item in value
            ]
            return torch.stack([tensor.float() for tensor in tensors]).sum()
        if isinstance(value, torch.Tensor):
            return value.detach().float()
        return torch.as_tensor(value, dtype=torch.float32)

    stats = {}
    for key in CRITIC_EXPLAINED_VARIANCE_STAT_KEYS:
        if key in metrics:
            stats[key] = sum_metric_values(metrics.pop(key))
    if stats:
        metrics.pop(CRITIC_EXPLAINED_VARIANCE_KEY, None)
    return stats


def _normalize_metric_shard(shard: object) -> torch.Tensor:
    """One rank's metric -> 1D float tensor on CPU."""
    if shard is None:
        return torch.tensor([], dtype=torch.float32)
    if isinstance(shard, torch.Tensor):
        return shard.detach().cpu().reshape(-1).float()
    if isinstance(shard, list):
        if not shard:
            return torch.tensor([], dtype=torch.float32)
        return torch.cat([x.detach().cpu().reshape(-1).float() for x in shard], dim=0)
    return torch.as_tensor(shard, dtype=torch.float32).cpu().reshape(-1)


def count_trajectories(metrics_dict):
    """
    Count the total number of trajectories from metrics dictionary.

    Args:
        metrics_dict: Dictionary of metrics where each value is a tensor after concatenation.
                     Each tensor's first dimension represents the number of trajectories.

    Returns:
        int: Total number of trajectories. If metrics_dict is empty, returns 0.
    """
    if not metrics_dict:
        return 0

    # Per-chunk diagnostics can have a different count from completed episodes.
    first_key = (
        "success_once" if "success_once" in metrics_dict else next(iter(metrics_dict))
    )
    first_tensor = metrics_dict[first_key]

    if isinstance(first_tensor, torch.Tensor):
        return first_tensor.shape[0]
    elif isinstance(first_tensor, list):
        # If it's a list of tensors, sum up all trajectory counts
        return sum(
            t.shape[0] if isinstance(t, torch.Tensor) else len(t) for t in first_tensor
        )
    else:
        raise TypeError(f"Unsupported tensor type: {type(first_tensor)}")


def compute_evaluate_metrics(eval_metrics_list):
    """
    List of evaluate metrics, list length stands for rollout process

    Returns:
        dict: Aggregated metrics with mean values and trajectory count
    """
    if not eval_metrics_list:
        return {}

    all_eval_metrics = {}
    env_info_keys: set[str] = set()
    for eval_metrics in eval_metrics_list:
        env_info_keys.update(eval_metrics.keys())

    # Count trajectories from each process
    trajectory_counts = []
    for eval_metrics in eval_metrics_list:
        count = count_trajectories(eval_metrics)
        trajectory_counts.append(count)

    for env_info_key in env_info_keys:
        metric = [
            eval_metrics[env_info_key]
            for eval_metrics in eval_metrics_list
            if env_info_key in eval_metrics
        ]
        if metric:
            all_eval_metrics[env_info_key] = metric

    for key in all_eval_metrics:
        shards = [_normalize_metric_shard(s) for s in all_eval_metrics[key]]
        stacked = torch.concat(shards).float()
        all_eval_metrics[key] = (
            stacked.mean().detach().cpu().numpy()
            if stacked.numel() > 0
            else np.asarray(0.0, dtype=np.float64)
        )

    # Add total trajectory count to metrics
    all_eval_metrics["num_trajectories"] = sum(trajectory_counts)

    return all_eval_metrics


def aggregate_tabero_dsrl_env_metrics(
    env_metrics_list: Sequence[dict],
) -> dict[str, float]:
    """Aggregate exact terminal-safe episode rows and reward sufficient stats."""

    has_reward_audit = any(
        any(key.startswith("reward_audit/") for key in metrics)
        for metrics in env_metrics_list
    )
    if not has_reward_audit:
        return {}

    audit_shards = []
    for metrics in env_metrics_list:
        audit = {}
        for field in DSRL_REWARD_AUDIT_FIELDS:
            key = f"reward_audit/{field}"
            if key not in metrics:
                raise ValueError(
                    f"Tabero DSRL env metrics are missing reward audit field {key!r}."
                )
            values = _normalize_metric_shard(metrics[key])
            audit[field] = (
                float(values.max().item())
                if field == "reward_max" and values.numel()
                else float(values.sum().item())
            )
        audit_shards.append(audit)
    combined_audit = combine_dsrl_reward_audits(audit_shards)

    def concatenate(field: str) -> torch.Tensor:
        shards = [
            _normalize_metric_shard(metrics[field])
            for metrics in env_metrics_list
            if field in metrics
        ]
        return torch.cat(shards, dim=0) if shards else torch.empty(0)

    success = concatenate("success_once")
    returns = concatenate("return")
    rewards = concatenate("reward")
    reward_sums = concatenate("reward_sum")
    terminal_step_rewards = concatenate("terminal_step_reward")
    terminations = concatenate("termination")
    truncations = concatenate("truncation")
    condition_ids = concatenate("condition_id")
    firm_success = concatenate("firm_success_once")
    firm_returns = concatenate("firm_return")
    firm_squeeze = concatenate("firm_squeeze_pred_mean")
    gentle_success = concatenate("gentle_success_once")
    gentle_returns = concatenate("gentle_return")
    gentle_squeeze = concatenate("gentle_squeeze_pred_mean")
    trajectory_mean_measured_squeeze = concatenate("trajectory_mean_measured_squeeze")
    force_valid_sample_count = concatenate("force_valid_sample_count")

    has_trajectory_force_metrics = bool(
        trajectory_mean_measured_squeeze.numel() or force_valid_sample_count.numel()
    )
    if has_trajectory_force_metrics and (
        trajectory_mean_measured_squeeze.numel() != success.numel()
        or force_valid_sample_count.numel() != success.numel()
    ):
        raise ValueError(
            "Tabero trajectory force metrics must align with exact episode rows; "
            f"success={success.numel()}, "
            f"mean_force={trajectory_mean_measured_squeeze.numel()}, "
            f"sample_count={force_valid_sample_count.numel()}."
        )

    completed_episode_count = int(success.numel())
    firm_episode_count = int(firm_success.numel())
    gentle_episode_count = int(gentle_success.numel())
    nonfirm_episode_count = int(condition_ids.ne(0).sum().item())

    def mean_or_nan(values: torch.Tensor) -> float:
        return float(values.mean().item()) if values.numel() else float("nan")

    metrics = {
        f"reward_audit/{field}": value for field, value in combined_audit.items()
    }
    metrics.update(
        {
            "num_trajectories": completed_episode_count,
            "completed_episode_count": completed_episode_count,
            "firm_episode_count": firm_episode_count,
            "firm_success_count": float(firm_success.sum().item()),
            "firm_success_rate": mean_or_nan(firm_success),
            "firm_success_once": mean_or_nan(firm_success),
            "firm_return": mean_or_nan(firm_returns),
            "firm_return_sum": float(firm_returns.sum().item()),
            "firm_squeeze_pred_mean": mean_or_nan(firm_squeeze),
            "gentle_episode_count": gentle_episode_count,
            "gentle_success_count": float(gentle_success.sum().item()),
            "gentle_success_rate": mean_or_nan(gentle_success),
            "gentle_success_once": mean_or_nan(gentle_success),
            "gentle_return": mean_or_nan(gentle_returns),
            "gentle_squeeze_pred_mean": mean_or_nan(gentle_squeeze),
            "nonfirm_episode_count": nonfirm_episode_count,
            "success_once": mean_or_nan(success),
            "return": mean_or_nan(returns),
            "reward": mean_or_nan(rewards),
            "reward_sum": float(reward_sums.sum().item()),
            "terminal_step_reward_sum": float(terminal_step_rewards.sum().item()),
            "termination_count": float(terminations.sum().item()),
            "truncation_count": float(truncations.sum().item()),
        }
    )
    if has_trajectory_force_metrics:
        force_valid_sample_count = force_valid_sample_count.to(torch.float32)
        invalid_force_rows = (force_valid_sample_count < 0) | (
            (force_valid_sample_count > 0)
            & ~torch.isfinite(trajectory_mean_measured_squeeze)
        )
        if invalid_force_rows.any():
            raise ValueError("Tabero trajectory force metric rows are invalid.")

        successful_rows = success.to(torch.bool)
        successful_force_rows = (
            successful_rows
            & (force_valid_sample_count > 0)
            & torch.isfinite(trajectory_mean_measured_squeeze)
        )
        successful_force_means = trajectory_mean_measured_squeeze[
            successful_force_rows
        ].to(torch.float32)
        successful_sample_counts = force_valid_sample_count[successful_rows]
        metrics.update(
            {
                "success_trajectory_mean_measured_squeeze": mean_or_nan(
                    successful_force_means
                ),
                "success_trajectory_mean_measured_squeeze_median": (
                    float(torch.quantile(successful_force_means, 0.5).item())
                    if successful_force_means.numel()
                    else float("nan")
                ),
                "success_force_valid_sample_count": mean_or_nan(
                    successful_sample_counts
                ),
                "success_force_trajectory_count": int(
                    successful_force_rows.sum().item()
                ),
                "success_force_missing_trajectory_count": int(
                    successful_rows.sum().item() - successful_force_rows.sum().item()
                ),
            }
        )
    return metrics


def compute_rollout_metrics(data_buffer: dict) -> dict:
    rollout_metrics = {}
    loss_mask = data_buffer.get("loss_mask", None)

    def reduce_metrics(values: torch.Tensor) -> tuple[float, float, float]:
        from rlinf.scheduler.worker.worker import Worker

        device = Worker.torch_platform.current_device()

        if values.numel() == 0:
            count = torch.tensor(0.0, device=device, dtype=torch.float32)
            values_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
            min_value = float("inf")
            max_value = float("-inf")
        else:
            values = values.to(device)
            count = torch.tensor(
                values.numel(), device=values.device, dtype=torch.float32
            )
            values_sum = values.to(dtype=torch.float32).sum()
            max_value = torch.max(values).detach().item()
            min_value = torch.min(values).detach().item()

        reduce_sum_count = torch.stack([values_sum, count])
        reduce_min_max = torch.as_tensor(
            [-min_value, max_value],
            device=device,
            dtype=torch.float32,
        )
        torch.distributed.all_reduce(
            reduce_sum_count, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(reduce_min_max, op=torch.distributed.ReduceOp.MAX)
        reduced_sum, reduced_count = reduce_sum_count.tolist()
        reduced_min, reduced_max = reduce_min_max.tolist()

        if reduced_count <= 0:
            return float("nan"), float("nan"), float("nan")
        return reduced_sum / reduced_count, -reduced_min, reduced_max

    def valid_values(values: torch.Tensor) -> torch.Tensor:
        if loss_mask is None:
            return values.reshape(-1)
        mask = loss_mask.to(device=values.device, dtype=torch.bool)
        if mask.ndim == values.ndim - 1:
            mask = mask.unsqueeze(-1)
        if mask.shape != values.shape:
            mask = torch.broadcast_to(mask, values.shape)
        return values[mask]

    if "rewards" in data_buffer:
        rewards = data_buffer["rewards"]
        rewards = valid_values(rewards)
        mean_rewards, _, _ = reduce_metrics(rewards)

        rewards_metrics = {
            "rewards": mean_rewards,
        }
        rollout_metrics.update(rewards_metrics)

    if "advantages" in data_buffer:
        advantages = data_buffer["advantages"]
        advantages = valid_values(advantages)
        mean_adv, min_adv, max_adv = reduce_metrics(advantages)

        advantages_metrics = {
            "advantages_mean": mean_adv,
            "advantages_max": max_adv,
            "advantages_min": min_adv,
        }
        rollout_metrics.update(advantages_metrics)

    if data_buffer.get("returns", None) is not None:
        returns = data_buffer["returns"]
        returns = valid_values(returns)
        mean_ret, min_ret, max_ret = reduce_metrics(returns)

        returns_metrics = {
            "returns_mean": mean_ret,
            "returns_max": max_ret,
            "returns_min": min_ret,
        }
        rollout_metrics.update(returns_metrics)

    primitive_loss_mask = data_buffer.get("primitive_loss_mask")
    if primitive_loss_mask is not None:
        from rlinf.scheduler.worker.worker import Worker

        primitive_loss_mask = primitive_loss_mask.to(dtype=torch.bool)
        if primitive_loss_mask.ndim != 3:
            raise ValueError(
                "primitive_loss_mask must have shape [T, B, action_chunk]; "
                f"got {tuple(primitive_loss_mask.shape)}."
            )
        boundary_counts = summarize_primitive_loss_mask(primitive_loss_mask)
        local_boundary_counts = torch.tensor(
            [
                boundary_counts["valid_primitive_actions"],
                boundary_counts["masked_post_done_actions"],
                boundary_counts["partial_chunk_count"],
            ],
            device=Worker.torch_platform.current_device(),
            dtype=torch.float64,
        )
        torch.distributed.all_reduce(
            local_boundary_counts, op=torch.distributed.ReduceOp.SUM
        )
        rollout_metrics.update(
            {
                "boundary/valid_primitive_actions": float(
                    local_boundary_counts[0].item()
                ),
                "boundary/masked_post_done_actions": float(
                    local_boundary_counts[1].item()
                ),
                "boundary/partial_chunk_count": float(local_boundary_counts[2].item()),
            }
        )

    return rollout_metrics


def append_to_dict(data, new_data):
    for key, val in new_data.items():
        if key not in data:
            data[key] = []
        data[key].append(val)


def compute_loss_mask(dones):
    _, actual_bsz, num_action_chunks = dones.shape
    n_chunk_step = dones.shape[0] - 1
    flattened_dones = dones.transpose(1, 2).reshape(
        -1, actual_bsz
    )  # [(n_chunk_step + 1) * num_action_chunks, rollout_epoch x bsz]
    flattened_dones = flattened_dones[
        -(n_chunk_step * num_action_chunks + 1) :
    ]  # [n_steps+1, actual-bsz]
    flattened_loss_mask = (flattened_dones.cumsum(dim=0) == 0)[
        :-1
    ]  # [n_steps, actual-bsz]

    loss_mask = flattened_loss_mask.reshape(n_chunk_step, num_action_chunks, actual_bsz)
    loss_mask = loss_mask.transpose(
        1, 2
    )  # [n_chunk_step, actual_bsz, num_action_chunks]

    loss_mask_sum = loss_mask.sum(dim=(0, 2), keepdim=True)  # [1, bsz, 1]
    loss_mask_sum = loss_mask_sum.expand_as(loss_mask)

    return loss_mask, loss_mask_sum


def compute_embodied_loss_masks(
    dones: torch.Tensor,
    *,
    reward_type: str,
    use_primitive_prefix_logprobs: bool,
) -> dict[str, torch.Tensor]:
    """Build aligned primitive and macro masks for embodied policy updates.

    ``compute_loss_mask`` accounts for RLinf's leading bootstrap done row. Its
    primitive mask includes the first done action and excludes every later
    action. Boundary-safe chunk-level PPO keeps this prefix for log-probability
    reduction while using one macro loss sample for every non-empty chunk.
    """

    primitive_loss_mask, primitive_loss_mask_sum = compute_loss_mask(dones)
    if reward_type != "chunk_level":
        result = {
            "loss_mask": primitive_loss_mask,
            "loss_mask_sum": primitive_loss_mask_sum,
        }
        if use_primitive_prefix_logprobs:
            result["primitive_loss_mask"] = primitive_loss_mask
        return result

    chunk_loss_mask = primitive_loss_mask.any(dim=-1, keepdim=True)
    if use_primitive_prefix_logprobs:
        chunk_loss_count = chunk_loss_mask.sum(dim=(0, 2), keepdim=True)
        chunk_loss_mask_sum = chunk_loss_count.expand_as(chunk_loss_mask)
        return {
            "primitive_loss_mask": primitive_loss_mask,
            "chunk_loss_mask": chunk_loss_mask,
            "loss_mask": chunk_loss_mask,
            "loss_mask_sum": chunk_loss_mask_sum,
        }

    return {
        "loss_mask": chunk_loss_mask,
        "loss_mask_sum": primitive_loss_mask_sum[..., -1:],
    }


def summarize_primitive_loss_mask(
    primitive_loss_mask: torch.Tensor,
) -> dict[str, int]:
    primitive_loss_mask = primitive_loss_mask.to(dtype=torch.bool)
    if primitive_loss_mask.ndim != 3:
        raise ValueError(
            "primitive_loss_mask must have shape [T, B, action_chunk]; "
            f"got {tuple(primitive_loss_mask.shape)}."
        )
    valid_per_chunk = primitive_loss_mask.sum(dim=-1)
    return {
        "valid_primitive_actions": int(primitive_loss_mask.sum().item()),
        "masked_post_done_actions": int((~primitive_loss_mask).sum().item()),
        "partial_chunk_count": int(
            ((valid_per_chunk > 0) & (valid_per_chunk < primitive_loss_mask.shape[-1]))
            .sum()
            .item()
        ),
    }


def print_metrics_table(
    step: int,
    total_steps: int,
    start_time: float,
    metrics: dict,
    start_step: int = 0,
    log_path: str | None = None,
):
    """Print training metrics in a simple, fast formatted table.

    The rendered table is written to stdout and, when ``log_path`` is given,
    also appended to ``<log_path>/metrics.log``.
    """
    # Accumulate the table into lines so the exact same rendering goes to both
    # stdout and the log file.
    lines: list[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)

    # Calculate progress info
    progress = (step + 1) / total_steps * 100
    elapsed_time = time.time() - start_time
    steps_done = step + 1 - start_step
    eta_seconds = (
        elapsed_time / steps_done * (total_steps - step - 1) if steps_done > 0 else 0
    )

    def format_time(seconds):
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"

    # Format elapsed time and ETA
    elapsed_str = format_time(elapsed_time)
    eta_str = format_time(eta_seconds)

    # Create progress bar
    bar_width = 40
    filled = int(bar_width * progress / 100)
    bar = "█" * filled + "░" * (bar_width - filled)

    # Print header with progress
    total_width = 120

    def _fit_line(text: str, width: int) -> str:
        if len(text) <= width:
            return text + (" " * (width - len(text)))
        if width <= 1:
            return text[:width]
        return text[: width - 1] + "…"

    def _fit_cell(text: str, width: int) -> str:
        return _fit_line(text, width)

    def _print_section_title(title: str) -> None:
        title_text = f" {title} "
        padding = total_width - 2 - len(title_text)
        left = padding // 2
        right = padding - left
        emit(f"├{'─' * left}{title_text}{'─' * right}┤")

    emit(f"\n╭{'─' * (total_width - 2)}╮")
    _print_section_title("Metric Table")

    # First line: Global Step and Progress
    step_str = f"Global Step: {step + 1:4d}/{total_steps}"
    progress_str = f"Progress: {bar} │ {progress:5.1f}%"
    line1 = f"│ {step_str} │ {progress_str}"
    line1 = _fit_line(line1, total_width - 2)
    emit(f"{line1} │")

    # Second line: Time information
    elapsed_str_formatted = f"Elapsed: {elapsed_str}"
    eta_str_formatted = f"ETA: {eta_str}"
    step_time_str = f"Step Time: {elapsed_time / steps_done:.3f}s"
    line2 = f"│ {elapsed_str_formatted} │ {eta_str_formatted} │ {step_time_str}"
    line2 = _fit_line(line2, total_width - 2)
    emit(f"{line2} │")

    # Group metrics by category
    categories = {
        "Time": {},
        "Environment": {},
        "Rollout": {},
        "Evaluation": {},
        "Replay Buffer": {},
        "Training/Actor": {},
        "Training/Critic": {},
        "Training/Other": {},
    }

    for key, value in metrics.items():
        if "/" in key:
            category, metric_name = key.split("/", 1)
            category_map = {
                "time": "Time",
                "env": "Environment",
                "rollout": "Rollout",
                "eval": "Evaluation",
                "replay_buffer": "Replay Buffer",
            }
            if category in category_map:
                categories[category_map[category]][metric_name] = value
            elif category == "train":
                if metric_name.startswith("actor/"):
                    categories["Training/Actor"][metric_name] = value
                elif metric_name.startswith("critic/"):
                    categories["Training/Critic"][metric_name] = value
                elif metric_name.startswith("replay_buffer/"):
                    categories["Replay Buffer"][
                        metric_name.replace("replay_buffer/", "")
                    ] = value
                else:
                    categories["Training/Other"][metric_name] = value

    # Print metrics by category - 3 metrics per row
    table_width = total_width  # Match header width
    base_col_width = (table_width - 4) // 3
    remainder = (table_width - 4) - (base_col_width * 3)
    col_widths = [
        base_col_width + (1 if remainder > 0 else 0),
        base_col_width + (1 if remainder > 1 else 0),
        base_col_width,
    ]

    for category_name, category_metrics in categories.items():
        if category_metrics:
            _print_section_title(category_name)
            # Blank line before metrics (except Global Step section, which is separate)
            emit(f"│{' ' * (table_width - 2)}│")

            # Sort metrics for consistent output
            sorted_metrics = sorted(category_metrics.items())

            # Print in 3-column layout
            for i in range(0, len(sorted_metrics), 3):
                # Get up to 3 metrics for this row
                row_metrics = []
                for j in range(3):
                    if i + j < len(sorted_metrics):
                        metric_name, metric_value = sorted_metrics[i + j]

                        # Format value
                        if isinstance(metric_value, float):
                            if abs(metric_value) < 0.001 and metric_value != 0:
                                formatted_value = f"{metric_value:.2e}"
                            elif abs(metric_value) < 0.01:
                                formatted_value = f"{metric_value:.4f}"
                            elif abs(metric_value) > 10000:
                                formatted_value = f"{metric_value:.2e}"
                            elif abs(metric_value) > 100:
                                formatted_value = f"{metric_value:.1f}"
                            else:
                                formatted_value = f"{metric_value:.3f}"
                        else:
                            formatted_value = str(metric_value)

                        display = f"{metric_name}={formatted_value}"
                        row_metrics.append(display)
                    else:
                        row_metrics.append("")

                # Create the line with exactly 3 columns
                line = (
                    f"│{_fit_cell(row_metrics[0], col_widths[0])}"
                    f"│{_fit_cell(row_metrics[1], col_widths[1])}"
                    f"│{_fit_cell(row_metrics[2], col_widths[2])}│"
                )
                emit(line)

            # Section separator (minimal)
            emit(f"│{' ' * (table_width - 2)}│")

    # Bottom border
    emit(f"╰{'─' * (table_width - 2)}╯")

    emit()

    table = "\n".join(lines)
    print(table)
    if log_path:
        os.makedirs(log_path, exist_ok=True)
        with open(os.path.join(log_path, "metrics.log"), "a") as metrics_file:
            metrics_file.write(table + "\n")
