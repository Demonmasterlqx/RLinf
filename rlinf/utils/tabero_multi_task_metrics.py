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

"""Metrics for Tabero's opt-in synchronous multi-task training.

Transport uses flat tensor keys so the shared EnvWorker collection stays unchanged.
"""

from collections.abc import Sequence

import numpy as np
import torch

from rlinf.utils.metric_utils import (
    _normalize_metric_shard,
    aggregate_tabero_dsrl_env_metrics,
    compute_evaluate_metrics,
)

TASK_METRIC_PREFIX = "multi_task/"


def attach_task_env_metrics(
    env_info: dict, records: dict, task_name: str, *, realworld: bool
) -> None:
    """Attach deduplicated episode rows and per-chunk diagnostics as tensors."""
    from rlinf.workers.env.env_worker import (
        realworld_first_episode_records_to_env_info,
        tabero_chunk_episode_records_to_env_info,
    )

    project = (
        realworld_first_episode_records_to_env_info
        if realworld
        else tabero_chunk_episode_records_to_env_info
    )
    metrics = project(records)
    metrics.setdefault("success_once", torch.empty(0))
    metrics.update(
        (key, value)
        for key, value in env_info.items()
        if key.startswith(("chunk_boundary/", "reward_audit/"))
    )
    env_info.update(
        {
            f"{TASK_METRIC_PREFIX}{task_name}/{key}": value
            for key, value in metrics.items()
        }
    )


def compute_task_evaluate_metrics(eval_metrics_list: Sequence[dict]) -> dict:
    """Pool task samples without changing the legacy global reductions."""
    task_shards: dict[str, list[dict]] = {}
    global_shards = []
    for worker_metrics in eval_metrics_list:
        global_metrics, tasks = {}, {}
        for key, value in worker_metrics.items():
            if key.startswith(TASK_METRIC_PREFIX):
                _, name, field = key.split("/", 2)
                tasks.setdefault(name, {})[field] = value
            else:
                global_metrics[key] = value
        global_shards.append(global_metrics)
        for name, metrics in tasks.items():
            task_shards.setdefault(name, []).append(metrics)
    all_eval_metrics = compute_evaluate_metrics(global_shards)
    for name, shards in task_shards.items():
        metrics = compute_evaluate_metrics(shards)
        metrics.update(aggregate_tabero_dsrl_env_metrics(shards))
        field_names = {key for shard in shards for key in shard}
        # Boundary diagnostics are per-chunk event counts, not episode means.
        for key in field_names:
            if key.startswith("chunk_boundary/"):
                metrics[key] = sum(
                    _normalize_metric_shard(shard[key]).sum().item()
                    for shard in shards
                    if key in shard
                )
        empty_fields = {
            key
            for key in field_names
            if not any(
                _normalize_metric_shard(other[key]).numel()
                for other in shards
                if key in other
            )
        }
        for key, value in metrics.items():
            # A task without observations has counts, not fabricated zero means.
            if key not in empty_fields and np.isfinite(value):
                all_eval_metrics[f"multi_task/{name}/{key}"] = value

    return all_eval_metrics


def compute_task_rollout_metrics(data_buffer: dict, task_names: Sequence[str]) -> dict:
    """Return only Tabero task statistics and additional PPO value/count metrics.

    The actor keeps the global metrics from the shared compute_rollout_metrics.
    Task labels are action-aligned; prev_values has one extra bootstrap row.
    """
    rollout_metrics = {}
    loss_mask = data_buffer.get("loss_mask")

    def reduction_device():
        if torch.distributed.get_backend() == "gloo":
            return torch.device("cpu")
        from rlinf.scheduler.worker.worker import Worker

        return Worker.torch_platform.current_device()

    def reduce_metrics(values: torch.Tensor) -> tuple[float, float, float, int]:
        device = reduction_device()

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
            return float("nan"), float("nan"), float("nan"), 0
        return (
            reduced_sum / reduced_count,
            -reduced_min,
            reduced_max,
            int(reduced_count),
        )

    indices = data_buffer.get("task_indices")
    if (
        not task_names
        or len(set(task_names)) != len(task_names)
        or indices is None
        or indices.ndim != 2
        or indices.dtype != torch.int64
        or (indices < 0).any()
        or (indices >= len(task_names)).any()
    ):
        raise ValueError("Rollout metrics require valid action-aligned task indices.")

    def valid_values(
        values: torch.Tensor, task_index: int | None = None
    ) -> torch.Tensor:
        mask = torch.ones_like(values, dtype=torch.bool)
        if loss_mask is not None:
            active = loss_mask.to(device=values.device, dtype=torch.bool)
            if active.ndim == values.ndim - 1:
                active = active.unsqueeze(-1)
            mask &= torch.broadcast_to(active, values.shape)
        if indices is not None:
            if values.shape[:2] != indices.shape:
                raise ValueError(
                    "Rollout metric rows must align with task_indices after bootstrap "
                    f"removal: {tuple(values.shape)} versus {tuple(indices.shape)}."
                )
            if task_index is not None:
                selected = indices.to(values.device).eq(task_index)
                while selected.ndim < values.ndim:
                    selected = selected.unsqueeze(-1)
                mask &= selected
        return values[mask]

    signals = {
        name: data_buffer[name]
        for name in ("rewards", "advantages", "returns")
        if data_buffer.get(name) is not None
    }
    if data_buffer.get("prev_values") is not None:
        values = data_buffer["prev_values"]
        if values.shape[:2] != (indices.shape[0] + 1, indices.shape[1]):
            raise ValueError(
                "PPO values must contain exactly one trailing bootstrap row."
            )
        signals["values"] = values[:-1]

    # Every actor visits every configured task, including locally empty tasks.
    for name, values in signals.items():
        mean, minimum, maximum, count = reduce_metrics(valid_values(values))
        if name == "values":
            rollout_metrics.update(
                {
                    f"{name}_mean": mean,
                    f"{name}_min": minimum,
                    f"{name}_max": maximum,
                }
            )
        rollout_metrics[f"{name}_count"] = count
        for task_index, task_name in enumerate(task_names):
            mean, minimum, maximum, count = reduce_metrics(
                valid_values(values, task_index)
            )
            prefix = f"multi_task/{task_name}/"
            rollout_metrics[f"{prefix}{name}_count"] = count
            if count:
                if name == "rewards":
                    rollout_metrics[f"{prefix}{name}"] = mean
                else:
                    rollout_metrics.update(
                        {
                            f"{prefix}{name}_mean": mean,
                            f"{prefix}{name}_min": minimum,
                            f"{prefix}{name}_max": maximum,
                        }
                    )

    return rollout_metrics
