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

"""Task metrics with synthetic episodes and real CPU collectives."""

import inspect
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from rlinf.utils.dsrl_reward import empty_dsrl_reward_audit
from rlinf.utils.metric_utils import (
    compute_evaluate_metrics as compute_legacy_evaluate_metrics,
)
from rlinf.utils.metric_utils import (
    compute_rollout_metrics as compute_legacy_rollout_metrics,
)
from rlinf.utils.tabero_multi_task_metrics import (
    compute_task_evaluate_metrics as compute_evaluate_metrics,
)
from rlinf.utils.tabero_multi_task_metrics import (
    compute_task_rollout_metrics,
)


def task_fields(**tasks):
    return {
        f"multi_task/{name}/{key}": value
        for name, metrics in tasks.items()
        for key, value in metrics.items()
    }


def finalize(env_metrics):
    return {
        key: torch.cat(value, dim=0).contiguous().cpu()
        for key, value in env_metrics.items()
    }


def compute_rollout_metrics(batch, task_names=None):
    # The production actor retains the shared metrics and adds Tabero statistics.
    from unittest.mock import patch

    from rlinf.scheduler.worker.worker import Worker

    with patch.object(
        Worker, "torch_platform", SimpleNamespace(current_device=lambda: "cpu")
    ):
        metrics = compute_legacy_rollout_metrics(batch)
    if task_names is not None:
        extra = compute_task_rollout_metrics(batch, task_names)
        assert not (metrics.keys() & extra.keys())
        metrics.update(extra)
    return metrics


def episodes(returns, *, forces=None, counts=None):
    returns = torch.tensor(returns, dtype=torch.float32)
    n = returns.numel()
    result = {
        "env_index": torch.arange(n),
        "success_once": (returns > 0).float(),
        "return": returns,
        "episode_len": torch.full((n,), 10.0),
        "reward": returns / 10,
        "termination": returns > 0,
        "truncation": returns == 0,
    }
    if forces is not None:
        result.update(
            trajectory_mean_measured_squeeze=torch.tensor(forces, dtype=torch.float32),
            force_valid_sample_count=torch.tensor(counts, dtype=torch.int64),
            force_bonus=torch.full((n,), 0.25),
        )
    return result


def projected(records):
    return {key: value for key, value in records.items() if key != "env_index"}


def test_task_pooling_preserves_global_and_uses_episode_denominators():
    a = projected(episodes([1, 2, 3], forces=[2, 4, 6], counts=[4, 8, 12]))
    b = projected(episodes([10], forces=[20], counts=[4]))
    c = projected(episodes([0, 0]))  # A different task has force bonus disabled.
    workers = [
        {**a, **task_fields(easy=a, hard=c)},
        {**b, **task_fields(easy=b)},
    ]
    baseline = compute_legacy_evaluate_metrics([a, b])
    metrics = compute_evaluate_metrics(workers)
    for key, value in baseline.items():
        assert metrics[key] == pytest.approx(value)
    prefix = "multi_task/easy/"
    assert metrics[prefix + "return"] == 4  # Not mean(2, 10).
    assert metrics[prefix + "reward"] == pytest.approx(0.4)
    assert metrics[prefix + "num_trajectories"] == 4
    assert metrics[prefix + "trajectory_mean_measured_squeeze_mean"] == 8
    assert metrics[prefix + "trajectory_mean_measured_squeeze_median"] == 5
    assert metrics[prefix + "force_valid_sample_count_mean"] == 7
    assert metrics[prefix + "force_bonus_sum"] == 1
    assert metrics[prefix + "force_bonus_max"] == 0.25
    assert metrics["multi_task/hard/num_trajectories"] == 2
    assert not any(key.startswith("multi_task/hard/force") for key in metrics)
    assert "multi_task/easy/return" in workers[0]  # Aggregation is non-mutating.
    assert not any(key.startswith("_") for key in metrics)


def test_empty_task_and_contactless_episodes_have_no_fabricated_force_mean():
    empty = projected(episodes([]))
    empty["chunk_boundary/done_envs"] = torch.tensor([1, 2])
    contactless = projected(episodes([0], forces=[0], counts=[0]))
    metrics = compute_evaluate_metrics(
        [task_fields(empty=empty, contactless=contactless)]
    )
    assert metrics["multi_task/empty/num_trajectories"] == 0
    assert "multi_task/empty/return" not in metrics
    assert metrics["multi_task/empty/chunk_boundary/done_envs"] == 3
    assert metrics["multi_task/contactless/force_missing_trajectory_count"] == 1
    assert "multi_task/contactless/trajectory_mean_measured_squeeze_mean" not in metrics


def test_task_reward_audit_sums_counts_and_takes_global_max():
    shards = []
    for reward in (2.0, 7.0):
        row = projected(episodes([reward]))
        row.update(
            condition_id=torch.tensor([0]),
            reward_sum=torch.tensor([reward]),
            terminal_step_reward=torch.tensor([reward]),
            firm_success_once=torch.ones(1),
            firm_return=torch.tensor([reward]),
            firm_squeeze_pred_mean=torch.tensor([3.0]),
        )
        audit = empty_dsrl_reward_audit()
        audit.update(reward_sum=reward, reward_max=reward, transition_count=1)
        row.update({f"reward_audit/{k}": torch.tensor([v]) for k, v in audit.items()})
        shards.append(task_fields(easy=row))
    metrics = compute_evaluate_metrics(shards)
    assert metrics["multi_task/easy/reward_audit/reward_sum"] == 9
    assert metrics["multi_task/easy/reward_audit/reward_max"] == 7
    assert metrics["multi_task/easy/reward_audit/transition_count"] == 2
    assert metrics["multi_task/easy/firm_return"] == 4.5


def worker_fixture(rank=0):
    from rlinf.workers.env.env_worker import EnvWorker

    worker = object.__new__(EnvWorker)
    worker._rank, worker.stage_num = rank, 2
    worker.multi_task_enabled = True
    tasks = [{"name": name, "init_params": {}} for name in ("easy", "hard", "third")]
    worker.cfg = OmegaConf.create(
        {
            "env": {
                split: {
                    "multi_task": {"tasks": tasks},
                    "env_type": "isaaclab",
                    "auto_reset": False,
                    "ignore_terminations": False,
                }
                for split in ("train", "eval")
            }
        }
    )
    worker._multi_task_seen = [set(), set()]
    worker._multi_task_eval_seen = [set(), set()]
    worker._multi_task_counts = torch.zeros((2, 3), dtype=torch.int64)
    worker._multi_task_eval_counts = torch.zeros((2, 3), dtype=torch.int64)
    return worker


@pytest.mark.parametrize("evaluation", [False, True])
def test_worker_stage_task_assignment_dedup_and_epoch_reset(evaluation):
    results = []
    for rank in (0, 1):
        worker = worker_fixture(rank)
        accumulated = {}
        for epoch in range(2):
            if evaluation:
                worker._multi_task_eval_seen = [set(), set()]
            else:
                worker._multi_task_seen = [set(), set()]
            for stage in (0, 1):
                records = episodes([rank + stage + 1, 0], forces=[4, 0], counts=[4, 0])
                for _ in range(2):  # A repeated terminal record must not double count.
                    selected = worker._record_multi_task_episodes(
                        stage, records, evaluation=evaluation
                    )
                    info = {"chunk_boundary/done_envs": torch.ones(1)}
                    worker._attach_task_env_metrics(
                        info, selected, stage, realworld=True, evaluation=evaluation
                    )
                    worker.record_env_metrics(accumulated, info)
        counts = (
            worker._multi_task_eval_counts if evaluation else worker._multi_task_counts
        )
        result = finalize(accumulated)
        for i, task in enumerate(("easy", "hard", "third")):
            assert (
                len(result.get(f"multi_task/{task}/success_once", [])) == counts[1, i]
            )
        results.append(result)
        unused = (
            worker._multi_task_counts if evaluation else worker._multi_task_eval_counts
        )
        assert unused.sum() == 0
    metrics = compute_evaluate_metrics(results)
    assert metrics["multi_task/easy/num_trajectories"] == 8
    assert metrics["multi_task/hard/num_trajectories"] == 4
    assert metrics["multi_task/third/num_trajectories"] == 4
    assert metrics["multi_task/easy/return"] == 1


@pytest.mark.parametrize("evaluation", [False, True])
def test_interact_and_evaluate_attach_task_envelope(monkeypatch, evaluation):
    from rlinf.workers.env import env_worker

    worker = worker_fixture()
    worker.model_cfg = OmegaConf.create(
        {
            "model_type": "openpi",
            "num_action_chunks": 2,
            "action_dim": 13,
        }
    )
    monkeypatch.setattr(
        env_worker, "prepare_actions", lambda **kwargs: kwargs["raw_chunk_actions"]
    )
    worker._build_chunk_final_obs = lambda *_: None
    records = episodes([1, 0], forces=[4, 0], counts=[4, 0])

    def chunk_step(_):
        return (
            [{}],
            torch.zeros(2, 2),
            torch.zeros(2, 2, dtype=torch.bool),
            torch.zeros(2, 2, dtype=torch.bool),
            [
                {
                    "_realworld_first_episode_records": records,
                    "chunk_boundary_metrics": {"done_envs": 2},
                }
            ],
        )

    worker.env_list = worker.eval_env_list = [
        SimpleNamespace(chunk_step=chunk_step)
    ] * 2
    method = worker.env_evaluate_step if evaluation else worker.env_interact_step
    result = inspect.unwrap(method)(worker, torch.zeros(2, 2, 13), 1)
    metrics = compute_evaluate_metrics([result[1]])
    assert metrics["multi_task/hard/return"] == 0.5
    assert metrics["multi_task/hard/force_bonus_sum"] == 0.5
    assert metrics["multi_task/hard/chunk_boundary/done_envs"] == 2


def _rollout_process(rank, rendezvous, output):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        # Rank 0 has only easy; rank 1 has easy and hard. "empty" is absent globally.
        indices = (
            torch.tensor([[0, 0], [0, 0]])
            if rank == 0
            else torch.tensor([[0, 1], [1, 1]])
        )
        values = (
            torch.tensor([[[1.0], [2.0]], [[3.0], [999.0]]])
            if rank == 0
            else torch.tensor([[[10.0], [20.0]], [[30.0], [40.0]]])
        )
        mask = (
            torch.tensor([[[True], [True]], [[True], [False]]])
            if rank == 0
            else torch.ones(2, 2, 1, dtype=torch.bool)
        )
        batch = {
            "task_indices": indices,
            "rewards": values.repeat(1, 1, 2),
            "advantages": values,
            "returns": values + 100,
            "prev_values": torch.cat([values + 200, torch.full((1, 2, 1), 9999.0)]),
            "loss_mask": mask,
            "primitive_loss_mask": mask.expand(-1, -1, 2).clone(),
        }
        batch["primitive_loss_mask"][0, 0, 1] = False  # A partial terminal chunk.
        baseline = compute_rollout_metrics(batch)
        metrics = compute_rollout_metrics(batch, ["easy", "hard", "empty"])
        for key in baseline:
            assert metrics[key] == baseline[key]
        Path(output, f"rank_{rank}.json").write_text(json.dumps(metrics))
    finally:
        dist.destroy_process_group()


def test_two_rank_rollout_metrics_pool_samples_and_exclude_bootstrap(tmp_path):
    mp.spawn(_rollout_process, args=(str(tmp_path / "rdzv"), str(tmp_path)), nprocs=2)
    metrics = json.loads((tmp_path / "rank_0.json").read_text())
    assert metrics == json.loads((tmp_path / "rank_1.json").read_text())
    assert metrics["multi_task/easy/rewards"] == 4
    assert metrics["multi_task/hard/rewards"] == 30
    assert metrics["multi_task/easy/rewards_count"] == 8
    assert metrics["multi_task/easy/returns_count"] == 4
    assert metrics["multi_task/easy/returns_mean"] == 104
    assert metrics["multi_task/easy/returns_min"] == 101
    assert metrics["multi_task/easy/returns_max"] == 110
    assert metrics["multi_task/hard/values_mean"] == 230
    assert metrics["multi_task/hard/values_max"] == 240
    assert metrics["multi_task/empty/rewards_count"] == 0
    assert "multi_task/empty/rewards" not in metrics
    assert metrics["returns_mean"] == pytest.approx((4 * 104 + 3 * 130) / 7)
    assert metrics["boundary/valid_primitive_actions"] == 12
    assert metrics["boundary/partial_chunk_count"] == 2


def test_multi_epoch_reorder_and_empty_masks(tmp_path):
    from rlinf.workers.actor.fsdp_actor_worker import process_nested_dict_for_adv

    # Each epoch has two action rows and a separate trailing bootstrap value.
    raw = {
        "task_indices": torch.tensor([[0], [0], [1], [1]]),
        "rewards": torch.tensor([1.0, 3.0, 10.0, 30.0]).reshape(4, 1, 1),
        "prev_values": torch.tensor([2.0, 4.0, 999.0, 20.0, 40.0, 999.0]).reshape(
            6, 1, 1
        ),
        "loss_mask": torch.ones(4, 1, 1, dtype=torch.bool),
    }
    batch = process_nested_dict_for_adv(raw, 2)
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'rdzv'}", rank=0, world_size=1
    )
    try:
        metrics = compute_rollout_metrics(batch, ["easy", "hard"])
        assert metrics["multi_task/easy/rewards"] == 2
        assert metrics["multi_task/hard/rewards"] == 20
        assert metrics["multi_task/easy/values_mean"] == 3
        assert metrics["multi_task/hard/values_mean"] == 30
        assert batch["prev_values"].shape == (3, 2, 1)
        batch["loss_mask"].zero_()
        empty = compute_rollout_metrics(batch, ["easy", "hard"])
        assert empty["multi_task/easy/values_count"] == 0
        assert "multi_task/easy/values_mean" not in empty
        assert "multi_task/hard/rewards" not in empty
    finally:
        dist.destroy_process_group()


def test_invalid_rollout_task_alignment_rejected_before_reductions(monkeypatch):
    batch = {
        "task_indices": torch.zeros(2, 1, dtype=torch.int64),
        "rewards": torch.zeros(3, 1, 1),
    }
    with pytest.raises(ValueError, match="after bootstrap"):
        compute_task_rollout_metrics(batch, ["task"])
    batch["task_indices"][:] = 1
    with pytest.raises(ValueError, match="valid action-aligned"):
        compute_task_rollout_metrics(batch, ["task"])


def test_logger_backends_receive_identical_task_keys_and_steps():
    from rlinf.utils.metric_logger import MetricLogger

    logger = object.__new__(MetricLogger)
    logger.log_start_step, logger.per_worker_log = None, False
    logger._all_loggers = []
    logger.logger = {"wandb": Mock(), "tensorboard": Mock()}
    data = {
        "env/" + key: value
        for key, value in compute_evaluate_metrics(
            [task_fields(easy=projected(episodes([1, 0])))]
        ).items()
    }
    logger.log(data, 42)
    for backend in logger.logger.values():
        backend.log.assert_called_once_with(data=data, step=42)
    assert "env/multi_task/easy/return" in data
    assert all(not key.startswith("env/_") for key in data)


def test_libero_condition_subsets_and_success_force_metrics():
    worker = worker_fixture()
    records = episodes([2, 0], forces=[4, 8], counts=[5, 10])
    records.pop("force_bonus")  # The Libero adapter exposes mean/count only.
    records.update(
        condition_id=torch.tensor([0, 1]),
        reward_sum=records["return"].clone(),
        terminal_step_reward=records["return"].clone(),
        squeeze_pred_mean=torch.tensor([3.0, 6.0]),
        task_id=torch.zeros(2, dtype=torch.int64),
        task_shard_id=torch.zeros(2, dtype=torch.int64),
    )
    accumulated = {}
    for _ in range(2):
        selected = worker._record_multi_task_episodes(0, records)
        info = {
            f"reward_audit/{key}": torch.tensor([value])
            for key, value in empty_dsrl_reward_audit().items()
        }
        worker._attach_task_env_metrics(info, selected, 0, realworld=False)
        worker.record_env_metrics(accumulated, info)
    result = compute_evaluate_metrics([finalize(accumulated)])
    assert result["multi_task/easy/num_trajectories"] == 2
    assert result["multi_task/easy/firm_episode_count"] == 1
    assert result["multi_task/easy/gentle_episode_count"] == 1
    assert result["multi_task/easy/firm_squeeze_pred_mean"] == 3
    assert result["multi_task/easy/gentle_squeeze_pred_mean"] == 6
    assert result["multi_task/easy/success_trajectory_mean_measured_squeeze"] == 4
    assert result["multi_task/easy/trajectory_mean_measured_squeeze_mean"] == 6


def test_runner_preserves_task_metrics_in_global_and_worker_views():
    from rlinf.runners.embodied_runner import EmbodiedRunner

    runner = object.__new__(EmbodiedRunner)
    runner.multi_task_controller = object()
    a = projected(episodes([1, 2]))
    b = projected(episodes([9]))
    results = [
        {"rank": 0, "env": {**a, **task_fields(easy=a)}},
        {"rank": 1, "env": {**b, **task_fields(easy=b)}},
    ]
    metrics, ranked = runner._process_ranked_eval_results(results, "env")
    assert metrics["multi_task/easy/return"] == 4
    assert ranked[0]["multi_task/easy/return"] == 1.5
    assert ranked[1]["multi_task/easy/return"] == 9


def test_single_task_runner_uses_shared_reducer(monkeypatch):
    from rlinf.runners.embodied_runner import EmbodiedRunner
    from rlinf.utils import tabero_multi_task_metrics

    runner = object.__new__(EmbodiedRunner)
    runner.multi_task_controller = None
    task_reducer = Mock(
        side_effect=AssertionError("single-task reached Tabero metrics")
    )
    monkeypatch.setattr(
        tabero_multi_task_metrics, "compute_task_evaluate_metrics", task_reducer
    )
    rows = [projected(episodes([1, 3]))]
    assert runner._compute_evaluate_metrics(rows) == compute_legacy_evaluate_metrics(
        rows
    )
    task_reducer.assert_not_called()
