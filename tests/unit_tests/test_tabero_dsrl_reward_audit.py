# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.utils.dsrl_reward import (
    DSRL_REWARD_AUDIT_FIELDS,
    combine_dsrl_reward_audits,
    compare_tabero_dsrl_reward_audits,
    summarize_dsrl_chunk_rewards,
)
from rlinf.utils.metric_utils import aggregate_tabero_dsrl_env_metrics
from rlinf.workers.env.env_worker import (
    tabero_chunk_episode_records_to_env_info,
)


def _matching_step_metrics(*, success: bool = True):
    rewards = torch.zeros(2, 10)
    terminations = torch.zeros(2, 10, dtype=torch.bool)
    truncations = torch.zeros(2, 10, dtype=torch.bool)
    if success:
        rewards[0, 3] = 1.0
        terminations[0, 3] = True
    else:
        truncations[0, 9] = True
    truncations[1, 9] = True

    audit = summarize_dsrl_chunk_rewards(rewards, terminations, truncations)
    episode_success = torch.tensor([float(success), 0.0])
    episode_returns = episode_success.clone()
    episode_lengths = torch.tensor([4.0 if success else 10.0, 10.0])
    env_metrics = {
        "success_once": [episode_success],
        "return": [episode_returns],
        "episode_len": [episode_lengths],
        "reward": [episode_returns / episode_lengths],
        "reward_sum": [episode_returns],
        "terminal_step_reward": [episode_returns],
        "termination": [torch.tensor([success, False])],
        "truncation": [torch.tensor([not success, True])],
        "condition_id": [torch.zeros(2, dtype=torch.int64)],
        "firm_success_once": [episode_success],
        "firm_return": [episode_returns],
        "firm_squeeze_pred_mean": [torch.tensor([2.0, 3.0])],
    }
    env_metrics.update(
        {
            f"reward_audit/{field}": [torch.tensor([value], dtype=torch.float64)]
            for field, value in audit.items()
        }
    )
    replay_metrics = {
        f"replay_buffer/last_insert_{field}": value for field, value in audit.items()
    }
    return env_metrics, replay_metrics


def test_reward_audit_detects_done_padding_nonfinite_and_unpaired_rewards():
    rewards = torch.tensor([[0.0, 1.0, 2.0, float("nan")]])
    terminations = torch.tensor([[False, True, False, False]])
    truncations = torch.zeros_like(terminations)

    audit = summarize_dsrl_chunk_rewards(rewards, terminations, truncations)

    assert audit["transition_count"] == 1
    assert audit["primitive_count"] == 4
    assert audit["nonzero_primitive_reward_count"] == 2
    assert audit["nonzero_macro_reward_count"] == 1
    assert audit["reward_sum"] == 3
    assert audit["termination_count"] == 1
    assert audit["post_done_nonzero_reward_count"] == 1
    assert audit["reward_without_termination_count"] == 1
    assert audit["nonfinite_reward_count"] == 1


def test_reward_audit_combines_sum_fields_and_reward_max():
    first, first_replay = _matching_step_metrics()
    second, _ = _matching_step_metrics(success=False)
    first_audit = {
        field: first_replay[f"replay_buffer/last_insert_{field}"]
        for field in DSRL_REWARD_AUDIT_FIELDS
    }
    second_audit = {
        field: second[f"reward_audit/{field}"][0].item()
        for field in DSRL_REWARD_AUDIT_FIELDS
    }

    combined = combine_dsrl_reward_audits([first_audit, second_audit])

    assert combined["transition_count"] == 4
    assert combined["primitive_count"] == 40
    assert combined["reward_sum"] == 1
    assert combined["reward_max"] == 1
    assert combined["termination_count"] == 1
    assert combined["truncation_count"] == 3


def test_episode_record_projection_preserves_early_firm_and_gentle_terminals():
    records = {
        "env_index": torch.tensor([3, 7]),
        "primitive_step_index": torch.tensor([1, 8]),
        "condition_id": torch.tensor([0, 1]),
        "termination": torch.tensor([True, False]),
        "truncation": torch.tensor([False, True]),
        "success_once": torch.tensor([1.0, 0.0]),
        "return": torch.tensor([1.0, 0.0]),
        "episode_len": torch.tensor([12.0, 360.0]),
        "reward": torch.tensor([1.0 / 12.0, 0.0]),
        "reward_sum": torch.tensor([1.0, 0.0]),
        "terminal_step_reward": torch.tensor([1.0, 0.0]),
        "squeeze_pred_mean": torch.tensor([2.0, 0.5]),
        "task_id": torch.tensor([0.0, 0.0]),
        "task_shard_id": torch.tensor([1.0, 1.0]),
    }

    projected = tabero_chunk_episode_records_to_env_info(records)

    assert projected["success_once"].tolist() == [1.0, 0.0]
    assert projected["firm_success_once"].tolist() == [1.0]
    assert projected["gentle_success_once"].tolist() == [0.0]
    assert projected["firm_return"].tolist() == [1.0]
    assert projected["gentle_return"].tolist() == [0.0]


def test_exact_env_aggregation_uses_episode_records_and_reward_sufficient_stats():
    env_metrics, replay_metrics = _matching_step_metrics()

    exact = aggregate_tabero_dsrl_env_metrics([env_metrics])
    audit_metrics, mismatches = compare_tabero_dsrl_reward_audits(exact, replay_metrics)

    assert exact["completed_episode_count"] == 2
    assert exact["firm_episode_count"] == 2
    assert exact["firm_success_count"] == 1
    assert exact["firm_success_rate"] == pytest.approx(0.5)
    assert exact["reward"] == pytest.approx(0.125)
    assert exact["reward_sum"] == 1
    assert exact["termination_count"] == 1
    assert exact["truncation_count"] == 1
    assert audit_metrics == {
        "audit/dsrl_reward_match": 1.0,
        "audit/dsrl_reward_mismatch_count": 0.0,
    }
    assert mismatches == ()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reward_sum", 0.0, "reward sum"),
        ("nonzero_primitive_reward_count", 2.0, "count"),
        ("post_done_nonzero_reward_count", 1.0, "must be zero"),
        ("nonfinite_reward_count", 1.0, "must be zero"),
    ],
)
def test_reward_comparison_rejects_lost_duplicate_or_invalid_replay_rewards(
    field,
    value,
    message,
):
    env_metrics, replay_metrics = _matching_step_metrics()
    exact = aggregate_tabero_dsrl_env_metrics([env_metrics])
    replay_metrics[f"replay_buffer/last_insert_{field}"] = value

    audit_metrics, mismatches = compare_tabero_dsrl_reward_audits(exact, replay_metrics)

    assert audit_metrics["audit/dsrl_reward_match"] == 0
    assert any(message in mismatch for mismatch in mismatches)


class _MetricSink:
    def __init__(self):
        self.records = []

    def log(self, metrics, step):
        self.records.append((step, dict(metrics)))


def _audit_runner(*, min_buffer_size: int) -> EmbodiedRunner:
    runner = EmbodiedRunner.__new__(EmbodiedRunner)
    runner.cfg = OmegaConf.create(
        {"algorithm": {"replay_buffer": {"min_buffer_size": min_buffer_size}}}
    )
    runner._dsrl_consecutive_zero_success_steps = 0
    runner.metric_logger = _MetricSink()
    runner.logger = SimpleNamespace(error=lambda _message: None)
    return runner


def test_runner_reward_gate_passes_matching_positive_firm_step():
    env_metrics, replay_metrics = _matching_step_metrics()
    runner = _audit_runner(min_buffer_size=1)

    audit = runner._validate_tabero_dsrl_reward_step(
        step=1,
        env_results=[env_metrics],
        actor_training_metrics=[replay_metrics],
    )

    assert audit["audit/dsrl_reward_match"] == 1


def test_runner_reward_gate_logs_and_blocks_replay_mismatch():
    env_metrics, replay_metrics = _matching_step_metrics()
    replay_metrics["replay_buffer/last_insert_reward_sum"] = 0.0
    runner = _audit_runner(min_buffer_size=1)

    with pytest.raises(RuntimeError, match="reward audit failed"):
        runner._validate_tabero_dsrl_reward_step(
            step=1,
            env_results=[env_metrics],
            actor_training_metrics=[replay_metrics],
        )

    assert any(
        record.get("audit/dsrl_reward_match") == 0
        for _, record in runner.metric_logger.records
    )


def test_runner_reward_gate_blocks_zero_reward_during_warmup():
    env_metrics, replay_metrics = _matching_step_metrics(success=False)
    runner = _audit_runner(min_buffer_size=10)

    with pytest.raises(RuntimeError, match="warm-up produced zero replay reward"):
        runner._validate_tabero_dsrl_reward_step(
            step=0,
            env_results=[env_metrics],
            actor_training_metrics=[replay_metrics],
        )


def test_runner_reward_gate_rejects_nonfirm_episode_metrics():
    env_metrics, replay_metrics = _matching_step_metrics()
    env_metrics["condition_id"] = [torch.tensor([0, 1])]
    env_metrics["firm_success_once"] = [torch.tensor([1.0])]
    env_metrics["firm_return"] = [torch.tensor([1.0])]
    env_metrics["firm_squeeze_pred_mean"] = [torch.tensor([2.0])]
    env_metrics["gentle_success_once"] = [torch.tensor([0.0])]
    env_metrics["gentle_return"] = [torch.tensor([0.0])]
    env_metrics["gentle_squeeze_pred_mean"] = [torch.tensor([3.0])]
    runner = _audit_runner(min_buffer_size=1)

    with pytest.raises(RuntimeError, match="nonfirm_episode_count must be zero"):
        runner._validate_tabero_dsrl_reward_step(
            step=1,
            env_results=[env_metrics],
            actor_training_metrics=[replay_metrics],
        )


def test_runner_reward_gate_blocks_three_post_warmup_zero_success_steps():
    env_metrics, replay_metrics = _matching_step_metrics(success=False)
    runner = _audit_runner(min_buffer_size=1)

    for step in (1, 2):
        audit = runner._validate_tabero_dsrl_reward_step(
            step=step,
            env_results=[env_metrics],
            actor_training_metrics=[replay_metrics],
        )
        assert audit["audit/dsrl_reward_match"] == 1
    with pytest.raises(RuntimeError, match="3 consecutive"):
        runner._validate_tabero_dsrl_reward_step(
            step=3,
            env_results=[env_metrics],
            actor_training_metrics=[replay_metrics],
        )
