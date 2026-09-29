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

"""Synthetic PPO multi-task contracts; no simulator, model or dataset assets."""

import copy
import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from rlinf.algorithms.losses import compute_ppo_actor_critic_loss
from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    Trajectory,
    convert_trajectories_to_batch,
)
from rlinf.utils.ppo_multi_task import (
    DEFAULTS,
    SuccessWeightController,
    batch_weight_stats,
    enabled,
    task_env_cfg,
    task_list,
    weighted_reduce,
)


def controller():
    return SuccessWeightController(["easy", "hard"], DEFAULTS.copy(), {"contract": 1})


def test_disabled_does_not_change_config():
    cfg = OmegaConf.create({"algorithm": {"loss_type": "actor_critic"}})
    before = OmegaConf.to_container(cfg)
    assert not enabled(cfg)
    assert OmegaConf.to_container(cfg) == before


def test_counts_are_pooled_before_rates_and_ema():
    tracker = controller()
    # Unequal worker denominators: easy=9/10, hard=1/10, not mean(1, 0).
    worker_a = torch.tensor([[9, 0], [9, 1]])
    worker_b = torch.tensor([[0, 1], [1, 9]])
    metrics = tracker.update(worker_a + worker_b, 1)
    torch.testing.assert_close(
        tracker.ema, torch.tensor([0.9, 0.1], dtype=torch.float64)
    )
    assert tracker.weights[1] > tracker.weights[0]
    assert metrics["multi_task/easy/episodes"] == 10
    tracker.update(torch.tensor([[0, 0], [10, 0]]), 2)
    torch.testing.assert_close(
        tracker.ema, torch.tensor([0.81, 0.1], dtype=torch.float64)
    )


def test_unobserved_task_uses_neutral_weights_until_initialized():
    tracker = controller()
    tracker.update(torch.tensor([[1, 0], [1, 0]]), 1)
    torch.testing.assert_close(tracker.weights, torch.ones(2))
    tracker.update(torch.tensor([[0, 0], [0, 1]]), 2)
    assert tracker.weights[1] > tracker.weights[0]


@pytest.mark.parametrize("success", [0, 3])
def test_equal_rates_have_equal_weights(success):
    tracker = controller()
    tracker.update(torch.tensor([[success, success], [3, 3]]), 1)
    torch.testing.assert_close(tracker.weights, torch.full((2,), 1.25))


@pytest.mark.parametrize(
    "counts",
    [
        [[2, 0], [1, 1]],
        [[-1, 0], [1, 1]],
        [[0.5, 0], [1, 1]],
        [[float("nan"), 0], [1, 1]],
        [[0, 0, 0], [1, 1, 1]],
    ],
)
def test_bad_counts_fail(counts):
    with pytest.raises(ValueError):
        controller().update(torch.tensor(counts), 1)


def test_round_is_updated_once():
    tracker = controller()
    counts = torch.tensor([[0, 1], [1, 1]])
    tracker.update(counts, 1)
    with pytest.raises(ValueError, match="exactly once"):
        tracker.update(counts, 1)


def test_resume_preserves_next_weights_and_rejects_contract_changes(tmp_path):
    original = controller()
    original.update(torch.tensor([[2, 1], [3, 5]]), 1)
    original.save(str(tmp_path), 1)
    restored = controller()
    restored.restore(str(tmp_path), 1)
    next_counts = torch.tensor([[1, 3], [4, 4]])
    original.update(next_counts, 2)
    restored.update(next_counts, 2)
    torch.testing.assert_close(original.weights, restored.weights, rtol=0, atol=0)
    changed = controller()
    changed.contract = {"contract": 2}
    with pytest.raises(ValueError, match="contract"):
        changed.restore(str(tmp_path), 1)
    with pytest.raises(ValueError, match="step"):
        controller().restore(str(tmp_path), 2)
    with pytest.raises(ValueError, match="requires"):
        controller().restore(str(tmp_path / "missing"), 1)


def test_corrupt_resume_is_rejected(tmp_path):
    tracker = controller()
    tracker.update(torch.tensor([[1, 0], [1, 1]]), 1)
    tracker.save(str(tmp_path), 1)
    path = tmp_path / "multi_task_state.json"
    payload = json.loads(path.read_text())
    payload["ema"][0] = 2
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="state"):
        controller().restore(str(tmp_path), 1)


def test_task_config_isolated_and_legacy_assignment_disabled():
    cfg = OmegaConf.create(
        {
            "init_params": {
                "id": "Isaac-Libero-Franka-Hybrid-Tactile-v0",
                "task_suite": "suite",
                "task_id": 9,
                "task_description": "base",
                "tabero_task_subset_path": "unused.json",
            },
            "multi_task": {
                "tasks": [
                    {
                        "name": "a",
                        "init_params": {"task_id": 1, "task_description": "one"},
                    },
                    {
                        "name": "b",
                        "init_params": {"task_id": 2, "task_description": "two"},
                    },
                ]
            },
        }
    )
    before = copy.deepcopy(cfg)
    first, second = task_env_cfg(cfg, 0), task_env_cfg(cfg, 1)
    first.init_params.task_description = "modified"
    assert second.init_params.task_description == "two"
    assert second.init_params.tasks[0].task_id == 2
    assert second.init_params.tabero_task_subset_path is None
    assert cfg == before


def test_duplicate_task_names_rejected():
    cfg = OmegaConf.create(
        {
            "multi_task": {
                "tasks": [
                    {"name": "a", "init_params": {}},
                    {"name": "a", "init_params": {}},
                ]
            }
        }
    )
    with pytest.raises(ValueError, match="unique"):
        task_list(cfg)


def ppo_inputs():
    return {
        "logprobs": torch.tensor([0.1, -0.2, 0.4, 0.3], requires_grad=True),
        "old_logprobs": torch.zeros(4),
        "advantages": torch.tensor([1.0, -2.0, 3.0, -1.0]),
        "values": torch.tensor([0.2, 0.7, -0.1, 0.5], requires_grad=True),
        "prev_values": torch.zeros(4),
        "returns": torch.tensor([1.0, 0.0, 0.4, -1.0]),
        "clip_ratio_low": 0.2,
        "clip_ratio_high": 0.2,
        "value_clip": 0.2,
        "huber_delta": 1.0,
        "loss_mask": torch.tensor([True, False, True, True]),
    }


@pytest.mark.parametrize("length_corrected", [False, True])
def test_neutral_weights_match_existing_loss_and_gradients(length_corrected):
    kwargs = ppo_inputs()
    if length_corrected:
        kwargs.update(
            loss_mask_sum=torch.tensor([3.0, 3.0, 1.0, 2.0]), max_episode_steps=4
        )
    old, _ = compute_ppo_actor_critic_loss(**kwargs)
    weighted, _ = compute_ppo_actor_critic_loss(**kwargs, sample_weights=torch.ones(4))
    torch.testing.assert_close(old, weighted)
    parameters = [kwargs["logprobs"], kwargs["values"]]
    for a, b in zip(
        torch.autograd.grad(old, parameters, retain_graph=True),
        torch.autograd.grad(weighted, parameters),
    ):
        torch.testing.assert_close(a, b)


def test_weighted_loss_matches_individual_clipped_losses_and_entropy():
    kwargs = ppo_inputs()
    weights = torch.tensor([0.5, 2.0, 2.0, 1.0], requires_grad=True)
    valid = kwargs["loss_mask"]
    normalized = weights.detach() / weights.detach()[valid].mean()
    actual, _ = compute_ppo_actor_critic_loss(**kwargs, sample_weights=normalized)
    expected = 0
    for i in torch.where(valid)[0].tolist():
        sample = {
            k: v[i : i + 1] if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        loss, _ = compute_ppo_actor_critic_loss(**sample)
        expected += normalized[i] * loss / valid.sum()
    torch.testing.assert_close(actual, expected)
    entropy = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    actual -= 0.005 * weighted_reduce(entropy, valid, sample_weights=normalized)
    expected -= 0.005 * (entropy[valid] * normalized[valid]).mean()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert weights.grad is None


def test_all_masked_loss_is_finite_and_differentiable():
    kwargs = ppo_inputs()
    kwargs.update(
        loss_mask=torch.zeros(4, dtype=torch.bool),
        loss_mask_sum=torch.zeros(4),
        max_episode_steps=4,
    )
    loss, _ = compute_ppo_actor_critic_loss(
        **kwargs, sample_weights=torch.ones(4), weighted_reduction_scale=1.0
    )
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(kwargs["logprobs"].grad).all()


def test_task_indices_survive_split_merge_and_bootstrap():
    rollout = EmbodiedRolloutResult()
    for _ in range(2):
        rollout.append_step_result(
            ChunkStepResult(
                task_indices=torch.tensor([0, 1]),
                prev_logprobs=torch.zeros(2, 3),
                prev_values=torch.zeros(2, 1),
            )
        )
        # Bootstrap contributes a value, but no task/action row.
        rollout.append_step_result(ChunkStepResult(prev_values=torch.zeros(2, 1)))
    split = rollout.to_splited_trajectories_by_sizes([1, 1])
    batch = convert_trajectories_to_batch(split)
    torch.testing.assert_close(batch["task_indices"], torch.tensor([[0, 1], [0, 1]]))
    assert batch["prev_values"].shape[0] == 4
    rollout.clear()
    assert not rollout.task_indices
    with pytest.raises(ValueError, match="with and without"):
        convert_trajectories_to_batch([split[0], Trajectory()])


def _distributed_gradient(rank, rendezvous, output_dir, micro_size):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        weights = torch.tensor([0.5, 2.0, 1.0, 2.0])
        valid = torch.tensor([True, False, True, True])
        sl = slice(rank * 2, rank * 2 + 2)
        stats = batch_weight_stats(weights[sl], valid[sl])
        dist.all_reduce(stats)
        normalized = weights[sl] / (stats[0] / stats[1])
        parameter = torch.tensor(0.7, requires_grad=True)
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])[sl]
        accumulation = 2 // micro_size
        for offset in range(0, 2, micro_size):
            part = slice(offset, offset + micro_size)
            loss = weighted_reduce(
                (parameter * x[part]) ** 2,
                valid[sl][part],
                sample_weights=normalized[part],
                reduction_scale=2 * accumulation / stats[1],
            )
            (loss / accumulation).backward()
        dist.all_reduce(parameter.grad)
        parameter.grad /= 2
        torch.save(parameter.grad, Path(output_dir) / f"rank_{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("micro_size", [1, 2])
def test_two_rank_accumulation_matches_full_batch(tmp_path, micro_size):
    mp.spawn(
        _distributed_gradient,
        args=(str(tmp_path / "rendezvous"), str(tmp_path), micro_size),
        nprocs=2,
    )
    parameter = torch.tensor(0.7, requires_grad=True)
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mask = torch.tensor([True, False, True, True])
    weights = torch.tensor([0.5, 2.0, 1.0, 2.0])
    normalized = weights / weights[mask].mean()
    weighted_reduce((parameter * x) ** 2, mask, sample_weights=normalized).backward()
    for rank in range(2):
        torch.testing.assert_close(
            torch.load(tmp_path / f"rank_{rank}.pt", weights_only=True), parameter.grad
        )


def test_worker_counts_first_episode_once_and_keeps_eval_separate():
    from rlinf.workers.env.env_worker import EnvWorker

    worker = object.__new__(EnvWorker)
    worker._rank, worker.stage_num = 0, 2
    worker._multi_task_seen = [set(), set()]
    worker._multi_task_eval_seen = [set(), set()]
    worker._multi_task_counts = torch.zeros((2, 2), dtype=torch.int64)
    worker._multi_task_eval_counts = torch.zeros((2, 2), dtype=torch.int64)
    records = {
        "env_index": torch.tensor([0, 1]),
        "success_once": torch.tensor([1.0, 0.0]),
        "termination": torch.tensor([True, False]),
        "truncation": torch.tensor([False, True]),
    }
    worker._record_multi_task_episodes(1, records)
    worker._record_multi_task_episodes(1, records)
    torch.testing.assert_close(
        worker._multi_task_counts, torch.tensor([[0, 1], [0, 2]])
    )
    worker._record_multi_task_episodes(0, records, evaluation=True)
    torch.testing.assert_close(
        worker._multi_task_eval_counts, torch.tensor([[1, 0], [2, 0]])
    )
    torch.testing.assert_close(
        worker._multi_task_counts, torch.tensor([[0, 1], [0, 2]])
    )


def test_actor_uses_global_batch_weights_and_rejects_stale_table(monkeypatch):
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

    actor = object.__new__(EmbodiedFSDPActor)
    actor.cfg = OmegaConf.create(
        {
            "actor": {"global_batch_size": 4},
            "env": {
                "train": {
                    "multi_task": {
                        "tasks": [
                            {"name": "a", "init_params": {}},
                            {"name": "b", "init_params": {}},
                        ]
                    }
                }
            },
        }
    )
    actor.device = "cpu"
    actor._world_size = 1
    actor.gradient_accumulation = 4
    actor.version = 3
    actor.multi_task_enabled = True
    actor.set_task_weights([0.5, 2.0], 4)
    batch = {
        "task_indices": torch.tensor([0, 1, 0, 1]),
        "loss_mask": torch.tensor([[True], [True], [False], [True]]),
        "loss_mask_sum": torch.tensor([[2], [2], [0], [2]]),
    }
    monkeypatch.setattr(dist, "all_reduce", lambda _: None)
    metrics = {}
    assert actor._prepare_task_weighted_batch(batch, metrics)
    torch.testing.assert_close(
        batch["sample_weights"], torch.tensor([1 / 3, 4 / 3, 1 / 3, 4 / 3])
    )
    assert batch["weighted_reduction_scale"][0] == 1
    assert batch["weighted_entropy_scale"][0] == pytest.approx(4 / 3)
    actor.version = 4
    with pytest.raises(ValueError, match="Missing current"):
        actor._prepare_task_weighted_batch(batch, {})
    actor.set_task_weights([0.5, 2.0], 5)
    batch["loss_mask"].zero_()
    assert not actor._prepare_task_weighted_batch(batch, {})


def test_task_indices_follow_multi_epoch_flattening_and_shuffle():
    from rlinf.workers.actor.fsdp_actor_worker import (
        process_nested_dict_for_adv,
        process_nested_dict_for_train,
    )

    source = {"task_indices": torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])}
    source["prev_logprobs"] = source["task_indices"].float().unsqueeze(-1)
    batch = process_nested_dict_for_adv(source, rollout_epoch=2)
    shuffled = process_nested_dict_for_train(
        batch, torch.tensor([7, 0, 5, 1, 6, 4, 2, 3])
    )
    torch.testing.assert_close(
        shuffled["task_indices"].float(), shuffled["prev_logprobs"].flatten()
    )


def test_real_actor_micro_batch_weights_all_three_terms(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from rlinf.algorithms.registry import policy_loss
    from rlinf.scheduler import Worker
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

    actor = object.__new__(EmbodiedFSDPActor)
    actor.cfg = OmegaConf.create(
        {
            "runner": {"task_type": "embodied"},
            "actor": {"model": {"model_type": "openpi", "action_dim": 1}},
            "env": {"train": {"max_episode_steps": 4}},
            "algorithm": {
                "adv_type": "gae",
                "loss_type": "actor_critic",
                "reward_type": "chunk_level",
                "logprob_type": "chunk_level",
                "clip_ratio_high": 0.2,
                "clip_ratio_low": 0.2,
                "value_clip": 0.2,
                "huber_delta": 1.0,
                "entropy_bonus": 0.005,
            },
        }
    )
    parameter = torch.nn.Parameter(torch.tensor(0.1))

    def model(**kwargs):
        n = kwargs["forward_inputs"]["action"].shape[0]
        return {
            "logprobs": parameter.expand(n, 1, 1),
            "values": parameter.expand(n),
            "entropy": (parameter + 2).expand(n, 1),
        }

    actor.model = model
    actor.device = "cpu"
    actor.amp_context = nullcontext()
    actor.before_micro_batch = lambda *args, **kwargs: nullcontext()
    actor._tabero_ppo_transition_boundary_semantics = None
    actor.optimizer_steps = 1
    actor.critic_warmup_steps = 0
    actor.enable_sft_co_train = False
    actor.gradient_accumulation = 2
    actor.grad_scaler = SimpleNamespace(scale=lambda loss: loss)
    monkeypatch.setattr(
        Worker, "torch_platform", SimpleNamespace(current_device=lambda: "cpu")
    )
    batch = {
        "advantages": torch.tensor([[1.0], [-1.0]]),
        "prev_logprobs": torch.zeros(2, 1, 1),
        "returns": torch.tensor([[1.0], [0.0]]),
        "prev_values": torch.zeros(2, 1),
        "loss_mask": torch.ones(2, 1, dtype=torch.bool),
        "forward_inputs": {"action": torch.zeros(2, 1)},
        "sample_weights": torch.tensor([0.4, 1.6]),
        "weighted_reduction_scale": torch.ones(2),
        "weighted_entropy_scale": torch.ones(2),
    }
    actor.train_micro_batch(batch, {}, is_last=True)
    actual_grad = parameter.grad.clone()
    parameter.grad = None
    reference, _ = policy_loss(
        loss_type="actor_critic",
        task_type="embodied",
        logprob_type="chunk_level",
        reward_type="chunk_level",
        single_action_dim=1,
        logprobs=parameter.expand(2, 1, 1),
        values=parameter.expand(2),
        old_logprobs=batch["prev_logprobs"],
        advantages=batch["advantages"],
        returns=batch["returns"],
        prev_values=batch["prev_values"],
        loss_mask=batch["loss_mask"],
        clip_ratio_high=0.2,
        clip_ratio_low=0.2,
        value_clip=0.2,
        huber_delta=1.0,
        sample_weights=batch["sample_weights"],
    )
    reference -= 0.005 * ((parameter + 2) * batch["sample_weights"]).mean()
    reference.backward()
    torch.testing.assert_close(actual_grad, parameter.grad)


def test_weighting_preserves_terminal_prefix_gradients():
    from rlinf.algorithms.registry import policy_loss

    logprobs = torch.zeros((2, 3, 1), requires_grad=True)
    mask = torch.tensor([[True, False, False], [True, True, False]])
    loss, _ = policy_loss(
        loss_type="actor_critic",
        task_type="embodied",
        reward_type="chunk_level",
        logprob_type="chunk_level",
        single_action_dim=1,
        logprobs=logprobs,
        old_logprobs=torch.zeros_like(logprobs),
        advantages=torch.ones(2, 1),
        returns=torch.zeros(2, 1),
        values=torch.zeros(2),
        prev_values=torch.zeros(2, 1),
        loss_mask=torch.ones(2, 1, dtype=torch.bool),
        primitive_loss_mask=mask,
        clip_ratio_high=0.2,
        clip_ratio_low=0.2,
        value_clip=0.2,
        huber_delta=1.0,
        sample_weights=torch.tensor([0.5, 1.5]),
    )
    loss.backward()
    torch.testing.assert_close(logprobs.grad.squeeze(-1)[~mask], torch.zeros(3))
    torch.testing.assert_close(logprobs.grad[0, 0, 0], torch.tensor(-0.25))
    torch.testing.assert_close(logprobs.grad[1, 0, 0], torch.tensor(-0.75))


def test_checkpoint_metadata_has_all_tasks_and_no_single_object():
    from rlinf.utils.ppo_multi_task import checkpoint_metadata

    cfg = OmegaConf.create(
        {
            "env": {
                "train": {
                    "init_params": {
                        "id": "Isaac-RealWorld-GentleGrasp-XarmUmi-Hybrid-Tactile-v0",
                        "task_suite": "gentle_grasp",
                        "task_id": 6,
                    },
                    "multi_task": {
                        "tasks": [
                            {
                                "name": "a",
                                "init_params": {
                                    "target_object": "a",
                                    "task_description": "pick a",
                                },
                            },
                            {
                                "name": "b",
                                "init_params": {
                                    "target_object": "b",
                                    "task_description": "pick b",
                                },
                            },
                        ]
                    },
                }
            }
        }
    )
    old = {"target_object": "a", "task_description": "pick a", "target_global_step": 10}
    result = checkpoint_metadata(old, cfg)
    assert result["target_global_step"] == 10
    assert "target_object" not in result and "task_description" not in result
    assert [task["target_object"] for task in result["tasks"]] == ["a", "b"]
    assert old["target_object"] == "a"
