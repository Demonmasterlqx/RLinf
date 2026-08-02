# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

import rlinf.workers.actor.fsdp_sac_policy_worker as sac_worker_module
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.dsrl_reward import (
    chunk_bootstrap_discount,
    discounted_alive_masked_chunk_rewards,
)
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy

GAMMA = 0.999
CHUNK_LENGTH = 10


class _SeparatedActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Linear(2, 2, bias=False)
        self.critic = nn.Linear(2, 1, bias=False)


def _gradient_partition_worker():
    worker = EmbodiedSACFSDPPolicy.__new__(EmbodiedSACFSDPPolicy)
    worker.model = _SeparatedActorCritic()
    worker.optimizer = torch.optim.SGD(worker.model.actor.parameters(), lr=0.1)
    worker.qf_optimizer = torch.optim.SGD(worker.model.critic.parameters(), lr=0.1)
    worker.use_dsrl = True
    worker._cache_optimizer_parameter_sets()
    return worker


def test_dsrl_optimizer_parameter_sets_are_disjoint_and_complete():
    worker = _gradient_partition_worker()

    assert {id(parameter) for parameter in worker._actor_parameters} == {
        id(worker.model.actor.weight)
    }
    assert {id(parameter) for parameter in worker._critic_parameters} == {
        id(worker.model.critic.weight)
    }


def test_dsrl_critic_and_actor_backward_keep_parameter_gradients_isolated():
    worker = _gradient_partition_worker()
    inputs = torch.tensor([[1.0, -2.0]])

    worker._zero_model_and_optimizer_gradients(worker.qf_optimizer)
    critic_loss = worker.model.critic(inputs).square().sum()
    critic_loss.backward()
    worker._assert_no_parameter_gradients(worker._actor_parameters, "critic")
    assert worker.model.critic.weight.grad is not None

    worker._zero_model_and_optimizer_gradients(worker.optimizer)
    action = worker.model.actor(inputs)
    action.retain_grad()
    with worker._temporarily_freeze_parameters(worker._critic_parameters):
        actor_loss = -worker.model.critic(action).sum()
        actor_loss.backward()

    worker._assert_no_parameter_gradients(worker._critic_parameters, "actor")
    assert action.grad is not None
    assert torch.count_nonzero(action.grad).item() > 0
    assert worker.model.actor.weight.grad is not None
    assert torch.count_nonzero(worker.model.actor.weight.grad).item() > 0


def test_dsrl_gradient_isolation_allows_fsdp_zero_gradient_views():
    worker = _gradient_partition_worker()
    worker.model.actor.weight.grad = torch.zeros_like(worker.model.actor.weight)

    worker._assert_no_parameter_gradients(worker._actor_parameters, "critic")

    worker.model.actor.weight.grad[0, 0] = torch.finfo(torch.float32).eps
    with pytest.raises(RuntimeError, match="1 non-zero gradients"):
        worker._assert_no_parameter_gradients(worker._actor_parameters, "critic")


def test_dsrl_phase_start_clears_stale_model_gradients():
    worker = _gradient_partition_worker()
    worker.model.actor.weight.grad = torch.ones_like(worker.model.actor.weight)
    worker.model.critic.weight.grad = torch.ones_like(worker.model.critic.weight)

    worker._zero_model_and_optimizer_gradients(worker.optimizer)

    assert worker.model.actor.weight.grad is None
    assert worker.model.critic.weight.grad is None


def test_dsrl_critic_requires_grad_is_restored_after_actor_exception():
    worker = _gradient_partition_worker()

    with pytest.raises(RuntimeError, match="actor failed"):
        with worker._temporarily_freeze_parameters(worker._critic_parameters):
            assert worker.model.critic.weight.requires_grad is False
            raise RuntimeError("actor failed")

    assert worker.model.critic.weight.requires_grad is True


def _chunk_tensors(dtype=torch.float32):
    return (
        torch.zeros(1, CHUNK_LENGTH, dtype=dtype),
        torch.zeros(1, CHUNK_LENGTH, dtype=torch.bool),
        torch.zeros(1, CHUNK_LENGTH, dtype=torch.bool),
    )


@pytest.mark.parametrize("reward_index", [0, 1, 7, 9])
def test_discounted_chunk_reward_keeps_each_primitive_position(reward_index):
    rewards, terminations, truncations = _chunk_tensors()
    rewards[0, reward_index] = 1

    result = discounted_alive_masked_chunk_rewards(
        rewards,
        terminations,
        truncations,
        gamma=GAMMA,
        num_action_chunks=CHUNK_LENGTH,
    )

    assert result.dtype == torch.float32
    assert result.shape == (1, 1)
    assert result.item() == pytest.approx(GAMMA**reward_index)


@pytest.mark.parametrize("done_field", ["terminations", "truncations"])
def test_discounted_chunk_reward_keeps_done_step_and_masks_later_episode(done_field):
    rewards, terminations, truncations = _chunk_tensors()
    rewards[0, 4] = 2
    rewards[0, 8] = 100
    done = terminations if done_field == "terminations" else truncations
    done[0, 4] = True

    result = discounted_alive_masked_chunk_rewards(
        rewards,
        terminations,
        truncations,
        gamma=GAMMA,
        num_action_chunks=CHUNK_LENGTH,
    )

    assert result.item() == pytest.approx(2 * GAMMA**4)


def test_discounted_chunk_reward_sums_complete_chunk_without_done():
    rewards, terminations, truncations = _chunk_tensors()
    rewards.fill_(1)

    result = discounted_alive_masked_chunk_rewards(
        rewards,
        terminations,
        truncations,
        gamma=GAMMA,
        num_action_chunks=CHUNK_LENGTH,
    )

    assert result.item() == pytest.approx(sum(GAMMA**i for i in range(CHUNK_LENGTH)))


def test_discounted_chunk_reward_preserves_zero_and_float32_discount_for_bfloat16():
    rewards, terminations, truncations = _chunk_tensors(dtype=torch.bfloat16)
    zero_result = discounted_alive_masked_chunk_rewards(
        rewards,
        terminations,
        truncations,
        gamma=GAMMA,
        num_action_chunks=CHUNK_LENGTH,
    )
    rewards[0, 9] = 1
    discounted_result = discounted_alive_masked_chunk_rewards(
        rewards,
        terminations,
        truncations,
        gamma=GAMMA,
        num_action_chunks=CHUNK_LENGTH,
    )
    bootstrap = chunk_bootstrap_discount(
        GAMMA,
        num_action_chunks=CHUNK_LENGTH,
        device=rewards.device,
    )

    assert torch.equal(zero_result, torch.zeros(1, 1, dtype=torch.float32))
    assert discounted_result.dtype == torch.float32
    assert discounted_result.item() == pytest.approx(GAMMA**9)
    assert bootstrap.dtype == torch.float32
    assert bootstrap.item() == pytest.approx(GAMMA**CHUNK_LENGTH)
    assert bootstrap.item() < 1.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda r, _t, _x: r.reshape(1, 2, 5), "shape.*batch, chunk"),
        (lambda _r, t, _x: t[:, :-1], "identical shapes"),
        (lambda _r, _t, x: x.repeat(2, 1), "identical shapes"),
    ],
)
def test_discounted_chunk_reward_rejects_invalid_shapes(mutation, message):
    rewards, terminations, truncations = _chunk_tensors()
    changed = mutation(rewards, terminations, truncations)
    if changed.ndim == 3:
        rewards = changed
    elif changed.dtype == torch.bool and changed.shape[0] == 2:
        truncations = changed
    else:
        terminations = changed

    with pytest.raises(ValueError, match=message):
        discounted_alive_masked_chunk_rewards(
            rewards,
            terminations,
            truncations,
            gamma=GAMMA,
            num_action_chunks=CHUNK_LENGTH,
        )


def test_discounted_chunk_reward_rejects_configured_chunk_length_mismatch():
    rewards, terminations, truncations = _chunk_tensors()

    with pytest.raises(ValueError, match="chunk length.*num_action_chunks"):
        discounted_alive_masked_chunk_rewards(
            rewards,
            terminations,
            truncations,
            gamma=GAMMA,
            num_action_chunks=9,
        )


class _CriticModel:
    def __init__(self, *, data_q=0.0, next_q=0.0, crossq=False):
        self.data_q = data_q
        self.next_q = next_q
        self.crossq = crossq
        self.calls = []

    def __call__(self, *, forward_type, obs, actions=None, **_kwargs):
        self.calls.append((forward_type, obs, _kwargs.get("next_obs")))
        batch_size = obs["state"].shape[0]
        if forward_type == ForwardType.SAC:
            return (
                torch.zeros(batch_size, 32),
                torch.zeros(batch_size, 1),
                None,
            )
        if forward_type == ForwardType.SAC_Q:
            value = self.data_q if obs["kind"] == "current" else self.next_q
            return torch.full((batch_size, 10), value)
        if forward_type == ForwardType.CROSSQ_Q and self.crossq:
            return (
                torch.full((batch_size, 10), self.data_q),
                torch.full((batch_size, 10), self.next_q),
            )
        raise AssertionError(forward_type)


def _critic_worker(*, use_dsrl, crossq=False, next_q=0.0):
    worker = EmbodiedSACFSDPPolicy.__new__(EmbodiedSACFSDPPolicy)
    worker.cfg = OmegaConf.create(
        {
            "actor": {
                "model": {
                    "model_type": "openpi",
                    "num_action_chunks": CHUNK_LENGTH,
                    "openpi": {"use_dsrl": use_dsrl},
                }
            },
            "algorithm": {
                "q_head_type": "crossq" if crossq else "default",
                "bootstrap_type": "standard",
                "agg_q": "mean",
                "backup_entropy": False,
                "gamma": GAMMA,
            },
        }
    )
    worker.torch_dtype = torch.bfloat16
    worker.critic_subsample_size = 0
    worker.model = _CriticModel(next_q=next_q, crossq=crossq)
    worker.target_model = _CriticModel(next_q=next_q)
    worker.entropy_temp = SimpleNamespace(alpha=torch.tensor(0.0))
    return worker


def _critic_batch(*, reward_width=CHUNK_LENGTH):
    current_wrist = torch.full((1, 2, 2, 3), 7, dtype=torch.uint8)
    next_wrist = torch.full((1, 2, 2, 3), 11, dtype=torch.uint8)
    return {
        "curr_obs": {
            "kind": "current",
            "state": torch.zeros(1, 7),
            "wrist_images": current_wrist,
        },
        "next_obs": {
            "kind": "next",
            "state": torch.zeros(1, 7),
            "wrist_images": next_wrist,
        },
        "actions": torch.zeros(1, 50, 32),
        "rewards": torch.zeros(1, reward_width),
        "terminations": torch.zeros(1, reward_width, dtype=torch.bool),
        "truncations": torch.zeros(1, reward_width, dtype=torch.bool),
    }


@pytest.mark.parametrize("crossq", [False, True])
def test_forward_critic_uses_position_seven_terminal_reward_without_bootstrap(
    monkeypatch, crossq
):
    worker = _critic_worker(use_dsrl=True, crossq=crossq, next_q=100.0)
    batch = _critic_batch()
    batch["rewards"][0, 7] = 1
    batch["rewards"][0, 9] = 100
    batch["terminations"][0, 7] = True
    captured = {}

    def capture_loss(data_q, target_q):
        captured["target"] = target_q.detach().clone()
        return torch.mean((data_q - target_q) ** 2)

    monkeypatch.setattr(sac_worker_module.F, "mse_loss", capture_loss)

    EmbodiedSACFSDPPolicy.forward_critic.__wrapped__.__wrapped__(worker, batch)

    assert captured["target"].shape == (1, 10)
    assert captured["target"].float().unique().item() == pytest.approx(GAMMA**7)
    observed = [
        obs
        for model in (worker.model, worker.target_model)
        for _, obs, _ in model.calls
    ]
    observed.extend(
        next_obs
        for model in (worker.model, worker.target_model)
        for _, _, next_obs in model.calls
        if next_obs is not None
    )
    assert observed
    assert all("wrist_images" in obs for obs in observed)
    assert any(obs is batch["curr_obs"] for obs in observed)
    assert any(obs is batch["next_obs"] for obs in observed)


def test_forward_critic_non_dsrl_reward_sum_and_bootstrap_are_unchanged(monkeypatch):
    worker = _critic_worker(use_dsrl=False, next_q=2.0)
    batch = _critic_batch(reward_width=3)
    batch["rewards"][:] = torch.tensor([[1.0, 2.0, 3.0]])
    captured = {}

    def capture_loss(data_q, target_q):
        captured["target"] = target_q.detach().clone()
        return torch.mean((data_q - target_q) ** 2)

    monkeypatch.setattr(sac_worker_module.F, "mse_loss", capture_loss)

    EmbodiedSACFSDPPolicy.forward_critic.__wrapped__.__wrapped__(worker, batch)

    expected = 6.0 + GAMMA * 2.0
    assert captured["target"].float().unique().item() == pytest.approx(expected)
