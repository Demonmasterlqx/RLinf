# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Residual action, gradient and checkpoint contracts on synthetic inputs."""

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
from rlinf.utils.rlt_action_contract import (
    build_action_contract,
    checkpoint_action_contract,
)
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACFSDPPolicy
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


def policy_and_obs():
    policy = RLTMLPPolicy(2, 1, 1, 2, normalized_action_bound=4.0)
    obs = {
        "z_rl": torch.ones(2, 2),
        "proprio": torch.ones(2, 1),
        "ref_chunk": torch.tensor([[[5.0], [-6.0], [99.0]]] * 2),
    }
    return policy, obs


@pytest.mark.parametrize("mode", ["absolute", "residual"])
def test_explicit_model_mode_preserves_complete_action_semantics(mode):
    model = RLTMLPPolicy(
        2, 1, 1, 2, normalized_action_bound=4.0, action_parameterization=mode
    )
    _, obs = policy_and_obs()
    with torch.no_grad():
        model.actor_mean.weight.zero_()
        model.actor_mean.bias.copy_(torch.tensor([2.0, -3.0]))
    actual, _, _ = model.sac_forward(obs, deterministic=True)
    bounded = (4 * torch.tanh(torch.tensor([2.0, -3.0]) / 4)).expand(2, -1)
    expected = (
        bounded + obs["ref_chunk"][:, :2].flatten(1) if mode == "residual" else bounded
    )
    torch.testing.assert_close(actual, expected)
    assert model.sft_forward({"obs": obs, "actions": expected}).sum() == 0


@pytest.mark.parametrize("mode", ["absolute", "residual"])
def test_unlabelled_weight_requires_explicit_mode_and_rejects_conflicts(tmp_path, mode):
    path = tmp_path / "weights.pt"
    with pytest.raises(ValueError, match="Missing RLT action contract"):
        checkpoint_action_contract(path, {}, bound=4.0)
    contract = checkpoint_action_contract(path, {}, bound=4.0, parameterization=mode)
    assert contract == build_action_contract(4.0, mode)
    payload = {"metadata": {"rlt_action_contract": contract}}
    assert checkpoint_action_contract(path, payload, bound=4.0) == contract
    opposite = "absolute" if mode == "residual" else "residual"
    with pytest.raises(ValueError, match="action contract mismatch"):
        checkpoint_action_contract(path, payload, bound=4.0, parameterization=opposite)


@pytest.mark.parametrize("saved_mode", [None, "absolute", "residual"])
@pytest.mark.parametrize("configured_mode", ["absolute", "residual"])
def test_resume_supports_explicit_modes_without_silent_conversion(
    tmp_path, monkeypatch, saved_mode, configured_mode
):
    worker = object.__new__(RLTACFSDPPolicy)
    worker._rank = 0
    worker.use_rlt_schedule = True
    worker.cfg = OmegaConf.create(
        {
            "actor": {
                "model": {
                    "normalized_action_bound": 4.0,
                    "action_parameterization": configured_mode,
                }
            }
        }
    )
    state = dict.fromkeys(worker._RLT_CHECKPOINT_FIELDS, 0)
    state["global_step"] = 1
    if saved_mode is not None:
        state["rlt_action_contract"] = build_action_contract(4.0, saved_mode)
    path = tmp_path / "sac_components/rlt_schedule/checkpoint_rank_0.pt"
    path.parent.mkdir(parents=True)
    torch.save(state, path)
    loaded = []
    monkeypatch.setattr(
        EmbodiedSACFSDPPolicy, "load_checkpoint", lambda *args: loaded.append(True)
    )
    if saved_mode is not None and saved_mode != configured_mode:
        with pytest.raises(ValueError, match="action contract mismatch"):
            worker.load_checkpoint(str(tmp_path))
        assert not loaded
    else:
        worker.load_checkpoint(str(tmp_path))
        assert loaded == [True]


def test_zero_residual_preserves_reference_even_outside_residual_bounds():
    policy, obs = policy_and_obs()
    actions, _, _ = policy.sac_forward(obs, deterministic=True)
    expected = obs["ref_chunk"][:, :2].flatten(1)
    torch.testing.assert_close(actions, expected, rtol=0, atol=0)
    evaluated, _ = policy.predict_action_batch(obs, mode="eval")
    torch.testing.assert_close(evaluated.flatten(1), expected, rtol=0, atol=0)
    assert policy.sft_forward({"obs": obs, "actions": expected}).sum() == 0


def test_zero_residual_preserves_reference_precision_under_autocast():
    policy, obs = policy_and_obs()
    obs["ref_chunk"] += 0.01
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual, _, _ = policy.sac_forward(obs, deterministic=True)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(
        actual, obs["ref_chunk"][:, :2].flatten(1), rtol=0, atol=0
    )


def test_residual_dropout_keeps_anchor_and_sft_has_same_output_and_gradient():
    policy, obs = policy_and_obs()
    with torch.no_grad():
        policy.actor_mean.bias.copy_(torch.tensor([2.0, -3.0]))
    captured = []
    hook = policy.backbone.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach())
    )
    actions, _, _ = policy.sac_forward(
        obs,
        deterministic=True,
        apply_reference_dropout=True,
        reference_dropout_prob=1.0,
    )
    hook.remove()
    assert torch.count_nonzero(captured[0][:, :2]) == 0
    ref = obs["ref_chunk"][:, :2].flatten(1)
    residual = 4.0 * torch.tanh(policy.actor_mean.bias / 4.0)
    torch.testing.assert_close(actions, ref + residual)
    loss = policy.sft_forward({"obs": obs, "actions": ref}).mean()
    torch.testing.assert_close(loss, residual.square().mean())
    loss.backward()
    expected_grad = (
        2 * residual / 2 * (1 - torch.tanh(policy.actor_mean.bias / 4).square())
    )
    torch.testing.assert_close(policy.actor_mean.bias.grad, expected_grad)


def test_training_exploration_is_added_to_reference(monkeypatch):
    policy, obs = policy_and_obs()
    monkeypatch.setattr(
        torch.distributions.Normal, "rsample", lambda self: self.loc + self.scale
    )
    action, logprob, _ = policy.sac_forward(obs)
    expected_delta = 4 * torch.tanh(torch.tensor(policy.fixed_std / 4))
    torch.testing.assert_close(
        action, obs["ref_chunk"][:, :2].flatten(1) + expected_delta
    )
    assert torch.isfinite(logprob).all()


@pytest.mark.parametrize("use_schedule", [False, True])
@pytest.mark.parametrize(
    "contract",
    [None, {"action_parameterization": "absolute"}, build_action_contract(2.0)],
)
def test_resume_rejects_wrong_semantics_before_loading_state(
    tmp_path, monkeypatch, use_schedule, contract
):
    worker = object.__new__(RLTACFSDPPolicy)
    worker._rank = 0
    worker.use_rlt_schedule = use_schedule
    worker.cfg = OmegaConf.create(
        {"actor": {"model": {"normalized_action_bound": 4.0}}}
    )
    state = dict.fromkeys(worker._RLT_CHECKPOINT_FIELDS, 0)
    state.update(global_step=1, rlt_action_contract=contract)
    path = tmp_path / "sac_components/rlt_schedule/checkpoint_rank_0.pt"
    path.parent.mkdir(parents=True)
    torch.save(state, path)
    loaded = []
    monkeypatch.setattr(
        EmbodiedSACFSDPPolicy, "load_checkpoint", lambda *args: loaded.append(True)
    )
    with pytest.raises(ValueError, match="action contract mismatch"):
        worker.load_checkpoint(str(tmp_path))
    assert loaded == []


def test_raw_weight_contract_requires_matching_checkpoint_sidecar(tmp_path):
    path = tmp_path / "actor/model_state_dict/full_weights.pt"
    sidecar = tmp_path / "actor/sac_components/rlt_schedule/checkpoint_rank_0.pt"
    contract = build_action_contract(4.0)
    with pytest.raises(ValueError, match="Missing RLT action contract"):
        checkpoint_action_contract(path, {}, bound=4.0)
    sidecar.parent.mkdir(parents=True)
    torch.save({"global_step": 3, "rlt_action_contract": contract}, sidecar)
    assert checkpoint_action_contract(path, {}, bound=4.0, step=3) == contract
    with pytest.raises(ValueError, match="global step mismatch"):
        checkpoint_action_contract(path, {}, bound=4.0, step=4)


def test_embedded_trainable_contract_is_validated_without_sidecar(tmp_path):
    contract = build_action_contract(4.0)
    payload = {"model": {}, "metadata": {"step": 3, "rlt_action_contract": contract}}
    assert (
        checkpoint_action_contract(
            tmp_path / "trainable.pt", payload, bound=4.0, step=3
        )
        == contract
    )
    with pytest.raises(ValueError, match="action contract mismatch"):
        checkpoint_action_contract(tmp_path / "trainable.pt", payload, bound=2.0)


def test_actor_bc_and_td_target_use_complete_actions():
    class LinearQPolicy(RLTMLPPolicy):
        def sac_forward(self, obs, **kwargs):
            return super().sac_forward(obs, **kwargs, deterministic=True)

        def sac_q_forward(self, obs, actions, **kwargs):
            self.last_q_actions = actions.detach().clone()
            q = actions.flatten(1).sum(1, keepdim=True)
            return torch.cat([q, 2 * q], dim=1)

    model = LinearQPolicy(2, 1, 1, 2, normalized_action_bound=4.0)
    _, obs = policy_and_obs()
    next_obs = {**obs, "ref_chunk": torch.tensor([[[2.0], [3.0]]] * 2)}
    worker = object.__new__(RLTACFSDPPolicy)
    worker.model = model
    worker.target_model = LinearQPolicy(2, 1, 1, 2, normalized_action_bound=4.0)
    worker.torch_dtype = torch.float32
    worker.multi_task_enabled = False
    worker.cfg = OmegaConf.create(
        {
            "actor": {
                "model": {
                    "num_action_chunks": 2,
                    "action_dim": 1,
                    "normalized_action_bound": 4.0,
                }
            },
            "algorithm": {
                "q_weight": 0.4,
                "bc_weight": 2.0,
                "gamma": 0.5,
                "rlt_schedule": {"transition_replay": True},
            },
        }
    )
    batch = {
        "curr_obs": obs,
        "next_obs": next_obs,
        "actions": obs["ref_chunk"][:, :2].flatten(1),
        "rewards": torch.tensor([[1.0, 2.0]] * 2),
        "dones": torch.zeros(2, 2, dtype=torch.bool),
        "terminations": torch.zeros(2, 2, dtype=torch.bool),
    }
    actor_loss, _, metrics = RLTACFSDPPolicy.forward_actor.__wrapped__(worker, batch)
    assert metrics["bc_loss"] == 0.0
    torch.testing.assert_close(actor_loss, torch.tensor(0.4))
    actor_loss.backward()
    torch.testing.assert_close(model.actor_mean.bias.grad, torch.full((2,), -0.4))
    critic_loss, _ = RLTACFSDPPolicy.forward_critic.__wrapped__(worker, batch)
    torch.testing.assert_close(
        worker.target_model.last_q_actions, next_obs["ref_chunk"].flatten(1)
    )
    # Reward 1 + .5*2, bootstrap .5**2 * min(5,10); data Q=(-1,-2).
    expected = ((-1 - 3.25) ** 2 + (-2 - 3.25) ** 2) / 2
    torch.testing.assert_close(critic_loss, torch.tensor(expected))
