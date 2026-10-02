# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Synthetic RLT tests for weighted gradients, task replay and continuation."""

import copy
import inspect
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.multi_task import DEFAULTS, SuccessWeightController, state_format
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import (
    RLTACFSDPPolicy,
)
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


def worker(batch_size=4, world_size=1):
    actor = object.__new__(RLTACFSDPPolicy)
    actor.cfg = OmegaConf.create(
        {
            "actor": {
                "global_batch_size": batch_size,
                "micro_batch_size": 1,
                "model": {"num_action_chunks": 2, "action_dim": 1},
            },
            "algorithm": {
                "loss_type": "rlt_ac",
                "multi_task": {"enabled": True},
                "rlt_schedule": {
                    "enable": True,
                    "transition_replay": True,
                    "warmup_min_size": 1,
                    "warmup_post_collect_updates": 4,
                    "train_every_transitions": 2,
                    "max_updates_per_train_step": 2,
                },
                "replay_buffer": {"min_buffer_size": 1},
                "update_epoch": 1,
                "gamma": 0.5,
                "bootstrap_type": "standard",
                "q_weight": 0.4,
                "bc_weight": 2.0,
            },
            "env": {
                "train": {
                    "auto_reset": False,
                    "multi_task": {
                        "tasks": [
                            {"name": "easy", "init_params": {}},
                            {"name": "hard", "init_params": {}},
                        ]
                    },
                }
            },
        }
    )
    actor.device = "cpu"
    actor.torch_dtype = torch.float32
    actor._rank = 0
    actor._world_size = world_size
    actor.version = 0
    actor.update_step = 0
    actor.multi_task_enabled = True
    actor.set_task_weights([0.5, 2.0], 1)
    actor._init_rlt_schedule_state(actor.cfg)
    actor.critic_sample_generator = torch.Generator().manual_seed(27)
    return actor


class Model(torch.nn.Module):
    """Closed-form policy and twin Q, with gradients through the action."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.7))

    def forward(self, *, forward_type, obs, actions=None, **kwargs):
        x = obs["z_rl"][:, :1]
        if forward_type == ForwardType.SAC:
            action = self.scale * x * torch.tensor([[1.0, 2.0]])
            return action, torch.zeros_like(action), None
        assert forward_type == ForwardType.SAC_Q
        return (self.scale * x + actions.mean(1, keepdim=True)) * torch.tensor(
            [[1.0, 3.0]]
        )


def batch(size=4):
    x = torch.arange(1, size + 1).float().reshape(-1, 1)
    obs = {
        "z_rl": x,
        "proprio": x / 10,
        "ref_chunk": torch.cat([x / 2, -x / 2], dim=1),
    }
    result = {
        "curr_obs": obs,
        "next_obs": {k: v + 1 for k, v in obs.items()},
        "actions": torch.cat([x, -x], dim=1),
        "rewards": torch.cat([x / 4, x / 2], dim=1),
        "terminations": torch.zeros(size, 2, dtype=torch.bool),
        "dones": torch.zeros(size, 2, dtype=torch.bool),
        "intervene_flags": torch.zeros(size, 2, dtype=torch.bool),
        "task_indices": torch.arange(size).remainder(2),
    }
    result["dones"][::2, -1] = True
    result["intervene_flags"][1::2, 0] = True
    return result


def expected_losses(actor, data, weights):
    x = data["curr_obs"]["z_rl"][:, 0]
    a = actor.model.scale * x[:, None] * torch.tensor([[1.0, 2.0]])
    target_action = torch.where(
        data["intervene_flags"], data["actions"], data["curr_obs"]["ref_chunk"]
    )
    bc = (a - target_action).square().mean(1)
    q1 = actor.model.scale * x + a.mean(1)
    pi_loss = ((-0.4 * q1 + 2.0 * bc) * weights).mean()
    with torch.no_grad():
        next_x = data["next_obs"]["z_rl"][:, 0]
        next_q = actor.target_model.scale * next_x + actor.model.scale * next_x * 1.5
        reward = data["rewards"][:, 0] + 0.5 * data["rewards"][:, 1]
        target = reward + (~data["dones"].any(1)) * 0.25 * next_q
    q = (actor.model.scale * x + data["actions"].mean(1))[:, None] * torch.tensor(
        [[1.0, 3.0]]
    )
    q_loss = ((q - target[:, None]).square().mean(1) * weights).mean()
    return pi_loss, q_loss


@pytest.mark.parametrize("neutral", [False, True])
@pytest.mark.parametrize("method,index", [("forward_actor", 0), ("forward_critic", 1)])
def test_actual_losses_and_gradients_match_hand_calculation(neutral, method, index):
    actor = worker()
    actor.model = Model()
    actor.target_model = Model()
    data = batch()
    weights = torch.ones(4) if neutral else torch.tensor([0.2, 0.4, 0.8, 2.6])
    weights.requires_grad_()
    forward = inspect.unwrap(getattr(RLTACFSDPPolicy, method))
    actual = forward(actor, {**data, "sample_weights": weights})[0]
    expected = expected_losses(actor, data, weights.detach())[index]
    torch.testing.assert_close(actual, expected)
    grad, weight_grad = torch.autograd.grad(
        actual, (actor.model.scale, weights), allow_unused=True
    )
    torch.testing.assert_close(
        grad, torch.autograd.grad(expected, actor.model.scale)[0]
    )
    assert weight_grad is None
    if neutral:
        torch.testing.assert_close(actual, forward(actor, data)[0])


def labelled_trajectory():
    """Two epochs, with a terminal followed by padding in each environment."""
    traj = Trajectory(max_episode_length=6)
    traj.task_indices = torch.tensor([[0, 1]]).expand(6, 2).clone()
    ids = torch.arange(6).reshape(6, 1).expand(6, 2) * 10 + traj.task_indices
    traj.actions = ids[..., None].float().expand(6, 2, 2).clone()
    # Three actions and one bootstrap slot in each epoch.
    traj.rewards = torch.full((8, 2, 2), -999.0)
    traj.dones = torch.zeros(8, 2, 2, dtype=torch.bool)
    for epoch in range(2):
        for t in range(3):
            traj.rewards[epoch * 4 + t + 1] = traj.actions[epoch * 3 + t]
        traj.dones[epoch * 4 + 2, :, -1] = True
    traj.terminations = traj.dones.clone()
    traj.truncations = torch.zeros_like(traj.dones)
    traj.forward_inputs = {"record_transition": torch.ones(6, 2, 1, dtype=torch.bool)}
    # An unrecorded non-terminal must remove its label as well.
    traj.forward_inputs["record_transition"][0, 1] = False
    traj.curr_obs = {
        "z_rl": traj.actions[..., :1].clone(),
        "proprio": torch.zeros(6, 2, 1),
        "ref_chunk": torch.zeros(6, 2, 2),
    }
    traj.next_obs = {k: v + 1 for k, v in traj.curr_obs.items()}
    return traj


def test_split_filter_replay_and_reordered_resume_keep_action_labels(tmp_path):
    actor = worker()
    actor.replay_buffer = TrajectoryReplayBuffer(
        task_names=["easy", "hard"], sample_window_size=100
    )
    restored = TrajectoryReplayBuffer(
        task_names=["hard", "easy"], sample_window_size=100
    )
    try:
        transitions, completed = actor._transition_replay_trajectories(
            labelled_trajectory()
        )
        assert completed == 4 and len(transitions) == 7
        for t in transitions:
            assert t.task_indices.shape == (1, 1)
            assert t.actions[0, 0, 0].long() % 10 == t.task_indices.item()
            torch.testing.assert_close(t.rewards, t.actions)
            assert int(t.actions[0, 0, 0]) // 10 not in (2, 5)
        actor.replay_buffer.add_trajectories(transitions)
        actor.replay_buffer.save_checkpoint(str(tmp_path))
        expected = actor.replay_buffer.sample(7)
        restored.load_checkpoint(str(tmp_path))
        actual = restored.sample(7)
        torch.testing.assert_close(actual["actions"], expected["actions"])
        torch.testing.assert_close(actual["task_indices"], 1 - expected["task_indices"])
    finally:
        actor.replay_buffer.close()
        restored.close()


@pytest.mark.parametrize("bad", [None, torch.ones(6, 2), torch.full((6, 2), 2)])
def test_split_rejects_missing_or_invalid_task_identity(bad):
    actor = worker()
    actor.replay_buffer = TrajectoryReplayBuffer(task_names=["easy", "hard"])
    try:
        traj = labelled_trajectory()
        traj.task_indices = bad
        with pytest.raises(ValueError, match="indices"):
            actor._transition_replay_trajectories(traj)
    finally:
        actor.replay_buffer.close()


def test_controller_is_shared_and_resume_remaps_by_task_name(tmp_path):
    controllers = [
        SuccessWeightController(
            ["easy", "hard"],
            DEFAULTS,
            {},
            state_format=state_format(
                OmegaConf.create({"algorithm": {"loss_type": loss}})
            ),
        )
        for loss in ("actor_critic", "embodied_sac", "rlt_ac")
    ]
    for step, counts in enumerate(
        (torch.tensor([[3, 1], [4, 4]]), torch.tensor([[0, 3], [0, 4]])), start=1
    ):
        for controller in controllers:
            controller.update(counts, step)
        for controller in controllers[1:]:
            torch.testing.assert_close(controller.weights, controllers[0].weights)
    controllers[-1].save(str(tmp_path), 2)
    restored = SuccessWeightController(
        ["hard", "easy"], DEFAULTS, {}, state_format="rlt_multitask_v1"
    )
    restored.restore(str(tmp_path), 2)
    torch.testing.assert_close(restored.weights, controllers[-1].weights.flip(0))


def cpu_reduce(dictionary, dtype=torch.float32, group=None, op=dist.ReduceOp.SUM):
    keys = sorted(dictionary)
    tensor = torch.tensor([dictionary[k] for k in keys], dtype=dtype)
    dist.all_reduce(tensor, group=group, op=op)
    return dict(zip(keys, tensor.tolist()))


def _distributed(rank, init_file, micro_size):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        actor = worker(8, 2)
        actor._rank = rank
        full = batch(8)
        full["task_indices"] = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1])
        local = slice_batch(full, rank * 4, (rank + 1) * 4)
        actor._prepare_sac_task_batch(local)
        raw = torch.tensor([0.5, 2.0])[full["task_indices"]]
        weights = raw / raw.mean()
        torch.testing.assert_close(
            local["sample_weights"], weights[rank * 4 : (rank + 1) * 4]
        )
        for method, index in (("forward_actor", 0), ("forward_critic", 1)):
            actor.model = DistributedDataParallel(Model())
            actor.target_model = Model()
            for start in range(0, 4, micro_size):
                loss = inspect.unwrap(getattr(RLTACFSDPPolicy, method))(
                    actor, slice_batch(local, start, start + micro_size)
                )[0]
                (loss / (4 // micro_size)).backward()
            reference = worker(8)
            reference.model, reference.target_model = Model(), Model()
            expected_losses(reference, full, weights)[index].backward()
            torch.testing.assert_close(
                actor.model.module.scale.grad, reference.model.scale.grad
            )
        # Old replay data is reweighted using this rollout, not collection version.
        actor.version = 1
        actor.set_task_weights([2.0, 0.5], 2)
        actor._prepare_sac_task_batch(local)
        raw = torch.tensor([2.0, 0.5])[full["task_indices"]]
        torch.testing.assert_close(
            local["sample_weights"], (raw / raw.mean())[rank * 4 : (rank + 1) * 4]
        )
        # RLT schedule must preserve budget when any rank lacks a full batch.
        import rlinf.workers.actor.fsdp_rlt_ac_policy_worker as rlt

        rlt.all_reduce_dict = cpu_reduce
        actor.replay_buffer = SimpleNamespace(
            total_samples=10, available_samples=3 if rank == 0 else 10
        )
        actor.demo_buffer = None
        actor.total_transitions_added = 10
        updates, metrics = actor._rlt_updates_to_run()
        assert updates == 0 and actor.pending_update_budget == 4
        assert metrics["rlt/skip_reason"] == 4
        actor.replay_buffer.available_samples = 10
        updates, _ = actor._rlt_updates_to_run()
        assert updates == 2 and actor.get_rollout_sync_version() == 0
    finally:
        dist.destroy_process_group()


def slice_batch(data, start, end):
    return {
        key: slice_batch(value, start, end)
        if isinstance(value, dict)
        else value[start:end]
        for key, value in data.items()
    }


@pytest.mark.parametrize("micro_size", [1, 2, 4])
def test_two_rank_actual_rlt_gradients_and_batch_readiness(tmp_path, micro_size):
    mp.spawn(
        _distributed, args=(str(tmp_path / "gloo"), micro_size), nprocs=2, join=True
    )


def test_rlt_checkpoint_without_dsrl_shadow_and_schedule_receipt(tmp_path, monkeypatch):
    actor = worker()
    actor.update_step = 6
    actor.pending_update_budget = 8
    (tmp_path / "sac_components").mkdir()
    actor._save_multi_task_sac_state(str(tmp_path), 1)
    state = actor._read_multi_task_sac_state(str(tmp_path))
    assert state["format"] == "rlt_multitask_v1"
    assert state["target_shadow_f32"] == {}
    expected_rng = torch.rand(3, generator=actor.critic_sample_generator)
    actor._restore_multi_task_sac_state(state)
    torch.testing.assert_close(
        torch.rand(3, generator=actor.critic_sample_generator), expected_rng
    )
    monkeypatch.setattr(EmbodiedSACFSDPPolicy, "save_checkpoint", lambda *args: None)
    receipt = {"multi_task_step": 1, "multi_task_update_step": 6}
    monkeypatch.setattr(
        EmbodiedSACFSDPPolicy, "load_checkpoint", lambda *args: receipt.copy()
    )
    actor.save_checkpoint(str(tmp_path), 1)
    actor.update_step = 0
    actual = actor.load_checkpoint(str(tmp_path))
    assert actual == receipt and actor.update_step == 6
    assert actor.pending_update_budget == 8
    receipt["multi_task_update_step"] = 7
    with pytest.raises(ValueError, match="steps disagree"):
        actor.load_checkpoint(str(tmp_path))


def test_task_metrics_reduce_by_sample_count_not_microbatch_mean(monkeypatch):
    actor = worker()
    actor.model, actor.target_model = Model(), Model()
    full = batch()
    full["task_indices"] = torch.tensor([0, 0, 0, 1])
    metrics = []
    for start in (0, 2):
        data = slice_batch(full, start, start + 2)
        metrics.append(inspect.unwrap(RLTACFSDPPolicy.forward_critic)(actor, data)[1])
    # Match the base worker's micro-batch averaging, then verify final ratios.
    reduced = {
        f"critic/{key}": sum(m[key] for m in metrics) / len(metrics)
        for key in metrics[0]
    }
    monkeypatch.setattr(
        EmbodiedSACFSDPPolicy,
        "process_train_metrics",
        lambda *args: copy.deepcopy(reduced),
    )
    actual = actor.process_train_metrics({})
    _, full_metrics = inspect.unwrap(RLTACFSDPPolicy.forward_critic)(actor, full)
    for name in ("easy", "hard"):
        assert actual[f"critic/task/{name}/td_mse"] == pytest.approx(
            full_metrics[f"task/{name}/td_mse_sum"] / full_metrics[f"task/{name}/count"]
        )


def _gpu_roundtrip(rank, directory, gpu_ids, shard):
    """Exercise real RLT optimization, DCP, target I/O and replay on two GPUs."""
    import os
    from contextlib import nullcontext
    from pathlib import Path

    from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy

    from rlinf.data.embodied_buffer_dataset import ReplayBufferDataset
    from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy
    from rlinf.hybrid_engines.fsdp.utils import create_device_mesh, get_fsdp_wrap_policy
    from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
    from rlinf.scheduler.hardware.accelerators.accelerator import AcceleratorType

    torch.set_num_threads(1)
    os.environ["LOCAL_RANK"] = str(gpu_ids[rank])
    torch.cuda.set_device(gpu_ids[rank])
    dist.init_process_group(
        "nccl",
        init_method=f"file://{directory}/rendezvous",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    actor = None
    try:
        torch.manual_seed(42)
        actor = worker(4, 2)
        actor._rank = rank
        actor.device = torch.device("cuda", gpu_ids[rank])
        actor._accelerator_type = AcceleratorType.NV_GPU
        actor.worker_timer = lambda *args, **kwargs: nullcontext()
        actor.use_dsrl = False
        actor.enable_drq = False
        actor.cfg.actor.optim = {"clip_grad": 10.0}
        actor.cfg.actor.critic_optim = {"clip_grad": 10.0}
        actor.cfg.actor.fsdp_config = {
            "checkpoint_format": "dcp",
            "save_full_model_weights": False,
            "save_trainable_model_weights": False,
        }
        actor._cfg = actor.cfg.actor
        actor.cfg.algorithm.tau = 0.005
        actor.critic_actor_ratio = 2
        actor.target_update_type = "all"
        actor.target_model_initialized = True
        actor.is_weight_offloaded = actor.is_optimizer_offloaded = False
        actor.alpha_optimizer = None
        actor.entropy_temp = SimpleNamespace(alpha=torch.tensor(0.0))
        model = RLTMLPPolicy(
            z_dim=1, proprio_dim=1, action_dim=1, num_action_chunks=2, fixed_std=0.05
        )
        target = copy.deepcopy(model)
        fsdp_args = {
            "device_id": gpu_ids[rank],
            "device_mesh": create_device_mesh(2),
            "sharding_strategy": ShardingStrategy[shard],
            "use_orig_params": False,
            "auto_wrap_policy": get_fsdp_wrap_policy(
                model, model_type="rlt_mlp_policy"
            ),
        }
        actor.model = FullyShardedDataParallel(model, **fsdp_args)
        actor.target_model = FullyShardedDataParallel(target, **fsdp_args)
        actor.target_model.requires_grad_(False)
        critic_params = [
            p for name, p in actor.model.named_parameters() if "q_head" in name
        ]
        actor_params = [
            p for name, p in actor.model.named_parameters() if "q_head" not in name
        ]
        actor.optimizer = torch.optim.Adam(actor_params, lr=1e-4)
        actor.qf_optimizer = torch.optim.Adam(critic_params, lr=1e-4)
        actor.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            actor.optimizer, lambda _: 1.0
        )
        actor.qf_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            actor.qf_optimizer, lambda _: 1.0
        )
        actor._cache_optimizer_parameter_sets()
        actor._strategy = FSDPStrategy(actor.cfg.actor, 2)
        actor.replay_buffer = TrajectoryReplayBuffer(
            task_names=["easy", "hard"], sample_window_size=100
        )
        actor.demo_buffer = None
        transitions, completed = actor._transition_replay_trajectories(
            labelled_trajectory()
        )
        actor.replay_buffer.add_trajectories(transitions)
        actor._update_rollout_ingest_counters(len(transitions), completed)
        dataset = ReplayBufferDataset(
            actor.replay_buffer,
            demo_buffer=None,
            batch_size=2,
            min_replay_buffer_size=1,
            min_demo_buffer_size=0,
        )
        actor.buffer_dataloader_iter = iter(dataset)
        metrics = actor.run_training()
        assert actor.update_step == 2
        assert metrics["rlt/actor_updates_run"] == 1
        assert metrics["rlt/ready_for_online"] == 0
        assert all(torch.isfinite(torch.as_tensor(v)) for v in metrics.values())
        assert not hasattr(actor, "_target_shadow_f32")
        actor_dir = str(Path(directory) / "global_step_1" / "actor")
        actor.save_checkpoint(actor_dir, 1)
        dist.barrier()
        saved_params = [p.detach().clone() for p in actor.model.parameters()]
        saved_targets = [p.detach().clone() for p in actor.target_model.parameters()]

        def continue_training():
            actor.set_global_step(1)
            actor.set_task_weights([2.0, 0.5], 2)
            return actor.run_training()

        expected_metrics = continue_training()
        expected_params = [p.detach().clone() for p in actor.model.parameters()]
        expected_targets = [p.detach().clone() for p in actor.target_model.parameters()]
        expected_optimizers = copy.deepcopy(
            [actor.optimizer.state_dict(), actor.qf_optimizer.state_dict()]
        )
        assert actor.update_step == 4
        assert expected_metrics["rlt/ready_for_online"] == 1
        receipt = actor.load_checkpoint(actor_dir)
        assert (
            receipt["multi_task_step"] == 1 and receipt["multi_task_update_step"] == 2
        )
        assert actor.update_step == 2 and actor.get_rollout_sync_version() == 2
        for module, expected_values, label in (
            (actor.model, saved_params, "restored model"),
            (actor.target_model, saved_targets, "restored target"),
        ):
            for (name, actual), expected in zip(
                module.named_parameters(), expected_values, strict=True
            ):
                torch.testing.assert_close(
                    actual, expected, rtol=0, atol=0, msg=f"{label}: {name}"
                )
        # Recreate the iterator without consuming samples; no prefetch is used.
        actor.buffer_dataloader_iter = iter(dataset)
        actual_metrics = continue_training()
        assert actor.update_step == 4 and actor.get_rollout_sync_version() == 4
        for actual, expected in zip(
            actor.model.parameters(), expected_params, strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(
            actor.target_model.parameters(), expected_targets, strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(
            [actor.optimizer.state_dict(), actor.qf_optimizer.state_dict()],
            expected_optimizers,
            strict=True,
        ):
            assert_tree_equal(actual, expected)
        for key in (
            "sac/critic_loss",
            "sac/actor_loss",
            "actor/weighted_bc",
            "actor/weighted_q",
        ):
            assert actual_metrics[key] == expected_metrics[key]
        assert not (
            Path(actor_dir) / "sac_components/target_model/checkpoint_rank_1.pt"
        ).exists()
    finally:
        if actor is not None and getattr(actor, "replay_buffer", None) is not None:
            actor.replay_buffer.close()
        dist.destroy_process_group()


def assert_tree_equal(actual, expected):
    if torch.is_tensor(expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            assert_tree_equal(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize("shard", ["NO_SHARD", "FULL_SHARD"])
def test_two_gpu_rlt_update_checkpoint_and_exact_next_update(tmp_path, shard):
    import os

    value = os.environ.get("RLINF_TEST_GPU_IDS")
    if not value:
        pytest.skip(
            "Set RLINF_TEST_GPU_IDS to two available CUDA ordinals for FSDP tests"
        )
    gpu_ids = [int(gpu) for gpu in value.split(",")]
    assert len(gpu_ids) == 2 and len(set(gpu_ids)) == 2
    assert all(0 <= gpu < torch.cuda.device_count() for gpu in gpu_ids)
    mp.spawn(_gpu_roundtrip, args=(str(tmp_path), gpu_ids, shard), nprocs=2, join=True)
