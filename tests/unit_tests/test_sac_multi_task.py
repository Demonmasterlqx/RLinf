# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Synthetic DSRL tests: actual loss methods, replay and distributed reduction."""

import hashlib
import inspect
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel

from rlinf.data.dsrl_replay_buffer import CompactDSRLReplayBuffer
from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    Trajectory,
)
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.dsrl_replay import get_dsrl_replay_field_specs
from rlinf.utils.multi_task import (
    DEFAULTS,
    SuccessWeightController,
    checkpoint_metadata,
)
from rlinf.utils.sac_multi_task import validate_config, weighted_sac_mean
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy
from rlinf.workers.env.env_worker import project_compact_dsrl_step_result


def worker(batch_size=4, world_size=1):
    result = object.__new__(EmbodiedSACFSDPPolicy)
    result.cfg = OmegaConf.create(
        {
            "actor": {"global_batch_size": batch_size},
            "env": {
                "train": {
                    "multi_task": {
                        "tasks": [
                            {"name": "easy", "init_params": {}},
                            {"name": "hard", "init_params": {}},
                        ]
                    }
                }
            },
        }
    )
    result.device = "cpu"
    result._world_size = world_size
    result._rank = 0
    result.version = 0
    result.multi_task_enabled = True
    result.set_task_weights([0.5, 2.0], 1)
    return result


def trajectory(offset=0):
    result = Trajectory(max_episode_length=20, model_weights_id="test")
    for key, (shape, dtype) in get_dsrl_replay_field_specs().items():
        value = torch.zeros(2, 2, *shape, dtype=dtype)
        if "." in key:
            group, name = key.split(".")
            getattr(result, group)[name] = value
        else:
            setattr(result, key, value)
    result.task_indices = torch.tensor([[0, 1], [1, 0]])
    result.actions[..., 0] = result.task_indices + offset
    return result


def replay(kind, names):
    if kind == "compact":
        return CompactDSRLReplayBuffer(
            task_names=names,
            capacity_transitions=5,
            checkpoint_shard_transitions=2,
            max_resident_gib=0.01,
        )
    return TrajectoryReplayBuffer(task_names=names, sample_window_size=2)


@pytest.mark.parametrize("kind", ["compact", "trajectory"])
@pytest.mark.parametrize("restored_names", [["hard", "easy"], ["hard", "easy", "new"]])
def test_replay_wrap_resume_reorder_and_resave(kind, restored_names, tmp_path):
    original = replay(kind, ["easy", "hard"])
    restored = replay(kind, restored_names)
    again = replay(kind, ["easy", "hard"])
    try:
        for offset in (0, 2, 4):
            original.add_trajectories([trajectory(offset)])
        original.save_checkpoint(str(tmp_path / "one"))
        expected = original.sample(4)
        restored.load_checkpoint(str(tmp_path / "one"))
        actual = restored.sample(4)
        torch.testing.assert_close(actual["actions"], expected["actions"])
        torch.testing.assert_close(actual["task_indices"], 1 - expected["task_indices"])
        restored.save_checkpoint(str(tmp_path / "two"))
        expected = restored.sample(4)
        again.load_checkpoint(str(tmp_path / "two"))
        actual = again.sample(4)
        torch.testing.assert_close(actual["actions"], expected["actions"])
        torch.testing.assert_close(actual["task_indices"], 1 - expected["task_indices"])
        torch.testing.assert_close(
            actual["actions"][:, 0].long() % 2, actual["task_indices"]
        )
    finally:
        original.close()
        restored.close()
        again.close()


@pytest.mark.parametrize("kind", ["compact", "trajectory"])
@pytest.mark.parametrize(
    "bad", [None, torch.tensor([[0, 2], [0, 0]]), torch.zeros(2, 2)]
)
def test_replay_rejects_missing_or_invalid_labels(kind, bad):
    buffer = replay(kind, ["easy", "hard"])
    traj = trajectory()
    traj.task_indices = bad
    try:
        with pytest.raises(ValueError, match="indices"):
            buffer.add_trajectories([traj])
    finally:
        buffer.close()


@pytest.mark.parametrize("kind", ["compact", "trajectory"])
@pytest.mark.parametrize(
    "names", [["hard"], ["new", "hard"], ["new"], ["new", "hard", "easy"]]
)
def test_replay_migrates_task_set(kind, names, tmp_path):
    original = replay(kind, ["easy", "hard"])
    changed = replay(kind, names)
    again = replay(kind, names)
    source = tmp_path / "source"
    try:
        for offset in (0, 2, 4):
            traj = trajectory(offset)
            # Verify every retained transition keeps its observation/reward data.
            traj.rewards[..., 0] = traj.actions[..., 0] + 10
            for group in (traj.curr_obs, traj.next_obs, traj.forward_inputs):
                for value in group.values():
                    value.reshape(2, 2, -1)[..., 0] = traj.actions[..., 0]
            original.add_trajectories([traj])
        original.save_checkpoint(str(source))
        hashes = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()
        }
        changed.load_checkpoint(str(source))
        assert hashes == {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()
        }
        expected = (
            (2 if kind == "compact" else 4)
            if "easy" not in names
            else (5 if kind == "compact" else 8)
        )
        if "hard" not in names:
            expected = 0
        assert changed.total_samples == expected
        assert changed.last_restore_migration["retained_samples"] == expected
        if kind == "trajectory":
            # Force lazy reload of the migrated files, bypassing the cache.
            changed._flat_trajectory_cache.clear()
        changed.save_checkpoint(str(tmp_path / "next"))
        again.load_checkpoint(str(tmp_path / "next"))
        assert again.total_samples == expected
        if expected:
            actual = changed.sample(20)
            repeated = again.sample(20)
            for key in ("actions", "task_indices", "rewards"):
                torch.testing.assert_close(actual[key], repeated[key])
            torch.testing.assert_close(
                actual["rewards"][:, 0], actual["actions"][:, 0].float() + 10
            )
            for group in ("curr_obs", "next_obs", "forward_inputs"):
                for value in actual.get(group, {}).values():
                    torch.testing.assert_close(
                        value.reshape(len(actual["actions"]), -1)[:, 0].float(),
                        actual["actions"][:, 0].float(),
                    )
            for action, task in zip(
                actual["actions"][:, 0].long(), actual["task_indices"]
            ):
                assert names[task] == ["easy", "hard"][action % 2]
        else:
            with pytest.raises(RuntimeError, match="empty"):
                changed.sample(1)
        if kind == "compact" and names == ["hard"]:
            order = changed._chronological_indices()
            assert changed._storage["actions"][order, 0].tolist() == [5, 5]
            assert changed._write_pos == 2
            assert changed.size == original.size
            assert changed._total_inserted_samples == original._total_inserted_samples
    finally:
        original.close()
        changed.close()
        again.close()


@pytest.mark.parametrize("kind", ["compact", "trajectory"])
def test_migration_rejects_invalid_old_task_ids(kind, tmp_path):
    original = replay(kind, ["easy", "hard"])
    changed = replay(kind, ["new"])
    try:
        original.add_trajectories([trajectory()])
        original.save_checkpoint(str(tmp_path))
        path = next(
            tmp_path.glob("shard_*.pt" if kind == "compact" else "trajectory_*.pt")
        )
        data = torch.load(path, weights_only=True)
        data["task_indices"].view(-1)[0] = 2
        torch.save(data, path)
        if kind == "compact":
            metadata_path = tmp_path / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            for shard in metadata["shards"]:
                if shard["name"] == path.name:
                    shard["size_bytes"] = path.stat().st_size
            metadata_path.write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match="indices"):
            changed.load_checkpoint(str(tmp_path))
    finally:
        original.close()
        changed.close()


def test_compact_projection_keeps_labels_without_logprobs():
    rollout = EmbodiedRolloutResult(max_episode_length=20)
    for _ in range(2):
        bootstrap = ChunkStepResult(
            actions=torch.zeros(2, 32), task_indices=torch.tensor([0, 1])
        )
        rollout.append_step_result(project_compact_dsrl_step_result(bootstrap))
        rollout.append_step_result(ChunkStepResult(rewards=torch.zeros(2, 10)))
    traj = rollout.to_trajectory()
    assert traj.prev_logprobs is None
    torch.testing.assert_close(traj.task_indices, torch.tensor([[0, 1], [0, 1]]))
    shards = rollout.to_splited_trajectories(2)
    assert all(t.task_indices.shape == (2, 1) for t in shards)


def test_old_replay_samples_use_current_weights_and_reject_stale_version(monkeypatch):
    monkeypatch.setattr(dist, "all_reduce", lambda tensor, **kwargs: None)
    actor = worker()
    batch = {"task_indices": torch.tensor([0, 0, 0, 1])}
    actor._prepare_sac_task_batch(batch)
    torch.testing.assert_close(batch["sample_weights"].mean(), torch.tensor(1.0))
    first = batch["sample_weights"].clone()
    actor.version = 1
    with pytest.raises(ValueError, match="current-rollout"):
        actor._prepare_sac_task_batch(batch)
    actor.set_task_weights([2.0, 0.5], 2)
    actor._prepare_sac_task_batch(batch)
    assert first[-1] > first[0]
    assert batch["sample_weights"][-1] < batch["sample_weights"][0]
    assert set(batch) == {"task_indices", "sample_weights"}


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.7))

    def forward(self, *, forward_type, obs, actions=None, **kwargs):
        x = obs["x"].reshape(-1, 1)
        if forward_type == ForwardType.SAC:
            return self.scale * x, self.scale * x * 2, None
        assert forward_type == ForwardType.SAC_Q
        return self.scale * x * torch.tensor([[1.0, 3.0]])


def loss_worker():
    actor = worker(3)
    actor.cfg.actor.model = {
        "model_type": "openpi",
        "num_action_chunks": 10,
        "openpi": {"use_dsrl": True, "dsrl_num_q_heads": 2},
    }
    actor.cfg.algorithm = {
        "actor_agg_q": "mean",
        "agg_q": "mean",
        "backup_entropy": False,
        "gamma": 0.99,
        "bootstrap_type": "standard",
    }
    actor.model = Model()
    actor.target_model = Model()
    actor.use_dsrl = True
    actor.torch_dtype = torch.float32
    actor.critic_subsample_size = 0
    actor.entropy_temp = SimpleNamespace(
        alpha=torch.tensor(0.2), compute_alpha=lambda: torch.tensor(0.2)
    )
    actor.target_entropy = -1
    return actor


@pytest.mark.parametrize("neutral", [True, False])
def test_real_sac_forward_losses_and_gradients(neutral):
    actor = loss_worker()
    x = torch.tensor([1.0, 2.0, 4.0])
    batch = {
        "curr_obs": {"x": x},
        "next_obs": {"x": x + 1},
        "actions": torch.zeros(3, 32),
        "rewards": torch.zeros(3, 10),
        "terminations": torch.ones(3, 10, dtype=torch.bool),
        "truncations": torch.zeros(3, 10, dtype=torch.bool),
    }
    batch["rewards"][:, 0] = torch.tensor([1.0, 0.0, 2.0])
    weights = torch.ones(3) if neutral else torch.tensor([0.3, 0.6, 2.1])
    for method in ("forward_actor", "forward_critic"):
        run = inspect.unwrap(getattr(EmbodiedSACFSDPPolicy, method))
        original = run(actor, batch)[0]
        weighted = run(actor, {**batch, "sample_weights": weights})[0]
        if method == "forward_actor":
            per_sample = 0.2 * actor.model.scale * x * 2 - actor.model.scale * x * 2
        else:
            q = actor.model.scale * x[:, None] * torch.tensor([[1.0, 3.0]])
            per_sample = (q - batch["rewards"][:, :1]).square().mean(1)
        expected = (per_sample * weights).mean()
        torch.testing.assert_close(weighted, expected)
        torch.testing.assert_close(
            torch.autograd.grad(weighted, actor.model.scale)[0],
            torch.autograd.grad(expected, actor.model.scale)[0],
        )
        if neutral:
            torch.testing.assert_close(weighted, original)
    alpha = inspect.unwrap(EmbodiedSACFSDPPolicy.forward_alpha)
    torch.testing.assert_close(
        alpha(actor, batch), alpha(actor, {**batch, "sample_weights": weights})
    )


def _distributed(rank, init_file, micro_size):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        actor = worker(8, 2)
        actor._rank = rank
        indices = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1])
        batch = {"task_indices": indices[rank * 4 : (rank + 1) * 4]}
        actor._prepare_sac_task_batch(batch)
        model = torch.nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.5)
        ddp = DistributedDataParallel(model)
        x = torch.arange(1, 9).float().reshape(-1, 1)
        for start in range(0, 4, micro_size):
            values = ddp(x[rank * 4 + start : rank * 4 + start + micro_size]).square()
            loss = weighted_sac_mean(
                values, batch["sample_weights"][start : start + micro_size]
            )
            (loss / (4 // micro_size)).backward()
        reference = torch.nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            reference.weight.fill_(0.5)
        weights = torch.tensor([0.5, 2.0])[indices]
        weighted_sac_mean(reference(x).square(), weights / weights.mean()).backward()
        torch.testing.assert_close(model.weight.grad, reference.weight.grad)
        actor.replay_buffer = SimpleNamespace(
            available_samples=2 if rank == 0 else 10, is_ready=lambda _: True
        )
        assert not actor._all_ranks_replay_ready(1)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("micro_size", [1, 2, 4])
def test_two_rank_accumulation_and_joint_warmup(tmp_path, micro_size):
    mp.spawn(
        _distributed, args=(str(tmp_path / "gloo"), micro_size), nprocs=2, join=True
    )


def test_sac_controller_resume_by_name_and_algorithm_format(tmp_path):
    tracker = SuccessWeightController(
        ["easy", "hard"], DEFAULTS, {}, state_format="sac_multitask_v1"
    )
    tracker.update(torch.tensor([[3, 1], [4, 4]]), 1)
    tracker.save(str(tmp_path), 1)
    resumed = SuccessWeightController(
        ["hard", "easy"], DEFAULTS, {}, state_format="sac_multitask_v1"
    )
    resumed.restore(str(tmp_path), 1)
    torch.testing.assert_close(resumed.weights, tracker.weights.flip(0))
    assert resumed.version == 1
    with pytest.raises(ValueError, match="format"):
        SuccessWeightController(["easy", "hard"], DEFAULTS, {}).restore(
            str(tmp_path), 1
        )


def test_sac_checkpoint_preserves_update_count_and_rng(tmp_path):
    actor = worker()
    actor.update_step = 7
    actor.critic_sample_generator = torch.Generator().manual_seed(27)
    actor._target_shadow_f32 = {"q": torch.tensor([0.123456789])}
    (tmp_path / "sac_components").mkdir()
    actor._save_multi_task_sac_state(str(tmp_path), 1)
    expected = torch.rand(3, generator=actor.critic_sample_generator)
    state = actor._read_multi_task_sac_state(str(tmp_path))
    actor.update_step = 0
    actor.version = 0
    actor._target_shadow_f32["q"].zero_()
    actor._restore_multi_task_sac_state(state)
    assert actor.update_step == 7 and actor.version == 1
    torch.testing.assert_close(
        torch.rand(3, generator=actor.critic_sample_generator), expected
    )
    assert state["update_step"] == 7
    torch.testing.assert_close(
        state["target_shadow_f32"]["q"], actor._target_shadow_f32["q"]
    )


def test_sac_metadata_lists_tasks_without_single_target():
    actor = worker()
    actor.cfg.algorithm = {"loss_type": "embodied_sac"}
    actor.cfg.env.train.init_params = {"id": "example", "task_id": 6}
    metadata = checkpoint_metadata({"task_id": 6, "target_object": "wrong"}, actor.cfg)
    assert "task_id" not in metadata and "target_object" not in metadata
    assert metadata["multi_task_format"] == "sac_multitask_v1"
    assert [task["name"] for task in metadata["tasks"]] == ["easy", "hard"]


@pytest.mark.parametrize(
    "env_id,boundary",
    [
        ("Isaac-Libero-Franka-Hybrid-Tactile-v0", "terminal_safe_hdf5_v1"),
        ("Isaac-RealWorld-GentleGrasp-XarmUmi-Hybrid-Tactile-v0", "terminal_safe_v1"),
    ],
)
def test_validation_accepts_dsrl_boundaries_and_rejects_unsupported_modes(
    env_id, boundary
):
    cfg = worker().cfg
    cfg.runner = {"task_type": "embodied", "val_check_interval": -1}
    cfg.actor.training_backend = "fsdp"
    cfg.actor.model = {"model_type": "openpi", "openpi": {"use_dsrl": True}}
    cfg.algorithm = {
        "adv_type": "embodied_sac",
        "loss_type": "embodied_sac",
        "reward_type": "chunk_level",
        "dsrl_transition_boundary_semantics": "explicit",
    }
    cfg.rollout = {
        "pipeline_stage_num": 2,
        "collect_transitions": True,
        "collect_prev_infos": False,
    }
    cfg.env.train.env_type = "isaaclab"
    cfg.env.train.auto_reset = False
    cfg.env.train.ignore_terminations = False
    cfg.env.train.max_episode_steps = 20
    cfg.env.train.max_steps_per_rollout_epoch = 20
    cfg.env.train.init_params = {
        "id": env_id,
        "chunk_boundary_mode": boundary,
        "task_suite": "example",
        "task_id": 0,
    }
    validate_config(cfg)
    cfg.algorithm.demo_buffer = {}
    with pytest.raises(ValueError, match="demo_buffer"):
        validate_config(cfg)
    del cfg.algorithm.demo_buffer
    cfg.algorithm.replay_buffer = {"enable_preload": True}
    with pytest.raises(ValueError, match="enable_preload"):
        validate_config(cfg)


def test_disk_backed_replay_remaps_cache_misses_and_new_samples(tmp_path):
    original = TrajectoryReplayBuffer(
        task_names=["easy", "hard"],
        enable_cache=False,
        auto_save=True,
        auto_save_path=str(tmp_path / "live"),
        sample_window_size=3,
    )
    restored = TrajectoryReplayBuffer(
        task_names=["hard", "easy"],
        enable_cache=False,
        auto_save=True,
        auto_save_path=str(tmp_path / "next"),
        sample_window_size=3,
    )
    again = replay("trajectory", ["easy", "hard"])
    try:
        for offset in (0, 2, 4):
            original.add_trajectories([trajectory(offset)])
        original.save_checkpoint(str(tmp_path / "one"))
        expected = original.sample(8)
        restored.load_checkpoint(str(tmp_path / "one"))
        actual = restored.sample(8)
        torch.testing.assert_close(actual["actions"], expected["actions"])
        torch.testing.assert_close(actual["task_indices"], 1 - expected["task_indices"])
        new = trajectory(6)
        new.task_indices = 1 - new.task_indices
        restored.add_trajectories([new])
        restored.save_checkpoint(str(tmp_path / "two"))
        again.load_checkpoint(str(tmp_path / "two"))
        batch = again.sample(8)
        torch.testing.assert_close(
            batch["actions"][:, 0].long() % 2, batch["task_indices"]
        )
    finally:
        original.close()
        restored.close()
        again.close()


def _distributed_migration_readiness(rank, directory, kind):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/gloo",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    original = replay(kind, ["easy", "hard"])
    restored = replay(kind, ["hard"])
    try:
        traj = trajectory()
        traj.task_indices.fill_(rank)
        original.add_trajectories([traj])
        path = Path(directory) / str(rank)
        original.save_checkpoint(str(path))
        restored.load_checkpoint(str(path))
        actor = worker(batch_size=8, world_size=2)
        actor.replay_buffer = restored
        assert restored.total_samples == (0 if rank == 0 else 4)
        assert not actor._all_ranks_replay_ready(1)
        # One rank collecting more must not permit any rank to update early.
        traj.task_indices.zero_()
        if rank == 1:
            restored.add_trajectories([traj])
        assert not actor._all_ranks_replay_ready(1)
        if rank == 0:
            restored.add_trajectories([traj])
        assert actor._all_ranks_replay_ready(1)
    finally:
        original.close()
        restored.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("kind", ["compact", "trajectory"])
def test_two_rank_migrated_replay_waits_for_collection(tmp_path, kind):
    mp.spawn(
        _distributed_migration_readiness,
        args=(str(tmp_path), kind),
        nprocs=2,
        join=True,
    )


def test_compact_deletion_preserves_ring_chronology_and_next_write(tmp_path):
    original = replay("compact", ["easy", "hard"])
    restored = replay("compact", ["hard"])
    try:
        for offset in (0, 4):
            traj = trajectory()
            traj.actions[..., 0] = torch.arange(offset, offset + 4).reshape(2, 2)
            original.add_trajectories([traj])
        original.save_checkpoint(str(tmp_path))
        restored.load_checkpoint(str(tmp_path))
        assert restored._storage["actions"][
            restored._chronological_indices(), 0
        ].tolist() == [5, 6]
        fresh = trajectory()
        fresh.task_indices.zero_()
        fresh.actions[..., 0] = torch.arange(8, 12).reshape(2, 2)
        restored.add_trajectories([fresh])
        assert restored._storage["actions"][
            restored._chronological_indices(), 0
        ].tolist() == [6, 8, 9, 10, 11]
    finally:
        original.close()
        restored.close()


def test_trajectory_migration_survives_eviction_and_new_collection(tmp_path):
    original = TrajectoryReplayBuffer(task_names=["easy", "hard"], sample_window_size=3)
    restored = TrajectoryReplayBuffer(
        task_names=["hard", "new"],
        sample_window_size=3,
        cache_size=1,
        auto_save=True,
        auto_save_path=str(tmp_path / "live"),
    )
    try:
        for offset, labels in (
            (0, [[1, 1], [1, 1]]),
            (4, [[0, 1], [1, 0]]),
            (8, [[0, 0], [1, 0]]),
        ):
            traj = trajectory(offset)
            traj.task_indices = torch.tensor(labels)
            traj.forward_inputs["alignment"] = traj.actions[..., :1].clone()
            original.add_trajectories([traj])
        original.save_checkpoint(str(tmp_path / "source"))
        restored.load_checkpoint(str(tmp_path / "source"))
        assert restored.total_samples == 7
        assert [
            restored._trajectory_index[i]["num_samples"]
            for i in restored._trajectory_id_list
        ] == [4, 2, 1]
        scratch = restored._migration_directory.name
        for tid in (0, 1, 2, 0, 2, 1):
            loaded = restored._load_trajectory(tid, "test")
            flat = restored._flatten_trajectory(loaded)
            restored._flat_trajectory_cache.put(tid, flat)
            assert not flat["task_indices"].any()
            torch.testing.assert_close(
                flat["forward_inputs"]["alignment"], flat["actions"][:, :1]
            )
        new = trajectory(12)
        new.task_indices.fill_(1)
        new.forward_inputs["alignment"] = new.actions[..., :1].clone()
        restored.add_trajectories([new])
        assert restored._trajectory_id_list == [0, 1, 2, 3]
        assert restored.available_samples == 7  # Recent IDs 1, 2, 3 only.
        restored.save_checkpoint(str(tmp_path / "next"))
        restored.load_checkpoint(str(tmp_path / "next"))
        assert not Path(scratch).exists()
        assert restored._trajectory_id_list == [1, 2, 3]
        samples = restored.sample(20)
        torch.testing.assert_close(
            samples["forward_inputs"]["alignment"], samples["actions"][:, :1]
        )
        assert (
            samples["task_indices"] == (samples["actions"][:, 0] >= 12).long()
        ).all()
    finally:
        original.close()
        restored.close()
