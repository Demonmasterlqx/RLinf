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

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import rlinf.models.embodiment.openpi as openpi_loader
import rlinf.workers.actor.fsdp_rlt_ac_policy_worker as rlt_worker_module
from rlinf.algorithms.rlt.rollout import predict_rlt_actions
from rlinf.algorithms.rlt.route import (
    RLTRouteContext,
    SimulatorRLTRoute,
    build_rlt_route,
)
from rlinf.algorithms.rlt.transition import use_simulator_transition_replay
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.models.embodiment.mlp_policy import get_model as get_mlp_model
from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
from rlinf.models.embodiment.modules.rlt_token_transformer import RLTTokenTransformer
from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0Config,
    OpenPi0ForRLActionPrediction,
)
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import (
    AsyncRLTACFSDPPolicy,
    RLTACFSDPPolicy,
    RLTACReplayMixin,
)
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


def _rlt_obs(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "z_rl": torch.zeros(batch_size, 2048),
        "proprio": torch.zeros(batch_size, 7),
        "ref_chunk": torch.full((batch_size, 10, 13), 0.25),
    }


def _route_result(batch_size: int = 2) -> dict:
    return {
        "forward_inputs": {
            "action": torch.zeros(batch_size, 130),
            "model_action": torch.zeros(batch_size, 130),
        }
    }


def test_tabero_rlt_stage2_config_defaults_preserve_existing_behavior():
    config = OpenPi0Config()

    assert config.rlt_action_space == "environment"
    assert config.rlt_use_normalized_proprio is False
    assert config.rlt_stage2_encoder_only is False


def test_scaled_tanh_uses_configured_normalized_action_bound():
    policy = RLTMLPPolicy(
        z_dim=2048,
        proprio_dim=7,
        action_dim=13,
        num_action_chunks=10,
        normalized_action_bound=4.0,
        fixed_std=0.05,
    )
    with torch.no_grad():
        policy.actor_mean.weight.zero_()
        policy.actor_mean.bias.fill_(4.0)

    actions, log_probs, _ = policy.sac_forward(_rlt_obs(), deterministic=True)

    expected = 4.0 * torch.tanh(torch.tensor(1.0))
    torch.testing.assert_close(actions, torch.full_like(actions, expected))
    assert torch.isfinite(log_probs).all()
    assert actions.abs().max() < 4.0


def test_rlt_policy_builder_forwards_normalized_action_bound():
    cfg = OmegaConf.create(
        {
            "model_type": "rlt_mlp_policy",
            "z_dim": 8,
            "proprio_dim": 7,
            "action_dim": 2,
            "num_action_chunks": 3,
            "add_q_head": True,
            "normalized_action_bound": 4.0,
        }
    )

    policy = get_mlp_model(cfg)

    assert policy.normalized_action_bound == 4.0


def test_explicit_transition_replay_enables_isaaclab_without_changing_auto_mode():
    explicit = OmegaConf.create(
        {
            "algorithm": {"rlt_schedule": {"transition_replay": True}},
            "env": {"train": {"env_type": "isaaclab"}},
        }
    )
    legacy = OmegaConf.create(
        {
            "algorithm": {"rlt_schedule": {}},
            "env": {"train": {"env_type": "isaaclab"}},
        }
    )

    assert use_simulator_transition_replay(explicit) is True
    assert use_simulator_transition_replay(legacy) is False


def test_full_task_route_records_reference_during_warmup_then_uses_actor():
    route = SimulatorRLTRoute(
        use_schedule=True,
        warmup_updates=2_000,
        actor_scope="full_task",
    )
    student = torch.full((2, 10, 13), 0.75)
    obs = _rlt_obs()

    warmup = route.route(
        RLTRouteContext(
            env_obs={},
            rlt_obs=obs,
            student_actions=student,
            result=_route_result(),
            mode="train",
            version=1_999,
        )
    )
    online = route.route(
        RLTRouteContext(
            env_obs={},
            rlt_obs=obs,
            student_actions=student,
            result=_route_result(),
            mode="train",
            version=2_000,
        )
    )

    torch.testing.assert_close(warmup.actions, obs["ref_chunk"])
    assert warmup.result["forward_inputs"]["record_transition"].all()
    assert not warmup.result["forward_inputs"]["actor_switch"].any()
    torch.testing.assert_close(online.actions, student)
    assert online.result["forward_inputs"]["record_transition"].all()
    assert online.result["forward_inputs"]["actor_switch"].all()


def test_full_task_route_uses_loaded_actor_during_standalone_eval():
    route = SimulatorRLTRoute(
        use_schedule=True,
        warmup_updates=2_000,
        actor_scope="full_task",
        standalone_eval=True,
    )
    student = torch.full((2, 10, 13), 0.75)

    evaluated = route.route(
        RLTRouteContext(
            env_obs={},
            rlt_obs=_rlt_obs(),
            student_actions=student,
            result=_route_result(),
            mode="eval",
            version=0,
        )
    )

    torch.testing.assert_close(evaluated.actions, student)
    assert evaluated.result["forward_inputs"]["actor_switch"].all()


def test_full_task_route_periodic_eval_still_obeys_warmup():
    route = SimulatorRLTRoute(
        use_schedule=True,
        warmup_updates=2_000,
        actor_scope="full_task",
        standalone_eval=False,
    )
    student = torch.full((2, 10, 13), 0.75)
    obs = _rlt_obs()

    evaluated = route.route(
        RLTRouteContext(
            env_obs={},
            rlt_obs=obs,
            student_actions=student,
            result=_route_result(),
            mode="eval",
            version=1_999,
        )
    )

    torch.testing.assert_close(evaluated.actions, obs["ref_chunk"])
    assert not evaluated.result["forward_inputs"]["actor_switch"].any()


def test_route_builder_reads_full_task_scope():
    cfg = OmegaConf.create(
        {
            "algorithm": {
                "rlt_schedule": {
                    "enable": True,
                    "transition_replay": True,
                    "warmup_post_collect_updates": 2_000,
                },
                "rlt_route": {"actor_scope": "full_task"},
            },
            "env": {"train": {"env_type": "isaaclab"}},
            "runner": {"only_eval": True},
            "rollout": {
                "rlt_feature_model": {
                    "openpi": {"rlt_action_space": "model_normalized"}
                }
            },
        }
    )

    route = build_rlt_route(cfg)

    assert isinstance(route, SimulatorRLTRoute)
    assert route.actor_scope == "full_task"
    assert route.standalone_eval is True
    assert route.action_space == "model_normalized"


def test_normalized_expert_takeover_uses_model_action_not_environment_action():
    class _ExpertModel:
        config = SimpleNamespace(action_dim=32)

        def predict_action_batch(self, env_obs, mode, compute_values):
            del env_obs, mode, compute_values
            environment_actions = torch.full((1, 10, 13), 99.0)
            model_actions = torch.full((1, 50 * 32), 0.5)
            return environment_actions, {
                "forward_inputs": {"model_action": model_actions}
            }

    route = SimulatorRLTRoute(
        use_schedule=False,
        warmup_updates=0,
        actor_scope="full_task",
        action_space="model_normalized",
    )
    routed = route.route(
        RLTRouteContext(
            env_obs={},
            rlt_obs=_rlt_obs(batch_size=1),
            student_actions=torch.full((1, 10, 13), 0.75),
            result=_route_result(batch_size=1),
            mode="train",
            intervene_requested=torch.tensor([True]),
            expert_model=_ExpertModel(),
        )
    )

    torch.testing.assert_close(routed.actions, torch.full((1, 10, 13), 0.5))
    torch.testing.assert_close(
        routed.result["forward_inputs"]["action"].reshape(1, 10, 13),
        routed.actions,
    )
    assert routed.result["intervene_flags"].all()


def test_normalized_expert_takeover_requires_model_action():
    class _ExpertModel:
        config = SimpleNamespace(action_dim=32)

        def predict_action_batch(self, env_obs, mode, compute_values):
            del env_obs, mode, compute_values
            return torch.zeros(1, 10, 13), {"forward_inputs": {}}

    route = SimulatorRLTRoute(
        use_schedule=False,
        warmup_updates=0,
        actor_scope="full_task",
        action_space="model_normalized",
    )

    with pytest.raises(ValueError, match="model_action"):
        route.route(
            RLTRouteContext(
                env_obs={},
                rlt_obs=_rlt_obs(batch_size=1),
                student_actions=torch.full((1, 10, 13), 0.75),
                result=_route_result(batch_size=1),
                mode="train",
                intervene_requested=torch.tensor([True]),
                expert_model=_ExpertModel(),
            )
        )


def test_encoder_only_mode_discards_decoder_without_changing_encoding():
    torch.manual_seed(0)
    module = RLTTokenTransformer(
        input_dim=8,
        embed_dim=8,
        num_rl_tokens=1,
        prefix_seq_len=4,
        num_layers=1,
        num_heads=2,
        mlp_ratio=2.0,
    ).eval()
    prefix = torch.randn(2, 4, 8)
    before = module.encode_flat(prefix)

    module.discard_decoder()
    after = module.encode_flat(prefix)

    assert module.decoder is None
    torch.testing.assert_close(after, before)


def test_stage2_feature_model_preparation_discards_decoder_after_load():
    module = RLTTokenTransformer(
        input_dim=8,
        embed_dim=8,
        num_rl_tokens=1,
        prefix_seq_len=4,
        num_layers=1,
        num_heads=2,
        mlp_ratio=2.0,
    )
    model = SimpleNamespace(rlt_module=module)
    config = SimpleNamespace(rlt_stage2_encoder_only=True, use_rlt=True)

    openpi_loader._prepare_stage2_feature_model(model, config)

    assert model.rlt_module.decoder is None


class _FeatureModel:
    def __init__(self):
        self.config = SimpleNamespace(rlt_action_space="model_normalized")
        self.decoded_input = None

    def extract_rlt_obs(self, env_obs):
        del env_obs
        return _rlt_obs(batch_size=1)

    def decode_rlt_actions(self, actions, env_obs):
        del env_obs
        self.decoded_input = actions.clone()
        return actions + 10.0


class _PolicyModel:
    def predict_action_batch(self, env_obs, mode, return_obs):
        del env_obs, mode, return_obs
        actions = torch.full((1, 10, 13), 0.75)
        return actions, _route_result(batch_size=1)


def test_predict_rlt_actions_keeps_replay_normalized_and_decodes_env_actions():
    feature_model = _FeatureModel()
    route = SimulatorRLTRoute(
        use_schedule=False,
        warmup_updates=0,
        actor_scope="full_task",
    )

    env_actions, result = predict_rlt_actions(
        policy_model=_PolicyModel(),
        feature_model=feature_model,
        rlt_route=route,
        env_obs={},
        final_obs=None,
        mode="train",
    )

    replay_actions = result["forward_inputs"]["action"].reshape(1, 10, 13)
    torch.testing.assert_close(feature_model.decoded_input, replay_actions)
    torch.testing.assert_close(env_actions, replay_actions + 10.0)
    torch.testing.assert_close(
        result["forward_inputs"]["environment_action"].reshape(1, 10, 13),
        env_actions,
    )


def test_transition_metrics_report_normalized_saturation_and_physical_force():
    trajectory = Trajectory(max_episode_length=1)
    trajectory.actions = torch.tensor(
        [[[[3.99] * 13] * 10]], dtype=torch.float32
    )
    environment_action = torch.zeros(1, 1, 130)
    environment_action.reshape(1, 1, 10, 13)[..., 7:] = 12.5
    trajectory.forward_inputs = {"environment_action": environment_action}
    worker = RLTACReplayMixin()
    worker.cfg = OmegaConf.create(
        {"actor": {"model": {"normalized_action_bound": 4.0}}}
    )

    metrics = worker._transition_replay_metrics([trajectory])

    assert metrics["replay/normalized_action_abs_max"] == pytest.approx(3.99)
    assert metrics["replay/normalized_action_saturation_ratio"] == 1.0
    assert metrics["replay/physical_force_abs_max"] == 12.5


def test_transition_metrics_reduce_counts_and_maxima_across_actor_ranks(monkeypatch):
    trajectory = Trajectory(max_episode_length=1)
    trajectory.actions = torch.full((1, 1, 10, 13), 3.0)
    environment_action = torch.zeros(1, 1, 130)
    environment_action.reshape(1, 1, 10, 13)[..., 7:] = 12.5
    trajectory.forward_inputs = {"environment_action": environment_action}
    worker = RLTACReplayMixin()
    worker.cfg = OmegaConf.create(
        {"actor": {"model": {"normalized_action_bound": 4.0}}}
    )

    def fake_all_reduce(values, op):
        if op == torch.distributed.ReduceOp.SUM:
            return {key: value * 2.0 for key, value in values.items()}
        if op == torch.distributed.ReduceOp.MAX:
            return {key: value + 1.0 for key, value in values.items()}
        raise AssertionError(f"unexpected reduction: {op}")

    monkeypatch.setattr(rlt_worker_module, "all_reduce_dict", fake_all_reduce)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    metrics = worker._transition_replay_metrics([trajectory])

    assert metrics["replay/transition_count"] == 2.0
    assert metrics["replay/normalized_action_abs_max"] == 4.0
    assert metrics["replay/physical_force_abs_max"] == 13.5


def test_async_drain_disables_cross_rank_metric_reductions():
    worker = object.__new__(AsyncRLTACFSDPPolicy)
    worker._recv_queue = rlt_worker_module.queue.Queue()
    worker._recv_queue.put("trajectory")
    calls = []
    counters = []

    def ingest(recv_list, *, reduce_metrics=True):
        calls.append((recv_list, reduce_metrics))
        return 3, 1

    worker._ingest_rollout_trajectories = ingest
    worker._update_rollout_ingest_counters = lambda added, completed: counters.append(
        (added, completed)
    )

    worker._drain_received_trajectories()

    assert calls == [(["trajectory"], False)]
    assert counters == [(3, 1)]


def test_transition_replay_keeps_all_non_auto_reset_rollout_epochs():
    num_epochs = 2
    chunks_per_epoch = 4
    traj_len = num_epochs * chunks_per_epoch
    trajectory = Trajectory(max_episode_length=40)
    trajectory.actions = torch.arange(traj_len, dtype=torch.float32).reshape(
        traj_len, 1, 1
    )
    # Real EmbodiedRolloutResult includes one bootstrap reward/done slot at
    # the beginning of each rollout epoch.
    epoch_rewards = torch.tensor(
        [-10.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32
    )
    trajectory.rewards = epoch_rewards.repeat(num_epochs).reshape(-1, 1, 1)
    trajectory.intervene_flags = torch.zeros(traj_len, 1, 1, dtype=torch.bool)
    trajectory.prev_logprobs = torch.zeros(traj_len, 1, 1)
    trajectory.prev_values = torch.zeros(traj_len, 1, 1)
    trajectory.versions = torch.zeros(traj_len, 1, 1)

    # Each non-auto-reset epoch contributes an initial bootstrap slot followed
    # by one terminal flag per action chunk.
    episode_dones = torch.tensor(
        [False, False, False, False, True], dtype=torch.bool
    )
    trajectory.dones = episode_dones.repeat(num_epochs).reshape(-1, 1, 1)
    trajectory.terminations = trajectory.dones.clone()
    trajectory.truncations = torch.zeros_like(trajectory.dones)
    trajectory.forward_inputs = {
        "record_transition": torch.ones(traj_len, 1, 1, dtype=torch.bool),
    }
    trajectory.curr_obs = {
        "z_rl": torch.arange(traj_len, dtype=torch.float32).reshape(
            traj_len, 1, 1
        ),
        "proprio": torch.zeros(traj_len, 1, 1),
        "ref_chunk": torch.zeros(traj_len, 1, 1),
    }
    trajectory.next_obs = {
        key: value + 1.0 for key, value in trajectory.curr_obs.items()
    }

    worker = RLTACReplayMixin()
    worker.cfg = OmegaConf.create({"env": {"train": {"auto_reset": False}}})
    worker.replay_buffer = TrajectoryReplayBuffer(auto_save=False)

    transitions, completed = worker._transition_replay_trajectories(trajectory)

    assert len(transitions) == traj_len
    assert completed == num_epochs
    assert [int(item.actions.item()) for item in transitions] == list(
        range(traj_len)
    )
    assert [bool(item.dones.item()) for item in transitions] == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
    ]
    assert [float(item.rewards.item()) for item in transitions] == [
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def test_unrecorded_terminal_still_stops_non_auto_reset_epoch():
    trajectory = Trajectory(max_episode_length=30)
    trajectory.actions = torch.arange(3, dtype=torch.float32).reshape(3, 1, 1)
    trajectory.rewards = torch.zeros(4, 1, 1)
    trajectory.dones = torch.tensor([False, False, True, False]).reshape(4, 1, 1)
    trajectory.terminations = trajectory.dones.clone()
    trajectory.truncations = torch.zeros_like(trajectory.dones)
    trajectory.forward_inputs = {
        "record_transition": torch.tensor([True, False, True]).reshape(3, 1, 1),
    }
    trajectory.curr_obs = {
        "z_rl": torch.arange(3, dtype=torch.float32).reshape(3, 1, 1),
    }
    trajectory.next_obs = {
        "z_rl": torch.arange(1, 4, dtype=torch.float32).reshape(3, 1, 1),
    }

    worker = RLTACReplayMixin()
    worker.cfg = OmegaConf.create({"env": {"train": {"auto_reset": False}}})
    worker.replay_buffer = TrajectoryReplayBuffer(auto_save=False)

    transitions, completed = worker._transition_replay_trajectories(trajectory)

    assert [int(item.actions.item()) for item in transitions] == [0]
    assert completed == 1


def test_model_normalized_reference_skips_output_transform_and_slices_env_dims():
    normalized = torch.arange(2 * 50 * 32, dtype=torch.float32).reshape(2, 50, 32)

    class _Model:
        config = SimpleNamespace(
            rlt_action_space="model_normalized",
            action_chunk=10,
            action_env_dim=13,
        )

        def output_transform(self, outputs):
            raise AssertionError("normalized reference must not be decoded")

        def _make_output_transform_input(self, actions, observation):
            del actions, observation
            raise AssertionError("normalized reference must not be decoded")

    reference = OpenPi0ForRLActionPrediction._prepare_rlt_reference_chunk(
        _Model(),
        {"actions": normalized},
        SimpleNamespace(state=torch.zeros(2, 32)),
    )

    torch.testing.assert_close(reference, normalized[:, :10, :13])


def test_normalized_proprio_uses_transformed_state_before_state_selection():
    class _Model:
        config = SimpleNamespace(rlt_use_normalized_proprio=True)

        def _select_configured_state(self, state):
            return state[..., [0, 2, 4]]

    raw_state = torch.full((2, 7), 100.0)
    normalized_state = torch.arange(2 * 32, dtype=torch.float32).reshape(2, 32)

    proprio = OpenPi0ForRLActionPrediction._prepare_rlt_proprio(
        _Model(), raw_state, normalized_state
    )

    torch.testing.assert_close(proprio, normalized_state[:, [0, 2, 4]])


def test_decode_rlt_actions_pads_model_width_and_uses_current_observation_state():
    normalized = torch.randn(2, 10, 13)
    captured = {}

    class _Model:
        config = SimpleNamespace(action_dim=32, action_env_dim=13)

        def obs_processor(self, env_obs):
            return env_obs

        def input_transform(self, env_obs, transpose=False):
            assert transpose is False
            return env_obs

        def precision_processor(self, env_obs):
            return env_obs

        def _observation_from_dict(self, env_obs):
            return SimpleNamespace(state=env_obs["states"] + 1.0)

        def _make_output_transform_input(self, actions, observation):
            captured["padded"] = actions.clone()
            captured["state"] = observation.state.clone()
            return {"actions": actions, "state": observation.state}

        def output_transform(self, outputs):
            return {"actions": outputs["actions"] + 10.0}

    decoded = OpenPi0ForRLActionPrediction.decode_rlt_actions(
        _Model(), normalized, {"states": torch.zeros(2, 7)}
    )

    assert captured["padded"].shape == (2, 10, 32)
    torch.testing.assert_close(captured["padded"][..., :13], normalized)
    assert torch.count_nonzero(captured["padded"][..., 13:]) == 0
    torch.testing.assert_close(captured["state"], torch.ones(2, 7))
    torch.testing.assert_close(decoded, normalized + 10.0)


def test_tabero_rlt_stage2_task5_firm_config_composes(monkeypatch):
    rlinf_root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("EMBODIED_PATH", str(rlinf_root / "examples" / "embodiment"))
    config_dir = rlinf_root / "examples" / "tabero"

    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg = compose(config_name="tabero_rlt_stage2_ac_task5_firm")

    assert cfg.cluster.component_placement.actor == "0-7"
    assert cfg.cluster.component_placement.rollout == "0:0,7:1"
    assert cfg.cluster.component_placement.env == "0:0,7:1"
    assert cfg.runner.max_epochs == 100
    assert cfg.runner.save_interval == 5
    assert list(cfg.runner.logger.logger_backends) == ["tensorboard", "wandb"]
    assert cfg.algorithm.rlt_schedule.transition_replay is True
    assert cfg.algorithm.rlt_schedule.warmup_min_size == 256
    assert cfg.algorithm.rlt_schedule.warmup_post_collect_updates == 2_000
    assert cfg.algorithm.rlt_route.actor_scope == "full_task"
    assert cfg.actor.micro_batch_size == 32
    assert cfg.actor.global_batch_size == 512
    assert cfg.actor.model.normalized_action_bound == 4.0
    assert cfg.actor.model.fixed_std == 0.05
    assert cfg.actor.fsdp_config.use_orig_params is False
    assert cfg.env.train.total_num_envs == 16
    assert cfg.env.train.rollout_epoch == 4
    assert cfg.env.train.max_episode_steps == 360
    assert cfg.env.train.init_params.task_id == 5
    assert list(cfg.env.train.init_params.prompt_conditions.condition_cycle) == [
        "firm"
    ]
    assert cfg.env.train.video_cfg.save_video is False
    assert cfg.env.eval.total_num_envs == 2
    assert cfg.env.eval.rollout_epoch == 25
    assert cfg.rollout.collect_transitions is True
    assert cfg.rollout.rlt_feature_model.openpi.rlt_action_space == (
        "model_normalized"
    )
    assert cfg.rollout.rlt_feature_model.openpi.rlt_use_normalized_proprio is True
    assert cfg.rollout.rlt_feature_model.openpi.rlt_stage2_encoder_only is True
    assert cfg.rollout.rlt_feature_model.model_path.endswith(
        "global_step_2000/actor"
    )


@pytest.mark.parametrize("worker_type", [RLTACFSDPPolicy, AsyncRLTACFSDPPolicy])
def test_rlt_checkpoint_restores_schedule_state(tmp_path, monkeypatch, worker_type):
    monkeypatch.setattr(
        EmbodiedSACFSDPPolicy,
        "save_checkpoint",
        lambda self, save_base_path, step: None,
    )
    monkeypatch.setattr(
        EmbodiedSACFSDPPolicy,
        "load_checkpoint",
        lambda self, load_base_path: None,
    )
    worker = worker_type.__new__(worker_type)
    worker._rank = 3
    worker.use_rlt_schedule = True
    expected = {
        "update_step": 2_000,
        "transitions_since_train": 17,
        "episodes_since_train": 2,
        "total_transitions_added": 12_345,
        "total_episodes_added": 98,
        "_warmup_ready_total_transitions": 4_096,
        "_warmup_ready_total_episodes": 32,
        "pending_update_budget": 7,
    }
    for key, value in expected.items():
        setattr(worker, key, value)

    actor_path = tmp_path / "global_step_5" / "actor"
    worker.save_checkpoint(str(actor_path), step=5)
    for key in expected:
        setattr(worker, key, 0)
    worker.load_checkpoint(str(actor_path))

    assert {key: getattr(worker, key) for key in expected} == expected
    state_path = (
        actor_path / "sac_components" / "rlt_schedule" / "checkpoint_rank_3.pt"
    )
    assert state_path.is_file()


def test_rlt_checkpoint_rejects_sidecar_from_different_global_step(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        EmbodiedSACFSDPPolicy,
        "load_checkpoint",
        lambda self, load_base_path: None,
    )
    worker = RLTACFSDPPolicy.__new__(RLTACFSDPPolicy)
    worker._rank = 0
    worker.use_rlt_schedule = True
    actor_path = tmp_path / "global_step_6" / "actor"
    state_path = Path(worker._rlt_checkpoint_path(str(actor_path)))
    state_path.parent.mkdir(parents=True)
    state = dict.fromkeys(worker._RLT_CHECKPOINT_FIELDS, 0)
    state["global_step"] = 5
    torch.save(state, state_path)
    gathered = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def gather_errors(output, local_error):
        gathered.append(local_error)
        output[:] = [local_error, None]

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather_errors)

    with pytest.raises(ValueError, match="global step mismatch"):
        worker.load_checkpoint(str(actor_path))
    assert gathered and "global step mismatch" in gathered[0]


def test_rlt_checkpoint_rejects_cross_rank_schedule_divergence(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        EmbodiedSACFSDPPolicy,
        "load_checkpoint",
        lambda self, load_base_path: None,
    )
    worker = RLTACFSDPPolicy.__new__(RLTACFSDPPolicy)
    worker._rank = 0
    worker.use_rlt_schedule = True
    actor_path = tmp_path / "global_step_5" / "actor"
    state_path = Path(worker._rlt_checkpoint_path(str(actor_path)))
    state_path.parent.mkdir(parents=True)
    state = dict.fromkeys(worker._RLT_CHECKPOINT_FIELDS, 0)
    state["global_step"] = 5
    torch.save(state, state_path)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, local_error: output.__setitem__(slice(None), [None, None]),
    )

    def divergent_reduce(values, *, dtype, op):
        del dtype
        reduced = dict(values)
        if op == torch.distributed.ReduceOp.MAX:
            reduced["update_step"] += 1
        return reduced

    monkeypatch.setattr(rlt_worker_module, "all_reduce_dict", divergent_reduce)

    with pytest.raises(ValueError, match="cross-rank schedule state mismatch"):
        worker.load_checkpoint(str(actor_path))


def test_rlt_checkpoint_requires_rank_schedule_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(
        EmbodiedSACFSDPPolicy,
        "load_checkpoint",
        lambda self, load_base_path: None,
    )
    worker = RLTACFSDPPolicy.__new__(RLTACFSDPPolicy)
    worker._rank = 2
    worker.use_rlt_schedule = True
    worker.logger = SimpleNamespace(warning=lambda *args, **kwargs: None)

    with pytest.raises(FileNotFoundError, match="checkpoint_rank_2.pt"):
        worker.load_checkpoint(str(tmp_path))


def test_rlt_checkpoint_without_schedule_preserves_legacy_resume(tmp_path, monkeypatch):
    load_calls = []
    monkeypatch.setattr(
        EmbodiedSACFSDPPolicy,
        "load_checkpoint",
        lambda self, load_base_path: load_calls.append(load_base_path),
    )
    worker = RLTACFSDPPolicy.__new__(RLTACFSDPPolicy)
    worker._rank = 2
    worker.use_rlt_schedule = False

    worker.load_checkpoint(str(tmp_path))

    assert load_calls == [str(tmp_path)]


def test_run_training_reports_post_update_schedule_step(monkeypatch):
    worker = RLTACFSDPPolicy.__new__(RLTACFSDPPolicy)
    worker.use_rlt_schedule = True
    worker.cfg = OmegaConf.create(
        {
            "actor": {
                "enable_offload": False,
                "global_batch_size": 1,
                "micro_batch_size": 1,
            }
        }
    )
    worker._world_size = 1
    worker.rlt_schedule_cfg = {"warmup_post_collect_updates": 2_000}
    worker.update_step = 0
    worker.critic_actor_ratio = 4
    worker.pending_update_budget = 1
    worker.transitions_since_train = 1
    worker.episodes_since_train = 0
    worker.model = SimpleNamespace(train=lambda: None)
    worker._rlt_updates_to_run = lambda: (
        1,
        {"rlt/update_step": 0.0, "rlt/ready_for_online": 0.0},
    )
    worker.update_one_epoch = lambda train_actor: {}
    worker.process_train_metrics = lambda metrics: metrics
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    metrics = worker.run_training()

    assert worker.update_step == 1
    assert metrics["rlt/update_step"] == 1.0
