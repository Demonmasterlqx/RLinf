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

"""Synthetic force shaping and task contracts; no simulator or local assets."""

import warnings
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.envs.isaaclab.tasks.realworld_tabero_tacfield import (
    _terminal_success_reward,
    _termination_term_items,
)
from rlinf.envs.isaaclab.tasks.tabero_force_reward import (
    TERMINAL_SUCCESS_KEY,
    make_trajectory_force_success_reward_term,
    record_success_metrics,
    valid_success_mask,
    validate_force_reward_cfg,
)
from rlinf.utils.multi_task import task_env_cfg, validate_environments


def normalized(**kwargs):
    return {"enabled": True, "min_effort": 10.0, "max_effort": 20.0, **kwargs}


def make_term(
    squeeze,
    *,
    direct,
    minimum_samples=1,
    terminal_reward=1.0,
    mode="normalized",
    **normalized_overrides,
):
    num_envs = len(squeeze)
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        step_dt=0.05,
        squeeze=torch.tensor(squeeze, dtype=torch.float32),
        grasp=torch.ones(num_envs, dtype=torch.bool),
    )
    terms = {
        name: torch.zeros(num_envs, dtype=torch.bool) for name in ("success", "failure")
    }
    env.termination_manager = SimpleNamespace(
        get_term=lambda name: terms[name],
        time_outs=torch.zeros(num_envs, dtype=torch.bool),
        active_terms=list(terms),
    )
    env.observation_manager = SimpleNamespace(
        compute_group=lambda *args, **kwargs: {"grasp": env.grasp}
    )

    def reader(env, **kwargs):
        force = torch.zeros(num_envs, 2, 3)
        force[:, :, 2] = env.squeeze[:, None] / 2
        return force[:, None] if direct else force

    cfg = {
        "normalize_effort_reward": normalized(
            min_valid_samples=minimum_samples, **normalized_overrides
        )
    }
    if mode == "old":
        cfg = {
            "force_bonus": {
                "enabled": True,
                "coefficient": 10.0,
                "max_bonus": 1.0,
                "min_valid_samples": minimum_samples,
            }
        }
    params = validate_force_reward_cfg(cfg, terminal_reward)
    params.pop("enabled")
    params.update(
        success_term_name="success",
        failure_term_names=("failure",),
        terminal_reward=terminal_reward,
    )
    if direct:
        params["direct_force_reader"] = reader
    else:
        params.update(force_reader=reader, force_sources=(("grasp", "sensor"),))

    class Base:
        def __init__(self, cfg, env):
            pass

    term = make_trajectory_force_success_reward_term(Base)(
        SimpleNamespace(params=params), env
    )
    return env, terms, term, params


@pytest.mark.parametrize("direct", [True, False])
@pytest.mark.parametrize("step_dt", [0.02, 0.05])
@pytest.mark.parametrize(
    "overrides,expected_bonus,expected_reward",
    [
        ({}, [1, 1, 0.5, 0, -0.5, -1, -1], [2, 4, 1.5, 1, 0.5, 0, 0]),
        ({"coefficient": 0}, [0] * 7, [1, 2, 1, 1, 1, 1, 1]),
        (
            {"coefficient": 0.5},
            [0.5, 0.5, 0.25, 0, -0.25, -0.5, -0.5],
            [1.5, 3, 1.25, 1, 0.75, 0.5, 0.5],
        ),
        ({"coefficient": 2}, [2, 2, 1, 0, -1, -2, -2], [3, 6, 2, 1, 0, -1, -1]),
        (
            {"min_reward": -0.25, "max_reward": 1.5, "coefficient": 2},
            [3, 2, 1, 0, -0.5, -0.5, -0.5],
            [4, 6, 2, 1, 0.5, 0.5, 0.5],
        ),
        (
            {"min_reward": 0},
            [1, 1, 0.5, 0, 0, 0, 0],
            [2, 4, 1.5, 1, 1, 1, 1],
        ),
        (
            {"max_reward": 0},
            [0, 0, 0, 0, -0.5, -1, -1],
            [1, 2, 1, 1, 0.5, 0, 0],
        ),
        (
            {"min_reward": 0.25, "max_reward": 0.25, "coefficient": 2},
            [0.5] * 7,
            [1.5, 3, 1.5, 1.5, 1.5, 1.5, 1.5],
        ),
    ],
)
def test_normalized_endpoints_clipping_and_condition_scaling(
    direct, step_dt, overrides, expected_bonus, expected_reward
):
    env, terms, term, params = make_term(
        [2, 10, 15, 20, 25, 30, 100], direct=direct, **overrides
    )
    terms["success"][:] = True
    env.step_dt = step_dt
    params["env_reward_multipliers"] = (1, 2, 1, 1, 1, 1, 1)
    reward = term(env, **params) * env.step_dt
    torch.testing.assert_close(
        term.current_force_bonus, torch.tensor(expected_bonus, dtype=torch.float32)
    )
    torch.testing.assert_close(
        reward, torch.tensor(expected_reward, dtype=torch.float32)
    )


@pytest.mark.parametrize("direct", [True, False])
@pytest.mark.parametrize("coefficient,expected", [(1.0, -0.5), (2.0, -1.5)])
def test_normalized_success_only_and_negative_total(direct, coefficient, expected):
    env, terms, term, params = make_term(
        [40] * 4, direct=direct, terminal_reward=0.5, coefficient=coefficient
    )
    terms["success"][:] = torch.tensor([True, True, True, False])
    terms["failure"][1] = True
    env.termination_manager.time_outs[2] = True
    torch.testing.assert_close(
        term(env, **params) * env.step_dt, torch.tensor([expected, 0, 0, 0])
    )


@pytest.mark.parametrize("reward_mode", ["binary", "old", "normalized"])
def test_gripper_slip_overrides_success_reward_and_episode_metrics(reward_mode):
    env, terms, term, params = make_term([10, 10], direct=True, mode=reward_mode)
    terms["gripper_slip"] = terms.pop("failure")
    terms["success"][:] = True
    terms["gripper_slip"][0] = True
    env.termination_manager.active_terms = list(terms)
    terminations_cfg = SimpleNamespace(
        success=SimpleNamespace(func=object(), time_out=False),
        gripper_slip=SimpleNamespace(func=object(), time_out=False),
        time_out=SimpleNamespace(func=object(), time_out=True),
    )
    failure_names = tuple(
        name
        for name, cfg in _termination_term_items(terminations_cfg)
        if name != "success" and not cfg.time_out
    )
    assert failure_names == ("gripper_slip",)
    params["failure_term_names"] = failure_names
    if reward_mode == "binary":
        reward = _terminal_success_reward(env, failure_term_names=failure_names)
    else:
        reward = term(env, **params)
        # The diagnostic bonus remains available, but must not be paid on failure.
        assert term.current_force_bonus[0] > 0
    reward *= env.step_dt
    assert reward[0] == 0
    assert reward[1] > 0
    valid_success = valid_success_mask(env)
    assert valid_success.tolist() == [False, True]
    env.returns = torch.zeros(2)
    env.success_once = torch.zeros(2, dtype=torch.bool)
    env.elapsed_steps = torch.ones(2, dtype=torch.long)
    infos = record_success_metrics(env, reward, {TERMINAL_SUCCESS_KEY: valid_success})
    assert infos["episode"]["success_once"].tolist() == [False, True]


@pytest.mark.parametrize("direct", [True, False])
def test_force_history_is_identical_between_reward_forms_and_partial_reset(direct):
    results = []
    for mode in ("old", "normalized"):
        env, terms, term, params = make_term(
            [0, 4], direct=direct, minimum_samples=2, mode=mode, coefficient=2.0
        )
        term(env, **params)
        env.squeeze[:] = torch.tensor([4, 8])
        term(env, **params)
        term.reset([1])
        # Released fingers do not contribute zero-valued samples.
        env.squeeze[:] = torch.tensor([0, 0])
        term(env, **params)
        env.squeeze[:] = torch.tensor([2, 6])
        terms["success"][:] = True
        term(env, **params)
        results.append(
            (term.trajectory_mean_force.clone(), term.valid_sample_count.clone())
        )
        assert term.current_force_bonus[1] == 0  # Not enough post-reset samples.
    for mean, count in results:
        torch.testing.assert_close(mean, torch.tensor([3.0, 6.0]))
        assert count.tolist() == [2, 1]


def test_grasp_gate_and_multiple_sources_keep_maximum_valid_force():
    env, _, term, params = make_term([20], direct=False)
    env.grasp[:] = False
    term(env, **params)
    assert term.valid_sample_count.item() == 0
    env.grasp[:] = True
    term._force_sources = (("grasp", "a"), ("grasp", "b"))
    term._grasp_started = torch.zeros(1, 2, dtype=torch.bool)

    def reader(env, contact_sensor_name):
        force = torch.zeros(1, 2, 3)
        force[..., 2] = 5 if contact_sensor_name == "a" else 15
        return force

    term._force_reader = reader
    term(env, **params)
    assert term.trajectory_mean_force.item() == 30
    assert term.current_force_bonus.item() == -1


@pytest.mark.parametrize(
    "field,value",
    [
        ("min_effort", -1),
        ("min_effort", True),
        ("min_effort", float("nan")),
        ("max_effort", 10),
        ("max_effort", float("inf")),
        ("max_effort", 9),
        ("max_effort", True),
        ("max_effort", float("nan")),
        ("max_effort", None),
        ("min_valid_samples", 0),
        ("min_valid_samples", 1.5),
        ("min_valid_samples", True),
        ("contact_epsilon", -1),
        ("contact_epsilon", float("nan")),
        ("enabled", "true"),
        ("coefficient", -1),
        ("coefficient", True),
        ("coefficient", float("nan")),
        ("coefficient", float("inf")),
        ("coefficient", "0.5"),
        ("coefficient", None),
    ],
)
def test_normalized_parameters_reject_invalid_values(field, value):
    with pytest.raises(ValueError):
        validate_force_reward_cfg(
            {"normalize_effort_reward": normalized(**{field: value})}, 1
        )


@pytest.mark.parametrize("field", ["min_effort", "max_effort"])
def test_normalized_bounds_are_required(field):
    cfg = normalized()
    del cfg[field]
    with pytest.raises(ValueError, match=field):
        validate_force_reward_cfg({"normalize_effort_reward": cfg}, 1)


@pytest.mark.parametrize("field", ["min_reward", "max_reward"])
@pytest.mark.parametrize("value", [True, None, "1", float("nan"), float("inf")])
def test_reward_clip_bounds_require_finite_numbers(field, value):
    with pytest.raises(ValueError, match=field):
        validate_force_reward_cfg(
            {"normalize_effort_reward": normalized(**{field: value})}, 1
        )


def test_reversed_reward_clip_bounds_are_rejected():
    with pytest.raises(ValueError, match="min_reward <= max_reward"):
        validate_force_reward_cfg(
            {"normalize_effort_reward": normalized(min_reward=1, max_reward=-1)}, 1
        )


@pytest.mark.parametrize("direct", [True, False])
def test_legacy_mid_effort_warns_and_preserves_rewards(direct):
    legacy = normalized()
    legacy["mid_effort"] = legacy.pop("max_effort")
    before = deepcopy(legacy)
    with pytest.warns(UserWarning, match="using mid_effort as max_effort"):
        resolved = validate_force_reward_cfg({"normalize_effort_reward": legacy}, 1)
    assert legacy == before
    assert resolved == validate_force_reward_cfg(
        {"normalize_effort_reward": normalized()}, 1
    )
    env, terms, term, params = make_term([10, 20, 30], direct=direct)
    term._normalize_effort_reward = resolved["normalize_effort_reward"]
    terms["success"][:] = True
    torch.testing.assert_close(
        term(env, **params) * env.step_dt, torch.tensor([2.0, 1.0, 0.0])
    )


@pytest.mark.parametrize("midpoint", [15.0, None, float("nan"), "ignored"])
def test_explicit_max_effort_takes_precedence_and_warns(midpoint):
    with pytest.warns(UserWarning, match="max_effort takes precedence"):
        env, terms, term, params = make_term(
            [10, 20, 30, 50, 60], direct=True, max_effort=30, mid_effort=midpoint
        )
    terms["success"][:] = True
    torch.testing.assert_close(
        term(env, **params) * env.step_dt, torch.tensor([2.0, 1.5, 1.0, 0.0, 0.0])
    )


@pytest.mark.parametrize("maximum", [None, float("nan"), 10.0])
def test_invalid_explicit_max_effort_does_not_fall_back_to_mid_effort(maximum):
    with pytest.warns(UserWarning, match="max_effort takes precedence"):
        with pytest.raises(ValueError, match="max_effort"):
            validate_force_reward_cfg(
                {
                    "normalize_effort_reward": normalized(
                        max_effort=maximum, mid_effort=20
                    )
                },
                1,
            )


def test_max_effort_only_is_quiet_and_does_not_derive_an_overflowing_bound():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        env, terms, term, params = make_term([10], direct=True, max_effort=1e308)
        terms["success"][:] = True
        torch.testing.assert_close(
            term(env, **params) * env.step_dt, torch.tensor([2.0])
        )


def test_disabled_rewards_and_mutual_exclusion():
    assert validate_force_reward_cfg({}, 1) == {"enabled": False}
    assert validate_force_reward_cfg(
        {"normalize_effort_reward": {"enabled": False}}, 1
    ) == {"enabled": False}
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_force_reward_cfg(
            {"force_bonus": {"enabled": True}, "normalize_effort_reward": normalized()},
            1,
        )


def task_config():
    return OmegaConf.create(
        {
            "runner": {"val_check_interval": 1},
            "rollout": {"pipeline_stage_num": 3},
            "env": {
                "train": {
                    "env_type": "isaaclab",
                    "auto_reset": False,
                    "ignore_terminations": False,
                    "max_episode_steps": 10,
                    "max_steps_per_rollout_epoch": 10,
                    "init_params": {
                        "id": "Isaac-RealWorld-GentleGrasp-XarmUmi-Hybrid-Tactile-v0",
                        "success": {
                            "terminal_reward": 1,
                            "required_consecutive_steps": 8,
                            "force_bonus": {"enabled": False},
                            "normalize_effort_reward": {"enabled": False},
                        },
                    },
                    "multi_task": {
                        "tasks": [
                            {
                                "name": "a",
                                "init_params": {
                                    "success": {"normalize_effort_reward": normalized()}
                                },
                            },
                            {
                                "name": "b",
                                "init_params": {
                                    "success": {
                                        "force_bonus": {
                                            "enabled": True,
                                            "coefficient": 10.0,
                                            "max_bonus": 1.0,
                                        }
                                    }
                                },
                            },
                            {"name": "c", "init_params": {}},
                        ]
                    },
                },
                "eval": "${env.train}",
            },
        }
    )


@pytest.mark.parametrize("ordinary_tabero", [False, True])
def test_tasks_may_mix_reward_forms_and_disabled_without_mutating_base(ordinary_tabero):
    cfg = task_config()
    if ordinary_tabero:
        cfg.env.train.init_params.update(
            id="Isaac-Libero-Franka-Hybrid-Tactile-v0",
            task_suite="synthetic",
            task_id=0,
        )
    before = deepcopy(cfg)
    validate_environments(cfg)
    assert cfg == before
    assert task_env_cfg(
        cfg.env.train, 0
    ).init_params.success.normalize_effort_reward.enabled
    assert task_env_cfg(cfg.env.train, 1).init_params.success.force_bonus.enabled
    cfg.env.train.multi_task.tasks[2].init_params.success = {
        "normalize_effort_reward": normalized(min_effort=5.0, max_effort=7.0)
    }
    validate_environments(cfg)
    assert (
        task_env_cfg(
            cfg.env.train, 0
        ).init_params.success.normalize_effort_reward.min_effort
        == 10
    )


def test_inherited_double_enable_reports_task_name():
    cfg = task_config()
    cfg.env.train.init_params.success.force_bonus.enabled = True
    with pytest.raises(ValueError, match="task a.*mutually exclusive"):
        validate_environments(cfg)


def test_other_success_settings_still_must_agree():
    cfg = task_config()
    cfg.env.train.multi_task.tasks[0].init_params.success.required_consecutive_steps = 3
    with pytest.raises(ValueError, match="interfaces"):
        validate_environments(cfg)


def test_evaluation_tasks_are_checked_only_when_enabled():
    cfg = task_config()
    cfg.env.eval = deepcopy(cfg.env.train)
    cfg.env.eval.multi_task.tasks[
        0
    ].init_params.success.normalize_effort_reward.max_effort = 0
    with pytest.raises(ValueError, match="env.eval task a"):
        validate_environments(cfg)
    cfg.runner.val_check_interval = -1
    validate_environments(cfg)


@pytest.mark.parametrize("only_eval", [False, True])
def test_single_task_preflight_checks_active_tabero_splits(only_eval):
    from rlinf.config import _validate_tabero_force_reward_contract

    cfg = task_config()
    cfg.runner.only_eval = only_eval
    cfg.env.train = task_env_cfg(cfg.env.train, 0)
    cfg.env.eval = deepcopy(cfg.env.train)
    split = "eval" if only_eval else "train"
    cfg.env[split].init_params.success.force_bonus.enabled = True
    with pytest.raises(ValueError, match=f"env.{split}.*mutually exclusive"):
        _validate_tabero_force_reward_contract(cfg)
    cfg.env[split].init_params.id = "Other-IsaacLab-Env"
    _validate_tabero_force_reward_contract(cfg)


@pytest.mark.parametrize("direct", [False, True])
def test_normalized_mode_rejects_nonfinite_force_and_wrong_multiplier_shape(direct):
    env, _, term, params = make_term([float("nan")], direct=direct)
    with pytest.raises(RuntimeError, match="non-finite measured contact force"):
        term(env, **params)
    env.squeeze[:] = 15
    params["env_reward_multipliers"] = (1.0, 2.0)
    with pytest.raises(ValueError, match="vector env reward shape"):
        term(env, **params)


def test_success_metrics_are_independent_of_reward_sign():
    env = SimpleNamespace(
        returns=torch.zeros(4),
        success_once=torch.zeros(4, dtype=torch.bool),
        elapsed_steps=torch.ones(4),
    )
    infos = record_success_metrics(
        env,
        torch.tensor([0.0, -0.5, 0.0, 2.0]),
        {
            TERMINAL_SUCCESS_KEY: torch.tensor([True, True, False, False]),
        },
    )
    assert infos["episode"]["success_once"].tolist() == [True, True, False, False]
    torch.testing.assert_close(
        infos["episode"]["return"], torch.tensor([0.0, -0.5, 0.0, 2.0])
    )


@pytest.mark.parametrize("realworld", [False, True])
def test_capture_success_before_manager_reset_and_clear_next_step(realworld):
    from rlinf.envs.isaaclab.tasks.realworld_tabero_tacfield import (
        _TerminalObservationCapture,
    )
    from rlinf.envs.isaaclab.tasks.tabero_tacfield import TaberoHdf5ResetWrapper

    class Simulator:
        num_envs = 4
        device = "cpu"

        def __init__(self):
            self.terms = {
                "success": torch.tensor([True, True, True, False]),
                "failure": torch.tensor([False, True, False, False]),
            }
            self.termination_manager = SimpleNamespace(
                active_terms=list(self.terms),
                get_term=lambda name: self.terms[name],
                time_outs=torch.tensor([False, False, True, False]),
            )
            self.observation_manager = SimpleNamespace(
                compute=lambda **kwargs: {"x": torch.ones(4)}
            )

        def _reset_idx(self, ids):
            for value in self.terms.values():
                value[ids] = False
            self.termination_manager.time_outs[ids] = False

        def step(self, action):
            terminated = self.terms["success"] | self.terms["failure"]
            truncated = self.termination_manager.time_outs.clone()
            ids = torch.where(terminated | truncated)[0]
            if ids.numel():
                self._reset_idx(ids)
            return {"x": torch.ones(4)}, torch.zeros(4), terminated, truncated, {}

    sim = Simulator()
    wrapper = (
        _TerminalObservationCapture(sim)
        if realworld
        else TaberoHdf5ResetWrapper(
            sim,
            dataset_handler=None,
            episode_names=[],
            shard_id=0,
            total_shards=1,
            capture_terminal_observation=True,
        )
    )
    info = wrapper.step(torch.zeros(4, 13))[-1]
    assert info[TERMINAL_SUCCESS_KEY].tolist() == [True, False, False, False]
    assert not sim.terms["success"].any()
    assert not wrapper.step(torch.zeros(4, 13))[-1][TERMINAL_SUCCESS_KEY].any()
