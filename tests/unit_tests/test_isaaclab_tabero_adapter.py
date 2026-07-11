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

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.data.embodied_io_struct import EnvOutput
from rlinf.envs import get_env_cls
from rlinf.envs.isaaclab.tasks import tabero_tacfield
from rlinf.envs.isaaclab.tasks.tabero_tacfield import (
    IsaaclabTaberoTacFieldEnv,
    TacManipMarkerMotionHistory,
    assign_tabero_task,
    build_tabero_state,
    ensure_tabero_root_task_description,
    resolve_tabero_tasks,
    validate_tabero_task_assignment,
)


def _marker_motion(num_envs: int, offset: float = 0.0) -> torch.Tensor:
    motion = torch.zeros((num_envs, 2, 2, 99, 2), dtype=torch.float32)
    base = torch.arange(2 * 99 * 2, dtype=torch.float32).reshape(2, 99, 2)
    for env_id in range(num_envs):
        motion[env_id, :, 0] = base + offset + env_id * 1000.0
        motion[env_id, :, 1] = base + offset + env_id * 1000.0 + 100.0
    return motion


def test_marker_motion_history_builds_tabero_prefix_with_front_padding():
    history = TacManipMarkerMotionHistory(num_envs=2, history_len=8)

    first = history.update(_marker_motion(2, offset=0.0))
    second = history.update(_marker_motion(2, offset=10.0))

    assert first.shape == (2, 9, 198, 2)
    assert second.shape == (2, 9, 198, 2)

    # Prefix slot 0 is the per-env reference marker position from the first frame.
    expected_init_env0 = _marker_motion(2, 0.0)[0, :, 0].reshape(198, 2)
    torch.testing.assert_close(second[0, 0], expected_init_env0)

    # History slots are front-padded by repeating the earliest current frame.
    expected_first_current = _marker_motion(2, 0.0)[0, :, 1].reshape(198, 2)
    expected_second_current = _marker_motion(2, 10.0)[0, :, 1].reshape(198, 2)
    torch.testing.assert_close(second[0, 1], expected_first_current)
    torch.testing.assert_close(second[0, 7], expected_first_current)
    torch.testing.assert_close(second[0, 8], expected_second_current)


def test_marker_motion_history_reset_reinitializes_selected_envs():
    history = TacManipMarkerMotionHistory(num_envs=2, history_len=8)
    history.update(_marker_motion(2, offset=0.0))
    history.reset(torch.tensor([1]))

    updated = history.update(_marker_motion(2, offset=20.0))

    original_init_env0 = _marker_motion(2, 0.0)[0, :, 0].reshape(198, 2)
    new_init_env1 = _marker_motion(2, 20.0)[1, :, 0].reshape(198, 2)
    torch.testing.assert_close(updated[0, 0], original_init_env0)
    torch.testing.assert_close(updated[1, 0], new_init_env1)


def test_build_tabero_state_uses_pose_axis_angle_and_gripper_scalar():
    policy_obs = {
        "eef_pose": torch.tensor(
            [
                [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
                [0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "gripper_pos": torch.tensor([[0.01, 0.02], [0.03, 0.04]], dtype=torch.float32),
    }

    state = build_tabero_state(policy_obs)

    assert state.shape == (2, 7)
    torch.testing.assert_close(state[:, :3], policy_obs["eef_pose"][:, :3])
    torch.testing.assert_close(state[:, 3:6], torch.zeros((2, 3)))
    torch.testing.assert_close(state[:, 6], torch.tensor([0.01, 0.03]))


def test_get_env_cls_resolves_tabero_tacfield_id_to_adapter():
    cfg = OmegaConf.create(
        {
            "init_params": {
                "id": "Isaac-Libero-Franka-Hybrid-Tactile-v0",
            }
        }
    )

    assert get_env_cls("isaaclab", cfg) is IsaaclabTaberoTacFieldEnv


def test_env_output_preserves_tabero_tactile_marker_motion_for_rollout_channel():
    tactile = torch.zeros((2, 9, 198, 2), dtype=torch.float32)
    custom_extra = torch.ones((2, 3), dtype=torch.float32)
    env_output = EnvOutput(
        obs={
            "main_images": torch.zeros((2, 4, 4, 3), dtype=torch.uint8),
            "wrist_images": torch.ones((2, 4, 4, 3), dtype=torch.uint8),
            "states": torch.zeros((2, 7), dtype=torch.float32),
            "task_descriptions": ["task a", "task b"],
            "tactile_marker_motion": tactile,
            "custom_extra": custom_extra,
        },
        final_obs={
            "main_images": torch.zeros((2, 4, 4, 3), dtype=torch.uint8),
            "wrist_images": torch.ones((2, 4, 4, 3), dtype=torch.uint8),
            "states": torch.zeros((2, 7), dtype=torch.float32),
            "task_descriptions": ["task a", "task b"],
            "tactile_marker_motion": tactile + 1,
            "custom_extra": custom_extra + 1,
        },
    )

    output_dict = env_output.to_dict()

    assert "tactile_marker_motion" in output_dict["obs"]
    assert "custom_extra" in output_dict["obs"]
    assert "tactile_marker_motion" in output_dict["final_obs"]
    assert "custom_extra" in output_dict["final_obs"]
    torch.testing.assert_close(output_dict["obs"]["tactile_marker_motion"], tactile)
    torch.testing.assert_close(output_dict["obs"]["custom_extra"], custom_extra)
    torch.testing.assert_close(
        output_dict["final_obs"]["tactile_marker_motion"], tactile + 1
    )
    torch.testing.assert_close(
        output_dict["final_obs"]["custom_extra"], custom_extra + 1
    )


def test_env_output_merge_flattens_mixed_task_descriptions_in_source_order():
    env_outputs = []
    for task_name in ("task a", "task b"):
        env_outputs.append(
            EnvOutput(
                obs={
                    "main_images": torch.zeros((1, 4, 4, 3), dtype=torch.uint8),
                    "wrist_images": torch.ones((1, 4, 4, 3), dtype=torch.uint8),
                    "states": torch.zeros((1, 7), dtype=torch.float32),
                    "task_descriptions": [task_name],
                    "tactile_marker_motion": torch.zeros(
                        (1, 9, 198, 2), dtype=torch.float32
                    ),
                },
                dones=torch.zeros(1, dtype=torch.bool),
                terminations=torch.zeros(1, dtype=torch.bool),
                truncations=torch.zeros(1, dtype=torch.bool),
                rewards=torch.zeros((1, 1), dtype=torch.float32),
            ).to_dict()
        )

    merged = EnvOutput.merge_env_outputs(env_outputs)

    assert merged["obs"]["task_descriptions"] == ["task a", "task b"]
    assert merged["obs"]["states"].shape == (2, 7)
    assert merged["obs"]["tactile_marker_motion"].shape == (2, 9, 198, 2)


def test_resolve_tabero_tasks_keeps_legacy_single_task_config():
    init_params = OmegaConf.create(
        {
            "task_suite": "libero_10",
            "task_id": 0,
            "task_description": "put both objects in the basket",
        }
    )

    tasks = resolve_tabero_tasks(init_params)

    assert len(tasks) == 1
    assert tasks[0].task_suite == "libero_10"
    assert tasks[0].task_id == 0
    assert tasks[0].task_description == "put both objects in the basket"


def test_resolve_tabero_tasks_loads_explicit_tasks_and_fills_descriptions(tmp_path):
    config_dir = tmp_path / "libero"
    config_dir.mkdir()
    (config_dir / "libero_object.json").write_text(
        """
        {
          "tasks": [
            {"task_id": 0, "language_instruction": "pick alphabet soup"},
            {"task_id": 1, "language_instruction": "pick cream cheese"}
          ]
        }
        """,
        encoding="utf-8",
    )
    init_params = OmegaConf.create(
        {
            "libero_config_dir": str(config_dir),
            "tasks": [
                {"task_suite": "libero_object", "task_id": 0},
                {
                    "task_suite": "libero_object",
                    "task_id": 1,
                    "task_description": "custom prompt",
                },
            ],
        }
    )

    tasks = resolve_tabero_tasks(init_params)

    assert [task.task_description for task in tasks] == [
        "pick alphabet soup",
        "custom prompt",
    ]


def test_resolve_tabero_tasks_loads_tabero_subset_mapping(tmp_path):
    config_dir = tmp_path / "libero"
    config_dir.mkdir()
    (config_dir / "libero_object.json").write_text(
        """
        {
          "tasks": [
            {"task_id": 0, "language_instruction": "pick alphabet soup"},
            {"task_id": 2, "language_instruction": "pick tomato sauce"}
          ]
        }
        """,
        encoding="utf-8",
    )
    subset_path = tmp_path / "tabero_tasks.json"
    subset_path.write_text(
        '{"libero_10": [], "libero_object": [0, 2], "libero_goal": []}',
        encoding="utf-8",
    )
    init_params = OmegaConf.create(
        {
            "libero_config_dir": str(config_dir),
            "tabero_task_subset_path": str(subset_path),
        }
    )

    tasks = resolve_tabero_tasks(init_params)

    assert [
        (task.task_suite, task.task_id, task.task_description) for task in tasks
    ] == [
        ("libero_object", 0, "pick alphabet soup"),
        ("libero_object", 2, "pick tomato sauce"),
    ]


def test_resolve_tabero_tasks_rejects_empty_task_lists():
    init_params = OmegaConf.create({"tasks": []})

    try:
        resolve_tabero_tasks(init_params)
    except ValueError as exc:
        assert "at least one task" in str(exc)
    else:
        raise AssertionError("Expected empty Tabero task list to fail.")


def test_assign_tabero_task_round_robins_by_logical_shard_id():
    tasks = resolve_tabero_tasks(
        OmegaConf.create(
            {
                "tasks": [
                    {
                        "task_suite": "libero_object",
                        "task_id": task_id,
                        "task_description": f"task {task_id}",
                    }
                    for task_id in range(3)
                ]
            }
        )
    )

    assigned = [assign_tabero_task(tasks, seed_offset=i).task_id for i in range(8)]

    assert assigned == [0, 1, 2, 0, 1, 2, 0, 1]


def test_ensure_tabero_root_task_description_backfills_assigned_prompt():
    cfg = OmegaConf.create({"init_params": {"tasks": []}})
    task = resolve_tabero_tasks(
        OmegaConf.create(
            {
                "tasks": [
                    {
                        "task_suite": "libero_object",
                        "task_id": 1,
                        "task_description": "pick cream cheese",
                    }
                ]
            }
        )
    )[0]

    ensure_tabero_root_task_description(cfg, task)

    assert cfg.init_params.task_description == "pick cream cheese"


def test_validate_tabero_task_assignment_fails_when_required_tasks_are_inactive():
    tasks = resolve_tabero_tasks(
        OmegaConf.create(
            {
                "tasks": [
                    {
                        "task_suite": "libero_object",
                        "task_id": task_id,
                        "task_description": f"task {task_id}",
                    }
                    for task_id in range(3)
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="only the first 2 task"):
        validate_tabero_task_assignment(
            tasks,
            total_num_processes=2,
            require_all_tasks_active=True,
        )


def test_tabero_record_metrics_adds_tensor_task_metadata():
    env = object.__new__(IsaaclabTaberoTacFieldEnv)
    env.num_envs = 2
    env.device = torch.device("cpu")
    env.returns = torch.zeros(2)
    env.success_once = torch.zeros(2, dtype=torch.bool)
    env._elapsed_steps = torch.ones(2, dtype=torch.int32)
    env._tabero_task = resolve_tabero_tasks(
        OmegaConf.create(
            {
                "tasks": [
                    {
                        "task_suite": "libero_object",
                        "task_id": 5,
                        "task_description": "pick ketchup",
                    }
                ]
            }
        )
    )[0]
    env._tabero_task_shard_id = 7

    infos = env._record_metrics(
        torch.tensor([0.0, 1.0]),
        torch.tensor([False, True]),
        {},
    )

    torch.testing.assert_close(infos["episode"]["task_id"], torch.tensor([5.0, 5.0]))
    torch.testing.assert_close(
        infos["episode"]["task_shard_id"], torch.tensor([7.0, 7.0])
    )


def test_tabero_wrap_obs_omits_force_observation_when_force_key_is_null():
    env = object.__new__(IsaaclabTaberoTacFieldEnv)
    env._main_image_key = "agentview_rgb"
    env._wrist_image_key = "eye_in_hand_rgb"
    env._marker_motion_key = "gripper_marker_motion"
    env._force_key = None
    env._marker_history = TacManipMarkerMotionHistory(num_envs=1, history_len=8)
    env.num_envs = 1
    env.task_description = "pick alphabet soup"

    wrapped = env._wrap_obs(
        {
            "policy": {
                "agentview_rgb": torch.zeros((1, 4, 4, 3), dtype=torch.uint8),
                "eye_in_hand_rgb": torch.zeros((1, 4, 4, 3), dtype=torch.uint8),
                "eef_pose": torch.tensor(
                    [[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32
                ),
                "gripper_pos": torch.zeros((1, 1), dtype=torch.float32),
                "gripper_marker_motion": _marker_motion(1),
                "gripper_net_force": torch.ones((1, 3), dtype=torch.float32),
            }
        }
    )

    assert "tactile_gripper_force" not in wrapped


def test_consecutive_success_tracker_triggers_only_on_eighth_success():
    tracker_cls = getattr(tabero_tacfield, "ConsecutiveSuccessTracker")
    tracker = tracker_cls(num_envs=1, required_steps=8, device="cpu")

    outputs = [tracker.update(torch.tensor([True])).item() for _ in range(8)]

    assert outputs == [False] * 7 + [True]
    assert tracker.streak.tolist() == [8]


def test_consecutive_success_tracker_keeps_legacy_single_step_default():
    tracker_cls = getattr(tabero_tacfield, "ConsecutiveSuccessTracker")
    tracker = tracker_cls(num_envs=1, device="cpu")

    stable = tracker.update(torch.tensor([True]))

    assert stable.tolist() == [True]
    assert tracker.streak.tolist() == [1]


def test_consecutive_success_tracker_resets_failed_and_selected_envs():
    tracker_cls = getattr(tabero_tacfield, "ConsecutiveSuccessTracker")
    tracker = tracker_cls(num_envs=3, required_steps=3, device="cpu")

    tracker.update(torch.tensor([True, True, True]))
    tracker.update(torch.tensor([True, False, True]))
    tracker.reset(torch.tensor([2]))
    stable = tracker.update(torch.tensor([True, True, True]))

    assert stable.tolist() == [True, False, False]
    assert tracker.streak.tolist() == [3, 1, 1]


class _FakeTerminationManager:
    def __init__(self, terms, time_outs):
        self._terms = terms
        self.time_outs = time_outs

    def get_term(self, name):
        return self._terms[name]


def test_stable_success_reward_rejects_failure_and_timeout_on_same_step():
    reward_fn = getattr(tabero_tacfield, "_stable_success_terminal_reward")
    env = type("FakeEnv", (), {})()
    env.step_dt = 0.05
    env.termination_manager = _FakeTerminationManager(
        terms={
            "success": torch.tensor([True, True, True, False]),
            "object_dropped": torch.tensor([False, True, False, False]),
        },
        time_outs=torch.tensor([False, False, True, False]),
    )

    reward = reward_fn(
        env,
        success_term_name="success",
        failure_term_names=("object_dropped",),
    )

    torch.testing.assert_close(reward, torch.tensor([20.0, 0.0, 0.0, 0.0]))


def test_stable_success_reward_applies_per_env_condition_multipliers():
    reward_fn = getattr(tabero_tacfield, "_stable_success_terminal_reward")
    env = type("FakeEnv", (), {})()
    env.step_dt = 0.05
    env.termination_manager = _FakeTerminationManager(
        terms={
            "success": torch.tensor([True, True, True, True]),
            "object_dropped": torch.tensor([False, False, True, False]),
        },
        time_outs=torch.tensor([False, False, False, True]),
    )

    reward = reward_fn(
        env,
        success_term_name="success",
        failure_term_names=("object_dropped",),
        env_reward_multipliers=(1.0, 3.0, 1.0, 3.0),
    )

    torch.testing.assert_close(reward, torch.tensor([20.0, 60.0, 0.0, 0.0]))


def test_install_success_reward_wraps_raw_success_and_lists_failure_terms():
    install_fn = getattr(tabero_tacfield, "_install_success_reward")

    class FakeManagerTermBase:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self.env = env

    class FakeTerm:
        def __init__(self, func, params=None, time_out=False):
            self.func = func
            self.params = params or {}
            self.time_out = time_out

    class FakeRewardTerm:
        def __init__(self, func, weight, params):
            self.func = func
            self.weight = weight
            self.params = params

    def raw_success(env, threshold):
        return env.raw >= threshold

    terminations = type("Terminations", (), {})()
    terminations.success = FakeTerm(raw_success, {"threshold": 1.0})
    terminations.object_dropped = FakeTerm(lambda env: env.dropped)
    terminations.time_out = FakeTerm(lambda env: env.timed_out, time_out=True)
    env_cfg = type("EnvCfg", (), {"terminations": terminations, "rewards": None})()

    install_fn(
        env_cfg,
        FakeRewardTerm,
        reward_coef=2.0,
        manager_term_base_cls=FakeManagerTermBase,
        required_steps=8,
    )

    assert env_cfg.terminations.success.func is not raw_success
    assert env_cfg.terminations.success.params["success_func"] is raw_success
    assert env_cfg.terminations.success.params["success_params"] == {"threshold": 1.0}
    assert env_cfg.terminations.success.params["required_steps"] == 8
    assert env_cfg.rewards["success"].weight == 2.0
    assert (
        env_cfg.rewards["success"].func
        is tabero_tacfield._stable_success_terminal_reward
    )
    assert env_cfg.rewards["success"].params == {
        "success_term_name": "success",
        "failure_term_names": ("object_dropped",),
    }


def test_install_success_reward_forwards_condition_multipliers():
    install_fn = getattr(tabero_tacfield, "_install_success_reward")

    class FakeManagerTermBase:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self.env = env

    class FakeTerm:
        def __init__(self, func, params=None, time_out=False):
            self.func = func
            self.params = params or {}
            self.time_out = time_out

    class FakeRewardTerm:
        def __init__(self, func, weight, params):
            self.func = func
            self.weight = weight
            self.params = params

    terminations = type("Terminations", (), {})()
    terminations.success = FakeTerm(lambda env: env.raw)
    env_cfg = type("EnvCfg", (), {"terminations": terminations, "rewards": None})()

    install_fn(
        env_cfg,
        FakeRewardTerm,
        reward_coef=1.0,
        manager_term_base_cls=FakeManagerTermBase,
        env_reward_multipliers=(1.0, 3.0),
    )

    assert env_cfg.rewards["success"].params["env_reward_multipliers"] == (
        1.0,
        3.0,
    )


def test_dynamic_consecutive_success_term_resets_through_manager_contract():
    make_term = getattr(tabero_tacfield, "_make_consecutive_success_term")

    class FakeManagerTermBase:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self.env = env

    env = type("FakeEnv", (), {})()
    env.num_envs = 2
    env.device = torch.device("cpu")
    env.raw = torch.tensor([True, True])
    cfg = type(
        "Cfg",
        (),
        {
            "params": {
                "success_func": lambda current_env: current_env.raw,
                "success_params": {},
                "required_steps": 2,
            }
        },
    )()
    term = make_term(FakeManagerTermBase)(cfg, env)

    assert term(env, **cfg.params).tolist() == [False, False]
    term.reset(env_ids=torch.tensor([1]))
    assert term(env, **cfg.params).tolist() == [True, False]


def test_prompt_conditions_pair_firm_and_gentle_and_rotate_deterministically():
    build_prompts = getattr(tabero_tacfield, "build_tabero_conditioned_prompts")
    prompt_cfg = OmegaConf.create(
        {
            "enabled": True,
            "assignment": "paired",
            "firm_adverbs": ["firmly", "tightly"],
            "gentle_adverbs": ["gently", "softly"],
            "prompt_seed": 0,
        }
    )

    first_prompts, condition_ids = build_prompts(
        instruction="pick up the soup",
        task_suite="libero_object",
        task_id=0,
        num_envs=2,
        prompt_cfg=prompt_cfg,
        rollout_round=0,
    )
    repeated_prompts, repeated_ids = build_prompts(
        instruction="pick up the soup",
        task_suite="libero_object",
        task_id=0,
        num_envs=2,
        prompt_cfg=prompt_cfg,
        rollout_round=0,
    )
    next_prompts, _ = build_prompts(
        instruction="pick up the soup",
        task_suite="libero_object",
        task_id=0,
        num_envs=2,
        prompt_cfg=prompt_cfg,
        rollout_round=1,
    )

    assert condition_ids == [0, 1]
    assert repeated_ids == condition_ids
    assert repeated_prompts == first_prompts
    assert any(word in first_prompts[0].lower() for word in ("firmly", "tightly"))
    assert any(word in first_prompts[1].lower() for word in ("gently", "softly"))
    assert next_prompts != first_prompts


def test_prompt_conditions_require_firm_gentle_pairs():
    build_prompts = getattr(tabero_tacfield, "build_tabero_conditioned_prompts")
    prompt_cfg = OmegaConf.create(
        {
            "enabled": True,
            "assignment": "paired",
            "firm_adverbs": ["firmly"],
            "gentle_adverbs": ["gently"],
            "prompt_seed": 0,
        }
    )

    with pytest.raises(ValueError, match="even number"):
        build_prompts(
            instruction="pick up the soup",
            task_suite="libero_object",
            task_id=0,
            num_envs=3,
            prompt_cfg=prompt_cfg,
            rollout_round=0,
        )


def test_predicted_squeeze_uses_tabero_13d_force_indices():
    squeeze_fn = getattr(tabero_tacfield, "compute_tabero_predicted_squeeze")
    actions = torch.zeros((2, 13), dtype=torch.float32)
    actions[0, 9] = -3.0
    actions[0, 12] = 5.0
    actions[1, 9] = 7.0
    actions[1, 12] = -2.0

    squeeze = squeeze_fn(actions)

    torch.testing.assert_close(squeeze, torch.tensor([6.0, 4.0]))


def test_tabero_condition_metrics_report_each_condition_without_half_scaling():
    env = object.__new__(IsaaclabTaberoTacFieldEnv)
    env.num_envs = 2
    env.device = torch.device("cpu")
    env.returns = torch.zeros(2)
    env.success_once = torch.zeros(2, dtype=torch.bool)
    env._elapsed_steps = torch.ones(2, dtype=torch.int32)
    env._tabero_task = resolve_tabero_tasks(
        OmegaConf.create(
            {
                "tasks": [
                    {
                        "task_suite": "libero_object",
                        "task_id": 0,
                        "task_description": "pick soup",
                    }
                ]
            }
        )
    )[0]
    env._tabero_task_shard_id = 0
    env._prompt_condition_ids = (0, 1)
    env._condition_squeeze_sum = torch.tensor([30.0, 4.0])
    env._condition_squeeze_count = torch.tensor([3.0, 2.0])

    infos = env._record_metrics(
        torch.tensor([1.0, 0.0]),
        torch.tensor([True, False]),
        {},
    )

    episode = infos["episode"]
    torch.testing.assert_close(episode["firm_success_once"], torch.ones(2))
    torch.testing.assert_close(episode["gentle_success_once"], torch.zeros(2))
    torch.testing.assert_close(episode["firm_return"], torch.ones(2))
    torch.testing.assert_close(episode["gentle_return"], torch.zeros(2))
    torch.testing.assert_close(
        episode["firm_squeeze_pred_mean"], torch.full((2,), 10.0)
    )
    torch.testing.assert_close(
        episode["gentle_squeeze_pred_mean"], torch.full((2,), 2.0)
    )
