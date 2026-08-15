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
from omegaconf import OmegaConf

from rlinf.data.embodied_io_struct import EnvOutput
from rlinf.envs import get_env_cls
from rlinf.envs.isaaclab.tasks import tabero_tacfield
from rlinf.envs.isaaclab.tasks.tabero_tacfield import (
    IsaaclabTaberoTacFieldEnv,
    TaberoHdf5ResetWrapper,
    TacManipMarkerMotionHistory,
    assign_tabero_reset_episode_names,
    assign_tabero_task,
    build_tabero_state,
    ensure_tabero_root_task_description,
    resolve_tabero_tasks,
    stack_tabero_initial_states,
    validate_tabero_chunk_boundary_mode,
    validate_tabero_firm_prompts,
    validate_tabero_task_assignment,
)


def _marker_motion(num_envs: int, offset: float = 0.0) -> torch.Tensor:
    motion = torch.zeros((num_envs, 2, 2, 99, 2), dtype=torch.float32)
    base = torch.arange(2 * 99 * 2, dtype=torch.float32).reshape(2, 99, 2)
    for env_id in range(num_envs):
        motion[env_id, :, 0] = base + offset + env_id * 1000.0
        motion[env_id, :, 1] = base + offset + env_id * 1000.0 + 100.0
    return motion


def _raw_tabero_obs(state_x: list[float], marker_offset: float = 0.0):
    num_envs = len(state_x)
    eef_pose = torch.zeros((num_envs, 7), dtype=torch.float32)
    eef_pose[:, 0] = torch.tensor(state_x)
    eef_pose[:, 3] = 1.0
    return {
        "policy": {
            "agentview_rgb": torch.zeros((num_envs, 4, 4, 3), dtype=torch.uint8),
            "eye_in_hand_rgb": torch.ones((num_envs, 4, 4, 3), dtype=torch.uint8),
            "eef_pose": eef_pose,
            "gripper_pos": torch.full((num_envs, 1), 0.03),
            "gripper_marker_motion": _marker_motion(num_envs, offset=marker_offset),
        }
    }


def _clone_nested_to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.clone().to(device=device)
    if isinstance(value, dict):
        return {
            key: _clone_nested_to_device(nested, device)
            for key, nested in value.items()
        }
    return value


class _TerminalSafeFakeEnv:
    device = torch.device("cpu")

    def __init__(self, schedule: list[dict[int, str]], num_envs: int):
        self.num_envs = num_envs
        self.schedule = schedule
        self.step_index = 0
        self.actions = []
        self.reset_calls = []

    def step(self, actions):
        step_index = self.step_index
        self.step_index += 1
        self.actions.append(actions.clone())
        done_spec = self.schedule[step_index]
        terminations = torch.zeros(self.num_envs, dtype=torch.bool)
        truncations = torch.zeros(self.num_envs, dtype=torch.bool)
        for env_id, done_kind in done_spec.items():
            if done_kind == "termination":
                terminations[env_id] = True
            elif done_kind == "truncation":
                truncations[env_id] = True
            else:
                raise AssertionError(done_kind)
        dones = terminations | truncations
        post_step_x = [10.0 + step_index + env_id for env_id in range(self.num_envs)]
        raw_obs = _raw_tabero_obs(post_step_x, marker_offset=10.0 + step_index)
        extras = {}
        if dones.any():
            terminal_x = [
                100.0 + step_index + env_id if dones[env_id] else post_step_x[env_id]
                for env_id in range(self.num_envs)
            ]
            extras[getattr(tabero_tacfield, "_TERMINAL_RAW_OBSERVATION_KEY")] = (
                _raw_tabero_obs(terminal_x, marker_offset=100.0 + step_index)
            )
            extras[getattr(tabero_tacfield, "_TERMINAL_OBSERVATION_MASK_KEY")] = (
                dones.clone()
            )
        rewards = terminations.to(dtype=torch.float32)
        return raw_obs, rewards, terminations, truncations, extras

    def reset(self, seed=None, env_ids=None):
        del seed
        env_ids = torch.as_tensor(env_ids, dtype=torch.long)
        self.reset_calls.append(env_ids.clone())
        return (
            _raw_tabero_obs(
                [200.0 + env_id for env_id in range(self.num_envs)],
                marker_offset=200.0,
            ),
            {},
        )


def _terminal_safe_adapter(
    schedule: list[dict[int, str]],
    num_envs: int = 1,
    prompt_condition: str = "firm",
):
    env = object.__new__(IsaaclabTaberoTacFieldEnv)
    env.num_envs = num_envs
    env.device = torch.device("cpu")
    env.cfg = SimpleNamespace(max_episode_steps=100)
    env.env = _TerminalSafeFakeEnv(schedule, num_envs)
    env._chunk_boundary_mode = getattr(tabero_tacfield, "_TERMINAL_SAFE_HDF5_MODE")
    env._main_image_key = "agentview_rgb"
    env._wrist_image_key = "eye_in_hand_rgb"
    env._marker_motion_key = "gripper_marker_motion"
    env._force_key = None
    env._marker_history = TacManipMarkerMotionHistory(num_envs=num_envs, history_len=8)
    prompt_enabled = prompt_condition != "none"
    env._prompt_cfg = OmegaConf.create(
        {
            "enabled": prompt_enabled,
            "assignment": "cyclic",
            "condition_cycle": [prompt_condition] if prompt_enabled else [],
            "firm_adverbs": ["firmly", "tightly"],
            "gentle_adverbs": ["gently", "softly"],
            "prompt_seed": 0,
        }
    )
    env._prompt_rollout_round = 0
    if prompt_condition == "firm":
        env._prompt_condition_ids = (0,) * num_envs
        env._conditioned_prompts = ["pick soup firmly"] * num_envs
    elif prompt_condition == "gentle":
        env._prompt_condition_ids = (1,) * num_envs
        env._conditioned_prompts = ["pick soup gently"] * num_envs
    elif prompt_condition == "none":
        env._prompt_condition_ids = ()
        env._conditioned_prompts = ["pick soup"] * num_envs
    else:
        raise AssertionError(prompt_condition)
    env._condition_squeeze_sum = torch.zeros(num_envs)
    env._condition_squeeze_count = torch.zeros(num_envs, dtype=torch.long)
    env.task_description = "pick soup"
    env.ignore_terminations = False
    env.auto_reset = False
    env.prev_step_reward = torch.zeros(num_envs)
    env.success_once = torch.zeros(num_envs, dtype=torch.bool)
    env.fail_once = torch.zeros(num_envs, dtype=torch.bool)
    env.returns = torch.zeros(num_envs)
    env._elapsed_steps = torch.zeros(num_envs, dtype=torch.int32)
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
    return env


def test_terminal_safe_firm_prompt_validator_rejects_condition_pollution():
    validate_tabero_firm_prompts(
        ["pick soup firmly", "pick soup tightly"],
        [0, 0],
    )

    with pytest.raises(ValueError, match="condition id 0"):
        validate_tabero_firm_prompts(["pick soup gently"], [1])
    with pytest.raises(ValueError, match="must contain 'firmly' or 'tightly'"):
        validate_tabero_firm_prompts(["pick soup"], [0])


def test_chunk_boundary_mode_rejects_unknown_values_without_legacy_fallback():
    assert validate_tabero_chunk_boundary_mode("legacy") == "legacy"
    assert (
        validate_tabero_chunk_boundary_mode("terminal_safe_hdf5_v1")
        == "terminal_safe_hdf5_v1"
    )
    with pytest.raises(ValueError, match="Unsupported Tabero chunk_boundary_mode"):
        validate_tabero_chunk_boundary_mode("terminal_safe_typo")


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


def test_marker_motion_history_update_mask_freezes_inactive_environment():
    history = TacManipMarkerMotionHistory(num_envs=2, history_len=8)
    initial = history.update(_marker_motion(2, offset=0.0))

    updated = history.update(
        _marker_motion(2, offset=20.0),
        update_mask=torch.tensor([False, True]),
    )

    torch.testing.assert_close(updated[0], initial[0])
    assert not torch.equal(updated[1], initial[1])


def test_hdf5_wrapper_captures_terminal_observation_before_internal_reset():
    class FakeObservationManager:
        def compute(self, update_history=False):
            assert update_history is False
            return _raw_tabero_obs([100.0, 101.0], marker_offset=100.0)

    class FakeEnv:
        num_envs = 2
        device = torch.device("cpu")

        def __init__(self):
            self.observation_manager = FakeObservationManager()
            self.reset_ids = []

        def _reset_idx(self, env_ids):
            self.reset_ids.append(torch.as_tensor(env_ids).clone())

        def step(self, action):
            del action
            self._reset_idx(torch.tensor([0]))
            return (
                _raw_tabero_obs([10.0, 11.0], marker_offset=10.0),
                torch.tensor([1.0, 0.0]),
                torch.tensor([True, False]),
                torch.tensor([False, False]),
                {},
            )

    wrapped = TaberoHdf5ResetWrapper(
        FakeEnv(),
        dataset_handler=SimpleNamespace(),
        episode_names=["demo_0"],
        shard_id=0,
        total_shards=1,
        capture_terminal_observation=True,
    )

    _, _, _, _, extras = wrapped.step(torch.zeros((2, 13)))

    terminal = extras[getattr(tabero_tacfield, "_TERMINAL_RAW_OBSERVATION_KEY")]
    mask = extras[getattr(tabero_tacfield, "_TERMINAL_OBSERVATION_MASK_KEY")]
    assert mask.tolist() == [True, False]
    assert terminal["policy"]["eef_pose"][0, 0].item() == 100.0


def test_hdf5_wrapper_captures_trajectory_force_metrics_before_internal_reset():
    class FakeObservationManager:
        def compute(self, update_history=False):
            assert update_history is False
            return _raw_tabero_obs([100.0, 101.0], marker_offset=100.0)

    class FakeForceRewardTerm:
        def __init__(self):
            self.trajectory_mean_force = torch.tensor([3.0, 5.0])
            self.valid_sample_count = torch.tensor([2, 4])

    class FakeRewardManager:
        active_terms = ["success"]

        def __init__(self, reward_term):
            self._reward_term = reward_term

        def get_term_cfg(self, term_name):
            assert term_name == "success"
            return SimpleNamespace(func=self._reward_term)

    class FakeEnv:
        num_envs = 2
        device = torch.device("cpu")

        def __init__(self):
            self.observation_manager = FakeObservationManager()
            self.force_reward_term = FakeForceRewardTerm()
            self.reward_manager = FakeRewardManager(self.force_reward_term)

        def _reset_idx(self, env_ids):
            self.force_reward_term.trajectory_mean_force[env_ids] = 0.0
            self.force_reward_term.valid_sample_count[env_ids] = 0

        def step(self, action):
            del action
            self._reset_idx(torch.tensor([0]))
            return (
                _raw_tabero_obs([10.0, 11.0], marker_offset=10.0),
                torch.tensor([1.0, 0.0]),
                torch.tensor([True, False]),
                torch.tensor([False, False]),
                {},
            )

    wrapped = TaberoHdf5ResetWrapper(
        FakeEnv(),
        dataset_handler=SimpleNamespace(),
        episode_names=["demo_0"],
        shard_id=0,
        total_shards=1,
        capture_terminal_observation=True,
    )

    _, _, _, _, extras = wrapped.step(torch.zeros((2, 13)))

    mean_force = extras[
        getattr(
            tabero_tacfield,
            "_TERMINAL_TRAJECTORY_MEAN_MEASURED_SQUEEZE_KEY",
        )
    ]
    sample_count = extras[
        getattr(tabero_tacfield, "_TERMINAL_FORCE_VALID_SAMPLE_COUNT_KEY")
    ]
    torch.testing.assert_close(mean_force, torch.tensor([3.0, 5.0]))
    assert sample_count.tolist() == [2, 4]


def test_hdf5_wrapper_accumulates_terminal_rows_across_multiple_internal_resets():
    class FakeObservationManager:
        def __init__(self):
            self.calls = 0

        def compute(self, update_history=False):
            assert update_history is False
            observations = (
                _raw_tabero_obs([100.0, 11.0], marker_offset=100.0)
                if self.calls == 0
                else _raw_tabero_obs([10.0, 101.0], marker_offset=200.0)
            )
            self.calls += 1
            return observations

    class FakeEnv:
        num_envs = 2
        device = torch.device("cpu")

        def __init__(self):
            self.observation_manager = FakeObservationManager()

        def _reset_idx(self, env_ids):
            del env_ids

        def step(self, action):
            del action
            self._reset_idx(torch.tensor([0]))
            self._reset_idx(torch.tensor([1]))
            return (
                _raw_tabero_obs([10.0, 11.0], marker_offset=10.0),
                torch.ones(2),
                torch.tensor([True, True]),
                torch.tensor([False, False]),
                {},
            )

    wrapped = TaberoHdf5ResetWrapper(
        FakeEnv(),
        dataset_handler=SimpleNamespace(),
        episode_names=["demo_0"],
        shard_id=0,
        total_shards=1,
        capture_terminal_observation=True,
    )

    _, _, _, _, extras = wrapped.step(torch.zeros((2, 13)))

    terminal = extras[getattr(tabero_tacfield, "_TERMINAL_RAW_OBSERVATION_KEY")]
    mask = extras[getattr(tabero_tacfield, "_TERMINAL_OBSERVATION_MASK_KEY")]
    assert mask.tolist() == [True, True]
    assert terminal["policy"]["eef_pose"][:, 0].tolist() == [100.0, 101.0]


@pytest.mark.parametrize("done_step", [0, 1, 3])
@pytest.mark.parametrize("done_kind", ["termination", "truncation"])
def test_terminal_safe_chunk_preserves_first_done_and_uses_hold_padding(
    done_step, done_kind
):
    chunk_size = 4
    schedule = [{} for _ in range(chunk_size)]
    schedule[done_step] = {0: done_kind}
    env = _terminal_safe_adapter(schedule)
    policy_actions = torch.full((1, chunk_size, 13), 9.0)

    obs_list, rewards, terminations, truncations, infos_list = env.chunk_step(
        policy_actions
    )

    expected_done = torch.zeros((1, chunk_size), dtype=torch.bool)
    expected_done[0, done_step] = True
    if done_kind == "termination":
        torch.testing.assert_close(terminations, expected_done)
        assert not truncations.any()
        assert rewards[0, done_step].item() == 1.0
    else:
        torch.testing.assert_close(truncations, expected_done)
        assert not terminations.any()
        assert rewards.sum().item() == 0.0
    assert rewards[0, done_step + 1 :].sum().item() == 0.0

    terminal_obs = infos_list[done_step]["final_observation"]
    assert terminal_obs["states"][0, 0].item() == 100.0 + done_step
    assert obs_list[-1]["states"][0, 0].item() == 200.0
    assert env.env.reset_calls[0].tolist() == [0]

    for padding_step in range(done_step + 1, chunk_size):
        executed = env.env.actions[padding_step][0]
        torch.testing.assert_close(executed[7:], torch.zeros(6))
        assert executed[0].item() == pytest.approx(9.0 + padding_step)

    metrics = infos_list[-1]["chunk_boundary_metrics"]
    assert metrics["done_envs"].item() == 1
    assert metrics["early_done_envs"].item() == int(done_step < chunk_size - 1)
    assert metrics["post_done_hold_steps"].item() == chunk_size - done_step - 1
    assert metrics["post_done_policy_actions"].item() == 0
    assert metrics["terminal_observation_captures"].item() == 1
    assert metrics["hdf5_reset_envs"].item() == 1

    records = infos_list[-1]["_tabero_chunk_episode_records"]
    assert records["env_index"].tolist() == [0]
    assert records["primitive_step_index"].tolist() == [done_step]
    assert records["condition_id"].tolist() == [0]
    assert records["termination"].tolist() == [done_kind == "termination"]
    assert records["truncation"].tolist() == [done_kind == "truncation"]
    assert records["success_once"].tolist() == [
        1.0 if done_kind == "termination" else 0.0
    ]
    assert records["return"].tolist() == [1.0 if done_kind == "termination" else 0.0]
    assert records["episode_len"].tolist() == [float(done_step + 1)]
    assert records["terminal_step_reward"].tolist() == [
        1.0 if done_kind == "termination" else 0.0
    ]


def test_terminal_safe_chunk_without_done_executes_policy_and_skips_reset():
    chunk_size = 3
    env = _terminal_safe_adapter([{} for _ in range(chunk_size)])
    policy_actions = torch.arange(chunk_size * 13, dtype=torch.float32).reshape(
        1, chunk_size, 13
    )

    _, rewards, terminations, truncations, infos_list = env.chunk_step(policy_actions)

    assert not terminations.any()
    assert not truncations.any()
    assert rewards.sum().item() == 0.0
    assert env.env.reset_calls == []
    torch.testing.assert_close(torch.stack(env.env.actions, dim=1), policy_actions)
    metrics = infos_list[-1]["chunk_boundary_metrics"]
    assert metrics["done_envs"].item() == 0
    assert metrics["post_done_hold_steps"].item() == 0
    assert infos_list[-1]["_tabero_chunk_episode_records"] == {}


def test_terminal_safe_chunk_records_measured_force_metrics_at_first_done():
    env = _terminal_safe_adapter([{0: "termination"}, {}])
    original_step = env.env.step

    def step_with_force_metrics(actions):
        obs, rewards, terminations, truncations, extras = original_step(actions)
        extras = dict(extras)
        extras[
            getattr(
                tabero_tacfield,
                "_TERMINAL_TRAJECTORY_MEAN_MEASURED_SQUEEZE_KEY",
            )
        ] = torch.tensor([3.5])
        extras[getattr(tabero_tacfield, "_TERMINAL_FORCE_VALID_SAMPLE_COUNT_KEY")] = (
            torch.tensor([6])
        )
        return obs, rewards, terminations, truncations, extras

    env.env.step = step_with_force_metrics

    _, _, _, _, infos_list = env.chunk_step(torch.ones((1, 2, 13)))

    records = infos_list[-1]["_tabero_chunk_episode_records"]
    assert records["trajectory_mean_measured_squeeze"].tolist() == [3.5]
    assert records["force_valid_sample_count"].tolist() == [6]


def test_terminal_safe_chunk_resets_simultaneous_done_environments_together():
    env = _terminal_safe_adapter(
        [{}, {0: "termination", 1: "truncation"}, {}], num_envs=2
    )

    _, _, terminations, truncations, infos_list = env.chunk_step(torch.ones((2, 3, 13)))

    assert terminations[0, 1]
    assert truncations[1, 1]
    assert env.env.reset_calls[0].tolist() == [0, 1]
    metrics = infos_list[-1]["chunk_boundary_metrics"]
    assert metrics["done_envs"].item() == 2
    assert metrics["post_done_hold_steps"].item() == 2
    assert metrics["terminal_observation_captures"].item() == 2
    records = infos_list[-1]["_tabero_chunk_episode_records"]
    assert records["env_index"].tolist() == [0, 1]
    assert records["termination"].tolist() == [True, False]
    assert records["truncation"].tolist() == [False, True]
    assert records["success_once"].tolist() == [1.0, 0.0]


def test_terminal_safe_chunk_handles_asynchronous_vector_done_and_partial_reset():
    env = _terminal_safe_adapter(
        [{0: "termination"}, {}, {1: "truncation"}, {}], num_envs=2
    )

    _, rewards, terminations, truncations, infos_list = env.chunk_step(
        torch.ones((2, 4, 13))
    )

    assert terminations[0].tolist() == [True, False, False, False]
    assert truncations[1].tolist() == [False, False, True, False]
    assert rewards[0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert env.env.reset_calls[0].tolist() == [0, 1]
    metrics = infos_list[-1]["chunk_boundary_metrics"]
    assert metrics["post_done_policy_actions"].item() == 0
    assert metrics["post_done_hold_steps"].item() == 4
    records = infos_list[-1]["_tabero_chunk_episode_records"]
    assert records["env_index"].tolist() == [0, 1]
    assert records["primitive_step_index"].tolist() == [0, 2]


@pytest.mark.parametrize(
    ("prompt_condition", "expected_condition_id", "expect_nan_squeeze"),
    [("none", -1, True), ("firm", 0, False), ("gentle", 1, False)],
)
def test_terminal_safe_chunk_is_independent_of_prompt_condition(
    prompt_condition, expected_condition_id, expect_nan_squeeze
):
    env = _terminal_safe_adapter(
        [{0: "termination"}, {}], prompt_condition=prompt_condition
    )

    _, _, _, _, infos_list = env.chunk_step(torch.ones((1, 2, 13)))

    records = infos_list[-1]["_tabero_chunk_episode_records"]
    assert records["condition_id"].tolist() == [expected_condition_id]
    squeeze = records["squeeze_pred_mean"]
    assert bool(torch.isnan(squeeze).all()) is expect_nan_squeeze


def test_terminal_safe_chunk_preserves_ignore_terminations_output_semantics():
    env = _terminal_safe_adapter([{0: "termination"}, {}])
    env.ignore_terminations = True

    _, _, terminations, truncations, infos_list = env.chunk_step(torch.ones((1, 2, 13)))

    assert not terminations.any()
    assert not truncations.any()
    assert infos_list[0]["episode"]["success_at_end"].item() is True
    assert env.env.reset_calls[0].tolist() == [0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_terminal_safe_chunk_accepts_cpu_actions_with_cuda_lifecycle_masks():
    env = _terminal_safe_adapter([{0: "termination"}, {}])
    env.device = torch.device("cuda")
    env._elapsed_steps = env._elapsed_steps.cuda()
    env.prev_step_reward = env.prev_step_reward.cuda()
    env.success_once = env.success_once.cuda()
    env.fail_once = env.fail_once.cuda()
    env.returns = env.returns.cuda()
    env._condition_squeeze_sum = env._condition_squeeze_sum.cuda()
    env._condition_squeeze_count = env._condition_squeeze_count.cuda()

    original_step = env.env.step

    def cuda_step(actions):
        raw_obs, rewards, terminations, truncations, extras = original_step(
            actions.cpu()
        )
        return (
            _clone_nested_to_device(raw_obs, env.device),
            rewards.cuda(),
            terminations.cuda(),
            truncations.cuda(),
            _clone_nested_to_device(extras, env.device),
        )

    env.env.step = cuda_step
    original_reset = env.env.reset

    def cuda_reset(seed=None, env_ids=None):
        raw_obs, extras = original_reset(seed=seed, env_ids=env_ids.cpu())
        return (
            _clone_nested_to_device(raw_obs, env.device),
            _clone_nested_to_device(extras, env.device),
        )

    env.env.reset = cuda_reset

    _, _, _, _, infos_list = env.chunk_step(torch.ones((1, 2, 13)))

    executed = infos_list[-1]["_tabero_executed_chunk_actions"]
    assert executed.device.type == "cpu"
    torch.testing.assert_close(executed[0, 1, 7:], torch.zeros(6))


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


def test_tabero_episode_horizon_configures_registry_cfg_for_exact_step_count():
    init_params = OmegaConf.create({"max_episode_steps": 300})
    isaac_env_cfg = SimpleNamespace(
        sim=SimpleNamespace(dt=1 / 60),
        decimation=3,
        episode_length_s=8.0,
    )

    expected_steps = tabero_tacfield.configure_tabero_episode_horizon(
        init_params, isaac_env_cfg
    )

    assert expected_steps == 300
    assert isaac_env_cfg.episode_length_s == pytest.approx(15.0)


@pytest.mark.parametrize("horizon", [None, True, 0, -1, 300.0, "300"])
def test_tabero_episode_horizon_rejects_missing_or_invalid_steps(horizon):
    init_params = OmegaConf.create(
        {} if horizon is None else {"max_episode_steps": horizon}
    )
    isaac_env_cfg = SimpleNamespace(
        sim=SimpleNamespace(dt=1 / 60),
        decimation=3,
        episode_length_s=8.0,
    )

    with pytest.raises(ValueError, match="max_episode_steps"):
        tabero_tacfield.configure_tabero_episode_horizon(init_params, isaac_env_cfg)


def test_tabero_episode_horizon_rejects_created_env_length_mismatch():
    tabero_tacfield.validate_tabero_episode_horizon(
        SimpleNamespace(max_episode_length=300), 300
    )

    with pytest.raises(ValueError, match="expected 300.*got 160"):
        tabero_tacfield.validate_tabero_episode_horizon(
            SimpleNamespace(max_episode_length=160), 300
        )


def test_tabero_env_build_path_applies_and_validates_episode_horizon():
    source = Path(tabero_tacfield.__file__).read_text()
    build_path = source.split("def make_env_isaaclab():", 1)[1]
    build_path = build_path.split("return make_env_isaaclab", 1)[0]

    configure_at = build_path.index("configure_tabero_episode_horizon(")
    make_at = build_path.index("env = gym.make(")
    validate_at = build_path.index("validate_tabero_episode_horizon(")
    assert configure_at < make_at < validate_at


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


def test_hdf5_reset_episode_assignment_covers_workers_without_overlap():
    episode_names = [f"demo_{index}" for index in range(50)]

    shard_zero = assign_tabero_reset_episode_names(
        episode_names,
        num_envs=28,
        shard_id=0,
        total_shards=2,
        rollout_round=0,
    )
    shard_one = assign_tabero_reset_episode_names(
        episode_names,
        num_envs=28,
        shard_id=1,
        total_shards=2,
        rollout_round=0,
    )
    next_round = assign_tabero_reset_episode_names(
        episode_names,
        num_envs=28,
        shard_id=0,
        total_shards=2,
        rollout_round=1,
    )

    assert shard_zero == [f"demo_{index}" for index in range(28)]
    assert shard_one[:22] == [f"demo_{index}" for index in range(28, 50)]
    assert shard_one[22:] == [f"demo_{index}" for index in range(6)]
    assert next_round[0] == "demo_6"


def test_stack_tabero_initial_states_concatenates_nested_tensor_batches():
    states = [
        {
            "articulation": {
                "robot": {"joint_position": torch.tensor([[float(index), index + 0.5]])}
            },
            "rigid_object": {
                "object": {"root_pose": torch.tensor([[float(index), 0.0, 1.0]])}
            },
        }
        for index in range(3)
    ]

    stacked = stack_tabero_initial_states(states)

    torch.testing.assert_close(
        stacked["articulation"]["robot"]["joint_position"],
        torch.tensor([[0.0, 0.5], [1.0, 1.5], [2.0, 2.5]]),
    )
    assert stacked["rigid_object"]["object"]["root_pose"].shape == (3, 3)


def test_hdf5_reset_wrapper_skips_bootstrap_then_applies_cyclic_states():
    class FakeEpisode:
        def __init__(self, value):
            self.data = {"initial_state": {}}
            self._value = value

        def get_initial_state(self):
            return {
                "rigid_object": {
                    "object": {"root_pose": torch.tensor([[float(self._value), 0.0]])}
                }
            }

    class FakeHandler:
        def __init__(self):
            self.closed = False

        def load_episode(self, episode_name, device):
            return FakeEpisode(int(episode_name.removeprefix("demo_")))

        def close(self):
            self.closed = True

    class FakeEnv:
        num_envs = 2
        device = torch.device("cpu")

        def __init__(self):
            self.reset_calls = []
            self.reset_to_calls = []

        def reset(self, seed=None, env_ids=None):
            self.reset_calls.append((seed, env_ids))
            return {"policy": "random"}, {"source": "random"}

        def reset_to(self, state, env_ids, is_relative=False):
            self.reset_to_calls.append((state, env_ids.clone(), is_relative))
            return {"policy": "dataset"}, {"source": "dataset"}

        def close(self):
            pass

    env = FakeEnv()
    handler = FakeHandler()
    wrapped = TaberoHdf5ResetWrapper(
        env,
        dataset_handler=handler,
        episode_names=["demo_0", "demo_1", "demo_2"],
        shard_id=0,
        total_shards=1,
    )

    bootstrap_obs, _ = wrapped.reset(seed=42)
    rollout_obs, _ = wrapped.reset(seed=43)

    assert bootstrap_obs["policy"] == "random"
    assert rollout_obs["policy"] == "dataset"
    assert len(env.reset_calls) == 2
    assert len(env.reset_to_calls) == 1
    state, env_ids, is_relative = env.reset_to_calls[0]
    torch.testing.assert_close(env_ids, torch.tensor([0, 1]))
    torch.testing.assert_close(
        state["rigid_object"]["object"]["root_pose"],
        torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
    )
    assert is_relative is True

    wrapped.close()
    assert handler.closed is True


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


class _FakeForceBonusObservationManager:
    def __init__(self, grasp_schedule: list[torch.Tensor]):
        self._grasp_schedule = list(grasp_schedule)
        self._step = 0

    def compute_group(self, group_name, update_history=False):
        assert group_name == "subtask_terms"
        assert update_history is False
        return {"grasp_1": self._grasp_schedule[self._step].clone()}


def _make_force_bonus_term(
    *,
    grasp_schedule: list[list[bool]],
    squeeze_schedule: list[list[float]],
    success_schedule: list[list[bool]],
    failure_schedule: list[list[bool]] | None = None,
    timeout_schedule: list[list[bool]] | None = None,
    coefficient: float = 0.3,
    epsilon: float = 0.1,
    max_bonus: float = 0.5,
    min_valid_samples: int = 1,
    multipliers: tuple[float, ...] | None = None,
):
    num_envs = len(grasp_schedule[0])
    steps = len(grasp_schedule)
    assert len(squeeze_schedule) == len(success_schedule) == steps
    failure_schedule = failure_schedule or [[False] * num_envs for _ in range(steps)]
    timeout_schedule = timeout_schedule or [[False] * num_envs for _ in range(steps)]

    class FakeManagerTermBase:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self._env = env

    observation_manager = _FakeForceBonusObservationManager(
        [torch.tensor(values) for values in grasp_schedule]
    )
    termination_manager = _FakeTerminationManager(
        terms={
            "success": torch.tensor(success_schedule[0]),
            "object_dropped": torch.tensor(failure_schedule[0]),
        },
        time_outs=torch.tensor(timeout_schedule[0]),
    )
    env = SimpleNamespace(
        num_envs=num_envs,
        device=torch.device("cpu"),
        step_dt=0.05,
        observation_manager=observation_manager,
        termination_manager=termination_manager,
    )
    force_step = {"value": 0}

    def force_reader(_env, contact_sensor_name):
        assert _env is env
        assert contact_sensor_name == "contact_gripper"
        step = force_step["value"]
        squeeze = torch.tensor(squeeze_schedule[step], dtype=torch.float32)
        # The production definition is 2 * min(|left_z|, |right_z|).
        force = torch.zeros((num_envs, 2, 3), dtype=torch.float32)
        force[:, :, 2] = squeeze[:, None] / 2.0
        force_step["value"] += 1
        return force

    params = {
        "success_term_name": "success",
        "failure_term_names": ("object_dropped",),
        "force_sources": (("grasp_1", "contact_gripper"),),
        "force_reader": force_reader,
        "terminal_reward": 1.0,
        "coefficient": coefficient,
        "epsilon": epsilon,
        "max_bonus": max_bonus,
        "min_valid_samples": min_valid_samples,
        "contact_epsilon": 1.0e-4,
    }
    if multipliers is not None:
        params["env_reward_multipliers"] = multipliers
    cfg = SimpleNamespace(params=params)
    reward_cls = tabero_tacfield._make_trajectory_force_success_reward_term(
        FakeManagerTermBase
    )
    reward_term = reward_cls(cfg, env)

    def run_step(step: int) -> torch.Tensor:
        observation_manager._step = step
        termination_manager._terms["success"] = torch.tensor(success_schedule[step])
        termination_manager._terms["object_dropped"] = torch.tensor(
            failure_schedule[step]
        )
        termination_manager.time_outs = torch.tensor(timeout_schedule[step])
        return reward_term(env, **params)

    return reward_term, run_step


def test_force_bonus_starts_at_grasp_and_ignores_zero_force_release_steps():
    reward_term, run_step = _make_force_bonus_term(
        grasp_schedule=[[False], [True], [False], [False]],
        squeeze_schedule=[[10.0], [4.0], [2.0], [0.0]],
        success_schedule=[[False], [False], [False], [True]],
    )

    rewards = [run_step(step) for step in range(4)]

    assert [reward.item() for reward in rewards[:3]] == [0.0, 0.0, 0.0]
    assert reward_term.valid_sample_count.tolist() == [2]
    torch.testing.assert_close(reward_term.trajectory_mean_force, torch.tensor([3.0]))
    # Raw term is divided by step_dt; RewardManager later multiplies by dt.
    assert rewards[-1].item() == pytest.approx((1.0 + 0.3 / 3.0) / 0.05)


def test_force_bonus_is_success_only_bounded_and_condition_scaled():
    reward_term, run_step = _make_force_bonus_term(
        grasp_schedule=[[True, True, True]],
        squeeze_schedule=[[0.2, 2.0, 2.0]],
        success_schedule=[[True, True, True]],
        failure_schedule=[[False, True, False]],
        timeout_schedule=[[False, False, True]],
        coefficient=1.0,
        epsilon=0.1,
        max_bonus=0.2,
        multipliers=(2.0, 2.0, 2.0),
    )

    reward = run_step(0)

    assert reward_term.valid_sample_count.tolist() == [1, 1, 1]
    torch.testing.assert_close(reward, torch.tensor([48.0, 0.0, 0.0]))


def test_force_bonus_requires_minimum_samples_and_resets_selected_envs_only():
    reward_term, run_step = _make_force_bonus_term(
        grasp_schedule=[[True, True], [False, False]],
        squeeze_schedule=[[2.0, 4.0], [2.0, 4.0]],
        success_schedule=[[False, False], [True, True]],
        min_valid_samples=2,
    )

    run_step(0)
    reward_term.reset(torch.tensor([0]))
    reward = run_step(1)

    assert reward_term.valid_sample_count.tolist() == [0, 2]
    assert reward[0].item() == pytest.approx(1.0 / 0.05)
    assert reward[1].item() == pytest.approx((1.0 + 0.3 / 4.0) / 0.05)


def test_force_bonus_rejects_non_finite_measured_force():
    _, run_step = _make_force_bonus_term(
        grasp_schedule=[[True]],
        squeeze_schedule=[[float("nan")]],
        success_schedule=[[False]],
    )

    with pytest.raises(RuntimeError, match="non-finite measured contact force"):
        run_step(0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("coefficient", -0.1, "coefficient"),
        ("epsilon", 0.0, "epsilon"),
        ("max_bonus", 0.0, "max_bonus"),
        ("max_bonus", -1.0, "max_bonus"),
        ("max_bonus", float("nan"), "finite"),
        ("max_bonus", float("inf"), "finite"),
        ("min_valid_samples", 0, "min_valid_samples"),
        ("contact_epsilon", -1.0, "contact_epsilon"),
    ],
)
def test_force_bonus_config_rejects_unsafe_values(field, value, message):
    config = {
        "enabled": True,
        "coefficient": 0.1,
        "epsilon": 0.1,
        "max_bonus": 0.2,
        "min_valid_samples": 1,
        "contact_epsilon": 1.0e-4,
    }
    config[field] = value

    with pytest.raises(ValueError, match=message):
        tabero_tacfield._validate_force_bonus_cfg(
            OmegaConf.create(config), terminal_reward=1.0
        )


def test_force_bonus_config_allows_positive_cap_above_terminal_reward():
    config = OmegaConf.create(
        {
            "enabled": True,
            "coefficient": 1.0,
            "epsilon": 1.0,
            "max_bonus": 10.0,
            "min_valid_samples": 4,
            "contact_epsilon": 1.0,
        }
    )

    normalized = tabero_tacfield._validate_force_bonus_cfg(
        config, terminal_reward=1.0
    )

    assert normalized["max_bonus"] == 10.0


@pytest.mark.parametrize(
    ("mean_force", "expected_bonus"),
    [(0.5, 1.0), (2.0, 0.5), (50.0, 0.02)],
)
def test_force_bonus_coef1_maxbonus10_values(mean_force, expected_bonus):
    reward_term, run_step = _make_force_bonus_term(
        grasp_schedule=[[True]],
        squeeze_schedule=[[mean_force]],
        success_schedule=[[True]],
        coefficient=1.0,
        epsilon=1.0,
        max_bonus=10.0,
    )

    reward = run_step(0)

    assert reward_term.trajectory_mean_force.item() == pytest.approx(mean_force)
    # The term divides by step_dt; RewardManager multiplies it back by step_dt.
    terminal_reward = reward.item() * 0.05
    assert terminal_reward == pytest.approx(1.0 + expected_bonus)


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


def test_install_success_reward_installs_force_tracking_term_and_params():
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

    def raw_success(env):
        return env.raw

    force_reader = object()
    grasp_term = FakeTerm(
        lambda env: env.grasped,
        {"object_cfg": SimpleNamespace(name="target_object")},
    )
    observations = SimpleNamespace(subtask_terms=SimpleNamespace(grasp_1=grasp_term))
    terminations = SimpleNamespace(success=FakeTerm(raw_success))
    env_cfg = SimpleNamespace(
        observations=observations,
        terminations=terminations,
        rewards=None,
    )
    force_bonus_cfg = OmegaConf.create(
        {
            "enabled": True,
            "coefficient": 0.1,
            "epsilon": 0.2,
            "max_bonus": 0.25,
            "min_valid_samples": 3,
            "contact_epsilon": 0.01,
        }
    )

    install_fn(
        env_cfg,
        FakeRewardTerm,
        reward_coef=2.0,
        manager_term_base_cls=FakeManagerTermBase,
        required_steps=8,
        terminal_reward=1.5,
        force_bonus_cfg=force_bonus_cfg,
        force_reader=force_reader,
    )

    reward_cfg = env_cfg.rewards["success"]
    assert issubclass(reward_cfg.func, FakeManagerTermBase)
    assert reward_cfg.func.__name__ == "TrajectoryForceSuccessRewardTerm"
    assert reward_cfg.weight == 2.0
    assert reward_cfg.params == {
        "success_term_name": "success",
        "failure_term_names": (),
        "force_sources": (("grasp_1", "contact_gripper"),),
        "force_reader": force_reader,
        "terminal_reward": 1.5,
        "coefficient": 0.1,
        "epsilon": 0.2,
        "max_bonus": 0.25,
        "min_valid_samples": 3,
        "contact_epsilon": 0.01,
    }


@pytest.mark.parametrize("missing", ["force_reader", "grasp_source"])
def test_install_success_reward_rejects_missing_force_bonus_source(missing):
    class FakeManagerTermBase:
        pass

    class FakeTerm:
        def __init__(self, func, params=None):
            self.func = func
            self.params = params or {}

    def raw_success(env):
        return env.raw

    observations = SimpleNamespace(
        subtask_terms=SimpleNamespace(
            grasp_1=FakeTerm(
                lambda env: env.grasped,
                {"object_cfg": SimpleNamespace(name="target_object")},
            )
        )
    )
    if missing == "grasp_source":
        observations.subtask_terms = SimpleNamespace()
    env_cfg = SimpleNamespace(
        observations=observations,
        terminations=SimpleNamespace(success=FakeTerm(raw_success)),
        rewards=None,
    )

    with pytest.raises(ValueError, match="reader|grasp observations"):
        tabero_tacfield._install_success_reward(
            env_cfg,
            lambda **kwargs: SimpleNamespace(**kwargs),
            reward_coef=1.0,
            manager_term_base_cls=FakeManagerTermBase,
            force_bonus_cfg=OmegaConf.create(
                {
                    "enabled": True,
                    "coefficient": 0.1,
                    "epsilon": 0.1,
                    "max_bonus": 0.2,
                }
            ),
            force_reader=None if missing == "force_reader" else object(),
        )

    # Validation is atomic: a rejected install must not wrap the success term.
    assert env_cfg.terminations.success.func is raw_success


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


def test_disabled_prompt_conditions_keep_original_instruction():
    build_prompts = getattr(tabero_tacfield, "build_tabero_conditioned_prompts")
    instruction = "pick up the butter and place it in the basket"

    prompts, condition_ids = build_prompts(
        instruction=instruction,
        task_suite="libero_object",
        task_id=6,
        num_envs=4,
        prompt_cfg=OmegaConf.create({"enabled": False}),
        rollout_round=0,
    )

    assert prompts == [instruction] * 4
    assert condition_ids == []


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


def test_prompt_conditions_support_cyclic_one_firm_three_gentle_assignment():
    build_prompts = getattr(tabero_tacfield, "build_tabero_conditioned_prompts")
    prompt_cfg = OmegaConf.create(
        {
            "enabled": True,
            "assignment": "cyclic",
            "condition_cycle": ["firm", "gentle", "gentle", "gentle"],
            "firm_adverbs": ["firmly", "tightly"],
            "gentle_adverbs": ["gently", "softly"],
            "prompt_seed": 0,
        }
    )

    prompts, condition_ids = build_prompts(
        instruction="pick up the soup",
        task_suite="libero_object",
        task_id=0,
        num_envs=4,
        prompt_cfg=prompt_cfg,
        rollout_round=0,
    )

    assert condition_ids == [0, 1, 1, 1]
    assert any(word in prompts[0].lower() for word in ("firmly", "tightly"))
    assert all(
        any(word in prompt.lower() for word in ("gently", "softly"))
        for prompt in prompts[1:]
    )


def test_prompt_conditions_require_complete_cyclic_assignments():
    build_prompts = getattr(tabero_tacfield, "build_tabero_conditioned_prompts")
    prompt_cfg = OmegaConf.create(
        {
            "enabled": True,
            "assignment": "cyclic",
            "condition_cycle": ["firm", "gentle", "gentle", "gentle"],
            "firm_adverbs": ["firmly"],
            "gentle_adverbs": ["gently"],
            "prompt_seed": 0,
        }
    )

    with pytest.raises(ValueError, match="divisible by condition cycle length"):
        build_prompts(
            instruction="pick up the soup",
            task_suite="libero_object",
            task_id=0,
            num_envs=6,
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
