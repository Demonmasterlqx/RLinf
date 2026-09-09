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

"""Regression coverage for first-episode logging across internal resets."""

import inspect
import json
from types import MethodType, SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.envs.isaaclab.tasks.realworld_tabero_tacfield import (
    IsaaclabRealWorldTaberoTacFieldEnv,
    _build_state,
)
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.workers.env import env_worker
from rlinf.workers.env.env_worker import (
    EnvWorker,
    realworld_first_episode_records_to_env_info,
)


def make_env(tmp_path):
    env = object.__new__(IsaaclabRealWorldTaberoTacFieldEnv)
    env.num_envs = 4
    env.device = torch.device("cpu")
    env.seed = 7
    env.cfg = SimpleNamespace(max_episode_steps=300)
    env._elapsed_steps = torch.zeros(4, dtype=torch.long)
    env.prev_step_reward = torch.zeros(4)
    env._init_metrics()
    env._action_filter = None
    env._marker_history = SimpleNamespace(reset=lambda _: None)
    env._tactile_image_history = SimpleNamespace(reset=lambda _: None)
    env._gripper_diagnostics_path = None
    env._episode_records_dir = tmp_path
    env._wrap_obs = lambda obs, marker_update_mask=None: {
        "states": _build_state(obs["policy"])
    }
    raw = {
        "policy": {
            "eef_pose": torch.tensor([[0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]]).repeat(
                4, 1
            ),
            "gripper_pos": torch.zeros(4, 1),
        }
    }

    class Simulator:
        step_index = 0

        def reset(self, seed=None, env_ids=None):
            if env_ids is None:
                self.step_index = 0
            return raw, {}

        def step(self, actions):
            self.step_index += 1
            success = torch.tensor(
                [
                    self.step_index in (120, 250),
                    self.step_index == 123,
                    self.step_index == 295,
                    False,
                ]
            )
            timeout = env._elapsed_steps + 1 >= 300
            return (
                raw,
                success.float(),
                success,
                timeout,
                {
                    "_realworld_terminal_raw_observation": raw,
                    "_realworld_terminal_observation_mask": success | timeout,
                    "_realworld_terminal_force_mean": torch.full((4,), 5.0),
                    "_realworld_terminal_force_count": torch.full(
                        (4,), self.step_index
                    ),
                    "_realworld_terminal_force_bonus": torch.full((4,), 0.25),
                },
            )

    env.env = Simulator()
    env.reset()
    return env


def run_rollout(env, monkeypatch):
    monkeypatch.setattr(
        env_worker, "prepare_actions", lambda raw_chunk_actions, **_: raw_chunk_actions
    )
    worker = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "env": {
                    "train": {
                        "env_type": "isaaclab",
                        "auto_reset": False,
                        "ignore_terminations": False,
                    }
                }
            }
        ),
        model_cfg=OmegaConf.create(
            {"model_type": "openpi", "num_action_chunks": 10, "action_dim": 13}
        ),
        env_list=[env],
        n_train_chunk_steps=30,
        _build_chunk_final_obs=None,
    )
    worker._build_chunk_final_obs = MethodType(EnvWorker._build_chunk_final_obs, worker)
    metrics, outputs = {}, []
    for chunk in range(30):
        output, info, _ = inspect.unwrap(EnvWorker.env_interact_step)(
            worker, torch.zeros(4, 10, 13), 0
        )
        outputs.append(output)
        if EnvWorker.should_record_env_metrics(worker, output, info, chunk):
            EnvWorker.record_env_metrics(worker, metrics, info)
    return compute_evaluate_metrics(
        [{key: torch.cat(value) for key, value in metrics.items()}]
    ), outputs


def test_first_success_survives_reset_and_next_rollout_starts_fresh(
    tmp_path, monkeypatch
):
    env = make_env(tmp_path)
    for _ in range(2):
        metrics, outputs = run_rollout(env, monkeypatch)
        assert metrics["num_trajectories"] == 4
        assert metrics["success_once"] == pytest.approx(0.75)
        assert metrics["return"] == pytest.approx(0.75)
        assert metrics["episode_len"] == pytest.approx((120 + 123 + 295 + 300) / 4)
        assert metrics["reward"] == pytest.approx((1 / 120 + 1 / 123 + 1 / 295) / 4)
        assert metrics["force_valid_sample_count"] == pytest.approx(
            (120 + 123 + 295 + 300) / 4
        )
        assert sum(float(out.rewards.sum()) for out in outputs) == 4
        assert outputs[12].rewards[1, 2] == 1
        assert not outputs[12].rewards[1, 3:].any()
        env.reset()
    rows = [
        json.loads(line)
        for path in tmp_path.glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert len(rows) == 8
    assert len({(row["rollout_index"], row["env_index"]) for row in rows}) == 8
    assert sorted(row["episode_len"] for row in rows) == [
        120,
        120,
        123,
        123,
        295,
        295,
        300,
        300,
    ]


def test_count_uses_episode_samples_even_when_chunk_diagnostics_arrive_first():
    metrics = {
        "chunk_boundary/done_envs": torch.ones(30),
        "success_once": torch.tensor([1.0, 0.0]),
    }
    assert compute_evaluate_metrics([metrics])["num_trajectories"] == 2
    assert (
        compute_evaluate_metrics([realworld_first_episode_records_to_env_info({})])[
            "num_trajectories"
        ]
        == 0
    )


def test_duplicate_records_are_rejected():
    records = {
        key: torch.ones(2)
        for key in (
            "env_index",
            "success_once",
            "return",
            "episode_len",
            "reward",
            "termination",
            "truncation",
        )
    }
    with pytest.raises(ValueError, match="duplicate"):
        realworld_first_episode_records_to_env_info(records)


def test_missing_terminal_is_detected(tmp_path):
    env = make_env(tmp_path)
    env._episode_rollout_steps = 299
    env._elapsed_steps[:] = 0
    with pytest.raises(RuntimeError, match="missing first-episode"):
        env.step(torch.zeros(4, 13))
