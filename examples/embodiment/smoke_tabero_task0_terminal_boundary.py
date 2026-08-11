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

"""Run a real one-env Tabero smoke for terminal-safe chunk boundaries."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from rlinf.envs.isaaclab.tasks.tabero_tacfield import IsaaclabTaberoTacFieldEnv

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT
    / "examples"
    / "embodiment"
    / "config"
    / "isaaclab_pi0_peft_lora_tacfield_tabero_task0_no_adverb_mass_friction_2gpu_100step.yaml"
)


def _tensor_list(value: torch.Tensor) -> list:
    return value.detach().cpu().tolist()


def main() -> None:
    os.environ.setdefault("TABERO_TASK0_NO_ADVERB_RUN_ID", "runtime_smoke")
    cfg = OmegaConf.load(CONFIG_PATH).env.train
    cfg.max_episode_steps = 5
    cfg.max_steps_per_rollout_epoch = 5
    cfg.init_params.max_episode_steps = 5

    env = IsaaclabTaberoTacFieldEnv(
        cfg=cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    try:
        obs, _ = env.reset(seed=42)
        state = obs["states"].to(device=env.device, dtype=torch.float32)
        actions = torch.zeros((1, 10, 13), device=env.device, dtype=torch.float32)
        actions[:, :5, :7] = state[:, None, :]
        actions[:, 5:, :7] = state[:, None, :] + 0.25
        actions[:, 5:, 7:] = 3.25

        obs_list, rewards, terminations, truncations, infos_list = env.chunk_step(
            actions
        )
        infos = infos_list[-1]
        metrics = infos["chunk_boundary_metrics"]
        records = infos["_tabero_chunk_episode_records"]
        executed = infos["_tabero_executed_chunk_actions"]

        done = terminations | truncations
        done_indices = torch.nonzero(done[0], as_tuple=False).reshape(-1)
        assert done_indices.tolist() == [4], done_indices.tolist()
        assert torch.count_nonzero(rewards[0, 5:]).item() == 0
        assert torch.count_nonzero(done[0, 5:]).item() == 0
        assert torch.count_nonzero(executed[0, 5:, 7:]).item() == 0
        assert not torch.allclose(executed[0, 5:, :7], actions[0, 5:, :7])
        assert int(metrics["post_done_policy_actions"].item()) == 0
        assert int(metrics["post_done_hold_steps"].item()) == 5
        assert int(metrics["terminal_observation_captures"].item()) == 1
        assert int(metrics["hdf5_reset_envs"].item()) == 1
        assert records["condition_id"].tolist() == [-1]
        assert math.isnan(float(records["squeeze_pred_mean"].item()))
        assert float(records["return"].item()) <= 1.0
        terminal_obs = infos_list[4]["final_observation"]["states"]
        assert torch.allclose(obs_list[4]["states"], terminal_obs)

        report = {
            "status": "passed",
            "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "chunk_boundary_mode": cfg.init_params.chunk_boundary_mode,
            "max_episode_steps": int(cfg.max_episode_steps),
            "done_indices_zero_based": _tensor_list(done_indices),
            "terminations": _tensor_list(terminations),
            "truncations": _tensor_list(truncations),
            "rewards": _tensor_list(rewards),
            "condition_id": _tensor_list(records["condition_id"]),
            "squeeze_pred_mean_is_nan": math.isnan(
                float(records["squeeze_pred_mean"].item())
            ),
            "episode_return": float(records["return"].item()),
            "episode_len": float(records["episode_len"].item()),
            "chunk_boundary_metrics": {
                key: int(torch.as_tensor(value).item())
                for key, value in metrics.items()
            },
            "planned_suffix_sentinel_nonzero": int(
                torch.count_nonzero(actions[0, 5:, 7:]).item()
            ),
            "executed_suffix_sentinel_nonzero": int(
                torch.count_nonzero(executed[0, 5:, 7:]).item()
            ),
            "terminal_observation_preserved": bool(
                torch.allclose(obs_list[4]["states"], terminal_obs)
            ),
        }
        print("TABERO_TERMINAL_BOUNDARY_SMOKE=" + json.dumps(report, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    main()
