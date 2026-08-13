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

import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from rlinf.utils.tabero_ppo_boundary import (
    TABERO_PPO_TRANSITION_BOUNDARY_SEMANTICS,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "examples" / "embodiment" / "config"
LAUNCHER = (
    REPO_ROOT
    / "examples"
    / "embodiment"
    / "run_tabero_task1_task5_no_adverb_uniform_physics_action_expert.sh"
)
MODEL_PATH = (
    "/data/home/sim6g/code/tabero/models/pi0_lora_tacfield_tabero_all_firm_safetensors"
)
PROFILE_DIR = Path(
    "/data/home/sim6g/code/tabero/Tabero/benchmarks/datasets/libero/"
    "config_profiles/"
    "alltask_damage_uniform_mass_05_16_friction_04_08_from_rlinf_sft_20k"
)

TASKS = {
    1: {
        "object": "cream_cheese_1",
        "instruction": "pick up the cream cheese and place it in the basket",
        "hdf5": "libero_object_task1_pick_up_the_cream_cheese_and_place_it_in_the_basket_demo.hdf5",
    },
    5: {
        "object": "tomato_sauce_1",
        "instruction": "pick up the tomato sauce and place it in the basket",
        "hdf5": "libero_object_task5_pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo.hdf5",
    },
}


def _config_name(task_id):
    return (
        "isaaclab_pi0_peft_lora_tacfield_tabero_"
        f"task{task_id}_no_adverb_uniform_physics_2gpu_100step"
    )


def _load_config(monkeypatch, task_id):
    monkeypatch.setenv("TABERO_TASK0_NO_ADVERB_RUN_ID", "20260812_120000_formal")
    return OmegaConf.load(CONFIG_DIR / f"{_config_name(task_id)}.yaml")


def _read_env(path):
    result = subprocess.run(
        ["bash", "-c", f"set -a; source {shlex.quote(str(path))}; env"],
        check=True,
        text=True,
        capture_output=True,
    )
    return dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )


@pytest.mark.parametrize("task_id", [1, 5])
def test_configs_preserve_two_gpu_boundary_safe_pirl_contract(monkeypatch, task_id):
    cfg = _load_config(monkeypatch, task_id)
    task = TASKS[task_id]

    assert OmegaConf.to_container(cfg.cluster.component_placement) == {
        "actor": "1",
        "rollout": "0",
        "env": "0",
    }
    assert cfg.runner.max_epochs == 100
    assert cfg.runner.save_interval == 10
    assert cfg.runner.logger.logger_backends == ["tensorboard", "wandb"]
    assert cfg.env.train.total_num_envs == 21
    assert cfg.env.train.rollout_epoch == 2
    assert cfg.actor.global_batch_size == 42
    assert cfg.actor.micro_batch_size == 2
    assert cfg.rollout.model.model_path == MODEL_PATH
    assert cfg.actor.model.model_path == MODEL_PATH
    assert cfg.actor.model.is_lora is True
    assert cfg.actor.model.lora_target == "action_expert"
    assert cfg.actor.model.lora_rank == 32
    assert cfg.actor.model.freeze_non_lora is True
    assert cfg.actor.model.openpi.train_expert_only is True
    assert cfg.actor.model.add_value_head is True
    assert cfg.algorithm.adv_type == "gae"
    assert cfg.algorithm.loss_type == "actor_critic"
    assert (
        cfg.algorithm.tabero_ppo_transition_boundary_semantics
        == TABERO_PPO_TRANSITION_BOUNDARY_SEMANTICS
    )

    for split_cfg in (cfg.env.train, cfg.env.eval):
        split = split_cfg.init_params
        assert split.task_suite == "libero_object"
        assert split.task_id == task_id
        assert OmegaConf.to_container(split.tasks) == [
            {"task_suite": "libero_object", "task_id": task_id}
        ]
        assert split.task_description == task["instruction"]
        assert split.prompt_conditions.enabled is False
        assert split.libero_config_dir == str(PROFILE_DIR)
        assert split.hdf5_initial_states_path.endswith(task["hdf5"])
        assert split.hdf5_reset_assignment == "cyclic"
        assert split.chunk_boundary_mode == "terminal_safe_hdf5_v1"
        assert split.success.required_consecutive_steps == 8
        assert split_cfg.auto_reset is False
        assert split_cfg.ignore_terminations is False

    metadata = cfg.actor.fsdp_config.trainable_checkpoint_metadata
    assert metadata.method == "pirl"
    assert metadata.task_id == task_id
    assert metadata.prompt_condition == "no_adverb"
    assert metadata.training_config == _config_name(task_id)
    assert metadata.target_global_step == 100
    assert (
        metadata.tabero_ppo_transition_boundary_semantics
        == TABERO_PPO_TRANSITION_BOUNDARY_SEMANTICS
    )


@pytest.mark.parametrize("task_id", [1, 5])
def test_configs_use_uniform_target_physics(monkeypatch, task_id):
    _load_config(monkeypatch, task_id)
    profile = json.loads((PROFILE_DIR / "libero_object.json").read_text())
    task = next(item for item in profile["tasks"] if item["task_id"] == task_id)
    target = task["physics"]["objects"][TASKS[task_id]["object"]]

    assert target == {
        "damage": {
            "threshold": {"mode": "mass_friction", "tolerance_factor": 1.1},
            "consecutive_frames": 4,
        },
        "friction": {
            "distribution": "uniform",
            "static_range": [0.4, 0.8],
            "dynamic_range": [0.3, 0.6],
            "apply_on": "reset",
            "num_buckets": 64,
        },
        "mass_kg": {
            "distribution": "uniform",
            "range": [0.5, 1.6],
            "apply_on": "reset",
        },
    }
    assert task["physics"]["gripper"]["friction"] == {
        "static": 0.5,
        "dynamic": 0.5,
    }


@pytest.mark.parametrize("task_id", [1, 5])
def test_launcher_dry_run_selects_task_specific_config_and_artifacts(tmp_path, task_id):
    run_id = f"20260812_12000{task_id}_smoke"
    env = os.environ.copy()
    env.update(
        {
            "TABERO_RESULTS_ROOT": str(tmp_path),
            "TABERO_PIRL_RUN_ID": run_id,
            "WANDB_RUN_ID": f"wandb-task{task_id}-test",
            "CUDA_VISIBLE_DEVICES": "0,1",
        }
    )
    result = subprocess.run(
        ["bash", str(LAUNCHER), str(task_id), "smoke", "--dry-run"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    output = (
        tmp_path
        / f"tabero_task{task_id}_no_adverb_uniform_physics_action_expert_lora_2gpu_capacity_smoke_{run_id}"
    )
    metadata = _read_env(output / "run.env")
    assert metadata["TABERO_TASK0_NO_ADVERB_RUN_ID"] == run_id
    assert metadata["TABERO_CONFIG_NAME"] == _config_name(task_id)
    assert metadata["TABERO_VISIBLE_GPUS"] == "0,1"
    assert (
        metadata["TABERO_PPO_TRANSITION_BOUNDARY_SEMANTICS"]
        == TABERO_PPO_TRANSITION_BOUNDARY_SEMANTICS
    )
    assert f"--config-name {_config_name(task_id)}" in result.stdout
    assert "runner.max_epochs=1" in result.stdout
    assert "actor.fsdp_config.trainable_checkpoint_metadata.target_global_step=1" in (
        result.stdout
    )


def test_launcher_rejects_tasks_outside_requested_pair():
    result = subprocess.run(
        ["bash", str(LAUNCHER), "0", "smoke", "--dry-run"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "Usage:" in result.stderr
