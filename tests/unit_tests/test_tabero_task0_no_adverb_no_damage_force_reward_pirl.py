# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import subprocess
from pathlib import Path

from omegaconf import OmegaConf

from rlinf.config import validate_embodied_cfg

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_NAME = (
    "isaaclab_pi0_peft_lora_tacfield_tabero_task0_no_adverb_uniform_physics_"
    "no_damage_force_reward_2gpu_100step"
)
CONFIG_PATH = REPO_ROOT / "examples" / "embodiment" / "config" / f"{CONFIG_NAME}.yaml"
LAUNCHER = (
    REPO_ROOT
    / "examples"
    / "embodiment"
    / "run_tabero_task0_no_adverb_uniform_physics_no_damage_force_reward_action_expert.sh"
)
PROFILE_DIR = Path(
    "/data/home/sim6g/code/tabero/Tabero/benchmarks/datasets/libero/"
    "config_profiles/"
    "alltask_fixed_damage_1000000_uniform_mass_05_16_friction_04_08_from_rlinf_sft_20k"
)
MODEL_PATH = (
    "/data/home/sim6g/code/tabero/models/pi0_lora_tacfield_tabero_all_firm_safetensors"
)


def _load(monkeypatch):
    monkeypatch.setenv("TABERO_TASK0_NO_ADVERB_RUN_ID", "20260814_010000_formal")
    return OmegaConf.load(CONFIG_PATH)


def test_config_uses_no_damage_uniform_physics_and_force_reward(monkeypatch):
    cfg = _load(monkeypatch)
    for split_cfg in (cfg.env.train, cfg.env.eval):
        split = split_cfg.init_params
        assert split.task_id == 0
        assert split.prompt_conditions.enabled is False
        assert split.libero_config_dir == str(PROFILE_DIR)
        assert split.chunk_boundary_mode == "terminal_safe_hdf5_v1"
        assert split.success.required_consecutive_steps == 8
        assert OmegaConf.to_container(split.success.force_bonus) == {
            "enabled": True,
            "coefficient": 4.0,
            "epsilon": 1.0,
            "max_bonus": 0.2,
            "min_valid_samples": 4,
            "contact_epsilon": 1.0,
        }
        assert split_cfg.auto_reset is False
        assert split_cfg.ignore_terminations is False

    profile = json.loads((PROFILE_DIR / "libero_object.json").read_text())
    task0 = next(task for task in profile["tasks"] if task["task_id"] == 0)
    physics = task0["physics"]
    target = physics["objects"]["alphabet_soup_1"]
    assert target["mass_kg"] == {
        "distribution": "uniform",
        "range": [0.5, 1.6],
        "apply_on": "reset",
    }
    assert target["friction"] == {
        "distribution": "uniform",
        "static_range": [0.4, 0.8],
        "dynamic_range": [0.3, 0.6],
        "apply_on": "reset",
        "num_buckets": 64,
    }
    assert target["damage"] == {
        "max_squeeze_force": 1000000.0,
        "consecutive_frames": 4,
    }
    assert physics["gripper"]["friction"] == {
        "static": 0.5,
        "dynamic": 0.5,
    }


def test_config_preserves_action_expert_only_boundary_safe_contract(monkeypatch):
    cfg = _load(monkeypatch)
    assert cfg.runner.logger.logger_backends == ["tensorboard", "wandb"]
    assert cfg.runner.max_epochs == 100
    assert cfg.runner.save_interval == 10
    assert cfg.env.train.total_num_envs == 21
    assert cfg.env.train.rollout_epoch == 2
    assert cfg.actor.global_batch_size == 42
    assert cfg.actor.micro_batch_size == 2
    assert cfg.actor.model.model_path == MODEL_PATH
    assert cfg.rollout.model.model_path == MODEL_PATH
    assert cfg.actor.model.is_lora is True
    assert cfg.actor.model.lora_target == "action_expert"
    assert cfg.actor.model.freeze_non_lora is True
    assert cfg.actor.model.openpi.train_expert_only is True
    assert cfg.actor.model.add_value_head is True
    assert cfg.actor.model.openpi.add_value_head is True
    assert cfg.actor.fsdp_config.trainable_checkpoint_metadata.training_config == (
        CONFIG_NAME
    )
    assert cfg.actor.fsdp_config.trainable_checkpoint_metadata.target_global_step == 100
    assert validate_embodied_cfg(cfg) is cfg


def test_launcher_dry_run_selects_exact_config_profile_and_output(tmp_path):
    run_id = "20260814_010001_smoke"
    env = os.environ.copy()
    env.update(
        {
            "TABERO_RESULTS_ROOT": str(tmp_path),
            "TABERO_TASK0_FORCE_RUN_ID": run_id,
            "WANDB_RUN_ID": "wandb-test-id",
            "CUDA_VISIBLE_DEVICES": "0,1",
        }
    )

    result = subprocess.run(
        ["bash", str(LAUNCHER), "smoke", "--dry-run"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    output = tmp_path / (
        "tabero_task0_no_adverb_uniform_physics_no_damage_force_reward_"
        f"action_expert_lora_2gpu_capacity_smoke_{run_id}"
    )
    assert output.is_dir()
    assert f"--config-name {CONFIG_NAME}" in result.stdout
    assert "runner.max_epochs=1" in result.stdout
    assert "runner.save_interval=1" in result.stdout
    assert f"runner.logger.log_path={output}" in result.stdout
