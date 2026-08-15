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

import pytest
from omegaconf import OmegaConf

from rlinf.config import validate_embodied_cfg

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "examples" / "embodiment" / "config"
PROFILE_DIR = Path(
    "/data/home/sim6g/code/tabero/Tabero/benchmarks/datasets/libero/"
    "config_profiles/"
    "task0_fixed_mean_mass_friction_no_control_override_from_rlinf_sft_20k"
)
MODEL_PATH = Path(
    "/data/home/sim6g/code/tabero/models/"
    "pi0_lora_tacfield_tabero_all_firm_safetensors"
)
LAUNCHER = (
    REPO_ROOT
    / "examples"
    / "embodiment"
    / "run_tabero_task0_no_adverb_fixed_mean_no_random_force_reward_"
    "tuning_action_expert.sh"
)

CANDIDATES = {
    "coef10": (10.0, ("0", "0", "1")),
    "coef20": (20.0, ("2", "2", "3")),
    "coef30": (30.0, ("4", "4", "5")),
}


def _config_name(candidate: str) -> str:
    return (
        "isaaclab_pi0_peft_lora_tacfield_tabero_task0_no_adverb_fixed_mean_"
        f"no_random_force_reward_{candidate}_maxbonus1_2gpu_100step"
    )


def _load(monkeypatch, candidate: str):
    monkeypatch.setenv("TABERO_TASK0_NO_ADVERB_RUN_ID", "20260815_020000_formal")
    monkeypatch.setenv("EMBODIED_PATH", str(REPO_ROOT / "examples" / "embodiment"))
    return OmegaConf.load(CONFIG_DIR / f"{_config_name(candidate)}.yaml")


def test_fixed_mean_profile_disables_randomization_and_control_override():
    profile = json.loads((PROFILE_DIR / "libero_object.json").read_text())
    assert profile["total_tasks"] == 1
    task = profile["tasks"][0]
    assert task["task_id"] == 0
    assert "control_overrides" not in task

    physics = task["physics"]
    target = physics["objects"]["alphabet_soup_1"]
    assert target["mass_kg"] == 1.05
    assert target["friction"] == {"static": 0.6, "dynamic": 0.45}
    assert target["damage"] == {
        "max_squeeze_force": 1_000_000,
        "consecutive_frames": 4,
    }
    assert physics["gripper"]["friction"] == {
        "static": 0.5,
        "dynamic": 0.5,
    }


@pytest.mark.parametrize("candidate", tuple(CANDIDATES))
def test_candidate_preserves_action_expert_contract(monkeypatch, candidate):
    coefficient, placement = CANDIDATES[candidate]
    cfg = _load(monkeypatch, candidate)

    for split in (cfg.env.train, cfg.env.eval):
        params = split.init_params
        assert params.libero_config_dir == str(PROFILE_DIR)
        assert params.task_id == 0
        assert params.prompt_conditions.enabled is False
        assert params.chunk_boundary_mode == "terminal_safe_hdf5_v1"
        assert params.success.required_consecutive_steps == 8
        assert params.success.terminal_reward == 1.0
        assert OmegaConf.to_container(params.success.force_bonus) == {
            "enabled": True,
            "coefficient": coefficient,
            "epsilon": 1.0,
            "max_bonus": 1.0,
            "min_valid_samples": 4,
            "contact_epsilon": 1.0,
        }

    assert (
        cfg.cluster.component_placement.rollout,
        cfg.cluster.component_placement.env,
        cfg.cluster.component_placement.actor,
    ) == placement
    assert cfg.runner.logger.logger_backends == ["tensorboard", "wandb"]
    assert cfg.runner.max_epochs == 100
    assert cfg.runner.save_interval == 10
    assert cfg.actor.global_batch_size == 42
    assert cfg.actor.micro_batch_size == 2
    assert Path(cfg.actor.model.model_path) == MODEL_PATH
    assert Path(cfg.rollout.model.model_path) == MODEL_PATH
    assert cfg.actor.model.is_lora is True
    assert cfg.actor.model.lora_target == "action_expert"
    assert cfg.actor.model.freeze_non_lora is True
    assert cfg.actor.model.openpi.train_expert_only is True
    assert cfg.actor.model.add_value_head is True
    assert cfg.actor.fsdp_config.trainable_checkpoint_metadata.training_config == (
        _config_name(candidate)
    )
    assert cfg.actor.fsdp_config.trainable_checkpoint_metadata.target_global_step == 100
    assert validate_embodied_cfg(cfg) is cfg


@pytest.mark.parametrize(
    ("candidate", "expected_gpu_pair"),
    (("coef10", "0,1"), ("coef20", "2,3"), ("coef30", "4,5")),
)
def test_launcher_dry_run_selects_candidate_and_fixed_profile(
    tmp_path, candidate, expected_gpu_pair
):
    run_id = "20260815_020001_smoke"
    env = os.environ.copy()
    env.update(
        {
            "TABERO_RESULTS_ROOT": str(tmp_path),
            "TABERO_TASK0_FORCE_RUN_ID": run_id,
            "WANDB_RUN_ID": f"wandb-{candidate}-test",
        }
    )

    result = subprocess.run(
        ["bash", str(LAUNCHER), candidate, "smoke", "--dry-run"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"--config-name {_config_name(candidate)}" in result.stdout
    assert f"CUDA_VISIBLE_DEVICES: {expected_gpu_pair}" in result.stdout
    assert "runner.max_epochs=1" in result.stdout
    assert "runner.save_interval=1" in result.stdout


@pytest.mark.parametrize(
    ("coefficient", "mean_force", "expected_bonus"),
    (
        (10.0, 50.0, 0.2),
        (20.0, 50.0, 0.4),
        (30.0, 50.0, 0.6),
        (30.0, 30.0, 1.0),
        (30.0, 20.0, 1.0),
    ),
)
def test_candidate_bonus_is_success_only_bounded_signal(
    coefficient, mean_force, expected_bonus
):
    bonus = min(1.0, coefficient / max(mean_force, 1.0))
    assert bonus == pytest.approx(expected_bonus)
    assert 1.0 <= 1.0 + bonus <= 2.0
