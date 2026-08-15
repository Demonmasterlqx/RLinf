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

import os
import subprocess
from pathlib import Path

from omegaconf import OmegaConf

from rlinf.config import validate_embodied_cfg

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "examples" / "embodiment" / "config"
BASE_CONFIG_NAME = (
    "isaaclab_pi0_peft_lora_tacfield_tabero_task0_no_adverb_uniform_physics_"
    "no_damage_force_reward_2gpu_100step"
)
CONFIG_NAME = (
    "isaaclab_pi0_peft_lora_tacfield_tabero_task0_no_adverb_uniform_physics_"
    "no_damage_force_reward_coef1_maxbonus10_2gpu_100step"
)
CONFIG_PATH = CONFIG_DIR / f"{CONFIG_NAME}.yaml"
BASE_CONFIG_PATH = CONFIG_DIR / f"{BASE_CONFIG_NAME}.yaml"
LAUNCHER = (
    REPO_ROOT
    / "examples"
    / "embodiment"
    / "run_tabero_task0_no_adverb_uniform_physics_no_damage_force_reward_"
    "coef1_maxbonus10_action_expert.sh"
)
OUTPUT_PREFIX = (
    "tabero_task0_no_adverb_uniform_physics_no_damage_force_reward_"
    "coef1_maxbonus10_action_expert_lora_2gpu"
)


def _load(monkeypatch):
    monkeypatch.setenv("TABERO_TASK0_NO_ADVERB_RUN_ID", "20260814_200000_formal")
    monkeypatch.setenv("EMBODIED_PATH", str(REPO_ROOT / "examples" / "embodiment"))
    return OmegaConf.load(CONFIG_PATH)


def test_config_changes_only_experiment_identity_and_force_bonus(monkeypatch):
    cfg = _load(monkeypatch)
    base = OmegaConf.load(BASE_CONFIG_PATH)

    base.runner.logger.log_path = cfg.runner.logger.log_path
    base.runner.logger.experiment_name = cfg.runner.logger.experiment_name
    base.env.train.init_params.success.force_bonus = (
        cfg.env.train.init_params.success.force_bonus
    )
    base.env.eval.init_params.success.force_bonus = (
        cfg.env.eval.init_params.success.force_bonus
    )
    base.actor.fsdp_config.trainable_checkpoint_metadata.training_config = (
        cfg.actor.fsdp_config.trainable_checkpoint_metadata.training_config
    )

    assert OmegaConf.to_container(base, resolve=True) == OmegaConf.to_container(
        cfg, resolve=True
    )


def test_config_uses_matching_coef1_maxbonus10_train_and_eval(monkeypatch):
    cfg = _load(monkeypatch)
    expected = {
        "enabled": True,
        "coefficient": 1.0,
        "epsilon": 1.0,
        "max_bonus": 10.0,
        "min_valid_samples": 4,
        "contact_epsilon": 1.0,
    }

    for split in (cfg.env.train, cfg.env.eval):
        assert OmegaConf.to_container(split.init_params.success.force_bonus) == expected
        assert split.init_params.success.terminal_reward == 1.0
        assert split.init_params.success.required_consecutive_steps == 8

    assert cfg.cluster.component_placement.env == "0"
    assert cfg.cluster.component_placement.rollout == "0"
    assert cfg.cluster.component_placement.actor == "1"
    assert cfg.runner.max_epochs == 100
    assert cfg.runner.save_interval == 10
    assert cfg.actor.fsdp_config.trainable_checkpoint_metadata.training_config == (
        CONFIG_NAME
    )
    assert validate_embodied_cfg(cfg) is cfg


def test_launcher_dry_run_selects_exact_config_gpus_and_output(tmp_path):
    run_id = "20260814_200001_smoke"
    env = os.environ.copy()
    env.update(
        {
            "TABERO_RESULTS_ROOT": str(tmp_path),
            "TABERO_TASK0_FORCE_RUN_ID": run_id,
            "WANDB_RUN_ID": "wandb-coef1-test",
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
    output = tmp_path / f"{OUTPUT_PREFIX}_capacity_smoke_{run_id}"
    assert output.is_dir()
    assert f"--config-name {CONFIG_NAME}" in result.stdout
    assert "CUDA_VISIBLE_DEVICES: 0,1" in result.stdout
    assert "runner.max_epochs=1" in result.stdout
    assert "runner.save_interval=1" in result.stdout
    assert f"runner.logger.log_path={output}" in result.stdout
