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

from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = "/data/home/sim6g/code/tabero/models/pi0_lora_tacfield_tabero_safetensors"
RUN_ID = "20260727_120000_formal"
STAGE1_PATH = (
    "/data/home/sim6g/code/tabero/results/tabero_rlt_stage1_20260721_tactile_fixed/"
    "tabero_rlt_stage1_tacfield_tactile_fixed/checkpoints/global_step_2000/actor"
)
TASKS = {
    0: (
        "pick up the alphabet soup and place it in the basket",
        "libero_object_task0_pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5",
    ),
    5: (
        "pick up the tomato sauce and place it in the basket",
        "libero_object_task5_pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo.hdf5",
    ),
}


def _load(path, monkeypatch):
    monkeypatch.setenv("TABERO_MATRIX_RUN_ID", RUN_ID)
    return OmegaConf.load(REPO_ROOT / path)


def _assert_task_firm_contract(cfg, task_id):
    description, hdf5_name = TASKS[task_id]
    for split in (cfg.env.train.init_params, cfg.env.eval.init_params):
        assert split.task_suite == "libero_object"
        assert split.task_id == task_id
        assert OmegaConf.to_container(split.tasks) == [
            {"task_suite": "libero_object", "task_id": task_id}
        ]
        assert split.task_description == description
        assert split.hdf5_initial_states_path.endswith(hdf5_name)
        assert split.hdf5_reset_assignment == "cyclic"
        assert split.success.required_consecutive_steps == 8
        assert split.prompt_conditions.enabled is True
        assert split.prompt_conditions.assignment == "cyclic"
        assert split.prompt_conditions.condition_cycle == ["firm"]
        assert split.prompt_conditions.firm_adverbs == ["firmly", "tightly"]
        assert split.prompt_conditions.prompt_seed == 0


def _assert_matrix_logger(cfg, experiment_stem):
    expected_name = f"{experiment_stem}_{RUN_ID}"
    assert cfg.runner.logger.log_path == (
        f"/data/home/sim6g/code/tabero/results/{expected_name}"
    )
    assert cfg.runner.logger.experiment_name == expected_name


def test_rlt_task0_formal_config(monkeypatch):
    cfg = _load("examples/tabero/tabero_rlt_stage2_ac_task0_firm.yaml", monkeypatch)

    _assert_task_firm_contract(cfg, 0)
    _assert_matrix_logger(cfg, "tabero_rlt_stage2_task0_firm")
    assert cfg.runner.max_epochs == 100
    assert cfg.runner.save_interval == 5
    assert cfg.runner.logger.logger_backends == ["tensorboard", "wandb"]
    assert cfg.algorithm.loss_type == "rlt_ac"
    assert cfg.env.train.total_num_envs == 16
    assert cfg.env.train.rollout_epoch == 4
    assert cfg.env.eval.total_num_envs == 2
    assert cfg.env.eval.rollout_epoch == 25
    assert cfg.rollout.rlt_feature_model.model_path == STAGE1_PATH
    assert cfg.rollout.rlt_feature_model.openpi.rlt_stage2_encoder_only is True


def test_pirl_task5_formal_config(monkeypatch):
    cfg = _load(
        "examples/embodiment/config/"
        "isaaclab_pi0_peft_lora_tacfield_tabero_task5_firm_8gpu_50step.yaml",
        monkeypatch,
    )

    _assert_task_firm_contract(cfg, 5)
    _assert_matrix_logger(cfg, "tabero_task5_firm_action_expert_lora_8gpu_50step")
    assert cfg.runner.max_epochs == 50
    assert cfg.runner.save_interval == 5
    assert cfg.runner.logger.logger_backends == ["tensorboard", "wandb"]
    assert cfg.algorithm.adv_type == "gae"
    assert cfg.algorithm.loss_type == "actor_critic"
    assert cfg.actor.model.model_path == MODEL_PATH
    assert cfg.rollout.model.model_path == MODEL_PATH
    assert cfg.actor.model.is_lora is True
    assert cfg.actor.model.lora_target == "action_expert"
    assert cfg.actor.model.freeze_non_lora is True
    assert cfg.actor.model.openpi.train_expert_only is True
    assert cfg.actor.model.add_value_head is True
    assert OmegaConf.to_container(
        cfg.actor.fsdp_config.trainable_checkpoint_metadata
    ) == {
        "method": "pirl",
        "task_id": 5,
        "training_config": (
            "isaaclab_pi0_peft_lora_tacfield_tabero_task5_firm_8gpu_50step"
        ),
        "target_global_step": 50,
    }
    assert cfg.env.train.total_num_envs == 84
    assert cfg.env.train.rollout_epoch == 2


def test_pirl_task0_historical_config_remains_without_retroactive_provenance():
    cfg = OmegaConf.load(
        REPO_ROOT / "examples/embodiment/config/"
        "isaaclab_pi0_peft_lora_tacfield_tabero_task0_firm_8gpu_50step.yaml"
    )

    assert "trainable_checkpoint_metadata" not in cfg.actor.fsdp_config
