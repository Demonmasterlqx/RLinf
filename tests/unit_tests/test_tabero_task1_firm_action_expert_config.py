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

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError


CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "embodiment"
    / "config"
    / "isaaclab_pi0_peft_lora_tacfield_tabero_task1_firm_8gpu_50step.yaml"
)
MODEL_PATH = (
    "/data/home/sim6g/code/tabero/models/pi0_lora_tacfield_tabero_safetensors"
)
HDF5_PATH = (
    "/data/home/sim6g/code/tabero/Tabero/benchmarks/datasets/libero/"
    "assembled_hdf5/"
    "libero_object_task1_pick_up_the_cream_cheese_and_place_it_in_the_basket_demo.hdf5"
)


def test_task1_firm_config_has_exact_8_gpu_training_contract(monkeypatch):
    monkeypatch.setenv("TABERO_TASK1_RUN_ID", "20260726_120000")
    cfg = OmegaConf.load(CONFIG_PATH)

    assert cfg.cluster.num_nodes == 1
    assert OmegaConf.to_container(cfg.cluster.component_placement) == {
        "actor": "4-7",
        "rollout": "0-3",
        "env": "0-3",
    }
    assert cfg.env.train.total_num_envs == 84
    assert cfg.env.train.rollout_epoch == 2
    assert cfg.env.train.max_steps_per_rollout_epoch == 360
    assert cfg.env.train.max_episode_steps == 360
    assert cfg.actor.global_batch_size == 168
    assert cfg.actor.micro_batch_size == 2
    assert (
        cfg.env.train.total_num_envs * cfg.env.train.rollout_epoch
        == cfg.actor.global_batch_size
    )

    assert cfg.runner.max_epochs == 50
    assert cfg.runner.max_steps == -1
    assert cfg.runner.val_check_interval == -1
    assert cfg.runner.save_interval == 5
    assert cfg.runner.resume_dir is None
    assert cfg.runner.ckpt_path is None
    assert cfg.runner.logger.project_name == "tabero-rlinf"
    assert cfg.runner.logger.logger_backends == ["tensorboard", "wandb"]
    assert "task1_firm_action_expert" in cfg.runner.logger.experiment_name
    assert cfg.runner.logger.log_path.startswith(
        "/data/home/sim6g/code/tabero/results/"
    )
    assert "task1_firm_action_expert" in cfg.runner.logger.log_path
    assert cfg.runner.logger.log_path.endswith("20260726_120000")


def test_task1_firm_config_has_exact_task_and_prompt_contract(monkeypatch):
    monkeypatch.setenv("TABERO_TASK1_RUN_ID", "20260726_120000")
    cfg = OmegaConf.load(CONFIG_PATH)
    train = cfg.env.train.init_params

    assert train.task_suite == "libero_object"
    assert train.task_id == 1
    assert OmegaConf.to_container(train.tasks) == [
        {"task_suite": "libero_object", "task_id": 1}
    ]
    assert train.require_all_tasks_active is True
    assert train.task_description == (
        "pick up the cream cheese and place it in the basket"
    )
    assert train.hdf5_initial_states_path == HDF5_PATH
    assert train.hdf5_reset_assignment == "cyclic"
    assert train.prompt_conditions.enabled is True
    assert train.prompt_conditions.assignment == "cyclic"
    assert train.prompt_conditions.condition_cycle == ["firm"]
    assert train.prompt_conditions.firm_adverbs == ["firmly", "tightly"]
    assert train.prompt_conditions.prompt_seed == 0
    assert train.success.required_consecutive_steps == 8
    assert train.success.terminal_reward == 1.0


def test_task1_firm_config_trains_only_action_expert_lora_and_value_head(monkeypatch):
    monkeypatch.setenv("TABERO_TASK1_RUN_ID", "20260726_120000")
    cfg = OmegaConf.load(CONFIG_PATH)
    model = cfg.actor.model

    assert cfg.rollout.model.model_path == MODEL_PATH
    assert model.model_path == MODEL_PATH
    assert model.is_lora is True
    assert model.lora_target == "action_expert"
    assert model.lora_target not in {"paligemma", "both"}
    assert model.lora_rank == 32
    assert model.freeze_non_lora is True
    assert model.openpi.train_expert_only is True
    assert model.add_value_head is True
    assert model.openpi.add_value_head is True
    assert model.num_action_chunks == 10
    assert model.num_steps == 10
    assert cfg.actor.optim.lr == 1.0e-6
    assert cfg.actor.optim.value_lr == 1.0e-4

    assert cfg.algorithm.adv_type == "gae"
    assert cfg.algorithm.loss_type == "actor_critic"
    assert cfg.algorithm.update_epoch == 1
    assert cfg.algorithm.kl_beta == 0.05
    assert cfg.algorithm.clip_ratio_high == 0.2
    assert cfg.algorithm.clip_ratio_low == 0.2
    assert cfg.algorithm.clip_ratio_c == 3.0
    assert cfg.algorithm.value_clip == 0.2
    assert cfg.algorithm.huber_delta == 10.0
    assert cfg.algorithm.gamma == 0.99
    assert cfg.algorithm.gae_lambda == 0.95
    assert cfg.actor.training_backend == "fsdp"
    assert cfg.actor.fsdp_config.strategy == "fsdp"
    assert cfg.actor.fsdp_config.sharding_strategy == "no_shard"
    assert cfg.actor.fsdp_config.use_orig_params is False
    assert cfg.actor.fsdp_config.checkpoint_format == "dcp"
    assert cfg.actor.fsdp_config.save_trainable_model_weights is True
    assert cfg.actor.fsdp_config.save_full_model_weights is False
    assert cfg.weight_syncer.type == "patch"
    assert cfg.weight_syncer.patch.snapshot_device == "cpu"
    assert cfg.weight_syncer.patch.delta_encoding is True
    assert cfg.weight_syncer.patch.compression == "none"
    assert cfg.weight_syncer.patch.init_sync.enabled is True
    assert cfg.weight_syncer.patch.init_sync.prefixes is None
    assert cfg.weight_syncer.patch.init_sync.bucket_size == 134217728


def test_task1_firm_config_requires_collision_free_runtime_run_id(monkeypatch):
    monkeypatch.delenv("TABERO_TASK1_RUN_ID", raising=False)
    cfg = OmegaConf.load(CONFIG_PATH)

    with pytest.raises(InterpolationResolutionError, match="TABERO_TASK1_RUN_ID"):
        _ = cfg.runner.logger.log_path
