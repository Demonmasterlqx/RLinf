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

CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/sft/config/replay_firm_tabero_pi05_tacfield_sft_2gpu.yaml"
)
XARM_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/sft/config/replay_firm_tabero_xarm_gripper_pi05_tacfield_sft_2gpu.yaml"
)
XARM_20K_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/sft/config/replay_firm_tabero_xarm_gripper_pi05_tacfield_sft_2gpu_20k.yaml"
)
XARM_PI0_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/sft/config/replay_firm_tabero_xarm_gripper_pi0_tacfield_sft_2gpu.yaml"
)


def test_replay_firm_pi05_tacfield_sft_2gpu_config_contract(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("EMBODIED_PATH", str(CONFIG_PATH.parents[1]))
    monkeypatch.setenv("REPLAY_FIRM_PI05_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("REPLAY_FIRM_PI05_RUN_NAME", "pi05-tacfield-test")
    monkeypatch.setenv(
        "REPLAY_FIRM_PI05_CHECKPOINT_ROOT", str(tmp_path / "checkpoints")
    )
    monkeypatch.setenv(
        "REPLAY_FIRM_PI05_NORM_STATS", str(tmp_path / "norm_stats.json")
    )

    cfg = OmegaConf.load(CONFIG_PATH)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    assert cfg.defaults[0] == "model/pi0_5@actor.model"
    assert cfg.cluster.component_placement.actor == "0-1"
    assert cfg.runner.max_steps == 20000
    assert cfg.runner.save_interval == 2000
    assert cfg.runner.logger.logger_backends == ["tensorboard", "wandb"]
    assert cfg.actor.micro_batch_size == 16
    assert cfg.actor.global_batch_size == 32
    assert cfg.data.train_data_paths[0].dataset_path.endswith(
        "datas/replay_firm_tabero"
    )
    assert cfg.data.train_data_paths[0].weight == pytest.approx(1.0)

    model = cfg.actor.model
    assert model.model_type == "openpi"
    assert model.model_path.endswith("models/pi05_libero_pytorch")
    assert model.num_action_chunks == 10
    assert model.action_dim == 13
    assert model.paligemma_lora_rank == 16
    assert model.action_expert_lora_rank == 32
    assert model.paligemma_lora_exclude_modules == ".*vision_tower.*"
    assert model.lora_target == "both"
    assert model.freeze_non_lora is True
    assert model.export_norm_asset_id == "replay_firm_tabero"
    assert model.extra_trainable_modules == [
        "paligemma_with_expert.paligemma.base_model.model.model.vision_tower",
        "action_in_proj",
        "time_mlp_in",
        "time_mlp_out",
        "action_out_proj",
        "tactile_prefix_encoder",
    ]

    openpi = model.openpi
    assert openpi.config_name == "pi05_lora_tacfield_tabero"
    assert openpi.pi05 is True
    assert openpi.action_horizon == 10
    assert openpi.discrete_state_input is True
    assert openpi.action_chunk == 10
    assert openpi.action_env_dim == 13
    assert openpi.effective_action_dim == 13
    assert openpi.num_images_in_input == 2
    assert openpi.tactile_type == "expert_his_c_fut"
    assert openpi.tactile_dim == 6
    assert openpi.tactile_dim_in == 0
    assert openpi.tactile_prefix_dim_in == 9 * 440 * 2
    assert openpi.tactile_prefix_history == 8
    assert openpi.tactile_prefix_encoder_type == "tcn"
    assert openpi.tactile_prefix_use_reference_frame is True
    assert openpi.tactile_prefix_diff_from_reference is False
    assert openpi.tactile_streams == ["tactile_prefix"]
    assert openpi.tactile_loss_weight == pytest.approx(0.01)
    assert openpi.padding_loss_weight == pytest.approx(1.0)
    assert openpi.expert_his_c_fut_loss_mode == "weighted_full"

    assert model.frozen_parameter_precision == "bf16"
    assert model.trainable_parameter_precision == "fp32"
    assert cfg.actor.optim.lr == pytest.approx(2.5e-5)
    assert cfg.actor.optim.adam_beta1 == pytest.approx(0.9)
    assert cfg.actor.optim.adam_beta2 == pytest.approx(0.95)
    assert cfg.actor.optim.adam_eps == pytest.approx(1.0e-8)
    assert cfg.actor.optim.weight_decay == pytest.approx(1.0e-10)
    assert cfg.actor.optim.clip_grad == pytest.approx(1.0)
    assert cfg.actor.optim.lr_warmup_steps == 1000
    assert cfg.actor.optim.lr_decay_steps == 30000
    assert cfg.actor.optim.min_lr == pytest.approx(2.5e-6)
    assert cfg.actor.fsdp_config.sharding_strategy == "full_shard"
    assert cfg.actor.fsdp_config.use_orig_params is False
    assert cfg.actor.fsdp_config.cpu_offload is False
    assert cfg.actor.fsdp_config.gradient_checkpointing is True
    assert cfg.actor.fsdp_config.amp_autocast.enabled is True
    assert cfg.actor.fsdp_config.amp_autocast.precision == "bf16"
    assert cfg.actor.fsdp_config.grad_scaler.enabled is False

    metadata = cfg.actor.fsdp_config.trainable_checkpoint_metadata
    assert metadata.model_family == "pi05"
    assert metadata.openpi_config_name == "pi05_lora_tacfield_tabero"
    assert metadata.deployment_config_name == "pi05_lora_tacfield_tabero"
    assert metadata.action_horizon == 10
    assert metadata.tactile_prefix_dim_in == 9 * 440 * 2
    assert metadata.target_global_step == 20000


def test_replay_firm_xarm_gripper_pi05_tacfield_sft_2gpu_config_contract(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("EMBODIED_PATH", str(XARM_CONFIG_PATH.parents[1]))
    monkeypatch.setenv("REPLAY_FIRM_XARM_PI05_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("REPLAY_FIRM_XARM_PI05_RUN_NAME", "xarm-gripper-test")
    monkeypatch.setenv(
        "REPLAY_FIRM_XARM_PI05_CHECKPOINT_ROOT", str(tmp_path / "checkpoints")
    )
    monkeypatch.setenv(
        "REPLAY_FIRM_XARM_PI05_NORM_STATS", str(tmp_path / "norm_stats.json")
    )

    cfg = OmegaConf.load(XARM_CONFIG_PATH)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    assert cfg.defaults[0] == "model/pi0_5@actor.model"
    assert cfg.runner.max_steps == 5000
    assert cfg.runner.save_interval == 1000
    assert cfg.actor.micro_batch_size == 16
    assert cfg.actor.global_batch_size == 32
    assert cfg.actor.optim.total_training_steps == 5000
    assert cfg.actor.optim.lr_warmup_steps == 1000
    assert cfg.actor.optim.lr_decay_steps == 30000
    assert cfg.data.train_data_paths[0].dataset_path.endswith(
        "datas/replay_firm_tabero_xarm_gripper"
    )

    model = cfg.actor.model
    assert model.openpi.config_name == "pi05_lora_tacfield_tabero_xarm_gripper"
    assert model.openpi.pi05 is True
    assert model.openpi.action_horizon == 10
    assert model.openpi.discrete_state_input is True
    assert model.openpi.effective_action_dim == 13
    assert model.openpi.tactile_prefix_dim_in == 9 * 440 * 2
    assert model.export_norm_asset_id == "replay_firm_tabero_xarm_gripper"

    metadata = cfg.actor.fsdp_config.trainable_checkpoint_metadata
    assert metadata.dataset == "datas/replay_firm_tabero_xarm_gripper"
    assert metadata.gripper_coordinate == "xarm_positive_open"
    assert metadata.training_config == XARM_CONFIG_PATH.stem
    assert metadata.target_global_step == 5000


def test_replay_firm_xarm_gripper_pi05_tacfield_sft_2gpu_20k_config_contract(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("EMBODIED_PATH", str(XARM_20K_CONFIG_PATH.parents[1]))
    monkeypatch.setenv("REPLAY_FIRM_XARM_PI05_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("REPLAY_FIRM_XARM_PI05_RUN_NAME", "xarm-gripper-20k-test")
    monkeypatch.setenv(
        "REPLAY_FIRM_XARM_PI05_CHECKPOINT_ROOT", str(tmp_path / "checkpoints")
    )
    monkeypatch.setenv(
        "REPLAY_FIRM_XARM_PI05_NORM_STATS", str(tmp_path / "norm_stats.json")
    )

    cfg = OmegaConf.load(XARM_20K_CONFIG_PATH)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    assert cfg.defaults[0] == "model/pi0_5@actor.model"
    assert cfg.cluster.component_placement.actor == "5-6"
    assert cfg.runner.resume_dir is None
    assert cfg.runner.max_steps == 20000
    assert cfg.runner.save_interval == 1000
    assert cfg.actor.micro_batch_size == 16
    assert cfg.actor.global_batch_size == 32
    assert cfg.actor.optim.total_training_steps == 20000
    assert cfg.actor.optim.lr_warmup_steps == 1000
    assert cfg.actor.optim.lr_decay_steps == 30000
    assert cfg.data.train_data_paths[0].dataset_path.endswith(
        "datas/replay_firm_tabero_xarm_gripper"
    )

    model = cfg.actor.model
    assert model.model_path.endswith("models/pi05_libero_pytorch")
    assert model.openpi.config_name == "pi05_lora_tacfield_tabero_xarm_gripper"
    assert model.openpi.pi05 is True
    assert model.openpi.action_horizon == 10
    assert model.openpi.discrete_state_input is True
    assert model.openpi.effective_action_dim == 13
    assert model.openpi.tactile_prefix_dim_in == 9 * 440 * 2
    assert model.lora_target == "both"
    assert model.freeze_non_lora is True
    assert model.extra_trainable_modules == [
        "paligemma_with_expert.paligemma.base_model.model.model.vision_tower",
        "action_in_proj",
        "time_mlp_in",
        "time_mlp_out",
        "action_out_proj",
        "tactile_prefix_encoder",
    ]
    assert model.export_norm_asset_id == "replay_firm_tabero_xarm_gripper"

    metadata = cfg.actor.fsdp_config.trainable_checkpoint_metadata
    assert metadata.dataset == "datas/replay_firm_tabero_xarm_gripper"
    assert metadata.model_family == "pi05"
    assert metadata.gripper_coordinate == "xarm_positive_open"
    assert metadata.training_config == XARM_20K_CONFIG_PATH.stem
    assert metadata.target_global_step == 20000


def test_replay_firm_xarm_gripper_pi0_tacfield_sft_2gpu_config_contract(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("EMBODIED_PATH", str(XARM_PI0_CONFIG_PATH.parents[1]))
    monkeypatch.setenv("REPLAY_FIRM_XARM_PI0_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("REPLAY_FIRM_XARM_PI0_RUN_NAME", "xarm-pi0-test")
    monkeypatch.setenv(
        "REPLAY_FIRM_XARM_PI0_CHECKPOINT_ROOT", str(tmp_path / "checkpoints")
    )
    monkeypatch.setenv(
        "REPLAY_FIRM_XARM_PI0_NORM_STATS", str(tmp_path / "norm_stats.json")
    )

    cfg = OmegaConf.load(XARM_PI0_CONFIG_PATH)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    assert cfg.defaults[0] == "model/pi0@actor.model"
    assert cfg.runner.max_steps == 5000
    assert cfg.runner.save_interval == 1000
    assert cfg.actor.micro_batch_size == 16
    assert cfg.actor.global_batch_size == 32
    assert cfg.data.train_data_paths[0].dataset_path.endswith(
        "datas/replay_firm_tabero_xarm_gripper"
    )

    model = cfg.actor.model
    assert model.model_path.endswith("models/pi0_base")
    assert model.num_action_chunks == 50
    assert model.extra_trainable_modules == [
        "paligemma_with_expert.paligemma.base_model.model.model.vision_tower",
        "state_proj",
        "action_in_proj",
        "action_time_mlp_in",
        "action_time_mlp_out",
        "action_out_proj",
        "tactile_prefix_encoder",
    ]
    assert model.openpi.config_name == "pi0_lora_tacfield_tabero_xarm_gripper"
    assert model.openpi.pi05 is False
    assert model.openpi.action_horizon == 50
    assert model.openpi.action_chunk == 50
    assert model.openpi.discrete_state_input is False
    assert model.openpi.effective_action_dim == 13
    assert model.openpi.tactile_prefix_dim_in == 9 * 440 * 2
    assert model.export_norm_asset_id == "replay_firm_tabero_xarm_gripper"

    metadata = cfg.actor.fsdp_config.trainable_checkpoint_metadata
    assert metadata.dataset == "datas/replay_firm_tabero_xarm_gripper"
    assert metadata.model_family == "pi0"
    assert metadata.action_horizon == 50
    assert metadata.control_hz == 20
    assert metadata.policy_horizon_seconds == pytest.approx(2.5)
    assert metadata.execution_steps == 10
    assert metadata.execution_horizon_seconds == pytest.approx(0.5)
    assert metadata.gripper_coordinate == "xarm_positive_open"
    assert metadata.training_config == XARM_PI0_CONFIG_PATH.stem
    assert metadata.target_global_step == 5000
