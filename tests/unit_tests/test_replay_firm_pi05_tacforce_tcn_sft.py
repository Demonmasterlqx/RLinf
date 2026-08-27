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
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.openpi import _validate_checkpoint_load_result
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0ForRLActionPrediction,
)
from rlinf.models.embodiment.openpi.policies.tabero_policy import (
    TaberoTacForceInputs,
)
from rlinf.models.embodiment.openpi.tactile_encoder import TactileTCNEncoder

CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "examples/sft/config/"
    "replay_firm_tabero_xarm_gripper_repaired_v1_"
    "pi05_tacforce_tcn_sft_2gpu_gb16_30k.yaml"
)

TASK820_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/sft/config/"
    "realworld_replayed_task820_firm_"
    "pi05_tacforce_tcn_sft_2gpu_gb32_gc_off_force0001_30k.yaml"
)

TASK820_GB16_MB8_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/sft/config/"
    "realworld_replayed_task820_firm_"
    "pi05_tacforce_tcn_sft_2gpu_gb16_mb8_gc_off_force0001_30k.yaml"
)

TASK820_GB16_MB8_NOEMA_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/sft/config/"
    "realworld_replayed_task820_firm_"
    "pi05_tacforce_tcn_sft_2gpu_gb16_mb8_gc_off_noema_force0001_30k.yaml"
)

TASK820_GB16_MB8_GC_ON_NOEMA_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/sft/config/"
    "realworld_replayed_task820_firm_"
    "pi05_tacforce_tcn_sft_2gpu_gb16_mb8_gc_on_noema_force0001_30k.yaml"
)


def test_pi05_tacforce_tcn_registry_and_transform_contract():
    config = get_openpi_config("pi05_lora_tacforce_tcn_real")

    assert config.model.pi05 is True
    assert config.model.action_horizon == 50
    assert config.model.action_chunk == 50
    assert config.model.effective_action_dim == 13
    assert config.model.tactile_prefix_dim_in == 8 * 6
    assert config.model.tactile_prefix_history == 8
    assert config.model.tactile_prefix_encoder_type == "tcn"
    assert config.model.tactile_prefix_use_reference_frame is False
    assert config.model.tactile_prefix_diff_from_reference is False
    assert config.model.tactile_streams == ("tactile_prefix",)
    assert config.data.repo_id == "replay_firm_tabero_xarm_gripper_repaired_v1"
    assert config.data.assets.asset_id == "pi05_horizon50_tacforce_tcn"
    assert config.batch_size == 32
    assert config.num_train_steps == 30_000
    assert config.seed == 42
    assert config.ema_decay == 0.99

    transformed = TaberoTacForceInputs(model_type=config.model.model_type)(
        {
            "image": np.zeros((224, 224, 3), dtype=np.uint8),
            "wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "tactile_image": np.ones((224, 224, 3), dtype=np.uint8),
            "tactile_marker_motion": np.ones((9, 440, 2), dtype=np.float32),
            "tactile_gripper_force": np.zeros((8, 6), dtype=np.float32),
            "state": np.zeros(7, dtype=np.float32),
            "actions": np.zeros((50, 13), dtype=np.float32),
            "prompt": "test",
        }
    )
    assert transformed["tactile_prefix"].shape == (8, 6)
    assert transformed["actions"].shape == (50, 13)
    assert "tactile_image" not in transformed
    assert "tactile_marker_motion" not in transformed
    assert transformed["image_mask"]["right_wrist_0_rgb"] == np.False_


def test_pi05_tacforce_tcn_yaml_contract(monkeypatch):
    monkeypatch.setenv("EMBODIED_PATH", str(CONFIG_PATH.parents[1]))
    cfg = OmegaConf.load(CONFIG_PATH)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    assert cfg.cluster.component_placement.actor == "5-6"
    assert cfg.runner.max_steps == 30_000
    assert cfg.runner.save_interval == 1_000
    assert cfg.runner.logger.logger_backends == ["tensorboard", "wandb"]
    assert cfg.actor.micro_batch_size == 8
    assert cfg.actor.global_batch_size == 16
    assert cfg.actor.global_batch_size // (cfg.actor.micro_batch_size * 2) == 1
    assert cfg.actor.seed == 42
    assert cfg.actor.model_weight_ema_decay == pytest.approx(0.99)

    model = cfg.actor.model
    assert model.model_path.endswith("models/pi05_base_pytorch")
    assert model.num_action_chunks == 50
    assert model.action_dim == 13
    assert model.paligemma_lora_rank == 16
    assert model.action_expert_lora_rank == 32
    assert model.freeze_non_lora is True
    assert model.checkpoint_load_allowed_missing_prefixes == ["tactile_prefix_encoder."]
    assert model.extra_trainable_modules == [
        "paligemma_with_expert.paligemma.base_model.model.model.vision_tower",
        "action_in_proj",
        "time_mlp_in",
        "time_mlp_out",
        "action_out_proj",
        "tactile_prefix_encoder",
    ]

    openpi = model.openpi
    assert openpi.config_name == "pi05_lora_tacforce_tcn_real"
    assert openpi.action_horizon == 50
    assert openpi.action_chunk == 50
    assert openpi.tactile_prefix_dim_in == 48
    assert openpi.tactile_prefix_history == 8
    assert openpi.tactile_prefix_use_reference_frame is False
    assert openpi.tactile_prefix_diff_from_reference is False
    assert openpi.tactile_streams == ["tactile_prefix"]
    assert openpi.tactile_loss_weight == pytest.approx(0.1)
    assert openpi.padding_loss_weight == pytest.approx(1.0)
    assert openpi.expert_his_c_fut_loss_mode == "weighted_full"

    optim = cfg.actor.optim
    assert optim.lr == pytest.approx(2.5e-5)
    assert optim.adam_beta1 == pytest.approx(0.9)
    assert optim.adam_beta2 == pytest.approx(0.95)
    assert optim.adam_eps == pytest.approx(1e-8)
    assert optim.weight_decay == pytest.approx(1e-10)
    assert optim.clip_grad == pytest.approx(1.0)
    assert optim.lr_warmup_steps == 10_000
    assert optim.lr_decay_steps == 1_000_000
    assert optim.min_lr == pytest.approx(2.5e-6)

    metadata = cfg.actor.fsdp_config.trainable_checkpoint_metadata
    assert metadata.execution_steps == 10
    assert metadata.target_global_step == 30_000
    assert metadata.tactile_input == "tactile_gripper_force"
    assert metadata.excluded_tactile_inputs == [
        "tactile_image",
        "tactile_marker_motion",
    ]
    assert metadata.weight_variant == "ema"


def test_task820_force0001_yaml_contract(monkeypatch):
    monkeypatch.setenv("EMBODIED_PATH", str(TASK820_CONFIG_PATH.parents[1]))
    cfg = OmegaConf.load(TASK820_CONFIG_PATH)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    assert cfg.cluster.component_placement.actor == "5-6"
    assert cfg.runner.max_steps == 30_000
    assert cfg.data.train_data_paths[0].dataset_path.endswith(
        "datas/realworld_replayed_task820_firm"
    )
    assert cfg.actor.micro_batch_size == 16
    assert cfg.actor.global_batch_size == 32
    assert cfg.actor.global_batch_size // (cfg.actor.micro_batch_size * 2) == 1
    assert cfg.actor.model.openpi.tactile_loss_weight == pytest.approx(0.001)
    assert cfg.actor.fsdp_config.gradient_checkpointing is False
    metadata = cfg.actor.fsdp_config.trainable_checkpoint_metadata
    assert metadata.future_force_loss_weight == pytest.approx(0.001)
    assert metadata.training_config == TASK820_CONFIG_PATH.stem
    assert metadata.target_global_step == 30_000
    assert cfg.actor.fsdp_config.amp_autocast.enabled is True
    assert cfg.actor.fsdp_config.amp_autocast.precision == "bf16"
    assert cfg.actor.fsdp_config.grad_scaler.enabled is False


def test_task820_gb16_mb8_force0001_yaml_contract(monkeypatch):
    monkeypatch.setenv("EMBODIED_PATH", str(TASK820_GB16_MB8_CONFIG_PATH.parents[1]))
    cfg = OmegaConf.load(TASK820_GB16_MB8_CONFIG_PATH)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    assert cfg.cluster.component_placement.actor == "5-6"
    assert cfg.actor.micro_batch_size == 8
    assert cfg.actor.global_batch_size == 16
    assert cfg.actor.global_batch_size // (cfg.actor.micro_batch_size * 2) == 1
    assert cfg.actor.fsdp_config.gradient_checkpointing is False
    assert cfg.actor.model.openpi.tactile_loss_weight == pytest.approx(0.001)
    metadata = cfg.actor.fsdp_config.trainable_checkpoint_metadata
    assert metadata.future_force_loss_weight == pytest.approx(0.001)
    assert metadata.training_config == TASK820_GB16_MB8_CONFIG_PATH.stem
    assert cfg.runner.max_steps == 30_000


def test_task820_gb16_mb8_noema_force0001_yaml_contract(monkeypatch):
    monkeypatch.setenv(
        "EMBODIED_PATH", str(TASK820_GB16_MB8_NOEMA_CONFIG_PATH.parents[1])
    )
    cfg = OmegaConf.load(TASK820_GB16_MB8_NOEMA_CONFIG_PATH)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    assert cfg.cluster.component_placement.actor == "5-6"
    assert cfg.actor.micro_batch_size == 8
    assert cfg.actor.global_batch_size == 16
    assert cfg.actor.model_weight_ema_decay is None
    assert cfg.actor.fsdp_config.gradient_checkpointing is False
    assert cfg.actor.model.openpi.tactile_loss_weight == pytest.approx(0.001)
    metadata = cfg.actor.fsdp_config.trainable_checkpoint_metadata
    assert metadata.future_force_loss_weight == pytest.approx(0.001)
    assert metadata.model_weight_ema_decay is None
    assert metadata.weight_variant == "online"
    assert metadata.training_config == TASK820_GB16_MB8_NOEMA_CONFIG_PATH.stem


def test_task820_gb16_mb8_gc_on_noema_force0001_yaml_contract(monkeypatch):
    monkeypatch.setenv(
        "EMBODIED_PATH", str(TASK820_GB16_MB8_GC_ON_NOEMA_CONFIG_PATH.parents[1])
    )
    cfg = OmegaConf.load(TASK820_GB16_MB8_GC_ON_NOEMA_CONFIG_PATH)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    assert cfg.cluster.component_placement.actor == "5-6"
    assert cfg.actor.micro_batch_size == 8
    assert cfg.actor.global_batch_size == 16
    assert cfg.actor.model_weight_ema_decay is None
    assert cfg.actor.fsdp_config.gradient_checkpointing is True
    assert cfg.actor.model.openpi.tactile_loss_weight == pytest.approx(0.001)
    metadata = cfg.actor.fsdp_config.trainable_checkpoint_metadata
    assert metadata.future_force_loss_weight == pytest.approx(0.001)
    assert metadata.model_weight_ema_decay is None
    assert metadata.weight_variant == "online"
    assert metadata.training_config == TASK820_GB16_MB8_GC_ON_NOEMA_CONFIG_PATH.stem


def test_tacforce_tcn_encodes_one_token_with_nonzero_gradients():
    encoder = TactileTCNEncoder(
        input_dim=6,
        hidden_dim=12,
        output_dim=16,
        history_len=8,
        has_reference_frame=False,
        diff_from_reference=False,
    )
    force_history = torch.randn(2, 8, 6, requires_grad=True)

    token = encoder(force_history)
    token.square().mean().backward()

    assert token.shape == (2, 16)
    assert torch.isfinite(token).all()
    assert force_history.grad is not None
    assert torch.isfinite(force_history.grad).all()
    assert force_history.grad.abs().max().item() > 0
    assert all(parameter.grad is not None for parameter in encoder.parameters())


def test_tacforce_online_observation_processor_selects_force_only():
    model = SimpleNamespace(
        config=SimpleNamespace(config_name="pi05_lora_tacforce_tcn_real")
    )
    marker_motion = torch.ones(2, 9, 440, 2)
    tactile_image = torch.ones(2, 3, 224, 224)
    force = torch.zeros(2, 8, 6)
    processed = OpenPi0ForRLActionPrediction.obs_processor(
        model,
        {
            "main_images": torch.zeros(2, 3, 224, 224),
            "wrist_images": torch.zeros(2, 3, 224, 224),
            "states": torch.zeros(2, 7),
            "task_descriptions": ["test", "test"],
            "tactile_gripper_force": force,
            "tactile_marker_motion": marker_motion,
            "tactile_images": tactile_image,
        },
    )

    assert processed["tactile_gripper_force"] is force
    assert "tactile_marker_motion" not in processed
    assert "tactile_image" not in processed


def test_base_checkpoint_allows_only_random_tcn_parameters_to_be_missing():
    valid = SimpleNamespace(
        missing_keys=[
            "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight",
            "tactile_prefix_encoder.blocks.0.kernels.0.weight",
            "tactile_prefix_encoder.out_proj.bias",
        ],
        unexpected_keys=[],
    )
    _validate_checkpoint_load_result(valid, ["tactile_prefix_encoder."])

    invalid = SimpleNamespace(
        missing_keys=["action_in_proj.weight"], unexpected_keys=[]
    )
    with pytest.raises(RuntimeError, match="action_in_proj.weight"):
        _validate_checkpoint_load_result(invalid, ["tactile_prefix_encoder."])
