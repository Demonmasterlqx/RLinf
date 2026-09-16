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

import numpy as np
import pytest
from omegaconf import OmegaConf
from openpi.models import model as _model

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi.policies.tabero_policy import (
    TaberoTacFieldInputs,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples/sft/config"


@pytest.mark.parametrize(
    ("config_name", "dataset_name", "placement", "asset_id", "variant", "openpi_name"),
    [
        (
            "tabero_firmly_tightly_no_adverb_pi05_tacfield_no_state_sft_2gpu_b64_gc_on_30k.yaml",
            "tabero_firmly_tightly_no_adverb",
            "6-7",
            "pi05_horizon50_tacfield_tabero_firmly_tightly_no_adverb",
            "firmly_tightly_no_adverb",
            "pi05_lora_tacfield_tabero_firmly_tightly_no_adverb_no_state",
        ),
        (
            "tabero_pi05_tacfield_no_state_sft_2gpu_b64_gc_on_30k.yaml",
            "tabero",
            "0-1",
            "pi05_horizon50_tacfield_tabero",
            "tabero",
            "pi05_lora_tacfield_tabero_no_state",
        ),
    ],
)
def test_tabero_pi05_tacfield_no_state_sft_contract(
    monkeypatch,
    config_name,
    dataset_name,
    placement,
    asset_id,
    variant,
    openpi_name,
):
    monkeypatch.setenv("EMBODIED_PATH", str(CONFIG_DIR.parent))
    cfg = OmegaConf.load(CONFIG_DIR / config_name)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    assert cfg.cluster.num_nodes == 1
    assert cfg.cluster.component_placement.actor == placement
    assert cfg.runner.max_epochs == -1
    assert cfg.runner.max_steps == 30_000
    assert cfg.runner.save_interval == 1_000
    assert cfg.runner.val_check_interval == -1
    assert cfg.runner.logger.project_name == "tabero-rlinf"
    assert cfg.runner.logger.logger_backends == ["tensorboard", "wandb"]
    assert cfg.data.train_data_paths[0].dataset_path.endswith(
        f"datasets/{dataset_name}"
    )
    assert cfg.data.train_data_paths[0].weight == pytest.approx(1.0)

    assert cfg.actor.training_backend == "fsdp"
    assert cfg.actor.micro_batch_size == 32
    assert cfg.actor.global_batch_size == 64
    assert cfg.actor.global_batch_size // (cfg.actor.micro_batch_size * 2) == 1
    assert cfg.actor.seed == 42
    assert cfg.actor.model_weight_ema_decay == pytest.approx(0.99)

    model = cfg.actor.model
    assert model.model_type == "openpi"
    assert model.model_path.endswith("models/pi05_base_pytorch")
    assert model.is_lora is True
    assert model.paligemma_lora_rank == 16
    assert model.action_expert_lora_rank == 32
    assert model.lora_target == "both"
    assert model.freeze_non_lora is True
    assert model.export_norm_asset_id == asset_id
    assert model.num_action_chunks == 50
    assert model.action_dim == 13
    assert "tactile_prefix_encoder" in model.extra_trainable_modules
    assert model.checkpoint_load_allowed_missing_prefixes == ["tactile_prefix_encoder."]

    openpi = model.openpi
    assert openpi.config_name == openpi_name
    assert openpi.pi05 is True
    assert openpi.discrete_state_input is False
    assert openpi.num_images_in_input == 2
    assert openpi.action_horizon == 50
    assert openpi.action_chunk == 50
    assert openpi.effective_action_dim == 13
    assert openpi.tactile_prefix_dim_in == 9 * 198 * 2
    assert openpi.tactile_prefix_history == 8
    assert openpi.tactile_prefix_encoder_type == "tcn"
    assert openpi.tactile_streams == ["tactile_prefix"]

    assert cfg.actor.optim.total_training_steps == 30_000
    assert cfg.actor.optim.lr == pytest.approx(2.5e-5)
    assert cfg.actor.optim.lr_warmup_steps == 10_000
    assert cfg.actor.optim.lr_decay_steps == 1_000_000
    assert cfg.actor.optim.min_lr == pytest.approx(2.5e-6)

    fsdp = cfg.actor.fsdp_config
    assert fsdp.sharding_strategy == "no_shard"
    assert fsdp.gradient_checkpointing is True
    assert fsdp.amp_autocast.enabled is True
    assert fsdp.amp_autocast.precision == "bf16"
    assert fsdp.save_trainable_model_weights is True
    assert fsdp.checkpoint_format == "dcp"

    metadata = fsdp.trainable_checkpoint_metadata
    assert metadata.method == "sft_full_lora_tacfield"
    assert metadata.dataset == f"datasets/{dataset_name}"
    assert metadata.openpi_config_name == openpi_name
    assert metadata.deployment_config_name == openpi_name
    assert metadata.discrete_state_input is False
    assert metadata.tactile_input == "tactile_marker_motion"
    assert metadata.tactile_prefix_dim_in == 9 * 198 * 2
    assert metadata.num_images_in_input == 2
    assert metadata.excluded_tactile_inputs == [
        "tactile_image",
        "tactile_gripper_force",
    ]
    assert metadata.target_global_step == 30_000
    assert metadata.dataset_variant == variant


@pytest.mark.parametrize(
    "config_name",
    [
        "pi05_lora_tacfield_tabero_no_state",
        "pi05_lora_tacfield_tabero_firmly_tightly_no_adverb_no_state",
    ],
)
def test_tabero_pi05_tacfield_no_state_openpi_contract(config_name):
    config = get_openpi_config(config_name)
    assert config.model.pi05 is True
    assert config.model.action_horizon == 50
    assert config.model.discrete_state_input is False
    assert config.model.num_images_in_input == 2
    assert config.model.tactile_prefix_dim_in == 9 * 198 * 2
    assert config.model.tactile_prefix_history == 8
    assert config.model.tactile_streams == ("tactile_prefix",)


def test_tabero_tacfield_transform_matches_198_marker_schema():
    transformed = TaberoTacFieldInputs(model_type=_model.ModelType.PI05)(
        {
            "image": np.zeros((224, 224, 3), dtype=np.uint8),
            "wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "state": np.zeros(7, dtype=np.float32),
            "actions": np.zeros((50, 13), dtype=np.float32),
            "tactile_marker_motion": np.zeros((9, 198, 2), dtype=np.float32),
            "prompt": "test",
        }
    )
    assert transformed["tactile_prefix"].shape == (9, 396)
    assert set(transformed["image"]) == {
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    }
