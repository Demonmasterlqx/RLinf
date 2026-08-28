from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from openpi.models import model as _model

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi.policies.tabero_policy import TaberoTacImgInputs
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker

CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/sft/config/"
    "realworld_replayed_task820_firm_pi05_tacimg_sft_2gpu_"
    "gb32_mb16_gc_on_ema099_force0001_30k.yaml"
)


def test_pi05_tacimg_registry_and_transform_contract():
    config = get_openpi_config(
        "pi05_lora_tacimg_realworld_replayed_task820_force"
    )

    assert config.model.model_type == _model.ModelType.PI05
    assert config.model.action_horizon == 50
    assert config.model.action_chunk == 50
    assert config.model.num_images_in_input == 3
    assert config.model.effective_action_dim == 13
    assert config.model.tactile_streams == ()
    assert config.model.tactile_loss_weight == pytest.approx(0.001)
    assert config.data.repo_id == "realworld_replayed_task820_firm"
    assert config.data.assets.asset_id == "pi05_horizon50_tacimg_task820_firm"
    assert config.ema_decay == pytest.approx(0.99)

    transformed = TaberoTacImgInputs(model_type=config.model.model_type)(
        {
            "image": np.zeros((224, 224, 3), dtype=np.uint8),
            "wrist_image": np.ones((224, 224, 3), dtype=np.uint8),
            "tactile_image": np.full((224, 224, 3), 2, dtype=np.uint8),
            "tactile_gripper_force": np.ones((8, 6), dtype=np.float32),
            "tactile_marker_motion": np.ones((9, 440, 2), dtype=np.float32),
            "state": np.zeros(7, dtype=np.float32),
            "actions": np.zeros((50, 13), dtype=np.float32),
            "prompt": "test",
        }
    )
    assert tuple(transformed["image"].keys()) == (
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    )
    assert all(bool(value) for value in transformed["image_mask"].values())
    assert transformed["actions"].shape == (50, 13)
    assert "tactile_prefix" not in transformed
    assert "tactile_gripper_force" not in transformed
    assert "tactile_marker_motion" not in transformed


def test_pi05_tacimg_yaml_contract(monkeypatch):
    monkeypatch.setenv("EMBODIED_PATH", str(CONFIG_PATH.parents[1]))
    cfg = OmegaConf.load(CONFIG_PATH)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    assert cfg.cluster.component_placement.actor == "5-6"
    assert cfg.runner.max_steps == 30_000
    assert cfg.runner.save_interval == 1_000
    assert cfg.runner.logger.logger_backends == ["tensorboard", "wandb"]
    assert cfg.actor.micro_batch_size == 16
    assert cfg.actor.global_batch_size == 32
    assert cfg.actor.global_batch_size // (cfg.actor.micro_batch_size * 2) == 1
    assert cfg.actor.model_weight_ema_decay == pytest.approx(0.99)
    assert cfg.actor.model.openpi.num_images_in_input == 3
    assert cfg.actor.model.openpi.tactile_streams == []
    assert cfg.actor.model.openpi.tactile_loss_weight == pytest.approx(0.001)
    assert cfg.actor.fsdp_config.gradient_checkpointing is True
    assert "tactile_prefix_encoder" not in cfg.actor.model.extra_trainable_modules
    assert cfg.actor.fsdp_config.trainable_checkpoint_metadata.method == (
        "sft_full_lora_tacimg"
    )


def test_vla_sft_releases_cuda_cache_only_before_first_training_forward(monkeypatch):
    worker = FSDPVlaSftWorker.__new__(FSDPVlaSftWorker)
    worker.amp_context = nullcontext()
    worker.model = lambda **_kwargs: torch.tensor(2.0)
    worker._logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    empty_cache_calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empty_cache_calls.append(1))

    first_loss, first_metrics = worker.get_train_model_output({})
    second_loss, second_metrics = worker.get_train_model_output({})

    assert empty_cache_calls == [1]
    assert first_loss.item() == pytest.approx(2.0)
    assert second_loss.item() == pytest.approx(2.0)
    assert first_metrics == {"loss": pytest.approx(2.0)}
    assert second_metrics == {"loss": pytest.approx(2.0)}
