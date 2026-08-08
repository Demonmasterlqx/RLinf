from pathlib import Path

import pytest
from omegaconf import OmegaConf

CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/sft/config/tabero_firm_sft_2gpu_selective_siglip_capacity.yaml"
)


def test_tabero_firm_2gpu_capacity_config_contract(monkeypatch, tmp_path):
    monkeypatch.setenv("EMBODIED_PATH", str(CONFIG_PATH.parents[1]))
    monkeypatch.setenv("TABERO_FIRM_2GPU_CAPACITY_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("TABERO_FIRM_2GPU_CAPACITY_RUN_NAME", "capacity-test")
    monkeypatch.setenv(
        "TABERO_FIRM_2GPU_CAPACITY_CHECKPOINT_ROOT",
        str(tmp_path / "checkpoints"),
    )
    monkeypatch.setenv("TABERO_FIRM_NORM_STATS", str(tmp_path / "norm_stats.json"))
    cfg = OmegaConf.load(CONFIG_PATH)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    assert cfg.cluster.component_placement.actor == "0-1"
    assert cfg.runner.max_steps == 1
    assert cfg.runner.save_interval == -1
    assert cfg.runner.logger.logger_backends == ["tensorboard", "wandb"]
    assert cfg.actor.micro_batch_size == 1
    assert cfg.actor.global_batch_size == 32
    assert cfg.actor.model.paligemma_lora_rank == 16
    assert cfg.actor.model.action_expert_lora_rank == 32
    assert cfg.actor.model.paligemma_lora_exclude_modules == ".*vision_tower.*"
    assert cfg.actor.model.freeze_non_lora is True
    assert cfg.actor.model.frozen_parameter_precision == "bf16"
    assert cfg.actor.model.trainable_parameter_precision == "fp32"
    assert cfg.actor.model.openpi.tactile_loss_weight == pytest.approx(0.01)
    assert cfg.actor.model.extra_trainable_modules == [
        "paligemma_with_expert.paligemma.base_model.model.model.vision_tower",
        "state_proj",
        "action_in_proj",
        "action_time_mlp_in",
        "action_time_mlp_out",
        "action_out_proj",
        "tactile_prefix_encoder",
    ]
    assert cfg.actor.fsdp_config.gradient_checkpointing is True
    assert cfg.actor.fsdp_config.mixed_precision.param_dtype is None
    assert cfg.actor.fsdp_config.mixed_precision.reduce_dtype is None
    assert cfg.actor.fsdp_config.mixed_precision.buffer_dtype is None
    assert cfg.actor.fsdp_config.amp_autocast.enabled is True
    assert cfg.actor.fsdp_config.grad_scaler.enabled is False
    assert cfg.actor.fsdp_config.trainable_checkpoint_metadata.siglip_lora is False
