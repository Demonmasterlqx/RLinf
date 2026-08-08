from pathlib import Path

import pytest
from omegaconf import OmegaConf

CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/sft/config/tabero_firm_sft_dual_rank_t2_precision_20k.yaml"
)


def test_tabero_firm_20k_config_contract(monkeypatch, tmp_path):
    monkeypatch.setenv("EMBODIED_PATH", str(CONFIG_PATH.parents[1]))
    monkeypatch.setenv("TABERO_FIRM_20K_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("TABERO_FIRM_20K_RUN_NAME", "formal-test")
    monkeypatch.setenv("TABERO_FIRM_20K_CHECKPOINT_ROOT", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("TABERO_FIRM_NORM_STATS", str(tmp_path / "norm_stats.json"))
    cfg = OmegaConf.load(CONFIG_PATH)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    assert cfg.cluster.component_placement.actor == "0-1,3-7"
    assert cfg.runner.max_steps == 20000
    assert cfg.runner.save_interval == 5000
    assert cfg.runner.logger.logger_backends == ["tensorboard", "wandb"]
    assert cfg.actor.micro_batch_size == 1
    assert cfg.actor.global_batch_size == 28
    assert cfg.actor.model.precision == "fp32"
    assert cfg.actor.model.lora_target == "both"
    assert cfg.actor.model.paligemma_lora_rank == 16
    assert cfg.actor.model.action_expert_lora_rank == 32
    assert cfg.actor.model.freeze_non_lora is False
    assert cfg.actor.model.openpi.tactile_loss_weight == pytest.approx(0.01)
    assert cfg.actor.optim.total_training_steps == 20000
    assert cfg.actor.optim.lr_warmup_steps == 1000
    assert cfg.actor.optim.lr_decay_steps == 30000
    assert cfg.actor.fsdp_config.mixed_precision.param_dtype == "fp32"
    assert cfg.actor.fsdp_config.mixed_precision.reduce_dtype == "fp32"
    assert cfg.actor.fsdp_config.mixed_precision.buffer_dtype == "fp32"
    assert cfg.actor.fsdp_config.amp_autocast.enabled is True
    assert cfg.actor.fsdp_config.grad_scaler.enabled is False
    metadata = cfg.actor.fsdp_config.trainable_checkpoint_metadata
    assert metadata.training_precision == "fp32"
    assert metadata.compute_precision == "bf16_amp"
    assert metadata.target_global_step == 20000
