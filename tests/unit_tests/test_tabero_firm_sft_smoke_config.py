from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples" / "sft" / "config"
WORKSPACE = Path("/data/home/sim6g/code/tabero")


def test_tabero_firm_full_lora_fp32_smoke_config(monkeypatch, tmp_path):
    norm_stats = tmp_path / "norm_stats.json"
    norm_stats.write_text("{}\n")
    monkeypatch.setenv("TABERO_FIRM_NORM_STATS", str(norm_stats))
    monkeypatch.setenv("TABERO_FIRM_SMOKE_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("EMBODIED_PATH", str(CONFIG_DIR.parent))
    with initialize_config_dir(version_base="1.1", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="tabero_firm_sft_full_lora_tacfield_fp32_smoke")
    resolved = OmegaConf.to_container(cfg, resolve=True)

    assert cfg.cluster.component_placement.actor == "all"
    assert cfg.runner.max_steps == 2
    assert cfg.runner.save_interval == 1
    assert cfg.runner.logger.logger_backends == ["tensorboard", "wandb"]
    assert cfg.runner.logger.log_path == str(tmp_path / "run")
    assert cfg.data.train_data_paths[0].dataset_path == str(
        WORKSPACE / "datas" / "tabero_firm"
    )
    assert cfg.actor.micro_batch_size == 1
    assert cfg.actor.global_batch_size == 8
    assert cfg.actor.model.precision == "fp32"
    assert cfg.actor.model.lora_rank == 32
    assert cfg.actor.model.lora_target == "both"
    assert cfg.actor.model.freeze_non_lora is True
    assert cfg.actor.model.extra_trainable_modules == ["tactile_prefix_encoder"]
    assert cfg.actor.model.openpi.config_name == "pi0_lora_tacfield_tabero"
    assert cfg.actor.model.openpi.action_chunk == 50
    assert cfg.actor.model.openpi.action_env_dim == 13
    assert resolved["actor"]["model"]["openpi_data"]["norm_stats_path"] == str(
        norm_stats
    )
    assert cfg.actor.fsdp_config.sharding_strategy == "full_shard"
    assert cfg.actor.fsdp_config.use_orig_params is False
    assert cfg.actor.fsdp_config.mixed_precision.param_dtype == "fp32"
    assert cfg.actor.fsdp_config.mixed_precision.reduce_dtype == "fp32"
    assert cfg.actor.fsdp_config.mixed_precision.buffer_dtype == "fp32"
    assert cfg.actor.fsdp_config.amp_autocast.enabled is False
    assert cfg.actor.fsdp_config.grad_scaler.enabled is False
    assert cfg.actor.fsdp_config.checkpoint_format == "none"
    metadata = cfg.actor.fsdp_config.trainable_checkpoint_metadata
    assert metadata.method == "sft_full_lora_tacfield"
    assert metadata.dataset == "datas/tabero_firm"
    assert metadata.training_precision == "fp32"
    assert metadata.target_global_step == 2
