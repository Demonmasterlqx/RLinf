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

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from toolkits.checkpoint.audit_tabero_firm_all_task_eval import audit_evaluation
from toolkits.checkpoint.audit_tabero_firm_sft_checkpoint import audit_checkpoint
from toolkits.checkpoint.audit_tabero_firm_sft_checkpoint_dir import (
    audit_checkpoint_dir,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT / "examples/sft/config/tabero_firm_sft_2gpu_selective_siglip_20k.yaml"
)
SCRIPT_DIR = REPO_ROOT / "examples/sft"


def test_2gpu_20k_config_contract(monkeypatch, tmp_path):
    monkeypatch.setenv("EMBODIED_PATH", str(CONFIG_PATH.parents[1]))
    monkeypatch.setenv("TABERO_FIRM_2GPU_20K_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("TABERO_FIRM_2GPU_20K_RUN_NAME", "formal-test")
    monkeypatch.setenv(
        "TABERO_FIRM_2GPU_20K_CHECKPOINT_ROOT", str(tmp_path / "checkpoints")
    )
    monkeypatch.setenv("TABERO_FIRM_NORM_STATS", str(tmp_path / "norm_stats.json"))
    cfg = OmegaConf.load(CONFIG_PATH)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    assert cfg.cluster.component_placement.actor == "0-1"
    assert cfg.runner.max_steps == 20000
    assert cfg.runner.save_interval == 2000
    assert cfg.runner.logger.logger_backends == ["tensorboard", "wandb"]
    assert cfg.actor.micro_batch_size == 16
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
    assert cfg.actor.optim.total_training_steps == 20000
    assert cfg.actor.optim.lr_warmup_steps == 1000
    assert cfg.actor.optim.lr_decay_steps == 30000
    assert cfg.actor.fsdp_config.gradient_checkpointing is True
    assert cfg.actor.fsdp_config.mixed_precision.param_dtype is None
    assert cfg.actor.fsdp_config.amp_autocast.enabled is True
    assert cfg.actor.fsdp_config.amp_autocast.precision == "bf16"
    metadata = cfg.actor.fsdp_config.trainable_checkpoint_metadata
    assert metadata.training_config == CONFIG_PATH.stem
    assert metadata.target_global_step == 20000
    assert metadata.siglip_lora is False


def _checkpoint_metadata(step: int, is_final: bool, parameter_count: int) -> dict:
    return {
        "format": "trainable_weights",
        "method": "sft_full_lora_tacfield",
        "dataset": "datas/tabero_firm",
        "training_precision": "fp32",
        "frozen_parameter_precision": "bf16",
        "trainable_parameter_precision": "fp32",
        "compute_precision": "bf16_amp",
        "export_precision": "bf16",
        "paligemma_lora_rank": 16,
        "action_expert_lora_rank": 32,
        "freeze_non_lora": True,
        "trainable_siglip": True,
        "siglip_lora": False,
        "training_config": "tabero_firm_sft_2gpu_selective_siglip_20k",
        "target_global_step": 20000,
        "step": step,
        "global_step": step,
        "is_final": is_final,
        "parameter_count": parameter_count,
    }


def test_checkpoint_audit_accepts_all_selective_trainable_groups(tmp_path):
    names = (
        "paligemma_with_expert.paligemma.base_model.model.model.vision_tower.layer.weight",
        "paligemma_with_expert.paligemma.layer.lora_A.weight",
        "paligemma_with_expert.gemma_expert.model.layer.lora_A.weight",
        "state_proj.weight",
        "action_in_proj.weight",
        "action_time_mlp_in.weight",
        "action_time_mlp_out.weight",
        "action_out_proj.weight",
        "tactile_prefix_encoder.weight",
    )
    checkpoint_path = tmp_path / "trainable_weights.pt"
    torch.save(
        {
            "model": {name: torch.ones(1, dtype=torch.float32) for name in names},
            "metadata": _checkpoint_metadata(20000, True, len(names)),
        },
        checkpoint_path,
    )

    result = audit_checkpoint(checkpoint_path, 20000, True)

    assert result["tensor_count"] == len(names)
    assert result["dtypes"] == ["torch.float32"]
    assert all(count > 0 for count in result["coverage"].values())


def test_checkpoint_audit_rejects_nonfinal_step_20000(tmp_path):
    checkpoint_path = tmp_path / "trainable_weights.pt"
    names = (
        "paligemma_with_expert.paligemma.base_model.model.model.vision_tower.x",
        "paligemma_with_expert.paligemma.x.lora_A.weight",
        "paligemma_with_expert.gemma_expert.model.x.lora_A.weight",
        "state_proj.x",
        "action_in_proj.x",
        "action_time_mlp_in.x",
        "action_time_mlp_out.x",
        "action_out_proj.x",
        "tactile_prefix_encoder.x",
    )
    torch.save(
        {
            "model": {name: torch.ones(1) for name in names},
            "metadata": _checkpoint_metadata(20000, False, len(names)),
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="is_final"):
        audit_checkpoint(checkpoint_path, 20000, True)


def test_checkpoint_dir_audit_requires_dcp_optimizer_scheduler_and_data(
    monkeypatch, tmp_path
):
    step_dir = tmp_path / "global_step_2000"
    dcp_dir = step_dir / "actor/dcp_checkpoint"
    sidecar_path = step_dir / "actor/model_state_dict/trainable_weights.pt"
    dcp_dir.mkdir(parents=True)
    sidecar_path.parent.mkdir(parents=True)
    (dcp_dir / ".metadata").write_bytes(b"metadata")
    (dcp_dir / "__0_0.distcp").write_bytes(b"rank0")
    (dcp_dir / "__1_0.distcp").write_bytes(b"rank1")
    (step_dir / "actor/data_state.json").write_text(
        json.dumps({"data_epoch": 1, "data_iter_offset": 10})
    )
    names = (
        "paligemma_with_expert.paligemma.base_model.model.model.vision_tower.x",
        "paligemma_with_expert.paligemma.x.lora_A.weight",
        "paligemma_with_expert.gemma_expert.model.x.lora_A.weight",
        "state_proj.x",
        "action_in_proj.x",
        "action_time_mlp_in.x",
        "action_time_mlp_out.x",
        "action_out_proj.x",
        "tactile_prefix_encoder.x",
    )
    torch.save(
        {
            "model": {name: torch.ones(1) for name in names},
            "metadata": _checkpoint_metadata(2000, False, len(names)),
        },
        sidecar_path,
    )
    monkeypatch.setattr(
        "toolkits.checkpoint.audit_tabero_firm_sft_checkpoint_dir._read_dcp_state_keys",
        lambda _path: {
            "fsdp_checkpoint.fsdp_version",
            "fsdp_checkpoint.model.weight",
            "fsdp_checkpoint.optimizers.state.weight.exp_avg",
            "fsdp_checkpoint.lr_schedulers.0.last_epoch",
            "fsdp_checkpoint.rng.cpu",
        },
    )

    result = audit_checkpoint_dir(step_dir, 2000, False)

    assert result["status"] == "complete_and_audited"
    assert result["dcp"]["shard_count"] == 2
    assert all(count == 1 for count in result["dcp"]["coverage"].values())
    assert result["data_state"]["data_epoch"] == 1


def test_all_task_eval_audit_cross_checks_450_episodes(tmp_path):
    results = {}
    txt_parts = []
    client_parts = []
    for task_id in (0, 1, 2, 3, 5, 6, 7, 8, 9):
        episodes = [
            {"experiment_index": index, "success": index < task_id + 10}
            for index in range(50)
        ]
        successes = task_id + 10
        results[f"libero_object_task{task_id}"] = {
            "status": "completed",
            "successful_experiments": successes,
            "total_experiments": 50,
            "success_rate": successes / 50 * 100,
            "metrics_status": "complete",
            "episodes": episodes,
            "execution_time": 1.0,
            "step_statistics": {},
        }
        txt_parts.append(f"Task {task_id} (task): 0.00% ({successes}/50)")
        client_parts.extend(
            f"[{index}/50] Starting experiment" for index in range(1, 51)
        )
        client_parts.append(f"TASK COMPLETED: libero_object - Task {task_id}")
    client_parts.append("Progress: 9/9 tasks completed")
    json_path = tmp_path / "result.json"
    txt_path = tmp_path / "result.txt"
    client_path = tmp_path / "client.log"
    json_path.write_text(json.dumps({"results": results}))
    txt_path.write_text("\n".join(txt_parts))
    client_path.write_text("\n".join(client_parts))

    result = audit_evaluation(json_path, txt_path, client_path)

    assert result["status"] == "completed_and_cross_checked"
    assert result["total_episodes"] == 450
    assert len(result["tasks"]) == 9


def test_pipeline_scripts_encode_retry_and_gpu_scope():
    launcher = (
        SCRIPT_DIR / "run_tabero_firm_sft_2gpu_selective_siglip_20k.sh"
    ).read_text()
    supervisor = (SCRIPT_DIR / "supervise_tabero_firm_sft_2gpu_20k.sh").read_text()
    evaluator = (SCRIPT_DIR / "evaluate_tabero_firm_sft_2gpu_all_tasks.sh").read_text()
    checkpoint_watcher = (
        SCRIPT_DIR / "audit_tabero_firm_sft_2gpu_checkpoints_until_done.sh"
    ).read_text()

    assert 'PHYSICAL_GPU_CSV="0,1"' in launcher
    assert "runner.save_interval=2" in launcher
    assert "--expected-step 20000" in launcher
    assert "recovering_once" in supervisor
    assert "train_attempt2.log" in supervisor
    assert "for attempt in 1 2" in evaluator
    assert "--task-ids 0 1 2 3 5 6 7 8 9" in evaluator
    assert "--send-dsrl-raw-image" not in evaluator
    assert "EXPECTED_STEPS=(2000 4000" in checkpoint_watcher
    assert "audit_tabero_firm_sft_checkpoint_dir.py" in checkpoint_watcher
    assert "LAST_SIZE" in checkpoint_watcher
