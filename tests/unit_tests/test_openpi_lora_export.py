# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from rlinf.utils.ckpt_convertor import export_openpi_lora_for_t2vla as exporter
from rlinf.utils.ckpt_convertor.export_openpi_lora_for_t2vla import _get_lora_modules


class _DummyOpenPI(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.paligemma = nn.Linear(1, 1)
        self.paligemma_with_expert.gemma_expert = nn.Module()
        self.paligemma_with_expert.gemma_expert.model = nn.Linear(1, 1)


def test_get_lora_modules_supports_dual_openpi_export_targets():
    model = _DummyOpenPI()

    modules = _get_lora_modules(model, "both")

    assert len(modules) == 2
    assert [item.adapter_dir_name for item in modules] == [
        "lora_adapter",
        "action_expert_lora_adapter",
    ]
    assert [item.target_name for item in modules] == [
        "paligemma",
        "action_expert",
    ]
    assert modules[0].module is model.paligemma_with_expert.paligemma
    assert modules[1].module is model.paligemma_with_expert.gemma_expert.model

    replacement_vlm = nn.Linear(1, 1)
    replacement_expert = nn.Linear(1, 1)
    modules[0].assign_module(replacement_vlm)
    modules[1].assign_module(replacement_expert)
    assert model.paligemma_with_expert.paligemma is replacement_vlm
    assert model.paligemma_with_expert.gemma_expert.model is replacement_expert


def test_canonical_peft_key_removes_nested_wrappers():
    assert (
        exporter._canonical_peft_key(  # noqa: SLF001
            "root.base_model.model.branch.base_model.model.proj.weight"
        )
        == "root.branch.proj.weight"
    )


def test_collect_extra_trainable_keys_requires_explicit_module_ownership():
    model = nn.Module()
    model.branch = nn.Module()
    model.branch.base_model = nn.Module()
    model.branch.base_model.model = nn.Module()
    model.branch.base_model.model.extra = nn.Linear(2, 2)
    model.register_parameter("lora_test", nn.Parameter(torch.ones(1)))
    model.value_head = nn.Linear(2, 1)

    key_map, prefixes = exporter._collect_extra_trainable_keys(  # noqa: SLF001
        model, ("branch.base_model.model.extra",)
    )

    assert prefixes == ("branch.extra",)
    assert key_map == {
        "branch.extra.weight": "branch.base_model.model.extra.weight",
        "branch.extra.bias": "branch.base_model.model.extra.bias",
    }

    model.unowned = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="unowned=.*unowned.weight"):
        exporter._collect_extra_trainable_keys(  # noqa: SLF001
            model, ("branch.base_model.model.extra",)
        )


def test_save_extra_trainable_uses_merged_canonical_values(tmp_path):
    model = nn.Module()
    model.branch = nn.Module()
    model.branch.extra = nn.Linear(2, 2)
    output = tmp_path / "extra_trainable.safetensors"

    exporter._save_extra_trainable(  # noqa: SLF001
        model,
        {
            "branch.extra.weight": "branch.base_model.model.extra.weight",
            "branch.extra.bias": "branch.base_model.model.extra.bias",
        },
        output,
        output_dtype=torch.bfloat16,
    )

    with safe_open(output, framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == {"branch.extra.weight", "branch.extra.bias"}
        assert {str(handle.get_slice(key).get_dtype()) for key in handle.keys()} == {
            "BF16"
        }


def test_bundle_staging_rejects_nonempty_target(tmp_path):
    target = tmp_path / "bundle"
    target.mkdir()
    (target / "user-data").write_text("preserve")

    with pytest.raises(FileExistsError, match="new or empty"):
        exporter._prepare_bundle_staging(target)  # noqa: SLF001

    assert (target / "user-data").read_text() == "preserve"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_resolve_norm_stats_uses_pirl_base_asset_contract(tmp_path):
    base = tmp_path / "base"
    norm_stats = base / "assets" / "replay_firm_tabero_xarm_gripper" / "norm_stats.json"
    norm_stats.parent.mkdir(parents=True)
    norm_stats.write_text('{"norm_stats": {}}')
    model_cfg = OmegaConf.create({"model_path": str(base)})
    checkpoint_meta = {
        "normalization_asset_id": "replay_firm_tabero_xarm_gripper",
        "base_norm_stats_sha256": _sha256(norm_stats),
    }

    source, asset_id = exporter._resolve_norm_stats_source(  # noqa: SLF001
        model_cfg, checkpoint_meta
    )

    assert source == norm_stats.resolve()
    assert asset_id == "replay_firm_tabero_xarm_gripper"


def test_resolve_norm_stats_rejects_pirl_hash_mismatch(tmp_path):
    base = tmp_path / "base"
    norm_stats = base / "assets" / "dataset" / "norm_stats.json"
    norm_stats.parent.mkdir(parents=True)
    norm_stats.write_text('{"norm_stats": {}}')

    with pytest.raises(ValueError, match="checkpoint contract"):
        exporter._resolve_norm_stats_source(  # noqa: SLF001
            OmegaConf.create({"model_path": str(base)}),
            {
                "normalization_asset_id": "dataset",
                "base_norm_stats_sha256": "0" * 64,
            },
        )


_ACTION_EXPERT_PROJECTIONS = (
    "mlp.down_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "self_attn.k_proj",
    "self_attn.o_proj",
    "self_attn.q_proj",
    "self_attn.v_proj",
)


def _action_expert_tensors(value: float) -> dict[str, torch.Tensor]:
    return {
        (
            "paligemma_with_expert.gemma_expert.model.layers."
            f"{layer}.{projection}.weight"
        ): torch.full((1,), value, dtype=torch.bfloat16)
        for layer in range(18)
        for projection in _ACTION_EXPERT_PROJECTIONS
    }


def _export_metadata_fixture(
    tmp_path: Path, *, changed_keys: set[str] | None = None
) -> dict:
    train_config = tmp_path / "isaaclab_pi0_task5_firm_8gpu_50step.yaml"
    train_config.write_text("actor: {}\n")
    checkpoint = tmp_path / "trainable_weights.pt"
    checkpoint_metadata = {
        "format": "trainable_weights",
        "method": "pirl",
        "task_id": 5,
        "training_config": train_config.stem,
        "step": 50,
        "global_step": 50,
        "target_global_step": 50,
        "is_final": True,
        "parameter_count": 1,
    }
    torch.save(
        {"model": {"lora.weight": torch.ones(1)}, "metadata": checkpoint_metadata},
        checkpoint,
    )
    base = tmp_path / "base"
    base.mkdir()
    base_tensors = _action_expert_tensors(1)
    save_file(base_tensors, base / "model.safetensors")
    model = tmp_path / "model.safetensors"
    changed_keys = set(base_tensors) if changed_keys is None else changed_keys
    model_tensors = {
        key: torch.full_like(value, 2) if key in changed_keys else value.clone()
        for key, value in base_tensors.items()
    }
    save_file(model_tensors, model)
    return {
        "train_config_path": str(train_config),
        "ckpt_path": str(checkpoint),
        "source_model_path": str(base),
        "checkpoint_meta": checkpoint_metadata,
        "model_path": model,
        "lora_target": "action_expert",
        "adapter_dirs": ["action_expert_lora_adapter"],
    }


def test_filtered_export_preserves_fixed_base_schema_dtypes(tmp_path):
    model = nn.Module()
    model.expert = nn.Linear(2, 2)
    base = tmp_path / "base"
    base.mkdir()
    save_file(
        {
            key: value.detach().to(torch.bfloat16)
            for key, value in model.state_dict().items()
        },
        base / "model.safetensors",
    )
    output = tmp_path / "model.safetensors"

    exporter._save_filtered_safetensors(model, str(output), base_model_path=str(base))

    with safe_open(output, framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == set(model.state_dict())
        assert {str(handle.get_slice(key).get_dtype()) for key in handle.keys()} == {
            "BF16"
        }


def test_tabero_sft_filtered_export_keeps_tcn_and_casts_all_bf16(tmp_path):
    model = nn.Module()
    model.base = nn.Linear(2, 2)
    model.tactile_prefix_encoder = nn.Linear(2, 3)
    base = tmp_path / "base"
    base.mkdir()
    save_file(
        {
            key: value.detach().to(torch.float32)
            for key, value in model.base.state_dict(prefix="base.").items()
        },
        base / "model.safetensors",
    )
    output = tmp_path / "model.safetensors"
    exporter._save_filtered_safetensors(
        model,
        str(output),
        base_model_path=str(base),
        allowed_extra_prefixes=("tactile_prefix_encoder.",),
        output_dtype=torch.bfloat16,
    )
    reference = tmp_path / "reference"
    reference.mkdir()
    save_file(
        {
            key: value.detach().to(torch.bfloat16)
            for key, value in model.state_dict().items()
        },
        reference / "model.safetensors",
    )

    exporter._validate_reference_schema(output, reference)
    with safe_open(output, framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == set(model.state_dict())
        assert {str(handle.get_slice(key).get_dtype()) for key in handle.keys()} == {
            "BF16"
        }


def test_tabero_sft_schema_is_base_plus_exact_tcn_and_not_fixed_total(tmp_path):
    base = tmp_path / "pi05_base"
    base.mkdir()
    base_tensors = {
        "base.weight": torch.ones((2, 2), dtype=torch.bfloat16),
        "base.bias": torch.ones(2, dtype=torch.bfloat16),
    }
    save_file(base_tensors, base / "model.safetensors")
    model = tmp_path / "model.safetensors"
    tcn_tensors = {
        f"tactile_prefix_encoder.layer_{index}.weight": torch.ones(
            1, dtype=torch.bfloat16
        )
        for index in range(16)
    }
    save_file({**base_tensors, **tcn_tensors}, model)

    counts = exporter._validate_tabero_sft_export_schema(
        model, base, method="sft_full_lora_tacfield"
    )

    assert counts == {
        "base_tensor_count": 2,
        "tactile_prefix_tensor_count": 16,
        "model_tensor_count": 18,
    }


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing_tcn", "extra_count=15"),
        ("wrong_prefix", "invalid_extra="),
        ("wrong_shape", "shape_mismatches="),
        ("wrong_dtype", "dtypes=.*F32"),
    ],
)
def test_tabero_sft_schema_rejects_invalid_pi05_export(tmp_path, mutation, match):
    base = tmp_path / "base"
    base.mkdir()
    base_tensors = {"base.weight": torch.ones(2, dtype=torch.bfloat16)}
    save_file(base_tensors, base / "model.safetensors")
    tensors = {
        **base_tensors,
        **{
            f"tactile_prefix_encoder.layer_{index}.weight": torch.ones(
                1, dtype=torch.bfloat16
            )
            for index in range(16)
        },
    }
    if mutation == "missing_tcn":
        tensors.pop("tactile_prefix_encoder.layer_15.weight")
    elif mutation == "wrong_prefix":
        tensors["unexpected.weight"] = tensors.pop(
            "tactile_prefix_encoder.layer_15.weight"
        )
    elif mutation == "wrong_shape":
        tensors["base.weight"] = torch.ones(3, dtype=torch.bfloat16)
    elif mutation == "wrong_dtype":
        tensors["base.weight"] = tensors["base.weight"].float()
    model = tmp_path / "model.safetensors"
    save_file(tensors, model)

    with pytest.raises(ValueError, match=match):
        exporter._validate_tabero_sft_export_schema(
            model, base, method="sft_full_lora_tacfield"
        )


def test_tabero_tacimg_schema_matches_fixed_base_exactly(tmp_path):
    base = tmp_path / "pi05_base"
    base.mkdir()
    base_tensors = {
        "base.weight": torch.ones((2, 2), dtype=torch.bfloat16),
        "base.bias": torch.ones(2, dtype=torch.bfloat16),
    }
    save_file(base_tensors, base / "model.safetensors")
    model = tmp_path / "model.safetensors"
    save_file(base_tensors, model)

    counts = exporter._validate_tabero_sft_export_schema(
        model, base, method="sft_full_lora_tacimg"
    )

    assert counts == {
        "base_tensor_count": 2,
        "tactile_prefix_tensor_count": 0,
        "model_tensor_count": 2,
    }


def test_tabero_tacimg_schema_rejects_extra_tensors(tmp_path):
    base = tmp_path / "pi05_base"
    base.mkdir()
    base_tensors = {"base.weight": torch.ones(2, dtype=torch.bfloat16)}
    save_file(base_tensors, base / "model.safetensors")
    model = tmp_path / "model.safetensors"
    save_file(
        {
            **base_tensors,
            "tactile_prefix_encoder.weight": torch.ones(
                1, dtype=torch.bfloat16
            ),
        },
        model,
    )

    with pytest.raises(ValueError, match="extra_count=1"):
        exporter._validate_tabero_sft_export_schema(
            model, base, method="sft_full_lora_tacimg"
        )


def test_trainable_sidecar_rejects_frozen_base_or_missing_tcn():
    model = nn.Module()
    model.base = nn.Linear(2, 2)
    model.tactile_prefix_encoder = nn.Linear(2, 2)
    for parameter in model.base.parameters():
        parameter.requires_grad = False
    valid = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    exporter._validate_trainable_checkpoint_keys(model, valid)

    missing = dict(valid)
    missing.pop("tactile_prefix_encoder.bias")
    with pytest.raises(RuntimeError, match="missing=.*tactile_prefix_encoder.bias"):
        exporter._validate_trainable_checkpoint_keys(model, missing)

    unexpected = dict(valid)
    unexpected["base.weight"] = model.base.weight.detach().clone()
    with pytest.raises(RuntimeError, match="unexpected=.*base.weight"):
        exporter._validate_trainable_checkpoint_keys(model, unexpected)


def test_export_metadata_carries_verified_fsdp_provenance_and_hashes(tmp_path):
    inputs = _export_metadata_fixture(tmp_path)

    metadata = exporter._build_export_metadata(**inputs)

    assert metadata["task_id"] == 5
    assert metadata["global_step"] == 50
    assert metadata["target_global_step"] == 50
    assert metadata["is_final"] is True
    assert metadata["base_model_sha256"] == _sha256(
        Path(inputs["source_model_path"]) / "model.safetensors"
    )
    assert metadata["source_ckpt_sha256"] == _sha256(Path(inputs["ckpt_path"]))
    assert metadata["source_ckpt_metadata"] == inputs["checkpoint_meta"]
    assert metadata["model_sha256"] == _sha256(inputs["model_path"])
    assert metadata["model_tensor_count"] == 126


@pytest.mark.parametrize(
    "method,dataset,config_stem",
    [
        (
            "sft_full_lora_tacfield",
            "datas/replay_firm_tabero",
            "replay_firm_tabero_pi05_tacfield_sft_2gpu",
        ),
        (
            "sft_full_lora_tacfield",
            "datas/replay_firm_tabero_xarm_gripper",
            "replay_firm_tabero_xarm_gripper_pi05_tacfield_sft_2gpu",
        ),
        (
            "sft_full_lora_tacfield",
            "datas/replay_firm_tabero_xarm_gripper",
            "replay_firm_tabero_xarm_gripper_pi0_tacfield_sft_2gpu",
        ),
        (
            "sft_full_lora_tacforce_tcn",
            "datas/replay_firm_tabero_xarm_gripper_repaired_v1",
            "replay_firm_tabero_xarm_gripper_repaired_v1_pi05_tacforce_tcn_sft_2gpu_gb16_30k",
        ),
        (
            "sft_full_lora_tacimg",
            "datas/realworld_replayed_task820_firm",
            "realworld_replayed_task820_firm_pi05_tacimg_sft_2gpu_gb32_mb16_gc_on_ema099_force0001_30k",
        ),
    ],
)
def test_export_metadata_accepts_replay_firm_pi05_precision_contract(
    tmp_path, method, dataset, config_stem
):
    train_config = tmp_path / f"{config_stem}.yaml"
    train_config.write_text("actor: {}\n")
    checkpoint = tmp_path / "trainable_weights.pt"
    checkpoint_metadata = {
        "format": "trainable_weights",
        "method": method,
        "dataset": dataset,
        "training_config": train_config.stem,
        "step": 20000,
        "global_step": 20000,
        "target_global_step": 20000,
        "is_final": True,
        "frozen_parameter_precision": "bf16",
        "trainable_parameter_precision": "fp32",
        "compute_precision": "bf16_amp",
        "export_precision": "bf16",
    }
    torch.save(
        {"model": {"weight": torch.ones(1)}, "metadata": checkpoint_metadata},
        checkpoint,
    )
    base = tmp_path / "base"
    base.mkdir()
    save_file(
        {"base.weight": torch.ones(1, dtype=torch.bfloat16)},
        base / "model.safetensors",
    )
    model = tmp_path / "model.safetensors"
    save_file(
        {
            "base.weight": torch.ones(1, dtype=torch.bfloat16),
            **{
                f"tactile_prefix_encoder.layer_{index}.weight": torch.ones(
                    1, dtype=torch.bfloat16
                )
                for index in range(16)
            },
        },
        model,
    )

    metadata = exporter._build_export_metadata(
        train_config_path=str(train_config),
        ckpt_path=str(checkpoint),
        source_model_path=str(base),
        checkpoint_meta=checkpoint_metadata,
        model_path=model,
        lora_target="both",
        adapter_dirs=["lora_adapter", "action_expert_lora_adapter"],
    )

    assert metadata["dataset"] == dataset
    assert metadata["base_model_tensor_count"] == 1
    assert metadata["extra_tensor_count"] == 16
    assert metadata["model_tensor_count"] == 17
    assert metadata["frozen_parameter_precision"] == "bf16"
    assert metadata["trainable_parameter_precision"] == "fp32"
    assert metadata["compute_precision"] == "bf16_amp"
    assert metadata["export_precision"] == "bf16"


@pytest.mark.parametrize(
    "changed_keys,actual_count",
    [
        (set(), 0),
        (
            {
                "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight"
            },
            1,
        ),
    ],
    ids=["unchanged", "partial"],
)
def test_export_metadata_rejects_incomplete_action_expert_delta(
    tmp_path, changed_keys, actual_count
):
    inputs = _export_metadata_fixture(tmp_path, changed_keys=changed_keys)

    with pytest.raises(
        ValueError,
        match=rf"expected_count=126, actual_count={actual_count}.*missing=",
    ):
        exporter._build_export_metadata(**inputs)


def test_export_metadata_rejects_checkpoint_without_formal_provenance(tmp_path):
    config = tmp_path / "task5.yaml"
    config.write_text("actor: {}\n")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    base = tmp_path / "base"
    base.mkdir()
    save_file({"weight": torch.ones(1)}, base / "model.safetensors")
    model = tmp_path / "model.safetensors"
    save_file({"weight": torch.ones(1)}, model)

    with pytest.raises(ValueError, match="provenance"):
        exporter._build_export_metadata(
            train_config_path=str(config),
            ckpt_path=str(checkpoint),
            source_model_path=str(base),
            checkpoint_meta={"format": "trainable_weights", "step": 50},
            model_path=model,
            lora_target="action_expert",
            adapter_dirs=[],
        )


def test_export_metadata_allows_explicit_task1_intermediate_checkpoint(tmp_path):
    inputs = _export_metadata_fixture(tmp_path)
    config = Path(inputs["train_config_path"])
    task1_config = config.with_name("task1_uniform_physics.yaml")
    task1_config.write_text(config.read_text())
    inputs["train_config_path"] = str(task1_config)
    inputs["checkpoint_meta"].update(
        task_id=1,
        training_config=task1_config.stem,
        step=30,
        global_step=30,
        target_global_step=100,
        is_final=False,
    )

    metadata = exporter._build_export_metadata(
        **inputs,
        allow_non_final=True,
    )

    assert metadata["task_id"] == 1
    assert metadata["global_step"] == 30
    assert metadata["target_global_step"] == 100
    assert metadata["is_final"] is False
    assert metadata["source_ckpt_metadata"]["is_final"] is False


def test_export_metadata_rejects_non_final_checkpoint_without_opt_in(tmp_path):
    inputs = _export_metadata_fixture(tmp_path)
    inputs["checkpoint_meta"].update(
        step=30,
        global_step=30,
        target_global_step=100,
        is_final=False,
    )

    with pytest.raises(ValueError, match="--allow_non_final"):
        exporter._build_export_metadata(**inputs)


def test_export_metadata_rejects_non_final_checkpoint_at_target(tmp_path):
    inputs = _export_metadata_fixture(tmp_path)
    inputs["checkpoint_meta"].update(
        step=50,
        global_step=50,
        target_global_step=50,
        is_final=False,
    )

    with pytest.raises(ValueError, match="less than target_global_step"):
        exporter._build_export_metadata(**inputs, allow_non_final=True)
