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

"""Synthetic TacCoRL export coverage; no local models, datasets, or GPUs."""

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file
from torch import nn

from rlinf.utils.ckpt_convertor import export_openpi_lora_for_t2vla as exporter

_METHODS = ("sft_full_lora_taccorl", "sft_full_lora_taccorl_cotrain")
_PREFIXES = ("marker_token_encoder.", "tactile_cross_attn.")


def _synthetic_model(depth: int = 1, cross_attn: bool = True) -> nn.Module:
    model = nn.Module()
    model.base = nn.Linear(2, 2).requires_grad_(False)
    model.marker_token_encoder = nn.Sequential(*(nn.Linear(2, 2) for _ in range(depth)))
    model.tactile_cross_attn = nn.Linear(2, 2) if cross_attn else None
    return model


def _write_schema(tmp_path: Path, model: nn.Module) -> tuple[Path, Path]:
    base_path = tmp_path / "base.safetensors"
    save_file(model.base.state_dict(prefix="base."), base_path)
    model_path = tmp_path / "model.safetensors"
    exporter._save_filtered_safetensors(
        model,
        str(model_path),
        base_model_path=str(base_path),
        allowed_extra_prefixes=_PREFIXES,
        output_dtype=torch.bfloat16,
    )
    return base_path, model_path


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("depth,cross_attn", [(1, True), (3, True), (2, False)])
def test_taccorl_schema_derives_extra_keys_from_model(
    tmp_path, method, depth, cross_attn
):
    model = _synthetic_model(depth, cross_attn)
    base_path, model_path = _write_schema(tmp_path, model)
    counts = exporter._validate_tabero_sft_export_schema(
        model_path,
        base_path,
        method=method,
        source_model=model,
        openpi_cfg={
            "tactile_fusion": "taccorl",
            "tactile_prefix_cross_attn": cross_attn,
        },
    )

    expected_keys = {key for key in model.state_dict() if key.startswith(_PREFIXES)}
    assert method in exporter.TABERO_SFT_METHODS
    assert counts["tactile_prefix_tensor_count"] == len(expected_keys)
    assert counts["model_tensor_count"] == len(model.state_dict())
    assert set(load_file(model_path)) == set(model.state_dict())
    assert {value.dtype for value in load_file(model_path).values()} == {torch.bfloat16}


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize(
    "mutation,match",
    [
        ("stray", "invalid_extra=.*unowned"),
        ("invented_marker_key", "invalid_extra=.*marker_token_encoder.invented"),
        ("missing_cross_attn", "missing_extra=.*tactile_cross_attn"),
        ("missing_base", "missing=.*base.weight"),
        ("wrong_shape", "shape_mismatches=.*marker_token_encoder"),
        ("wrong_dtype", "dtypes=.*F32"),
    ],
)
def test_taccorl_schema_rejects_invalid_extra_state(tmp_path, method, mutation, match):
    model = _synthetic_model()
    base_path, model_path = _write_schema(tmp_path, model)
    tensors = load_file(model_path)
    if mutation == "stray":
        tensors["unowned.weight"] = tensors.pop("tactile_cross_attn.weight")
    elif mutation == "invented_marker_key":
        tensors["marker_token_encoder.invented"] = tensors.pop(
            "marker_token_encoder.0.bias"
        )
    elif mutation == "missing_cross_attn":
        tensors.pop("tactile_cross_attn.bias")
    elif mutation == "missing_base":
        tensors.pop("base.weight")
    elif mutation == "wrong_shape":
        tensors["marker_token_encoder.0.weight"] = torch.ones(3, dtype=torch.bfloat16)
    elif mutation == "wrong_dtype":
        tensors["tactile_cross_attn.weight"] = tensors[
            "tactile_cross_attn.weight"
        ].float()
    save_file(tensors, model_path)

    with pytest.raises(ValueError, match=match):
        exporter._validate_tabero_sft_export_schema(
            model_path,
            base_path,
            method=method,
            source_model=model,
            openpi_cfg={"tactile_fusion": "taccorl"},
        )


@pytest.mark.parametrize("method", _METHODS)
def test_taccorl_requires_fusion_config_and_source_model(tmp_path, method):
    model = _synthetic_model()
    base_path, model_path = _write_schema(tmp_path, model)
    with pytest.raises(ValueError, match="require tactile_fusion"):
        exporter._validate_tabero_sft_export_schema(
            model_path, base_path, method=method, source_model=model
        )
    with pytest.raises(ValueError, match="requires the source model"):
        exporter._validate_tabero_sft_export_schema(
            model_path,
            base_path,
            method=method,
            openpi_cfg={"tactile_fusion": "taccorl"},
        )


@pytest.mark.parametrize(
    "method",
    [
        "sft_full_lora_tacfield",
        "sft_full_lora_tacforce_tcn",
        "sft_full_lora_tacimg",
        "sft_full_lora_vision",
        "pirl",
    ],
)
def test_non_taccorl_methods_reject_taccorl_models(tmp_path, method):
    model = _synthetic_model()
    base_path, model_path = _write_schema(tmp_path, model)
    with pytest.raises(ValueError, match="TacCoRL models require a TacCoRL SFT method"):
        exporter._validate_tabero_sft_export_schema(
            model_path,
            base_path,
            method=method,
            source_model=model,
            openpi_cfg={"tactile_fusion": "taccorl"},
        )


def test_taccorl_filtered_export_rejects_stray_module(tmp_path):
    model = _synthetic_model()
    model.unowned = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="unexpected=.*unowned"):
        _write_schema(tmp_path, model)


class _SyntheticAdapter(nn.Module):
    """Minimal adapter protocol for testing export orchestration without PEFT."""

    def __init__(self):
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.model = nn.Linear(2, 2).requires_grad_(False)
        self.lora_A = nn.Parameter(torch.ones(1))

    def merge_and_unload(self):
        return self.base_model.model

    def save_pretrained(self, path, *, safe_serialization):
        assert safe_serialization
        target = Path(path)
        target.mkdir(parents=True)
        (target / "adapter_config.json").write_text(
            json.dumps({"r": 1, "lora_alpha": 1, "target_modules": ["proj"]})
        )
        save_file(
            {"lora_A": self.lora_A.detach()}, target / "adapter_model.safetensors"
        )


@pytest.mark.parametrize("method", _METHODS)
def test_taccorl_sidecar_exports_bundle_with_serving_name_and_both_modules(
    tmp_path, monkeypatch, method
):
    model = _synthetic_model()
    model.paligemma_with_expert = nn.Module()
    model.paligemma_with_expert.paligemma = _SyntheticAdapter()
    model.paligemma_with_expert.gemma_expert = nn.Module()
    model.paligemma_with_expert.gemma_expert.model = _SyntheticAdapter()
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    base_tensors = {
        exporter._canonical_peft_key(key): value
        for key, value in model.state_dict().items()
        if "lora_" not in key and not key.startswith(_PREFIXES)
    }
    save_file(base_tensors, base_dir / "model.safetensors")
    norm_path = tmp_path / "norm_stats.json"
    norm_path.write_text('{"norm_stats": {}}')
    train_path = tmp_path / "synthetic_taccorl.yaml"
    model_cfg = {
        "is_lora": True,
        "model_path": str(base_dir),
        "lora_target": "both",
        "lora_rank": 1,
        "extra_trainable_modules": ["marker_token_encoder", "tactile_cross_attn"],
        "export_norm_asset_id": "synthetic_cotrain",
        "openpi_data": {"norm_stats_path": str(norm_path)},
        "openpi": {
            "config_name": "rlinf_training_config",
            "tactile_fusion": "taccorl",
            "tactile_prefix_num_tokens": 4,
            "tactile_suffix_num_tokens": 6,
            "tactile_prefix_cross_attn": True,
            "tactile_cross_attn_heads": 2,
            "image_noise_std": 0.03,
        },
    }
    OmegaConf.save(
        OmegaConf.create(
            {
                "actor": {
                    "model": model_cfg,
                    "fsdp_config": {
                        "trainable_checkpoint_metadata": {
                            "deployment_config_name": "t2vla_serving_config"
                        }
                    },
                }
            }
        ),
        train_path,
    )
    metadata = {
        "format": "trainable_weights",
        "method": method,
        "dataset": "datasets/realworld_replay_task820_firm_mixed",
        "training_config": train_path.stem,
        "deployment_config_name": "old_training_name",
        "step": 150,
        "global_step": 150,
        "target_global_step": 3000,
        "is_final": False,
        "frozen_parameter_precision": "bf16",
        "trainable_parameter_precision": "fp32",
        "compute_precision": "bf16_amp",
        "export_precision": "bf16",
    }
    checkpoint = tmp_path / "trainable_weights.pt"
    torch.save(
        {
            "model": {
                key: value.detach().clone()
                for key, value in model.named_parameters()
                if value.requires_grad
            },
            "metadata": metadata,
        },
        checkpoint,
    )
    monkeypatch.setattr(exporter, "get_model", lambda cfg: model)
    monkeypatch.setattr(
        exporter.importlib_metadata, "version", lambda name: "synthetic"
    )
    output_dir, bundle_dir = tmp_path / "export", tmp_path / "bundle"
    exporter.export_checkpoint(
        str(train_path),
        str(checkpoint),
        str(output_dir),
        save_adapter=True,
        allow_non_final=True,
        bundle_dir=str(bundle_dir),
    )

    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    policy_contract = manifest["policy_contract"]
    assert policy_contract["config_name"] == "t2vla_serving_config"
    assert policy_contract["norm_asset_id"] == "synthetic_cotrain"
    for key, value in model_cfg["openpi"].items():
        if key != "config_name":
            assert policy_contract[key] == value
    assert manifest["checkpoint"]["metadata"] == metadata
    assert manifest["checkpoint"]["method"] == method
    assert manifest["checkpoint"]["is_final"] is False
    extra_contract = manifest["extra_trainable"]
    assert extra_contract["canonical_prefixes"] == [
        "marker_token_encoder",
        "tactile_cross_attn",
    ]
    expected_extra = {key for key in model.state_dict() if key.startswith(_PREFIXES)}
    assert set(extra_contract["keys"]) == expected_extra
    assert extra_contract["tensor_count"] == len(expected_extra)
    assert manifest["export_metadata"]["extra_tensor_count"] == len(expected_extra)
    extra_state = load_file(bundle_dir / extra_contract["path"])
    merged_state = load_file(output_dir / "model.safetensors")
    assert {value.dtype for value in merged_state.values()} == {torch.bfloat16}
    for key in expected_extra:
        assert torch.equal(extra_state[key], merged_state[key])
    assert set(manifest["adapters"]) == {"paligemma", "action_expert"}
    assert (bundle_dir / "assets/synthetic_cotrain/norm_stats.json").read_bytes() == (
        norm_path.read_bytes()
    )
