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

import json

import pytest
import torch
from safetensors.torch import load_file

from rlinf.models.embodiment.modules.rlt_token_transformer import RLTTokenEncoder
from rlinf.utils.ckpt_convertor import export_tabero_rlt_for_t2vla


def _small_encoder_state() -> dict[str, torch.Tensor]:
    encoder = RLTTokenEncoder(
        input_dim=2,
        embed_dim=2,
        num_rl_tokens=1,
        prefix_seq_len=4,
        num_layers=1,
        num_heads=1,
        mlp_ratio=1.0,
    )
    return {
        f"rlt_module.encoder.{key}": value
        for key, value in encoder.state_dict().items()
    }


def _small_actor_state() -> dict[str, torch.Tensor]:
    return {
        "backbone.0.weight": torch.ones(2, 5),
        "backbone.0.bias": torch.zeros(2),
        "backbone.2.weight": torch.eye(2),
        "backbone.2.bias": torch.zeros(2),
        "backbone.4.weight": torch.eye(2),
        "backbone.4.bias": torch.zeros(2),
        "actor_mean.weight": torch.ones(2, 2),
        "actor_mean.bias": torch.zeros(2),
    }


def _export_small_bundle(stage1_path, stage2_path, output_dir):
    base_model = stage1_path.parent / "base-model"
    base_model.mkdir(exist_ok=True)
    (base_model / "model.safetensors").write_bytes(b"test base checkpoint")
    return export_tabero_rlt_for_t2vla.export_tabero_rlt_bundle(
        stage1_checkpoint=stage1_path,
        stage2_checkpoint=stage2_path,
        output_dir=output_dir,
        base_model=str(base_model),
        stage2_global_step=75,
        normalized_action_bound=4.0,
        proprio_dim=1,
        action_dim=1,
        num_action_chunks=2,
        ref_num_action_chunks=3,
        actor_hidden_dim=2,
        rlt_input_dim=2,
        rlt_embed_dim=2,
        rlt_num_rl_tokens=1,
        rlt_prefix_seq_len=4,
        rlt_num_layers=1,
        rlt_num_heads=1,
        rlt_mlp_ratio=1.0,
        base_config_name="pi0_test",
        base_action_horizon=3,
        base_model_action_dim=1,
        base_effective_action_dim=1,
        base_prefix_hidden_dim=2,
        base_norm_asset_id="test_asset",
        base_use_quantile_norm=False,
        rlt_use_normalized_proprio=True,
        state_indices=None,
        reference_num_steps=10,
        reference_sampling_method="flow_ode",
    )


def test_export_filters_stage1_encoder_and_stage2_actor(tmp_path):
    stage1_path = tmp_path / "stage1.pt"
    stage2_path = tmp_path / "stage2.pt"
    output_dir = tmp_path / "bundle"
    torch.save(
        {
            "model": {
                **_small_encoder_state(),
                "rlt_module.decoder.token": torch.tensor([2.0]),
            },
            "metadata": {"step": 2000},
        },
        stage1_path,
    )
    torch.save(
        {
            **_small_actor_state(),
            "actor_logstd.weight": torch.ones(2, 2),
            "q_head.weight": torch.ones(1),
        },
        stage2_path,
    )

    manifest = _export_small_bundle(stage1_path, stage2_path, output_dir)

    encoder = load_file(output_dir / "rlt_encoder.safetensors")
    actor = load_file(output_dir / "rlt_actor.safetensors")
    assert len(encoder) == 17
    assert set(actor) == {
        "backbone.0.weight",
        "backbone.0.bias",
        "backbone.2.weight",
        "backbone.2.bias",
        "backbone.4.weight",
        "backbone.4.bias",
        "actor_mean.weight",
        "actor_mean.bias",
    }
    assert manifest["action_space"] == "model_normalized"
    assert manifest["normalized_action_bound"] == 4.0
    assert manifest["stage2_global_step"] == 75
    assert manifest["z_dim"] == 2
    assert manifest["proprio_dim"] == 1
    assert manifest["num_action_chunks"] == 2
    assert manifest["ref_num_action_chunks"] == 3
    assert manifest["base_config_name"] == "pi0_test"
    assert manifest["base_action_horizon"] == 3
    assert manifest["base_model_action_dim"] == 1
    assert manifest["base_effective_action_dim"] == 1
    assert manifest["base_prefix_hidden_dim"] == 2
    assert manifest["base_norm_asset_id"] == "test_asset"
    assert manifest["base_use_quantile_norm"] is False
    assert manifest["rlt_use_normalized_proprio"] is True
    assert manifest["state_indices"] is None
    assert manifest["reference_num_steps"] == 10
    assert manifest["reference_sampling_method"] == "flow_ode"
    assert manifest["base_model_sha256"] == (
        export_tabero_rlt_for_t2vla.checkpoint_sha256(stage1_path.parent / "base-model")
    )
    assert json.loads((output_dir / "manifest.json").read_text()) == manifest


def test_export_rejects_unsupported_proprio_or_reference_semantics(tmp_path):
    stage1_path = tmp_path / "stage1.pt"
    stage2_path = tmp_path / "stage2.pt"
    torch.save({"model": _small_encoder_state()}, stage1_path)
    torch.save(_small_actor_state(), stage2_path)

    with pytest.raises(ValueError, match="normalized proprio"):
        export_tabero_rlt_for_t2vla.export_tabero_rlt_bundle(
            stage1_checkpoint=stage1_path,
            stage2_checkpoint=stage2_path,
            output_dir=tmp_path / "raw-proprio",
            base_model="/models/pi0_tabero",
            stage2_global_step=1,
            rlt_use_normalized_proprio=False,
            base_model_sha256="a" * 64,
        )

    with pytest.raises(ValueError, match="flow_ode"):
        export_tabero_rlt_for_t2vla.export_tabero_rlt_bundle(
            stage1_checkpoint=stage1_path,
            stage2_checkpoint=stage2_path,
            output_dir=tmp_path / "bad-reference",
            base_model="/models/pi0_tabero",
            stage2_global_step=1,
            reference_sampling_method="flow_sde",
            base_model_sha256="a" * 64,
        )


def test_checkpoint_sha256_is_content_based_and_relocation_stable(tmp_path):
    first = tmp_path / "first" / "model.safetensors"
    second = tmp_path / "second" / "model.safetensors"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"same checkpoint")
    second.write_bytes(b"same checkpoint")

    first_hash = export_tabero_rlt_for_t2vla.checkpoint_sha256(first.parent)
    second_hash = export_tabero_rlt_for_t2vla.checkpoint_sha256(second.parent)

    assert first_hash == second_hash
    second.write_bytes(b"different checkpoint")
    assert export_tabero_rlt_for_t2vla.checkpoint_sha256(second.parent) != first_hash


def test_export_rejects_incorrect_expected_base_checksum(tmp_path):
    stage1_path = tmp_path / "stage1.pt"
    stage2_path = tmp_path / "stage2.pt"
    base_model = tmp_path / "base-model"
    base_model.mkdir()
    (base_model / "model.safetensors").write_bytes(b"real checkpoint")
    torch.save({"model": _small_encoder_state()}, stage1_path)
    torch.save(_small_actor_state(), stage2_path)

    with pytest.raises(ValueError, match="does not match"):
        export_tabero_rlt_for_t2vla.export_tabero_rlt_bundle(
            stage1_checkpoint=stage1_path,
            stage2_checkpoint=stage2_path,
            output_dir=tmp_path / "bundle",
            base_model=str(base_model),
            stage2_global_step=1,
            base_model_sha256="a" * 64,
        )


def test_export_loads_checkpoints_in_weights_only_mode(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": {"weight": torch.ones(1)}}, checkpoint)
    real_load = torch.load
    calls = []

    def tracked_load(*args, **kwargs):
        calls.append(kwargs.get("weights_only"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(export_tabero_rlt_for_t2vla.torch, "load", tracked_load)

    export_tabero_rlt_for_t2vla._load_checkpoint(checkpoint)

    assert calls == [True]


def test_export_rejects_actor_shape_inconsistent_with_manifest(tmp_path):
    stage1_path = tmp_path / "stage1.pt"
    stage2_path = tmp_path / "stage2.pt"
    torch.save({"model": _small_encoder_state()}, stage1_path)
    actor = _small_actor_state()
    actor["backbone.0.weight"] = torch.ones(2, 4)
    torch.save(actor, stage2_path)

    with pytest.raises(ValueError, match="backbone.0.weight"):
        _export_small_bundle(stage1_path, stage2_path, tmp_path / "bundle")


def test_export_rejects_encoder_shape_inconsistent_with_manifest(tmp_path):
    stage1_path = tmp_path / "stage1.pt"
    stage2_path = tmp_path / "stage2.pt"
    encoder = _small_encoder_state()
    encoder["rlt_module.encoder.prefix_pos_enc"] = torch.ones(3, 2)
    torch.save({"model": encoder}, stage1_path)
    torch.save(_small_actor_state(), stage2_path)

    with pytest.raises(ValueError, match="encoder.prefix_pos_enc"):
        _export_small_bundle(stage1_path, stage2_path, tmp_path / "bundle")


def test_export_rejects_missing_encoder_weights(tmp_path):
    stage1_path = tmp_path / "stage1.pt"
    stage2_path = tmp_path / "stage2.pt"
    base_model = tmp_path / "base-model"
    base_model.mkdir()
    (base_model / "model.safetensors").write_bytes(b"test base checkpoint")
    torch.save({"model": {"rlt_module.decoder.token": torch.ones(1)}}, stage1_path)
    torch.save({"actor_mean.weight": torch.ones(1, 1)}, stage2_path)

    try:
        export_tabero_rlt_for_t2vla.export_tabero_rlt_bundle(
            stage1_checkpoint=stage1_path,
            stage2_checkpoint=stage2_path,
            output_dir=tmp_path / "bundle",
            base_model=str(base_model),
            stage2_global_step=1,
        )
    except ValueError as exc:
        assert "encoder" in str(exc)
    else:
        raise AssertionError("Expected missing encoder weights to fail")
