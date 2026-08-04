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
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from openpi.models import model as openpi_model

from rlinf.models.embodiment.modules.rlt_token_transformer import (
    RLTTokenTransformer,
)
from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0Config,
    OpenPi0ForRLActionPrediction,
)
from rlinf.workers.sft import fsdp_vla_sft_worker


def _bare_model(config) -> OpenPi0ForRLActionPrediction:
    model = OpenPi0ForRLActionPrediction.__new__(OpenPi0ForRLActionPrediction)
    nn.Module.__init__(model)
    model.config = config
    return model


class _TinyRLT(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.last_mask = None

    def forward(self, prefix, mask):
        self.last_mask = mask
        loss = torch.square(prefix * self.scale).mean()
        return loss, {"mse": loss}


def test_rlt_train_module_only_defaults_to_false():
    assert OpenPi0Config().rlt_train_module_only is False


def test_rlt_train_module_only_requires_rlt():
    with pytest.raises(ValueError, match="requires use_rlt=True"):
        OpenPi0Config(use_rlt=False, rlt_train_module_only=True)


def test_freeze_non_rlt_parameters_only_keeps_rlt_trainable():
    model = _bare_model(SimpleNamespace())
    model.backbone = nn.Linear(4, 4)
    model.tactile_prefix_encoder = nn.Linear(4, 4)
    model.rlt_module = nn.Linear(4, 4)

    model.freeze_non_rlt_parameters()

    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable == {"rlt_module.weight", "rlt_module.bias"}


def test_rlt_only_sft_skips_vla_path_and_backpropagates_only_rlt():
    config = SimpleNamespace(
        use_rlt=True,
        rlt_train_module_only=True,
        rlt_use_mask=True,
    )
    model = _bare_model(config)
    model.frozen_backbone = nn.Linear(4, 4)
    model.rlt_module = _TinyRLT()
    for parameter in model.frozen_backbone.parameters():
        parameter.requires_grad = False

    prefix = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    mask = torch.tensor([[True, True, False], [True, False, False]])

    def extract_prefix(self, observation, *, train):
        assert train is True
        return prefix, mask

    def forbidden_vla_path(self, observation, actions):
        pytest.fail("RLT-only Stage 1 must not execute the action-flow SFT path")

    model._extract_rlt_prefix_embeddings = MethodType(extract_prefix, model)
    model._sft_forward_with_rlt_prefix = MethodType(forbidden_vla_path, model)
    model.gradient_checkpointing_disable = MethodType(lambda self: None, model)

    output = model.sft_forward(data=({"dummy": torch.ones(2, 1)}, torch.zeros(2, 1)))
    output["loss"].backward()

    assert set(output) == {"loss", "rlt_loss"}
    assert output["loss"] is output["rlt_loss"]
    torch.testing.assert_close(model.rlt_module.last_mask, mask)
    assert model.rlt_module.scale.grad is not None
    assert all(
        parameter.grad is None for parameter in model.frozen_backbone.parameters()
    )


def test_rlt_only_sft_preserves_tactile_field_during_device_move():
    config = SimpleNamespace(
        use_rlt=True,
        rlt_train_module_only=True,
        rlt_use_mask=True,
    )
    model = _bare_model(config)
    model.anchor = nn.Parameter(torch.zeros(()))
    model.rlt_module = _TinyRLT()
    tactile = torch.randn(2, 9, 396)
    observation = openpi_model.Observation(
        images={},
        image_masks={},
        state=torch.zeros(2, 32),
        tokenized_prompt=torch.ones(2, 4, dtype=torch.long),
        tokenized_prompt_mask=torch.ones(2, 4, dtype=torch.bool),
    )
    object.__setattr__(observation, "tactile_prefix", tactile)
    captured = {}

    def extract_prefix(self, moved_observation, *, train):
        assert train is True
        captured["tactile_prefix"] = getattr(moved_observation, "tactile_prefix", None)
        return torch.zeros(2, 3, 4), torch.ones(2, 3, dtype=torch.bool)

    model._extract_rlt_prefix_embeddings = MethodType(extract_prefix, model)
    model.gradient_checkpointing_disable = MethodType(lambda self: None, model)

    model.sft_forward(data=(observation, torch.zeros(2, 1)))

    torch.testing.assert_close(captured["tactile_prefix"], tactile)


def test_non_rlt_sft_routes_tactile_prefix_through_trainable_encoder():
    config = SimpleNamespace(
        use_rlt=False,
        rlt_train_module_only=False,
        action_chunk=50,
        action_env_dim=13,
    )
    model = _bare_model(config)
    model.anchor = nn.Parameter(torch.zeros(()))
    model.tactile_prefix_encoder = nn.Linear(1, 1, bias=False)
    tactile = torch.ones(2, 1, 1)
    observation = openpi_model.Observation(
        images={},
        image_masks={},
        state=torch.zeros(2, 32),
        tokenized_prompt=torch.ones(2, 4, dtype=torch.long),
        tokenized_prompt_mask=torch.ones(2, 4, dtype=torch.bool),
    )
    object.__setattr__(observation, "tactile_prefix", tactile)

    def prefix_forward(self, moved_observation, actions):
        encoded = self.tactile_prefix_encoder(moved_observation.tactile_prefix)
        return encoded.square(), torch.empty(0), torch.empty(0)

    model._sft_forward_with_rlt_prefix = MethodType(prefix_forward, model)
    model.gradient_checkpointing_disable = MethodType(lambda self: None, model)

    loss = model.sft_forward(data=(observation, torch.zeros(2, 1, 1)))
    loss.backward()

    assert model.tactile_prefix_encoder.weight.grad is not None
    assert model.tactile_prefix_encoder.weight.grad.abs().max().item() > 0


def test_rlt_prefix_cache_passes_tactile_prefix_to_backbone():
    model = _bare_model(SimpleNamespace())
    model.anchor = nn.Parameter(torch.zeros(()))
    model.tactile_prefix_encoder = nn.Identity()
    tactile = torch.randn(2, 9, 396)
    captured = {}

    def preprocess_with_tactile(self, observation, *, train):
        assert observation == "observation"
        assert train is True
        return (
            [torch.zeros(2, 3, 8, 8)],
            [torch.ones(2, dtype=torch.bool)],
            torch.ones(2, 4, dtype=torch.long),
            torch.ones(2, 4, dtype=torch.bool),
            torch.zeros(2, 7),
            tactile,
        )

    def forbidden_legacy_preprocess(self, observation, *, train):
        pytest.fail("RLT prefix extraction must use the tactile-aware preprocess path")

    def build_prefix_cache(
        self, images, img_masks, lang_tokens, lang_masks, tactile_prefix
    ):
        captured["tactile_prefix"] = tactile_prefix
        return torch.zeros(2, 5, 8), torch.ones(2, 5, dtype=torch.bool), "cache"

    model._preprocess_observation_with_tactile = MethodType(
        preprocess_with_tactile, model
    )
    model._preprocess_observation = MethodType(forbidden_legacy_preprocess, model)
    model._build_prefix_cache = MethodType(build_prefix_cache, model)

    _, _, cache, _, state, tactile_token_count = model._build_rlt_prefix_cache(
        "observation", train=True
    )

    assert cache == "cache"
    assert state.shape == (2, 7)
    assert tactile_token_count == 1
    assert captured["tactile_prefix"] is tactile


def test_rlt_image_only_keeps_tactile_token_and_excludes_language():
    model = _bare_model(SimpleNamespace(rlt_image_only=True))
    prefix = torch.arange(7, dtype=torch.float32).reshape(1, 7, 1)
    mask = torch.ones(1, 7, dtype=torch.bool)
    language = torch.ones(1, 2, dtype=torch.long)

    selected, selected_mask = model._select_rlt_prefix_embeddings(
        prefix,
        mask,
        language,
        tactile_token_count=1,
    )

    torch.testing.assert_close(
        selected.flatten(), torch.tensor([0.0, 1.0, 2.0, 3.0, 6.0])
    )
    assert selected_mask.shape == (1, 5)


def test_rlt_mask_excludes_padding_from_reconstruction_loss():
    torch.manual_seed(0)
    module = RLTTokenTransformer(
        input_dim=8,
        embed_dim=8,
        num_rl_tokens=1,
        prefix_seq_len=3,
        num_layers=1,
        num_heads=2,
        mlp_ratio=1.0,
    )
    prefix = torch.randn(1, 3, 8)
    mask = torch.tensor([[True, True, False]])
    altered = prefix.clone()
    altered[:, 2] = 1000.0

    loss, _ = module(prefix, mask)
    altered_loss, _ = module(altered, mask)

    torch.testing.assert_close(loss, altered_loss)


def test_tabero_rlt_stage1_config_composes_for_five_gpus(monkeypatch):
    rlinf_root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("EMBODIED_PATH", str(rlinf_root / "examples" / "sft"))
    config_dir = rlinf_root / "examples" / "tabero"

    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg = compose(config_name="tabero_rlt_stage1_sft_openpi_pi0")

    assert cfg.cluster.component_placement["actor,env,rollout"] == "0-4"
    assert cfg.actor.micro_batch_size == 1
    assert cfg.actor.global_batch_size == 40
    assert cfg.actor.global_batch_size % (cfg.actor.micro_batch_size * 5) == 0
    assert cfg.actor.model.precision is None
    assert cfg.actor.fsdp_config.mixed_precision.param_dtype is None
    assert cfg.actor.fsdp_config.mixed_precision.reduce_dtype is None
    assert cfg.actor.fsdp_config.mixed_precision.buffer_dtype is None
    assert cfg.actor.model.openpi.rlt_train_module_only is True
    assert cfg.actor.model.openpi.rlt_image_only is False
    assert cfg.actor.model.openpi.rlt_use_mask is True
    assert cfg.actor.model.openpi_data.norm_stats_path.endswith(
        "assets/NathanWu7/tabero/norm_stats.json"
    )
    assert list(cfg.runner.logger.logger_backends) == ["tensorboard", "wandb"]


def test_openpi_sft_dataloader_preserves_tactile_prefix():
    tactile = torch.randn(2, 9, 396)
    actions = torch.randn(2, 10, 13)
    raw_batch = {
        "image": {
            "base_0_rgb": torch.zeros(2, 224, 224, 3, dtype=torch.uint8),
            "left_wrist_0_rgb": torch.zeros(2, 224, 224, 3, dtype=torch.uint8),
            "right_wrist_0_rgb": torch.zeros(2, 224, 224, 3, dtype=torch.uint8),
        },
        "image_mask": {
            "base_0_rgb": torch.ones(2, dtype=torch.bool),
            "left_wrist_0_rgb": torch.ones(2, dtype=torch.bool),
            "right_wrist_0_rgb": torch.zeros(2, dtype=torch.bool),
        },
        "state": torch.zeros(2, 32),
        "tokenized_prompt": torch.ones(2, 8, dtype=torch.long),
        "tokenized_prompt_mask": torch.ones(2, 8, dtype=torch.bool),
        "tactile_prefix": tactile,
        "actions": actions,
    }

    class _Inner:
        def __iter__(self):
            yield raw_batch

    class _Delegate:
        _data_loader = _Inner()

        def data_config(self):
            return "data-config"

    loader = fsdp_vla_sft_worker._OpenPiTactileDataLoader(_Delegate())
    payload = next(iter(loader))

    assert loader.data_config() == "data-config"
    assert loader._data_loader is _Delegate._data_loader
    assert set(payload) == {"observation", "actions", "tactile_prefix"}
    assert not hasattr(payload["observation"], "tactile_prefix")
    torch.testing.assert_close(payload["tactile_prefix"], tactile)
    torch.testing.assert_close(payload["actions"], actions)
