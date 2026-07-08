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

from pathlib import Path

from omegaconf import OmegaConf
import torch
from torch import nn

from rlinf.models import (
    _apply_openpi_lora,
    _get_openpi_lora_target_module,
)


class DummyActionExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)
        self.o_proj = nn.Linear(4, 4)

    def forward(self, x):
        return self.o_proj(self.v_proj(self.k_proj(self.q_proj(x))))


class DummyPaligemma(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)


class DummyOpenPI(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.paligemma = DummyPaligemma()
        self.paligemma_with_expert.gemma_expert = nn.Module()
        self.paligemma_with_expert.gemma_expert.model = DummyActionExpert()
        self.action_in_proj = nn.Linear(4, 4)
        self.tactile_prefix_encoder = nn.Linear(4, 4)
        self.value_head = nn.Linear(4, 1)


def _cfg(target: str):
    return OmegaConf.create(
        {
            "is_lora": True,
            "lora_rank": 2,
            "lora_path": None,
            "lora_target": target,
            "freeze_non_lora": True,
        }
    )


def test_get_openpi_lora_target_action_expert():
    model = DummyOpenPI()

    target, assign = _get_openpi_lora_target_module(model, "action_expert")

    assert target is model.paligemma_with_expert.gemma_expert.model
    replacement = DummyActionExpert()
    assign(replacement)
    assert model.paligemma_with_expert.gemma_expert.model is replacement


def test_get_openpi_lora_target_paligemma():
    model = DummyOpenPI()

    target, assign = _get_openpi_lora_target_module(model, "paligemma")

    assert target is model.paligemma_with_expert.paligemma
    replacement = DummyPaligemma()
    assign(replacement)
    assert model.paligemma_with_expert.paligemma is replacement


def test_apply_openpi_action_expert_lora_freezes_non_lora_params():
    model = DummyOpenPI()

    _apply_openpi_lora(model, _cfg("action_expert"))

    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable
    assert any("lora_" in name for name in trainable)
    assert all("paligemma_with_expert.paligemma" not in name for name in trainable)
    assert all(
        "lora_" in name or name.startswith("value_head.") for name in trainable
    )
    assert all(
        "paligemma_with_expert.gemma_expert.model" in name
        or name.startswith("value_head.")
        for name in trainable
    )


def test_tabero_peft_config_uses_action_expert_lora_only():
    path = Path(
        "/data/home/sim6g/code/tabero/RLinf/examples/embodiment/config/"
        "isaaclab_pi0_peft_lora_tacfield_tabero.yaml"
    )
    cfg = OmegaConf.load(path)

    assert cfg.actor.model.is_lora is True
    assert cfg.actor.model.lora_target == "action_expert"
    assert cfg.actor.model.freeze_non_lora is True
    assert cfg.actor.model.openpi.train_expert_only is True
    assert cfg.actor.fsdp_config.save_trainable_model_weights is True
    assert cfg.actor.fsdp_config.checkpoint_format == "none"
