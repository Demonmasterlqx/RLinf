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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import rlinf.hybrid_engines.fsdp.fsdp_model_manager as fsdp_model_manager
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager


class _Logger:
    def info(self, *_args, **_kwargs):
        pass


def test_normalize_fsdp_param_name_strips_wrappers():
    assert (
        FSDPModelManager._normalize_fsdp_param_name(
            "module._fsdp_wrapped_module.foo._fsdp_wrapped_module.weight"
        )
        == "foo.weight"
    )


def test_save_trainable_model_weights_fails_when_no_trainable_params(
    monkeypatch, tmp_path
):
    model = nn.Linear(2, 2)
    for param in model.parameters():
        param.requires_grad = False

    manager = FSDPModelManager.__new__(FSDPModelManager)
    manager.model = model
    manager._cfg = SimpleNamespace(
        fsdp_config={"save_trainable_model_weights": True}
    )
    manager._logger = _Logger()

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    with pytest.raises(RuntimeError, match="no trainable parameters"):
        manager._save_trainable_model_weights(str(tmp_path), step=0)


class _FlatParamWrappedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._flat_param = nn.Parameter(torch.ones(4))


def test_save_trainable_model_weights_uses_pre_wrap_trainable_names_for_fsdp_flat_params(
    monkeypatch, tmp_path
):
    model = _FlatParamWrappedModel()

    manager = FSDPModelManager.__new__(FSDPModelManager)
    manager.model = model
    manager._cfg = SimpleNamespace(
        fsdp_config={"save_trainable_model_weights": True}
    )
    manager._logger = _Logger()
    manager.trainable_param_names = [
        "paligemma.adapter.lora_A.weight",
        "gemma_expert.adapter.lora_B.weight",
    ]

    def fake_full_state_dict(*_args, **_kwargs):
        return {
            "_fsdp_wrapped_module.paligemma.adapter.lora_A.weight": torch.tensor(
                [[1.0, 2.0]]
            ),
            "_fsdp_wrapped_module.gemma_expert.adapter.lora_B.weight": torch.tensor(
                [[3.0], [4.0]]
            ),
            "_fsdp_wrapped_module.frozen.weight": torch.tensor([5.0]),
            "_fsdp_wrapped_module.persistent_buffer": torch.tensor([6.0]),
        }

    manager.get_model_state_dict = fake_full_state_dict

    monkeypatch.setattr(fsdp_model_manager, "FSDP", _FlatParamWrappedModel)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    manager._save_trainable_model_weights(str(tmp_path), step=7)

    checkpoint = torch.load(
        tmp_path / "model_state_dict" / "trainable_weights.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert checkpoint["metadata"]["step"] == 7
    assert checkpoint["metadata"]["parameter_count"] == 2
    assert list(checkpoint["model"].keys()) == [
        "paligemma.adapter.lora_A.weight",
        "gemma_expert.adapter.lora_B.weight",
    ]
    torch.testing.assert_close(
        checkpoint["model"]["paligemma.adapter.lora_A.weight"],
        torch.tensor([[1.0, 2.0]]),
    )
    torch.testing.assert_close(
        checkpoint["model"]["gemma_expert.adapter.lora_B.weight"],
        torch.tensor([[3.0], [4.0]]),
    )
