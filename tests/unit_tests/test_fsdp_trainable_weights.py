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
