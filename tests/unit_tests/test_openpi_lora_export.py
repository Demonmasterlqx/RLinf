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

from torch import nn

from rlinf.utils.ckpt_convertor.export_openpi_lora_for_t2vla import (
    _get_lora_modules,
)


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
    assert modules[0].module is model.paligemma_with_expert.paligemma
    assert modules[1].module is model.paligemma_with_expert.gemma_expert.model

    replacement_vlm = nn.Linear(1, 1)
    replacement_expert = nn.Linear(1, 1)
    modules[0].assign_module(replacement_vlm)
    modules[1].assign_module(replacement_expert)
    assert model.paligemma_with_expert.paligemma is replacement_vlm
    assert model.paligemma_with_expert.gemma_expert.model is replacement_expert
