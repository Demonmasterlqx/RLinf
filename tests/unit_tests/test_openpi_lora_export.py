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

import torch

from rlinf.utils.ckpt_convertor.export_openpi_lora_for_t2vla import (
    _filter_action_expert_overlay_state_dict,
)


def test_filter_action_expert_overlay_keeps_only_expert_and_tactile_weights():
    state = {
        "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight": torch.ones(
            1
        ),
        "tactile_prefix_encoder.out_proj.weight": torch.ones(1),
        "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight": torch.ones(
            1
        ),
        "value_head.net.0.weight": torch.ones(1),
    }

    out = _filter_action_expert_overlay_state_dict(state)

    assert sorted(out) == [
        "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight",
        "tactile_prefix_encoder.out_proj.weight",
    ]
