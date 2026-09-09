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

"""Check activation recomputation with a real, small Gemma prefix cache."""

import copy
from types import SimpleNamespace

import torch
from torch import nn
from transformers import GemmaConfig, GemmaModel

from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0ForRLActionPrediction,
)


class _PairedGemma(nn.Module):
    def __init__(self):
        super().__init__()
        config = GemmaConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=32,
            attention_dropout=0.0,
        )
        self.paligemma = nn.Module()
        self.paligemma.language_model = GemmaModel(config)
        self.paligemma.language_model.requires_grad_(False)
        self.gemma_expert = nn.Module()
        self.gemma_expert.model = GemmaModel(config)
        self.suffix_calls = 0

    def forward(self, *, inputs_embeds, adarms_cond=None, **kwargs):
        if inputs_embeds[0] is not None:
            result = self.paligemma.language_model(
                inputs_embeds=inputs_embeds[0], **kwargs
            )
            return (result.last_hidden_state, None), result.past_key_values
        self.suffix_calls += 1
        result = self.gemma_expert.model(inputs_embeds=inputs_embeds[1], **kwargs)
        return (None, result.last_hidden_state), None


def _policy():
    model = OpenPi0ForRLActionPrediction.__new__(OpenPi0ForRLActionPrediction)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(action_horizon=2)
    model.paligemma_with_expert = _PairedGemma()
    model.gradient_checkpointing_enabled = False
    prefix = torch.randn(1, 3, 16)
    suffix = torch.randn(1, 2, 16)
    model.embed_prefix = lambda *args: (
        prefix,
        torch.ones(1, 3, dtype=torch.bool),
        torch.zeros(1, 3, dtype=torch.bool),
    )
    model.embed_suffix = lambda *args: (
        suffix,
        torch.ones(1, 2, dtype=torch.bool),
        torch.zeros(1, 2, dtype=torch.bool),
        None,
    )
    return model.train()


def test_pirl_checkpointing_preserves_cache_outputs_and_parameter_gradients():
    torch.manual_seed(7)
    eager = _policy()
    checkpointed = copy.deepcopy(eager)
    checkpointed.gradient_checkpointing_enabled = True
    checkpointed.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
    checkpointed.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
    results = []
    for model in (eager, checkpointed):
        _, mask, cache = model._build_prefix_cache(None, None, None, None)
        assert cache is not None and cache.get_seq_length() == 3
        cache_before = cache[0][0].clone()
        result = model.get_suffix_out(None, mask, cache, None, None)
        result.square().sum().backward()
        assert cache.get_seq_length() == 3
        torch.testing.assert_close(cache[0][0], cache_before)
        results.append(result)
    torch.testing.assert_close(results[0], results[1])
    for (name, p), (_, q) in zip(
        eager.named_parameters(), checkpointed.named_parameters()
    ):
        if p.grad is not None:
            assert q.grad is not None, name
            torch.testing.assert_close(p.grad, q.grad)
    assert eager.paligemma_with_expert.suffix_calls == 1
    assert checkpointed.paligemma_with_expert.suffix_calls == 2
    assert checkpointed.paligemma_with_expert.paligemma.language_model.gradient_checkpointing
