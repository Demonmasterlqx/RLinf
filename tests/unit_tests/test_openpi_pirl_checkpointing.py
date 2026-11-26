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

import pytest
import torch
from torch import nn
from transformers import GemmaConfig, GemmaModel

from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0ForRLActionPrediction,
)


def _count_layer_calls(module, args):
    module.forward_calls += 1


class _PairedGemma(nn.Module):
    def __init__(self, frozen_prefix):
        super().__init__()
        config = GemmaConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=32,
            attention_dropout=0.0,
        )
        self.paligemma = nn.Module()
        self.paligemma.language_model = GemmaModel(config)
        self.paligemma.language_model.requires_grad_(not frozen_prefix)
        for layer in self.paligemma.language_model.layers:
            layer.forward_calls = 0
            layer.register_forward_pre_hook(_count_layer_calls)
        self.gemma_expert = nn.Module()
        self.gemma_expert.model = GemmaModel(config)
        self.suffix_calls = 0
        self.prefix_calls = 0

    def forward(self, *, inputs_embeds, adarms_cond=None, **kwargs):
        if inputs_embeds[0] is not None:
            self.prefix_calls += 1
            result = self.paligemma.language_model(
                inputs_embeds=inputs_embeds[0], **kwargs
            )
            return (result.last_hidden_state, None), result.past_key_values
        self.suffix_calls += 1
        result = self.gemma_expert.model(inputs_embeds=inputs_embeds[1], **kwargs)
        return (None, result.last_hidden_state), None


def _policy(frozen_prefix):
    model = OpenPi0ForRLActionPrediction.__new__(OpenPi0ForRLActionPrediction)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(action_horizon=2, tactile_fusion="taccorl")
    model.paligemma_with_expert = _PairedGemma(frozen_prefix)
    model.gradient_checkpointing_enabled = False
    prefix = torch.randn(1, 3, 16)
    suffix = torch.randn(1, 2, 16)
    model.embed_prefix = lambda *args, **kwargs: (
        prefix,
        torch.ones(1, 3, dtype=torch.bool),
        torch.zeros(1, 3, dtype=torch.bool),
    )
    model.embed_suffix = lambda *args, **kwargs: (
        suffix,
        torch.ones(1, 2, dtype=torch.bool),
        torch.zeros(1, 2, dtype=torch.bool),
        None,
    )
    return model.train()


@pytest.mark.parametrize("frozen_prefix", [True, False])
def test_pirl_checkpointing_preserves_cache_outputs_and_parameter_gradients(
    frozen_prefix,
):
    torch.manual_seed(7)
    eager = _policy(frozen_prefix)
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
    assert all(
        layer.forward_calls == (1 if frozen_prefix else 2)
        for layer in checkpointed.paligemma_with_expert.paligemma.language_model.layers
    )
    if not frozen_prefix:
        assert any(
            p.grad is not None and torch.count_nonzero(p.grad)
            for p in checkpointed.paligemma_with_expert.paligemma.parameters()
        )
    assert checkpointed.paligemma_with_expert.paligemma.language_model.gradient_checkpointing


def test_taccorl_sft_checkpoints_both_streams_with_matching_gradients():
    torch.manual_seed(17)
    eager = _policy(frozen_prefix=False)
    eager.action_out_proj = nn.Linear(16, 16)
    eager._preprocess_observation_with_tactile = lambda *args, **kwargs: (None,) * 6
    eager._encode_marker_tokens = lambda _: None
    checkpointed = copy.deepcopy(eager)
    checkpointed.gradient_checkpointing_enabled = True
    actions = torch.randn(1, 2, 16)
    losses = []
    for model in (eager, checkpointed):
        torch.manual_seed(19)
        loss = model._sft_forward_taccorl(None, actions).mean()
        loss.backward()
        losses.append(loss)
    torch.testing.assert_close(*losses)
    for (name, p), (_, q) in zip(
        eager.named_parameters(), checkpointed.named_parameters()
    ):
        if p.grad is not None:
            assert q.grad is not None, name
            torch.testing.assert_close(p.grad, q.grad)
    assert eager.paligemma_with_expert.prefix_calls == 1
    assert eager.paligemma_with_expert.suffix_calls == 1
    assert all(
        layer.forward_calls == 2
        for layer in checkpointed.paligemma_with_expert.paligemma.language_model.layers
    )
    assert checkpointed.paligemma_with_expert.suffix_calls == 2


def test_pirl_forward_moves_transformed_tactile_and_prompt_to_chain_device():
    model = OpenPi0ForRLActionPrediction.__new__(OpenPi0ForRLActionPrediction)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(action_chunk=2, action_env_dim=3)
    model.input_transform = lambda data, **kwargs: data
    model._observation_from_dict = lambda data: SimpleNamespace()
    model._preprocess_observation_with_tactile = lambda *args, **kwargs: (
        [torch.zeros(1, 3, 2, 2)],
        [torch.ones(1, dtype=torch.bool)],
        torch.ones(1, 2, dtype=torch.long),
        torch.ones(1, 2, dtype=torch.bool),
        torch.zeros(1, 3),
        torch.zeros(1, 3, 8),
    )

    def get_log_prob_value(
        images, masks, tokens, token_masks, state, tactile, chains, *args, **kwargs
    ):
        for tensor in [*images, *masks, tokens, token_masks, state, tactile]:
            assert tensor.device == chains.device
        scores = torch.zeros(1, 1, 2, 3, device=chains.device)
        return scores, torch.zeros(1, 1, device=chains.device), scores

    model.get_log_prob_value = get_log_prob_value
    result = model.default_forward(
        {
            "chains": torch.zeros(1, 2, 2, 3, device="meta"),
            "denoise_inds": torch.zeros(1, 1, dtype=torch.long, device="meta"),
        }
    )
    assert all(value.device.type == "meta" for value in result.values())


class _ShardedMarkerPolicy(nn.Module):
    _encode_marker_tokens = OpenPi0ForRLActionPrediction._encode_marker_tokens
    _apply_checkpoint = OpenPi0ForRLActionPrediction._apply_checkpoint

    def __init__(self, rank):
        super().__init__()
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        from rlinf.models.embodiment.openpi.tactile_encoder import MarkerTokenEncoder

        self.gradient_checkpointing_enabled = True
        self.gradient_checkpointing_use_reentrant = False
        self.marker_token_encoder = FSDP(
            MarkerTokenEncoder(
                in_dim=8,
                history_len=2,
                has_reference_frame=True,
                diff_from_reference=False,
                prefix_width=16,
                suffix_width=8,
                prefix_num_tokens=2,
                suffix_num_tokens=2,
            ).cuda(),
            device_id=rank,
            use_orig_params=False,
        )

    def forward(self, tactile):
        return self._encode_marker_tokens(tactile)


def _train_across_cpu_checkpoint(rank, init_method):
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        "nccl", init_method=init_method, rank=rank, world_size=2
    )
    try:
        model = FSDP(_ShardedMarkerPolicy(rank), device_id=rank, use_orig_params=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        tactile = torch.randn(2, 3, 8, device="cuda")
        for step in range(2):
            loss = sum(value.square().mean() for value in model(tactile))
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if step == 0:
                get_model_state_dict(
                    model,
                    options=StateDictOptions(full_state_dict=True, cpu_offload=True),
                )
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_taccorl_fsdp_trains_after_cpu_checkpoint(tmp_path):
    torch.multiprocessing.spawn(
        _train_across_cpu_checkpoint,
        args=(f"file://{tmp_path / 'process_group'}",),
        nprocs=2,
    )
