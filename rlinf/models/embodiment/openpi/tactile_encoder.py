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
from torch import nn
import torch.nn.functional as F


class TactileTCNBlock(nn.Module):
    """Causal TCN block matching the Tabero JAX tactile-prefix encoder."""

    def __init__(self, input_dim: int, output_dim: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1.")
        self.kernel_size = kernel_size
        self.kernels = nn.ModuleList(
            nn.Linear(input_dim, output_dim) for _ in range(kernel_size)
        )
        self.residual_proj = (
            nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"TactileTCNBlock expects rank 3, got shape {x.shape}.")

        batch_size, steps, input_dim = x.shape
        pad = torch.zeros(
            batch_size,
            self.kernel_size - 1,
            input_dim,
            dtype=x.dtype,
            device=x.device,
        )
        x_pad = torch.cat([pad, x], dim=1)

        y = None
        for idx, linear in enumerate(self.kernels):
            start = self.kernel_size - 1 - idx
            x_slice = x_pad[:, start : start + steps, :]
            projected = linear(x_slice)
            y = projected if y is None else y + projected

        residual = x if self.residual_proj is None else self.residual_proj(x)
        return F.silu(y + residual)


class TactileTCNEncoder(nn.Module):
    """Encode a Tabero marker-motion sequence into one prefix token."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        history_len: int,
        has_reference_frame: bool,
        diff_from_reference: bool,
        num_layers: int = 2,
        kernel_size: int = 3,
    ):
        super().__init__()
        if history_len <= 0:
            raise ValueError("history_len must be > 0.")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")

        self.input_dim = input_dim
        self.history_len = history_len
        self.has_reference_frame = has_reference_frame
        self.diff_from_reference = diff_from_reference

        blocks = []
        for idx in range(num_layers):
            block_input_dim = input_dim if idx == 0 else hidden_dim
            blocks.append(TactileTCNBlock(block_input_dim, hidden_dim, kernel_size))
        self.blocks = nn.ModuleList(blocks)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, tactile: torch.Tensor) -> torch.Tensor:
        if tactile.ndim == 2:
            tactile_seq = tactile[:, None, :]
        elif tactile.ndim == 3:
            tactile_seq = tactile
        else:
            raise ValueError(
                f"TactileTCNEncoder expects rank 2 or 3, got shape {tactile.shape}."
            )

        if tactile_seq.shape[-1] != self.input_dim:
            raise ValueError(
                f"TactileTCNEncoder expected input_dim={self.input_dim}, "
                f"got {tactile_seq.shape[-1]}."
            )

        if self.has_reference_frame:
            if tactile_seq.shape[1] < 2:
                raise ValueError(
                    "Reference-frame tactile input requires at least 2 frames."
                )
            if self.diff_from_reference:
                max_hist = min(self.history_len, tactile_seq.shape[1] - 1)
                baseline = tactile_seq[:, 0:1, :]
                seq_for_tcn = tactile_seq[:, 1 : 1 + max_hist, :] - baseline
            else:
                max_steps = min(self.history_len + 1, tactile_seq.shape[1])
                seq_for_tcn = tactile_seq[:, :max_steps, :]
        else:
            seq_for_tcn = tactile_seq[:, -self.history_len :, :]

        h = seq_for_tcn
        for block in self.blocks:
            h = block(h)
        return self.out_proj(h[:, -1, :])


class MarkerTokenEncoder(nn.Module):
    """Encode marker frames into TacCoRL prefix and suffix token sequences."""

    def __init__(
        self,
        *,
        in_dim: int,
        history_len: int,
        has_reference_frame: bool,
        diff_from_reference: bool,
        prefix_width: int,
        suffix_width: int,
        prefix_num_tokens: int,
        suffix_num_tokens: int,
        num_layers: int = 2,
        kernel_size: int = 3,
    ):
        super().__init__()
        if in_dim <= 0 or in_dim % 2:
            raise ValueError("MarkerTokenEncoder.in_dim must be a positive even number.")
        if history_len <= 0 or min(prefix_num_tokens, suffix_num_tokens) <= 0:
            raise ValueError("MarkerTokenEncoder requires positive history and token counts.")
        if num_layers != 2:
            raise ValueError("MarkerTokenEncoder uses exactly two TCN blocks for JAX parity.")
        self.input_dim = in_dim
        self.history_len = history_len
        self.has_reference_frame = has_reference_frame
        self.diff_from_reference = diff_from_reference
        self.prefix_width = prefix_width
        self.suffix_width = suffix_width
        self.prefix_num_tokens = prefix_num_tokens
        self.suffix_num_tokens = suffix_num_tokens
        self.blocks = nn.ModuleList(
            TactileTCNBlock(in_dim if idx == 0 else suffix_width, suffix_width, kernel_size)
            for idx in range(num_layers)
        )
        self.prefix_proj = nn.Linear(suffix_width, prefix_num_tokens * prefix_width)
        self.suffix_proj = nn.Linear(suffix_width, suffix_num_tokens * suffix_width)
        self.prefix_token_embedding = nn.Parameter(
            torch.randn(prefix_num_tokens, prefix_width) * 0.02
        )
        self.suffix_token_embedding = nn.Parameter(
            torch.randn(suffix_num_tokens, suffix_width) * 0.02
        )

    def forward(self, tactile: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tactile.ndim != 3 or tactile.shape[-1] != self.input_dim:
            raise ValueError(
                "MarkerTokenEncoder expects [B, T, input_dim] marker vectors; "
                f"got {tuple(tactile.shape)} with input_dim={self.input_dim}."
            )
        parameter_dtype = self.blocks[0].kernels[0].weight.dtype
        hidden = _select_tcn_frames_torch(
            tactile.to(dtype=parameter_dtype),
            history_len=self.history_len,
            has_reference_frame=self.has_reference_frame,
            diff_from_reference=self.diff_from_reference,
        )
        for block in self.blocks:
            hidden = block(hidden)
        pooled = hidden[:, -1, :]
        prefix = self.prefix_proj(pooled).reshape(-1, self.prefix_num_tokens, self.prefix_width)
        suffix = self.suffix_proj(pooled).reshape(-1, self.suffix_num_tokens, self.suffix_width)
        return prefix + self.prefix_token_embedding, suffix + self.suffix_token_embedding


def _select_tcn_frames_torch(
    tactile: torch.Tensor,
    *,
    history_len: int,
    has_reference_frame: bool,
    diff_from_reference: bool,
) -> torch.Tensor:
    if tactile.ndim != 3:
        raise ValueError(f"MarkerTokenEncoder expects rank 3 input, got {tuple(tactile.shape)}.")
    num_steps = tactile.shape[1]
    if has_reference_frame:
        if num_steps < 2:
            raise ValueError("Reference-frame tactile input requires at least 2 frames.")
        if diff_from_reference:
            max_hist = min(history_len, num_steps - 1)
            return tactile[:, 1 : 1 + max_hist, :] - tactile[:, 0:1, :]
        return tactile[:, : min(history_len + 1, num_steps), :]
    return tactile[:, -history_len:, :] if num_steps >= history_len else tactile


class TactilePrefixCrossAttention(nn.Module):
    """Pre-LN, fp32-softmax, zero-gated cross-attention for prefix tokens."""

    def __init__(self, width: int, num_heads: int, *, eps: float = 1e-6):
        super().__init__()
        if num_heads <= 0 or width % num_heads:
            raise ValueError("Cross-attention num_heads must divide the embedding width.")
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.norm = nn.LayerNorm(width, eps=eps)
        self.query = nn.Linear(width, width, bias=True)
        self.key = nn.Linear(width, width, bias=True)
        self.value = nn.Linear(width, width, bias=True)
        self.out = nn.Linear(width, width, bias=True)
        # FSDP requires a one-element vector instead of a scalar parameter.
        self.alpha = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(
        self, base: torch.Tensor, tactile: torch.Tensor, tactile_mask: torch.Tensor
    ) -> torch.Tensor:
        if base.ndim != 3 or tactile.ndim != 3:
            raise ValueError("TactilePrefixCrossAttention expects rank-3 inputs.")
        if tuple(tactile_mask.shape) != tuple(tactile.shape[:2]):
            raise ValueError(
                f"tactile_mask must have shape {tuple(tactile.shape[:2])}, "
                f"got {tuple(tactile_mask.shape)}."
            )
        parameter_dtype = self.query.weight.dtype
        base_work = base.to(dtype=parameter_dtype)
        tactile_work = tactile.to(dtype=parameter_dtype)
        query = self.query(self.norm(base_work)).reshape(
            base.shape[0], base.shape[1], self.num_heads, self.head_dim
        )
        key = self.key(tactile_work).reshape(
            tactile.shape[0], tactile.shape[1], self.num_heads, self.head_dim
        )
        value = self.value(tactile_work).reshape(
            tactile.shape[0], tactile.shape[1], self.num_heads, self.head_dim
        )
        logits = torch.einsum("bqhd,bkhd->bhqk", query, key).float()
        logits = logits * (self.head_dim**-0.5)
        key_mask = tactile_mask.to(device=logits.device, dtype=torch.bool)[:, None, None, :]
        weights = torch.softmax(logits.masked_fill(~key_mask, torch.finfo(torch.float32).min), dim=-1)
        weights = torch.where(key_mask, weights, torch.zeros_like(weights))
        attended = torch.einsum(
            "bhqk,bkhd->bqhd", weights.to(dtype=value.dtype), value
        ).reshape_as(base_work)
        residual = self.out(attended)
        contact = tactile_mask.to(device=residual.device, dtype=torch.bool).any(dim=-1, keepdim=True)
        scale = torch.tanh(self.alpha.to(dtype=residual.dtype)) * contact[:, :, None]
        return base + (scale * residual).to(dtype=base.dtype)
