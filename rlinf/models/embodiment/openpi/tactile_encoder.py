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
