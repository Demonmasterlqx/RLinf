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

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn


class ModelWeightEMA:
    """Maintain FP32 EMA shadows for optimizer-owned parameter shards."""

    _STATE_VERSION = 1

    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        decay: float,
        *,
        parameter_names: Sequence[str] | None = None,
    ) -> None:
        if isinstance(decay, bool) or not isinstance(decay, (float, int)):
            raise TypeError("Model EMA decay must be a float in [0, 1).")
        self.decay = float(decay)
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("Model EMA decay must be in [0, 1).")

        self.parameters = tuple(parameters)
        if not self.parameters:
            raise ValueError("Model EMA requires at least one trainable parameter.")
        if any(not parameter.is_floating_point() for parameter in self.parameters):
            raise TypeError("Model EMA supports only floating-point parameters.")

        if parameter_names is None:
            parameter_names = tuple(
                f"optimizer_parameter_{index}" for index in range(len(self.parameters))
            )
        if len(parameter_names) != len(self.parameters):
            raise ValueError(
                "Model EMA parameter_names must match the parameter count."
            )
        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError("Model EMA parameter names must be unique.")
        self.parameter_names = tuple(parameter_names)
        self.shadows = tuple(
            parameter.detach().to(dtype=torch.float32).clone()
            for parameter in self.parameters
        )
        self.num_updates = 0

    @torch.no_grad()
    def update(self) -> None:
        """Update every local EMA shard after a successful optimizer step."""
        one_minus_decay = 1.0 - self.decay
        for name, parameter, shadow in zip(
            self.parameter_names, self.parameters, self.shadows, strict=True
        ):
            if tuple(parameter.shape) != tuple(shadow.shape):
                raise RuntimeError(
                    f"Model EMA shape changed for {name!r}: "
                    f"{tuple(parameter.shape)} != {tuple(shadow.shape)}."
                )
            current = parameter.detach().to(device=shadow.device, dtype=torch.float32)
            shadow.mul_(self.decay).add_(current, alpha=one_minus_decay)
        self.num_updates += 1

    def state_dict(self) -> dict[str, Any]:
        """Return a rank-local, CPU-serializable EMA state."""
        return {
            "version": self._STATE_VERSION,
            "decay": self.decay,
            "num_updates": self.num_updates,
            "parameter_names": list(self.parameter_names),
            "shadows": [shadow.detach().cpu().clone() for shadow in self.shadows],
        }

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore an EMA state with strict topology and decay validation."""
        required_keys = {
            "version",
            "decay",
            "num_updates",
            "parameter_names",
            "shadows",
        }
        if set(state) != required_keys:
            raise ValueError(
                "Model EMA checkpoint keys do not match the required schema: "
                f"expected={sorted(required_keys)}, got={sorted(state)}."
            )
        if state["version"] != self._STATE_VERSION:
            raise ValueError(
                "Model EMA checkpoint version mismatch: "
                f"{state['version']} != {self._STATE_VERSION}."
            )
        if float(state["decay"]) != self.decay:
            raise ValueError(
                f"Model EMA decay mismatch: {state['decay']} != {self.decay}."
            )
        if tuple(state["parameter_names"]) != self.parameter_names:
            raise ValueError(
                "Model EMA parameter topology does not match the checkpoint."
            )
        saved_shadows = state["shadows"]
        if len(saved_shadows) != len(self.shadows):
            raise ValueError(
                "Model EMA shadow count does not match the current optimizer."
            )
        for name, saved, shadow in zip(
            self.parameter_names, saved_shadows, self.shadows, strict=True
        ):
            if not isinstance(saved, torch.Tensor):
                raise TypeError(f"Model EMA shadow {name!r} is not a tensor.")
            if tuple(saved.shape) != tuple(shadow.shape):
                raise ValueError(
                    f"Model EMA shape mismatch for {name!r}: "
                    f"{tuple(saved.shape)} != {tuple(shadow.shape)}."
                )
            shadow.copy_(saved.to(device=shadow.device, dtype=torch.float32))
        num_updates = state["num_updates"]
        if isinstance(num_updates, bool) or not isinstance(num_updates, int):
            raise TypeError("Model EMA num_updates must be an integer.")
        if num_updates < 0:
            raise ValueError("Model EMA num_updates must be non-negative.")
        self.num_updates = num_updates

    @contextmanager
    @torch.no_grad()
    def apply_to_parameters(self) -> Iterator[None]:
        """Temporarily expose EMA values through the live parameter shards."""
        backups = [parameter.detach().cpu().clone() for parameter in self.parameters]
        try:
            for parameter, shadow in zip(self.parameters, self.shadows, strict=True):
                parameter.copy_(
                    shadow.to(device=parameter.device, dtype=parameter.dtype)
                )
            yield
        finally:
            for parameter, backup in zip(self.parameters, backups, strict=True):
                parameter.copy_(
                    backup.to(device=parameter.device, dtype=parameter.dtype)
                )
