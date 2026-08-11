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

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

TABERO_PPO_TRANSITION_BOUNDARY_SEMANTICS = (
    "terminal_observation_first_done_prefix_logprob_hdf5_reset_v1"
)
TABERO_PPO_CHUNK_BOUNDARY_MODE = "terminal_safe_hdf5_v1"
TABERO_PPO_CHECKPOINT_METADATA_KEY = "tabero_ppo_transition_boundary_semantics"


def validate_tabero_ppo_checkpoint_boundary_metadata(
    load_base_path: str | Path,
    *,
    expected_semantics: str,
) -> dict[str, Any]:
    """Reject PPO checkpoints that predate or mismatch the configured boundary."""

    checkpoint_path = Path(load_base_path) / "model_state_dict" / "trainable_weights.pt"
    if not checkpoint_path.is_file():
        raise ValueError(
            "Tabero PPO boundary-safe resume requires checkpoint sidecar "
            f"{checkpoint_path}; legacy checkpoints must restart from the base model."
        )

    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(
            "Tabero PPO boundary-safe resume could not read checkpoint sidecar "
            f"{checkpoint_path}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise ValueError(
            "Tabero PPO boundary-safe checkpoint sidecar must contain a mapping; "
            f"got {type(payload).__name__}."
        )
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(
            "Tabero PPO boundary-safe checkpoint sidecar is missing mapping metadata."
        )

    actual_semantics = metadata.get(TABERO_PPO_CHECKPOINT_METADATA_KEY)
    if actual_semantics != expected_semantics:
        raise ValueError(
            "Tabero PPO checkpoint boundary semantics mismatch: expected "
            f"{expected_semantics!r}, got {actual_semantics!r}. Legacy or mismatched "
            "checkpoints must restart from the base model."
        )
    return dict(metadata)
