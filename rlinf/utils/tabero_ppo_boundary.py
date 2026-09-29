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
TABERO_PPO_DEFAULT_RESET_TRANSITION_BOUNDARY_SEMANTICS = "realworld_terminal_prefix"
TABERO_PPO_DEFAULT_RESET_CHUNK_BOUNDARY_MODE = "terminal_safe_v1"
TABERO_XARM_GRIPPER_TRAVEL_M = 0.045
TABERO_REALWORLD_CHECKPOINT_FORMAT = "realworld_pirl_v1"
TABERO_PPO_BOUNDARY_CONTRACTS = {
    TABERO_PPO_TRANSITION_BOUNDARY_SEMANTICS: {
        "chunk_boundary_mode": TABERO_PPO_CHUNK_BOUNDARY_MODE,
        "requires_hdf5": True,
    },
    TABERO_PPO_DEFAULT_RESET_TRANSITION_BOUNDARY_SEMANTICS: {
        "chunk_boundary_mode": TABERO_PPO_DEFAULT_RESET_CHUNK_BOUNDARY_MODE,
        "requires_hdf5": False,
    },
}
TABERO_PPO_CHECKPOINT_METADATA_KEY = "tabero_ppo_transition_boundary_semantics"
TABERO_PI05_TACIMG_CONFIG_NAME = "pi05_lora_tacimg_realworld_replayed_task820_force"


def uses_tabero_primitive_prefix_boundary(semantics: str | None) -> bool:
    return semantics in TABERO_PPO_BOUNDARY_CONTRACTS


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

    if expected_semantics == TABERO_PPO_DEFAULT_RESET_TRANSITION_BOUNDARY_SEMANTICS:
        validate_realworld_pirl_checkpoint_format(metadata)
        return dict(metadata)

    actual_semantics = metadata.get(TABERO_PPO_CHECKPOINT_METADATA_KEY)
    if actual_semantics != expected_semantics:
        raise ValueError(
            "Tabero PPO checkpoint boundary semantics mismatch: expected "
            f"{expected_semantics!r}, got {actual_semantics!r}. Legacy or mismatched "
            "checkpoints must restart from the base model."
        )
    return dict(metadata)


def build_realworld_pirl_checkpoint_metadata(cfg) -> dict[str, Any]:
    """Describe the actual runtime configuration, not a handwritten YAML copy."""
    model = cfg.actor.model
    openpi = model.openpi
    env = cfg.env.train.init_params
    target_step = cfg.runner.max_epochs
    if cfg.runner.get("max_steps", -1) >= 0:
        target_step = min(target_step, cfg.runner.max_steps)
    metadata = {
        "checkpoint_format": TABERO_REALWORLD_CHECKPOINT_FORMAT,
        "method": "pirl",
        "task_domain": "realworld",
        "task_suite": env.get("task_suite"),
        "task_id": env.get("task_id"),
        "target_object": env.get("target_object"),
        "task_description": env.get("task_description"),
        "reset_source": env.reset_source,
        "deployment_config_name": model.deployment_config_name,
        "normalization_asset_id": model.export_norm_asset_id,
        "model_family": "pi05",
        "action_horizon": openpi.action_horizon,
        "execution_horizon": model.num_action_chunks,
        "effective_action_dim": openpi.effective_action_dim,
        "tactile_prefix_dim_in": openpi.get("tactile_prefix_dim_in"),
        "tactile_prefix_history": openpi.get("tactile_prefix_history"),
        "discrete_state_input": openpi.discrete_state_input,
        "target_global_step": target_step,
        "gradient_checkpointing": cfg.actor.fsdp_config.gradient_checkpointing,
    }

    from rlinf.utils.ppo_multi_task import checkpoint_metadata, enabled

    if enabled(cfg):
        metadata = checkpoint_metadata(metadata, cfg)
    return metadata


def validate_realworld_pirl_checkpoint_format(metadata: Mapping) -> None:
    """Only the current runtime-generated RealWorld format is supported."""
    if metadata.get("checkpoint_format") != TABERO_REALWORLD_CHECKPOINT_FORMAT:
        raise ValueError(
            "Unsupported RealWorld PiRL checkpoint format; old checkpoints are not supported."
        )
