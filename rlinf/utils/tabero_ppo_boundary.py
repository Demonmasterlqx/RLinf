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

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

TABERO_PPO_TRANSITION_BOUNDARY_SEMANTICS = (
    "terminal_observation_first_done_prefix_logprob_hdf5_reset_v1"
)
TABERO_PPO_CHUNK_BOUNDARY_MODE = "terminal_safe_hdf5_v1"
TABERO_PPO_DEFAULT_RESET_TRANSITION_BOUNDARY_SEMANTICS = (
    "terminal_observation_first_done_prefix_logprob_default_reset_v1"
)
TABERO_PPO_DEFAULT_RESET_CHUNK_BOUNDARY_MODE = "terminal_safe_v1"
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
TABERO_PI05_PIRL_CONFIG_NAME = "pi05_lora_tacfield_tabero"
TABERO_PI05_PIRL_NORM_ASSET_ID = "replay_firm_tabero"
TABERO_PI05_PIRL_DATASET = "datas/replay_firm_tabero"


def _load_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Tabero PiRL {label} file does not exist: {path}.")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Tabero PiRL could not read {label} {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(
            f"Tabero PiRL {label} must contain a JSON object; "
            f"got {type(payload).__name__}."
        )
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"Tabero PiRL could not hash {path}: {error}") from error
    return digest.hexdigest()


def _validate_expected_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            f"Tabero PiRL {field} must be a lowercase 64-character SHA-256."
        )
    return value


def _require_exact_values(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    mismatches = {
        key: {"expected": expected_value, "actual": payload.get(key)}
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"Tabero PiRL {label} mismatch: {mismatches}.")


def _validate_norm_stats_vector(
    stats: Mapping[str, Any],
    *,
    name: str,
    expected_dim: int,
) -> None:
    for statistic in ("mean", "std", "q01", "q99"):
        values = stats.get(statistic)
        if not isinstance(values, list) or len(values) != expected_dim:
            actual_dim = len(values) if isinstance(values, list) else None
            raise ValueError(
                f"Tabero PiRL normalization {name}.{statistic} must have "
                f"dimension {expected_dim}; got {actual_dim}."
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError(
                f"Tabero PiRL normalization {name}.{statistic} contains "
                "a non-finite or non-numeric value."
            )


def validate_tabero_pi05_pirl_deployment_checkpoint(
    model_path: str | Path,
    *,
    expected_model_sha256: str,
    expected_norm_stats_sha256: str,
    require_final: bool,
) -> dict[str, Any]:
    """Validate a local merged PI0.5 TacField checkpoint before Ray starts.

    This gate deliberately verifies the model and normalization files rather
    than accepting a directory name as checkpoint provenance. A non-final SFT
    export may be used only when the caller explicitly marks a smoke run.
    """

    if not isinstance(require_final, bool):
        raise ValueError("Tabero PiRL checkpoint require_final must be boolean.")
    expected_model_sha256 = _validate_expected_sha256(
        expected_model_sha256, field="expected_model_sha256"
    )
    expected_norm_stats_sha256 = _validate_expected_sha256(
        expected_norm_stats_sha256, field="expected_norm_stats_sha256"
    )

    checkpoint_dir = Path(model_path).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise ValueError(
            "Tabero PiRL model_path must be an existing local checkpoint "
            f"directory; got {checkpoint_dir}."
        )

    model_file = checkpoint_dir / "model.safetensors"
    if not model_file.is_file():
        raise ValueError(
            f"Tabero PiRL checkpoint is missing model.safetensors: {model_file}."
        )
    export_metadata = _load_json_mapping(
        checkpoint_dir / "export_meta.json", label="export metadata"
    )
    model_config = _load_json_mapping(
        checkpoint_dir / "config.json", label="model config"
    )
    norm_stats_path = (
        checkpoint_dir / TABERO_PI05_PIRL_NORM_ASSET_ID / "norm_stats.json"
    )
    norm_payload = _load_json_mapping(norm_stats_path, label="normalization statistics")

    actual_model_sha256 = _sha256(model_file)
    metadata_model_sha256 = export_metadata.get("model_sha256")
    if (
        actual_model_sha256 != expected_model_sha256
        or metadata_model_sha256 != expected_model_sha256
    ):
        raise ValueError(
            "Tabero PiRL model SHA-256 mismatch: "
            f"expected={expected_model_sha256!r}, "
            f"export_meta={metadata_model_sha256!r}, "
            f"actual={actual_model_sha256!r}."
        )

    actual_norm_stats_sha256 = _sha256(norm_stats_path)
    if actual_norm_stats_sha256 != expected_norm_stats_sha256:
        raise ValueError(
            "Tabero PiRL normalization SHA-256 mismatch: "
            f"expected={expected_norm_stats_sha256!r}, "
            f"actual={actual_norm_stats_sha256!r}."
        )

    _require_exact_values(
        export_metadata,
        {
            "format": "t2vla_openpi_pytorch_merged_lora",
            "method": "sft_full_lora_tacfield",
            "dataset": TABERO_PI05_PIRL_DATASET,
        },
        label="export metadata",
    )
    source_metadata = export_metadata.get("source_ckpt_metadata")
    if not isinstance(source_metadata, Mapping):
        raise ValueError(
            "Tabero PiRL export metadata must include source_ckpt_metadata."
        )
    _require_exact_values(
        source_metadata,
        {
            "dataset": TABERO_PI05_PIRL_DATASET,
            "model_family": "pi05",
            "openpi_config_name": TABERO_PI05_PIRL_CONFIG_NAME,
            "deployment_config_name": TABERO_PI05_PIRL_CONFIG_NAME,
            "action_horizon": 10,
            "effective_action_dim": 13,
            "tactile_prefix_dim_in": 9 * 440 * 2,
            "tactile_prefix_history": 8,
        },
        label="source checkpoint metadata",
    )
    is_final = export_metadata.get("is_final")
    global_step = export_metadata.get("global_step")
    target_global_step = export_metadata.get("target_global_step")
    if (
        not isinstance(is_final, bool)
        or isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step <= 0
        or isinstance(target_global_step, bool)
        or not isinstance(target_global_step, int)
        or target_global_step <= 0
    ):
        raise ValueError(
            "Tabero PiRL export metadata must contain valid is_final, "
            "global_step, and target_global_step fields."
        )
    if require_final and (not is_final or global_step != target_global_step):
        raise ValueError(
            "Tabero PiRL formal training requires a final SFT export; got "
            f"is_final={is_final}, global_step={global_step}, "
            f"target_global_step={target_global_step}."
        )

    _require_exact_values(
        model_config,
        {
            "action_dim": 32,
            "action_horizon": 10,
            "pi05": True,
            "discrete_state_input": True,
            "config_name": TABERO_PI05_PIRL_CONFIG_NAME,
            "num_images_in_input": 2,
            "action_chunk": 10,
            "action_env_dim": 13,
            "num_steps": 10,
            "tactile_type": "expert_his_c_fut",
            "tactile_dim": 6,
            "tactile_dim_in": 0,
            "effective_action_dim": 13,
            "tactile_prefix_dim_in": 9 * 440 * 2,
            "tactile_prefix_history": 8,
            "tactile_prefix_encoder_type": "tcn",
            "tactile_prefix_use_reference_frame": True,
            "tactile_prefix_diff_from_reference": False,
            "tactile_streams": ["tactile_prefix"],
        },
        label="model config",
    )

    norm_stats = norm_payload.get("norm_stats")
    if not isinstance(norm_stats, Mapping):
        raise ValueError(
            "Tabero PiRL normalization file must contain a norm_stats mapping."
        )
    for name, expected_dim in {
        "state": 7,
        "actions": 13,
        # TaberoTacFieldInputs keeps history as the leading axis and flattens
        # each marker frame to 440 * 2. OpenPI Normalize broadcasts this
        # per-frame statistic across all nine reference/history frames.
        "tactile_prefix": 440 * 2,
    }.items():
        stats = norm_stats.get(name)
        if not isinstance(stats, Mapping):
            raise ValueError(f"Tabero PiRL normalization is missing mapping {name!r}.")
        _validate_norm_stats_vector(stats, name=name, expected_dim=expected_dim)

    return {
        "checkpoint_dir": str(checkpoint_dir),
        "model_sha256": actual_model_sha256,
        "norm_stats_path": str(norm_stats_path),
        "norm_stats_sha256": actual_norm_stats_sha256,
        "global_step": global_step,
        "target_global_step": target_global_step,
        "is_final": is_final,
    }


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

    actual_semantics = metadata.get(TABERO_PPO_CHECKPOINT_METADATA_KEY)
    if actual_semantics != expected_semantics:
        raise ValueError(
            "Tabero PPO checkpoint boundary semantics mismatch: expected "
            f"{expected_semantics!r}, got {actual_semantics!r}. Legacy or mismatched "
            "checkpoints must restart from the base model."
        )
    return dict(metadata)
