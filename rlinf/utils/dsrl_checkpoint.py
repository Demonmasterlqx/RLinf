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

"""Compact checkpoint helpers for Tabero OpenPI DSRL training."""

from collections.abc import Mapping, Sequence
from math import prod
from types import MappingProxyType
from typing import Any

import torch
from torch import nn

from rlinf.utils.dsrl_observation import DSRL_OBSERVATION_SEMANTICS
from rlinf.utils.dsrl_reward import DSRL_REWARD_SEMANTICS
from rlinf.utils.dsrl_rollout_sync import (
    DSRL_ROLLOUT_SYNC_MANIFEST_V2,
    normalize_fsdp_parameter_name,
)

DSRL_TARGET_FORMAT = "tabero_dsrl_target"
DSRL_TARGET_VERSION = 3
DSRL_TRAINABLE_MANIFEST_VERSION = 2
DSRL_TARGET_MANIFEST_VERSION = 2
DSRL_TARGET_PREFIXES = (
    "critic_image_encoder.",
    "critic_state_encoder.",
    "critic_tactile_encoder.",
    "q_head.",
)
DSRL_TRAINABLE_PREFIXES = (
    "dsrl_action_noise_net.",
    "actor_image_encoder.",
    "actor_state_encoder.",
    "actor_tactile_encoder.",
    *DSRL_TARGET_PREFIXES,
)
DSRL_TRAINABLE_TENSOR_COUNT = 220
DSRL_TRAINABLE_PARAMETER_COUNT = 5_273_866
DSRL_Q_STATE_DIM = 128
DSRL_Q_IMAGE_DIM = 128
DSRL_Q_ACTION_DIM = 32
DSRL_Q_HIDDEN_DIMS = (128, 128, 128)
DSRL_Q_OUTPUT_DIM = 1
DSRL_Q_HEAD_COUNT = 10


def _build_dsrl_trainable_manifest() -> Mapping[str, tuple[int, ...]]:
    manifest = dict(DSRL_ROLLOUT_SYNC_MANIFEST_V2)
    manifest.update(
        {
            name.replace("actor_", "critic_", 1): shape
            for name, shape in DSRL_ROLLOUT_SYNC_MANIFEST_V2.items()
            if name.startswith("actor_")
        }
    )

    input_dim = DSRL_Q_STATE_DIM + DSRL_Q_IMAGE_DIM + DSRL_Q_ACTION_DIM
    q_head_shapes: dict[str, tuple[int, ...]] = {}
    in_dim = input_dim
    for hidden_index, out_dim in enumerate(DSRL_Q_HIDDEN_DIMS):
        linear_index = hidden_index * 3
        norm_index = linear_index + 1
        q_head_shapes[f"net.{linear_index}.weight"] = (out_dim, in_dim)
        q_head_shapes[f"net.{linear_index}.bias"] = (out_dim,)
        q_head_shapes[f"net.{norm_index}.weight"] = (out_dim,)
        q_head_shapes[f"net.{norm_index}.bias"] = (out_dim,)
        in_dim = out_dim
    output_index = len(DSRL_Q_HIDDEN_DIMS) * 3
    q_head_shapes[f"net.{output_index}.weight"] = (DSRL_Q_OUTPUT_DIM, in_dim)
    q_head_shapes[f"net.{output_index}.bias"] = (DSRL_Q_OUTPUT_DIM,)
    for head_index in range(DSRL_Q_HEAD_COUNT):
        manifest.update(
            {
                f"q_head.q_heads.{head_index}.{suffix}": shape
                for suffix, shape in q_head_shapes.items()
            }
        )
    return MappingProxyType(manifest)


DSRL_TRAINABLE_MANIFEST_V2 = _build_dsrl_trainable_manifest()
DSRL_TARGET_MANIFEST_V2: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {
        name: shape
        for name, shape in DSRL_TRAINABLE_MANIFEST_V2.items()
        if name.startswith(DSRL_TARGET_PREFIXES)
    }
)

assert len(DSRL_TRAINABLE_MANIFEST_V2) == DSRL_TRAINABLE_TENSOR_COUNT
assert (
    sum(prod(shape) for shape in DSRL_TRAINABLE_MANIFEST_V2.values())
    == DSRL_TRAINABLE_PARAMETER_COUNT
)
assert len(DSRL_TARGET_MANIFEST_V2) == 172
assert sum(prod(shape) for shape in DSRL_TARGET_MANIFEST_V2.values()) == 2_954_026


def _require_strict_int(
    value: Any,
    *,
    label: str,
    minimum: int,
) -> int:
    if type(value) is not int or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(
            f"OpenPI DSRL compact target {label} must be a {qualifier} integer; "
            f"got {value!r}."
        )
    return value


def _normalized_named_parameters(
    model: nn.Module,
    *,
    prefixes: Sequence[str] | None = None,
    requires_grad_only: bool = False,
) -> dict[str, nn.Parameter]:
    """Select parameters by normalized name without materializing a state dict."""
    selected: dict[str, nn.Parameter] = {}
    prefix_tuple = tuple(prefixes) if prefixes is not None else None
    for wrapped_name, parameter in model.named_parameters(remove_duplicate=False):
        if requires_grad_only and not parameter.requires_grad:
            continue
        name = normalize_fsdp_parameter_name(wrapped_name)
        if prefix_tuple is not None and not name.startswith(prefix_tuple):
            continue
        if name in selected:
            raise ValueError(
                "OpenPI DSRL checkpoint produced duplicate normalized parameter "
                f"name {name!r}."
            )
        selected[name] = parameter
    return selected


def select_target_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    """Return the runtime DSRL target critic/Q parameter mapping."""
    selected = _normalized_named_parameters(model, prefixes=DSRL_TARGET_PREFIXES)
    if not selected:
        raise ValueError("OpenPI DSRL target has no critic/Q parameters.")
    return selected


def select_compact_target_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    """Return target parameters matching the canonical DSRL critic/Q manifest."""
    selected = select_target_parameters(model)
    expected_keys = set(DSRL_TARGET_MANIFEST_V2)
    actual_keys = set(selected)
    missing_keys = sorted(expected_keys - actual_keys)
    unexpected_keys = sorted(actual_keys - expected_keys)
    shape_mismatches = {
        name: {
            "expected": DSRL_TARGET_MANIFEST_V2[name],
            "actual": tuple(selected[name].shape),
        }
        for name in expected_keys & actual_keys
        if tuple(selected[name].shape) != DSRL_TARGET_MANIFEST_V2[name]
    }
    if missing_keys or unexpected_keys or shape_mismatches:
        raise ValueError(
            "OpenPI DSRL target does not match canonical manifest v2; "
            f"missing keys: {missing_keys}; unexpected keys: {unexpected_keys}; "
            f"shape mismatches: {shape_mismatches}."
        )
    return selected


def _normalize_tensor_mapping(
    tensors: Mapping[str, torch.Tensor], *, label: str
) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for wrapped_name, tensor in tensors.items():
        name = normalize_fsdp_parameter_name(wrapped_name)
        if name in normalized:
            raise ValueError(
                f"OpenPI DSRL {label} has duplicate normalized key {name!r}."
            )
        normalized[name] = tensor
    return normalized


def _validate_tensor_mapping(
    tensors: Mapping[str, Any],
    runtime: Mapping[str, nn.Parameter],
    *,
    label: str,
    require_runtime_dtype: bool,
    require_float32: bool = False,
) -> None:
    expected_keys = set(runtime)
    actual_keys = set(tensors)
    missing_keys = sorted(expected_keys - actual_keys)
    unexpected_keys = sorted(actual_keys - expected_keys)
    shape_mismatches = {
        name: {
            "expected": tuple(runtime[name].shape),
            "actual": tuple(tensors[name].shape),
        }
        for name in expected_keys & actual_keys
        if isinstance(tensors[name], torch.Tensor)
        and tuple(tensors[name].shape) != tuple(runtime[name].shape)
    }
    dtype_mismatches = {
        name: {
            "expected": (torch.float32 if require_float32 else runtime[name].dtype),
            "actual": tensors[name].dtype,
        }
        for name in expected_keys & actual_keys
        if isinstance(tensors[name], torch.Tensor)
        and (
            (require_float32 and tensors[name].dtype != torch.float32)
            or (require_runtime_dtype and tensors[name].dtype != runtime[name].dtype)
            or not tensors[name].is_floating_point()
        )
    }
    non_tensors = sorted(
        name
        for name in expected_keys & actual_keys
        if not isinstance(tensors[name], torch.Tensor)
    )
    nonfinite = sorted(
        name
        for name in expected_keys & actual_keys
        if isinstance(tensors[name], torch.Tensor)
        and tensors[name].is_floating_point()
        and not torch.isfinite(tensors[name]).all().item()
    )
    if (
        missing_keys
        or unexpected_keys
        or shape_mismatches
        or dtype_mismatches
        or non_tensors
        or nonfinite
    ):
        raise ValueError(
            f"OpenPI DSRL {label} does not match runtime parameters; "
            f"missing keys: {missing_keys}; unexpected keys: {unexpected_keys}; "
            f"shape mismatches: {shape_mismatches}; dtype mismatches: "
            f"{dtype_mismatches}; non-tensor keys: {non_tensors}; non-finite "
            f"keys: {nonfinite}."
        )


def _cpu_clone(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().contiguous().clone()


def build_compact_target_payload(
    target_model: nn.Module,
    target_shadow_f32: Mapping[str, torch.Tensor],
    *,
    step: int,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    """Build and validate a v3 compact target payload from live tensors."""
    _require_strict_int(step, label="step", minimum=0)
    _require_strict_int(rank, label="rank", minimum=0)
    _require_strict_int(world_size, label="world_size", minimum=1)
    runtime = select_target_parameters(target_model)
    shadow = _normalize_tensor_mapping(target_shadow_f32, label="target shadow")
    _validate_tensor_mapping(
        runtime,
        runtime,
        label="target parameters",
        require_runtime_dtype=True,
    )
    _validate_tensor_mapping(
        shadow,
        runtime,
        label="target shadow",
        require_runtime_dtype=False,
        require_float32=True,
    )
    model_state = {name: _cpu_clone(parameter) for name, parameter in runtime.items()}
    shadow_state = {name: _cpu_clone(tensor) for name, tensor in shadow.items()}
    _validate_tensor_mapping(
        model_state,
        runtime,
        label="saved target parameters",
        require_runtime_dtype=True,
    )
    _validate_tensor_mapping(
        shadow_state,
        runtime,
        label="saved target shadow",
        require_runtime_dtype=False,
        require_float32=True,
    )
    parameter_count = sum(tensor.numel() for tensor in model_state.values())
    shadow_parameter_count = sum(tensor.numel() for tensor in shadow_state.values())
    return {
        "format": DSRL_TARGET_FORMAT,
        "version": DSRL_TARGET_VERSION,
        "metadata": {
            "step": step,
            "rank": rank,
            "world_size": world_size,
            "reward_semantics": DSRL_REWARD_SEMANTICS,
            "observation_semantics": DSRL_OBSERVATION_SEMANTICS,
            "manifest_version": DSRL_TARGET_MANIFEST_VERSION,
            "tensor_count": len(model_state),
            "parameter_count": parameter_count,
            "shadow_tensor_count": len(shadow_state),
            "shadow_parameter_count": shadow_parameter_count,
        },
        "model": model_state,
        "target_shadow_f32": shadow_state,
    }


def _validate_compact_metadata(
    metadata: Any,
    *,
    rank: int,
    world_size: int,
    model_state: Mapping[str, torch.Tensor],
    shadow_state: Mapping[str, torch.Tensor],
) -> None:
    if not isinstance(metadata, Mapping):
        raise ValueError("OpenPI DSRL compact target metadata must be a mapping.")
    checkpoint_rank = _require_strict_int(metadata.get("rank"), label="rank", minimum=0)
    checkpoint_world_size = _require_strict_int(
        metadata.get("world_size"), label="world_size", minimum=1
    )
    _require_strict_int(metadata.get("step"), label="step", minimum=0)
    reward_semantics = metadata.get("reward_semantics")
    if reward_semantics != DSRL_REWARD_SEMANTICS:
        raise ValueError(
            "OpenPI DSRL compact target reward semantics mismatch: "
            f"expected {DSRL_REWARD_SEMANTICS!r}, got {reward_semantics!r}."
        )
    observation_semantics = metadata.get("observation_semantics")
    if observation_semantics != DSRL_OBSERVATION_SEMANTICS:
        raise ValueError(
            "OpenPI DSRL compact target observation semantics mismatch: "
            f"expected {DSRL_OBSERVATION_SEMANTICS!r}, got "
            f"{observation_semantics!r}."
        )
    manifest_version = _require_strict_int(
        metadata.get("manifest_version"), label="manifest_version", minimum=1
    )
    if manifest_version != DSRL_TARGET_MANIFEST_VERSION:
        raise ValueError(
            "OpenPI DSRL compact target manifest version mismatch: "
            f"expected {DSRL_TARGET_MANIFEST_VERSION}, got {manifest_version}."
        )
    if checkpoint_rank != rank:
        raise ValueError(
            "OpenPI DSRL compact target rank mismatch: "
            f"checkpoint={checkpoint_rank!r}, runtime={rank}."
        )
    if checkpoint_world_size != world_size:
        raise ValueError(
            "OpenPI DSRL compact target world-size mismatch: "
            f"checkpoint={checkpoint_world_size!r}, runtime={world_size}."
        )
    expected_counts = {
        "tensor_count": len(model_state),
        "parameter_count": sum(tensor.numel() for tensor in model_state.values()),
        "shadow_tensor_count": len(shadow_state),
        "shadow_parameter_count": sum(
            tensor.numel() for tensor in shadow_state.values()
        ),
    }
    count_mismatches = {}
    for name, expected in expected_counts.items():
        actual = _require_strict_int(metadata.get(name), label=name, minimum=1)
        if actual != expected:
            count_mismatches[name] = {"expected": expected, "actual": actual}
    if count_mismatches:
        raise ValueError(
            f"OpenPI DSRL compact target metadata count mismatch: {count_mismatches}."
        )


def _copy_target_and_build_shadow(
    runtime: Mapping[str, nn.Parameter],
    model_state: Mapping[str, torch.Tensor],
    shadow_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        for name, parameter in runtime.items():
            parameter.copy_(model_state[name].to(device=parameter.device))
    return {
        name: shadow_state[name]
        .to(device=runtime[name].device, dtype=torch.float32)
        .contiguous()
        .clone()
        for name in runtime
    }


def validate_target_payload_contract(payload: Any) -> None:
    """Validate the non-tensor contract needed before any checkpoint restore."""
    if not isinstance(payload, Mapping):
        raise ValueError("OpenPI DSRL target checkpoint must be a mapping.")
    if payload.get("format") != DSRL_TARGET_FORMAT:
        raise ValueError(
            "OpenPI DSRL target checkpoint has wrong format: "
            f"{payload.get('format')!r}."
        )
    version = _require_strict_int(payload.get("version"), label="version", minimum=1)
    if version != DSRL_TARGET_VERSION:
        raise ValueError(
            f"OpenPI DSRL target checkpoint has unsupported version: {version!r}."
        )
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("OpenPI DSRL compact target metadata must be a mapping.")
    reward_semantics = metadata.get("reward_semantics")
    if reward_semantics != DSRL_REWARD_SEMANTICS:
        raise ValueError(
            "OpenPI DSRL compact target reward semantics mismatch: "
            f"expected {DSRL_REWARD_SEMANTICS!r}, got {reward_semantics!r}."
        )
    observation_semantics = metadata.get("observation_semantics")
    if observation_semantics != DSRL_OBSERVATION_SEMANTICS:
        raise ValueError(
            "OpenPI DSRL compact target observation semantics mismatch: "
            f"expected {DSRL_OBSERVATION_SEMANTICS!r}, got "
            f"{observation_semantics!r}."
        )
    manifest_version = _require_strict_int(
        metadata.get("manifest_version"), label="manifest_version", minimum=1
    )
    if manifest_version != DSRL_TARGET_MANIFEST_VERSION:
        raise ValueError(
            "OpenPI DSRL compact target manifest version mismatch: "
            f"expected {DSRL_TARGET_MANIFEST_VERSION}, got {manifest_version}."
        )


def restore_target_payload(
    payload: Any,
    target_model: nn.Module,
    *,
    rank: int,
    world_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Validate/copy a current compact target and return shadow/receipt."""
    validate_target_payload_contract(payload)
    _require_strict_int(rank, label="runtime rank", minimum=0)
    _require_strict_int(world_size, label="runtime world_size", minimum=1)
    runtime = select_target_parameters(target_model)

    model_state = payload.get("model")
    shadow_state = payload.get("target_shadow_f32")
    if not isinstance(model_state, Mapping) or not isinstance(shadow_state, Mapping):
        raise ValueError(
            "OpenPI DSRL compact target model and target_shadow_f32 must be mappings."
        )
    _validate_tensor_mapping(
        model_state,
        runtime,
        label="compact target parameters",
        require_runtime_dtype=True,
    )
    _validate_tensor_mapping(
        shadow_state,
        runtime,
        label="compact target shadow",
        require_runtime_dtype=False,
        require_float32=True,
    )
    _validate_compact_metadata(
        payload.get("metadata"),
        rank=rank,
        world_size=world_size,
        model_state=model_state,
        shadow_state=shadow_state,
    )
    shadow = _copy_target_and_build_shadow(runtime, model_state, shadow_state)
    receipt = {
        "format": "compact_v3",
        "reward_semantics": DSRL_REWARD_SEMANTICS,
        "observation_semantics": DSRL_OBSERVATION_SEMANTICS,
        "tensor_count": len(model_state),
        "parameter_count": sum(tensor.numel() for tensor in model_state.values()),
        "shadow_tensor_count": len(shadow_state),
    }
    return shadow, receipt


def select_dsrl_trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect and validate the canonical DSRL trainable sidecar tensors."""
    trainable = _normalized_named_parameters(model, requires_grad_only=True)
    disallowed = sorted(
        name for name in trainable if not name.startswith(DSRL_TRAINABLE_PREFIXES)
    )
    if disallowed:
        raise ValueError(
            "OpenPI DSRL trainable parameters must use only the allowed prefixes "
            f"{list(DSRL_TRAINABLE_PREFIXES)}; got {disallowed}."
        )
    tensor_count = len(trainable)
    parameter_count = sum(parameter.numel() for parameter in trainable.values())
    if (
        tensor_count != DSRL_TRAINABLE_TENSOR_COUNT
        or parameter_count != DSRL_TRAINABLE_PARAMETER_COUNT
    ):
        raise ValueError(
            "OpenPI DSRL trainable sidecar requires exactly 220 tensors and "
            "5,273,866 parameters; got "
            f"{tensor_count} tensors and {parameter_count:,} parameters."
        )
    expected_keys = set(DSRL_TRAINABLE_MANIFEST_V2)
    actual_keys = set(trainable)
    missing_keys = sorted(expected_keys - actual_keys)
    unexpected_keys = sorted(actual_keys - expected_keys)
    shape_mismatches = {
        name: {
            "expected": DSRL_TRAINABLE_MANIFEST_V2[name],
            "actual": tuple(trainable[name].shape),
        }
        for name in expected_keys & actual_keys
        if tuple(trainable[name].shape) != DSRL_TRAINABLE_MANIFEST_V2[name]
    }
    if missing_keys or unexpected_keys or shape_mismatches:
        raise ValueError(
            "OpenPI DSRL trainable sidecar does not match canonical manifest v2; "
            f"missing keys: {missing_keys}; unexpected keys: {unexpected_keys}; "
            f"shape mismatches: {shape_mismatches}."
        )
    _validate_tensor_mapping(
        trainable,
        trainable,
        label="trainable sidecar",
        require_runtime_dtype=True,
    )
    state = {name: _cpu_clone(parameter) for name, parameter in trainable.items()}
    _validate_tensor_mapping(
        state,
        trainable,
        label="saved trainable sidecar",
        require_runtime_dtype=True,
    )
    return state
