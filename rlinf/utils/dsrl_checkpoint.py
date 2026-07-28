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
from typing import Any

import torch
from torch import nn

from rlinf.utils.dsrl_rollout_sync import normalize_fsdp_parameter_name

DSRL_TARGET_FORMAT = "tabero_dsrl_target"
DSRL_TARGET_VERSION = 1
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
DSRL_TRAINABLE_PARAMETER_COUNT = 5_183_754


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
    """Build and validate a v1 compact target payload from live tensors."""
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
    if metadata.get("rank") != rank:
        raise ValueError(
            "OpenPI DSRL compact target rank mismatch: "
            f"checkpoint={metadata.get('rank')!r}, runtime={rank}."
        )
    if metadata.get("world_size") != world_size:
        raise ValueError(
            "OpenPI DSRL compact target world-size mismatch: "
            f"checkpoint={metadata.get('world_size')!r}, runtime={world_size}."
        )
    expected_counts = {
        "tensor_count": len(model_state),
        "parameter_count": sum(tensor.numel() for tensor in model_state.values()),
        "shadow_tensor_count": len(shadow_state),
        "shadow_parameter_count": sum(
            tensor.numel() for tensor in shadow_state.values()
        ),
    }
    count_mismatches = {
        name: {"expected": expected, "actual": metadata.get(name)}
        for name, expected in expected_counts.items()
        if metadata.get(name) != expected
    }
    step = metadata.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(
            f"OpenPI DSRL compact target step must be a non-negative integer; got {step!r}."
        )
    if count_mismatches:
        raise ValueError(
            f"OpenPI DSRL compact target metadata count mismatch: {count_mismatches}."
        )


def _copy_target_and_build_shadow(
    runtime: Mapping[str, nn.Parameter],
    model_state: Mapping[str, torch.Tensor],
    shadow_state: Mapping[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        for name, parameter in runtime.items():
            parameter.copy_(model_state[name].to(device=parameter.device))
    if shadow_state is None:
        return {
            name: parameter.detach().float().clone()
            for name, parameter in runtime.items()
        }
    return {
        name: shadow_state[name]
        .to(device=runtime[name].device, dtype=torch.float32)
        .contiguous()
        .clone()
        for name in runtime
    }


def restore_target_payload(
    payload: Any,
    target_model: nn.Module,
    *,
    rank: int,
    world_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Validate/copy compact or legacy target tensors and return shadow/receipt."""
    if not isinstance(payload, Mapping):
        raise ValueError("OpenPI DSRL target checkpoint must be a mapping.")
    runtime = select_target_parameters(target_model)

    if "format" in payload:
        if payload.get("format") != DSRL_TARGET_FORMAT:
            raise ValueError(
                "OpenPI DSRL target checkpoint has wrong format: "
                f"{payload.get('format')!r}."
            )
        if payload.get("version") != DSRL_TARGET_VERSION:
            raise ValueError(
                "OpenPI DSRL target checkpoint has unsupported version: "
                f"{payload.get('version')!r}."
            )
        model_state = payload.get("model")
        shadow_state = payload.get("target_shadow_f32")
        if not isinstance(model_state, Mapping) or not isinstance(
            shadow_state, Mapping
        ):
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
            "format": "compact_v1",
            "tensor_count": len(model_state),
            "parameter_count": sum(tensor.numel() for tensor in model_state.values()),
            "shadow_tensor_count": len(shadow_state),
        }
        return shadow, receipt

    legacy_state = _normalize_tensor_mapping(payload, label="legacy target")
    selected_legacy = {
        name: tensor
        for name, tensor in legacy_state.items()
        if name.startswith(DSRL_TARGET_PREFIXES)
    }
    _validate_tensor_mapping(
        selected_legacy,
        runtime,
        label="legacy target parameters",
        require_runtime_dtype=False,
    )
    shadow = _copy_target_and_build_shadow(runtime, selected_legacy, None)
    receipt = {
        "format": "legacy_full",
        "tensor_count": len(selected_legacy),
        "parameter_count": sum(tensor.numel() for tensor in selected_legacy.values()),
        "shadow_tensor_count": len(shadow),
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
            "5,183,754 parameters; got "
            f"{tensor_count} tensors and {parameter_count:,} parameters."
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
