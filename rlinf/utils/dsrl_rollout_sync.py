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

from collections.abc import Mapping, Sequence
from math import prod
from types import MappingProxyType

import torch

DSRL_ROLLOUT_SYNC_PREFIXES = (
    "dsrl_action_noise_net.",
    "actor_image_encoder.",
    "actor_state_encoder.",
    "actor_tactile_encoder.",
)
DSRL_ROLLOUT_SYNC_MANIFEST_VERSION = 1
DSRL_ROLLOUT_SYNC_MANIFEST_V1: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {
        "dsrl_action_noise_net.shared_net.0.weight": (128, 192),
        "dsrl_action_noise_net.shared_net.0.bias": (128,),
        "dsrl_action_noise_net.shared_net.1.weight": (128,),
        "dsrl_action_noise_net.shared_net.1.bias": (128,),
        "dsrl_action_noise_net.shared_net.3.weight": (128, 128),
        "dsrl_action_noise_net.shared_net.3.bias": (128,),
        "dsrl_action_noise_net.shared_net.4.weight": (128,),
        "dsrl_action_noise_net.shared_net.4.bias": (128,),
        "dsrl_action_noise_net.shared_net.6.weight": (128, 128),
        "dsrl_action_noise_net.shared_net.6.bias": (128,),
        "dsrl_action_noise_net.shared_net.7.weight": (128,),
        "dsrl_action_noise_net.shared_net.7.bias": (128,),
        "dsrl_action_noise_net.mean_layer.weight": (32, 128),
        "dsrl_action_noise_net.mean_layer.bias": (32,),
        "dsrl_action_noise_net.log_std_layer.weight": (32, 128),
        "dsrl_action_noise_net.log_std_layer.bias": (32,),
        "actor_image_encoder.encoder.0.weight": (32, 3, 3, 3),
        "actor_image_encoder.encoder.0.bias": (32,),
        "actor_image_encoder.encoder.2.weight": (32, 32, 3, 3),
        "actor_image_encoder.encoder.2.bias": (32,),
        "actor_image_encoder.encoder.4.weight": (32, 32, 3, 3),
        "actor_image_encoder.encoder.4.bias": (32,),
        "actor_image_encoder.encoder.6.weight": (32, 32, 3, 3),
        "actor_image_encoder.encoder.6.bias": (32,),
        "actor_image_encoder.bottleneck.1.weight": (64, 32768),
        "actor_image_encoder.bottleneck.1.bias": (64,),
        "actor_image_encoder.bottleneck.2.weight": (64,),
        "actor_image_encoder.bottleneck.2.bias": (64,),
        "actor_state_encoder.encoder.0.weight": (64, 7),
        "actor_state_encoder.encoder.0.bias": (64,),
        "actor_state_encoder.encoder.1.weight": (64,),
        "actor_state_encoder.encoder.1.bias": (64,),
        "actor_tactile_encoder.blocks.0.kernels.0.weight": (64, 396),
        "actor_tactile_encoder.blocks.0.kernels.0.bias": (64,),
        "actor_tactile_encoder.blocks.0.kernels.1.weight": (64, 396),
        "actor_tactile_encoder.blocks.0.kernels.1.bias": (64,),
        "actor_tactile_encoder.blocks.0.kernels.2.weight": (64, 396),
        "actor_tactile_encoder.blocks.0.kernels.2.bias": (64,),
        "actor_tactile_encoder.blocks.0.residual_proj.weight": (64, 396),
        "actor_tactile_encoder.blocks.0.residual_proj.bias": (64,),
        "actor_tactile_encoder.blocks.1.kernels.0.weight": (64, 64),
        "actor_tactile_encoder.blocks.1.kernels.0.bias": (64,),
        "actor_tactile_encoder.blocks.1.kernels.1.weight": (64, 64),
        "actor_tactile_encoder.blocks.1.kernels.1.bias": (64,),
        "actor_tactile_encoder.blocks.1.kernels.2.weight": (64, 64),
        "actor_tactile_encoder.blocks.1.kernels.2.bias": (64,),
        "actor_tactile_encoder.out_proj.weight": (64, 64),
        "actor_tactile_encoder.out_proj.bias": (64,),
    }
)
DSRL_ROLLOUT_SYNC_TENSOR_COUNT = len(DSRL_ROLLOUT_SYNC_MANIFEST_V1)
DSRL_ROLLOUT_SYNC_PARAMETER_COUNT = 2_311_648

assert DSRL_ROLLOUT_SYNC_TENSOR_COUNT == 48
assert (
    sum(prod(shape) for shape in DSRL_ROLLOUT_SYNC_MANIFEST_V1.values())
    == DSRL_ROLLOUT_SYNC_PARAMETER_COUNT
)


def validate_dsrl_rollout_sync_config(actor_cfg) -> tuple[str, ...] | None:
    """Validate opt-in DSRL sync, or return ``None`` for the legacy path."""
    if "rollout_sync_prefixes" not in actor_cfg:
        return None

    configured_prefixes = tuple(actor_cfg.rollout_sync_prefixes)
    if configured_prefixes != DSRL_ROLLOUT_SYNC_PREFIXES:
        raise ValueError(
            "OpenPI DSRL actor.rollout_sync_prefixes must contain exactly "
            f"{list(DSRL_ROLLOUT_SYNC_PREFIXES)} in this order; got "
            f"{list(configured_prefixes)}."
        )

    training_backend = actor_cfg.get("training_backend")
    if training_backend != "fsdp":
        raise ValueError(
            "OpenPI DSRL selective rollout sync requires actor.training_backend: "
            f"fsdp; got {training_backend!r}."
        )

    openpi_cfg = actor_cfg.get("model", {}).get("openpi", {})
    if openpi_cfg.get("dsrl_use_tactile") is not True:
        raise ValueError(
            "OpenPI DSRL selective rollout sync requires "
            "actor.model.openpi.dsrl_use_tactile: true."
        )

    fsdp_cfg = actor_cfg.get("fsdp_config", {})
    sharding_strategy = fsdp_cfg.get("sharding_strategy")
    if sharding_strategy != "no_shard":
        raise ValueError(
            "OpenPI DSRL selective rollout sync requires actor.fsdp_config."
            "sharding_strategy: no_shard; got "
            f"{sharding_strategy!r}."
        )
    if fsdp_cfg.get("use_orig_params") is not True:
        raise ValueError(
            "OpenPI DSRL selective rollout sync requires actor.fsdp_config."
            "use_orig_params: true."
        )
    return configured_prefixes


def normalize_fsdp_parameter_name(name: str) -> str:
    """Remove transparent FSDP/compile wrapper components from a parameter name."""
    for prefix in ("_orig_mod.", "module.", "_fsdp_wrapped_module."):
        while name.startswith(prefix):
            name = name[len(prefix) :]
    return name.replace("._fsdp_wrapped_module.", ".")


def select_named_parameters_by_prefix(
    model: torch.nn.Module, prefixes: Sequence[str]
) -> dict[str, torch.nn.Parameter]:
    """Select parameters directly from ``named_parameters`` by normalized name."""
    selected: dict[str, torch.nn.Parameter] = {}
    prefix_tuple = tuple(prefixes)
    for wrapped_name, parameter in model.named_parameters(remove_duplicate=False):
        name = normalize_fsdp_parameter_name(wrapped_name)
        if not name.startswith(prefix_tuple):
            continue
        if name in selected:
            raise ValueError(
                "OpenPI DSRL rollout sync produced duplicate normalized parameter "
                f"name {name!r}."
            )
        selected[name] = parameter
    return selected


def filter_state_dict_by_prefix(
    state_dict: Mapping[str, torch.Tensor], prefixes: Sequence[str]
) -> dict[str, torch.Tensor]:
    """Filter a state dictionary to normalized keys matching ``prefixes``."""
    selected: dict[str, torch.Tensor] = {}
    prefix_tuple = tuple(prefixes)
    for wrapped_name, tensor in state_dict.items():
        name = normalize_fsdp_parameter_name(wrapped_name)
        if not name.startswith(prefix_tuple):
            continue
        if name in selected:
            raise ValueError(
                "OpenPI DSRL rollout sync produced duplicate normalized state-dict "
                f"key {name!r}."
            )
        selected[name] = tensor
    return selected


def validate_dsrl_rollout_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    """Require the versioned Tabero DSRL rollout synchronization manifest."""
    actual_key_set = set(state_dict)
    expected_key_set = set(DSRL_ROLLOUT_SYNC_MANIFEST_V1)
    missing_keys = sorted(expected_key_set - actual_key_set)
    unexpected_keys = sorted(actual_key_set - expected_key_set)
    shape_mismatches = {
        key: {
            "expected": expected_shape,
            "actual": tuple(state_dict[key].shape),
        }
        for key, expected_shape in DSRL_ROLLOUT_SYNC_MANIFEST_V1.items()
        if key in state_dict and tuple(state_dict[key].shape) != expected_shape
    }
    if missing_keys or unexpected_keys or shape_mismatches:
        raise ValueError(
            "OpenPI DSRL rollout sync state dict does not match canonical manifest "
            f"v{DSRL_ROLLOUT_SYNC_MANIFEST_VERSION}; missing keys: {missing_keys}; "
            "unexpected keys: "
            f"{unexpected_keys}; shape mismatches: {shape_mismatches}."
        )

    tensor_count = len(state_dict)
    parameter_count = sum(tensor.numel() for tensor in state_dict.values())
    if (
        tensor_count != DSRL_ROLLOUT_SYNC_TENSOR_COUNT
        or parameter_count != DSRL_ROLLOUT_SYNC_PARAMETER_COUNT
    ):
        raise ValueError(
            "OpenPI DSRL rollout sync requires exactly 48 tensors and "
            "2,311,648 parameters selected by actor.rollout_sync_prefixes; "
            f"got {tensor_count} tensors and {parameter_count:,} parameters."
        )
