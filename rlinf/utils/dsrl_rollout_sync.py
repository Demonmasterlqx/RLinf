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

import torch

DSRL_ROLLOUT_SYNC_PREFIXES = (
    "dsrl_action_noise_net.",
    "actor_image_encoder.",
    "actor_state_encoder.",
    "actor_tactile_encoder.",
)
DSRL_ROLLOUT_SYNC_TENSOR_COUNT = 48
DSRL_ROLLOUT_SYNC_PARAMETER_COUNT = 2_311_648


def validate_dsrl_rollout_sync_config(actor_cfg) -> tuple[str, ...]:
    """Validate the explicit DSRL actor-to-rollout synchronization contract."""
    training_backend = actor_cfg.get("training_backend")
    if training_backend != "fsdp":
        raise ValueError(
            "OpenPI DSRL selective rollout sync requires actor.training_backend: "
            f"fsdp; got {training_backend!r}."
        )

    configured_prefixes = tuple(actor_cfg.get("rollout_sync_prefixes", ()))
    if configured_prefixes != DSRL_ROLLOUT_SYNC_PREFIXES:
        raise ValueError(
            "OpenPI DSRL actor.rollout_sync_prefixes must contain exactly "
            f"{list(DSRL_ROLLOUT_SYNC_PREFIXES)} in this order; got "
            f"{list(configured_prefixes)}."
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
    """Require the Tabero DSRL actor rollout synchronization keyspace."""
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
