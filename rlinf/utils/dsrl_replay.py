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

"""Versioned compact replay contracts for Tabero DSRL."""

from collections.abc import Mapping
from types import MappingProxyType

import torch
import torch.nn.functional as F

from rlinf.utils.dsrl_observation import (
    DSRL_OBSERVATION_SEMANTICS,
    REALWORLD_TACIMG_DSRL_OBSERVATION_SEMANTICS,
)
from rlinf.utils.dsrl_transition import (
    DSRL_TRANSITION_BOUNDARY_SEMANTICS,
    REALWORLD_TACIMG_DSRL_TRANSITION_BOUNDARY_SEMANTICS,
)

DSRL_REPLAY_SEMANTICS = "main_wrist_bf16_64_transition_ring_v1"
REALWORLD_TACIMG_DSRL_REPLAY_SEMANTICS = "main_wrist_tactile_bf16_64_transition_ring_v1"
DSRL_REPLAY_BACKEND = "compact_dsrl_ring"
DSRL_REPLAY_FORMAT = "tabero_dsrl_compact_replay"
DSRL_REPLAY_FORMAT_VERSION = 1
DSRL_REPLAY_IMAGE_SIZE = 64
DSRL_REPLAY_NUM_IMAGES = 2
DSRL_REPLAY_VIEW_ORDER = ("main", "wrist")
DSRL_REPLAY_CAPACITY_TRANSITIONS = 100_000
DSRL_REPLAY_CHECKPOINT_SHARD_TRANSITIONS = 4096
DSRL_REPLAY_MAX_RESIDENT_GIB = 12.0

REALWORLD_TACIMG_DSRL_REPLAY_NUM_IMAGES = 3
REALWORLD_TACIMG_DSRL_REPLAY_VIEW_ORDER = ("main", "wrist", "tactile")
REALWORLD_TACIMG_DSRL_REPLAY_CAPACITY_TRANSITIONS = 80_000
REALWORLD_TACIMG_DSRL_REPLAY_CHECKPOINT_SHARD_TRANSITIONS = 4096
REALWORLD_TACIMG_DSRL_REPLAY_MAX_RESIDENT_GIB = 12.0


def _build_field_specs(
    *, num_images: int, include_marker_motion: bool
) -> MappingProxyType:
    specs: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
        "curr_obs.dsrl_images": (
            (num_images, 3, DSRL_REPLAY_IMAGE_SIZE, DSRL_REPLAY_IMAGE_SIZE),
            torch.bfloat16,
        ),
        "curr_obs.states": ((7,), torch.bfloat16),
        "next_obs.dsrl_images": (
            (num_images, 3, DSRL_REPLAY_IMAGE_SIZE, DSRL_REPLAY_IMAGE_SIZE),
            torch.bfloat16,
        ),
        "next_obs.states": ((7,), torch.bfloat16),
        "actions": ((32,), torch.bfloat16),
        "rewards": ((10,), torch.float32),
        "terminations": ((10,), torch.bool),
        "truncations": ((10,), torch.bool),
    }
    if include_marker_motion:
        specs["curr_obs.tactile_marker_motion"] = (
            (9, 198, 2),
            torch.bfloat16,
        )
        specs["next_obs.tactile_marker_motion"] = (
            (9, 198, 2),
            torch.bfloat16,
        )
    return MappingProxyType(specs)


# Trailing shapes and dtypes of one flattened SAC transition. Rewards remain
# FP32 because discounted macro-return arithmetic must not round gamma in BF16.
DSRL_REPLAY_FIELD_SPECS = _build_field_specs(
    num_images=DSRL_REPLAY_NUM_IMAGES,
    include_marker_motion=True,
)
REALWORLD_TACIMG_DSRL_REPLAY_FIELD_SPECS = _build_field_specs(
    num_images=REALWORLD_TACIMG_DSRL_REPLAY_NUM_IMAGES,
    include_marker_motion=False,
)

DSRL_REPLAY_CONTRACTS = MappingProxyType(
    {
        DSRL_REPLAY_SEMANTICS: {
            "field_specs": DSRL_REPLAY_FIELD_SPECS,
            "observation_semantics": DSRL_OBSERVATION_SEMANTICS,
            "transition_boundary_semantics": DSRL_TRANSITION_BOUNDARY_SEMANTICS,
            "view_order": DSRL_REPLAY_VIEW_ORDER,
            "num_images": DSRL_REPLAY_NUM_IMAGES,
            "capacity_transitions": DSRL_REPLAY_CAPACITY_TRANSITIONS,
            "checkpoint_shard_transitions": DSRL_REPLAY_CHECKPOINT_SHARD_TRANSITIONS,
            "max_resident_gib": DSRL_REPLAY_MAX_RESIDENT_GIB,
            "raw_image_shapes": {
                "main_images": (256, 256),
                "wrist_images": (256, 256),
            },
            "raw_image_fields": ("main_images", "wrist_images"),
            "include_marker_motion": True,
        },
        REALWORLD_TACIMG_DSRL_REPLAY_SEMANTICS: {
            "field_specs": REALWORLD_TACIMG_DSRL_REPLAY_FIELD_SPECS,
            "observation_semantics": (REALWORLD_TACIMG_DSRL_OBSERVATION_SEMANTICS),
            "transition_boundary_semantics": (
                REALWORLD_TACIMG_DSRL_TRANSITION_BOUNDARY_SEMANTICS
            ),
            "view_order": REALWORLD_TACIMG_DSRL_REPLAY_VIEW_ORDER,
            "num_images": REALWORLD_TACIMG_DSRL_REPLAY_NUM_IMAGES,
            "capacity_transitions": (REALWORLD_TACIMG_DSRL_REPLAY_CAPACITY_TRANSITIONS),
            "checkpoint_shard_transitions": (
                REALWORLD_TACIMG_DSRL_REPLAY_CHECKPOINT_SHARD_TRANSITIONS
            ),
            "max_resident_gib": REALWORLD_TACIMG_DSRL_REPLAY_MAX_RESIDENT_GIB,
            "raw_image_shapes": {
                "main_images": (480, 640),
                "wrist_images": (480, 640),
                "tactile_images": (224, 224),
            },
            "raw_image_fields": (
                "main_images",
                "wrist_images",
                "tactile_images",
            ),
            "include_marker_motion": False,
        },
    }
)


def get_dsrl_replay_contract(replay_semantics: str) -> dict[str, object]:
    """Return a shallow copy of a supported compact replay contract."""

    contract = DSRL_REPLAY_CONTRACTS.get(replay_semantics)
    if contract is None:
        raise ValueError(
            f"Unsupported Tabero DSRL replay semantics {replay_semantics!r}; "
            f"expected one of {sorted(DSRL_REPLAY_CONTRACTS)!r}."
        )
    return dict(contract)


def is_compact_dsrl_replay_semantics(replay_semantics: object) -> bool:
    """Return whether ``replay_semantics`` selects a supported compact schema."""

    return replay_semantics in DSRL_REPLAY_CONTRACTS


def get_dsrl_replay_field_specs(
    replay_semantics: str = DSRL_REPLAY_SEMANTICS,
) -> Mapping[str, tuple[tuple[int, ...], torch.dtype]]:
    """Return the immutable field specification for ``replay_semantics``."""

    return get_dsrl_replay_contract(replay_semantics)["field_specs"]


def dsrl_replay_bytes_per_transition(
    replay_semantics: str = DSRL_REPLAY_SEMANTICS,
) -> int:
    """Return the exact tensor bytes occupied by one compact transition."""

    total = 0
    for shape, dtype in get_dsrl_replay_field_specs(replay_semantics).values():
        numel = 1
        for dim in shape:
            numel *= dim
        total += numel * torch.empty((), dtype=dtype).element_size()
    return total


def _require_raw_rgb(
    name: str,
    image: object,
    *,
    batch_size: int | None,
    expected_hw: tuple[int, int],
) -> torch.Tensor:
    if not torch.is_tensor(image) or image.ndim != 4:
        shape = tuple(image.shape) if hasattr(image, "shape") else None
        raise ValueError(f"Tabero DSRL replay {name} must be rank-4; got {shape}.")
    if image.dtype != torch.uint8:
        raise ValueError(
            f"Tabero DSRL replay {name} must use uint8; got {image.dtype}."
        )
    if image.shape[-1] != 3:
        raise ValueError(
            f"Tabero DSRL replay {name} must be NHWC RGB; got {tuple(image.shape)}."
        )
    expected_shape = (*expected_hw, 3)
    if tuple(image.shape[1:]) != expected_shape:
        raise ValueError(
            f"Tabero DSRL replay {name} must have trailing shape "
            f"{expected_shape}; got {tuple(image.shape[1:])}."
        )
    if batch_size is not None and image.shape[0] != batch_size:
        raise ValueError(
            f"Tabero DSRL replay {name} batch mismatch: expected {batch_size}, "
            f"got {image.shape[0]}."
        )
    return image


def _require_tensor_shape(
    name: str,
    value: object,
    *,
    expected_shape: tuple[int, ...],
) -> torch.Tensor:
    if not torch.is_tensor(value) or tuple(value.shape) != expected_shape:
        shape = tuple(value.shape) if hasattr(value, "shape") else None
        raise ValueError(
            f"Tabero DSRL replay {name} expected shape {expected_shape}; got {shape}."
        )
    return value


@torch.no_grad()
def compact_tabero_dsrl_observation(
    obs: Mapping[str, object],
    *,
    replay_semantics: str = DSRL_REPLAY_SEMANTICS,
) -> dict[str, torch.Tensor]:
    """Project one raw Tabero observation batch to a compact SAC schema.

    The source observation is not mutated. Each configured RGB view is resized
    independently with the same bilinear operation used by the DSRL model, then
    stored in model-input BF16 to avoid retaining native-resolution images.
    """

    contract = get_dsrl_replay_contract(replay_semantics)
    raw_image_fields = tuple(contract["raw_image_fields"])
    include_marker_motion = bool(contract["include_marker_motion"])
    required = {*raw_image_fields, "states"}
    if include_marker_motion:
        required.add("tactile_marker_motion")
    missing = sorted(required - set(obs))
    if missing:
        raise ValueError(
            f"Tabero DSRL replay observation missing required fields: {missing}."
        )

    states = obs["states"]
    if not torch.is_tensor(states) or states.ndim != 2:
        shape = tuple(states.shape) if hasattr(states, "shape") else None
        raise ValueError(
            f"Tabero DSRL replay states must be rank-2 [B,7]; got {shape}."
        )
    batch_size = int(states.shape[0])
    states = _require_tensor_shape("states", states, expected_shape=(batch_size, 7))
    if not states.is_floating_point():
        raise ValueError(
            f"Tabero DSRL replay states must be floating point; got {states.dtype}."
        )

    raw_image_shapes = contract["raw_image_shapes"]
    raw_images = [
        _require_raw_rgb(
            field,
            obs[field],
            batch_size=batch_size,
            expected_hw=tuple(raw_image_shapes[field]),
        )
        for field in raw_image_fields
    ]

    compact: dict[str, torch.Tensor] = {
        "states": states.to(device="cpu", dtype=torch.bfloat16).contiguous()
    }
    if include_marker_motion:
        tactile = _require_tensor_shape(
            "tactile_marker_motion",
            obs["tactile_marker_motion"],
            expected_shape=(batch_size, 9, 198, 2),
        )
        if not tactile.is_floating_point():
            raise ValueError(
                "Tabero DSRL replay tactile_marker_motion must be floating point; "
                f"got {tactile.dtype}."
            )
        compact["tactile_marker_motion"] = tactile.to(
            device="cpu", dtype=torch.bfloat16
        ).contiguous()

    resized_views = []
    for image in raw_images:
        image_bchw = image.permute(0, 3, 1, 2).to(dtype=torch.float32) / 255.0
        resized = F.interpolate(
            image_bchw,
            size=(DSRL_REPLAY_IMAGE_SIZE, DSRL_REPLAY_IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        resized_views.append((resized * 2.0 - 1.0).to(dtype=torch.bfloat16))
    compact["dsrl_images"] = torch.stack(resized_views, dim=1).cpu().contiguous()
    return compact


def validate_compact_dsrl_observation(
    obs: Mapping[str, object],
    *,
    batch_size: int,
    prefix: str,
    replay_semantics: str = DSRL_REPLAY_SEMANTICS,
) -> None:
    """Validate one flattened or batched compact observation dictionary."""

    field_specs = get_dsrl_replay_field_specs(replay_semantics)
    expected_keys = {
        name.removeprefix(f"{prefix}.")
        for name in field_specs
        if name.startswith(f"{prefix}.")
    }
    actual_keys = set(obs)
    if actual_keys != expected_keys:
        raise ValueError(
            f"Tabero DSRL compact replay {prefix} fields mismatch: expected "
            f"{sorted(expected_keys)}, got {sorted(actual_keys)}."
        )
    for key in sorted(expected_keys):
        shape, dtype = field_specs[f"{prefix}.{key}"]
        tensor = obs[key]
        expected_shape = (batch_size, *shape)
        if not torch.is_tensor(tensor) or tuple(tensor.shape) != expected_shape:
            actual = tuple(tensor.shape) if hasattr(tensor, "shape") else None
            raise ValueError(
                f"Tabero DSRL compact replay {prefix}.{key} expected shape "
                f"{expected_shape}; got {actual}."
            )
        if tensor.dtype != dtype:
            raise ValueError(
                f"Tabero DSRL compact replay {prefix}.{key} expected dtype "
                f"{dtype}; got {tensor.dtype}."
            )
