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

"""Versioned compact replay contract for Tabero tactile DSRL."""

from collections.abc import Mapping
from types import MappingProxyType

import torch
import torch.nn.functional as F

DSRL_REPLAY_SEMANTICS = "main_wrist_bf16_64_transition_ring_v1"
DSRL_REPLAY_BACKEND = "compact_dsrl_ring"
DSRL_REPLAY_FORMAT = "tabero_dsrl_compact_replay"
DSRL_REPLAY_FORMAT_VERSION = 1
DSRL_REPLAY_IMAGE_SIZE = 64
DSRL_REPLAY_NUM_IMAGES = 2
DSRL_REPLAY_VIEW_ORDER = ("main", "wrist")
DSRL_REPLAY_CAPACITY_TRANSITIONS = 100_000
DSRL_REPLAY_CHECKPOINT_SHARD_TRANSITIONS = 4096
DSRL_REPLAY_MAX_RESIDENT_GIB = 12.0

# Trailing shapes and dtypes of one flattened SAC transition. Rewards remain
# FP32 because discounted macro-return arithmetic must not round gamma in BF16.
DSRL_REPLAY_FIELD_SPECS = MappingProxyType(
    {
        "curr_obs.dsrl_images": (
            (DSRL_REPLAY_NUM_IMAGES, 3, DSRL_REPLAY_IMAGE_SIZE, DSRL_REPLAY_IMAGE_SIZE),
            torch.bfloat16,
        ),
        "curr_obs.states": ((7,), torch.bfloat16),
        "curr_obs.tactile_marker_motion": ((9, 198, 2), torch.bfloat16),
        "next_obs.dsrl_images": (
            (DSRL_REPLAY_NUM_IMAGES, 3, DSRL_REPLAY_IMAGE_SIZE, DSRL_REPLAY_IMAGE_SIZE),
            torch.bfloat16,
        ),
        "next_obs.states": ((7,), torch.bfloat16),
        "next_obs.tactile_marker_motion": ((9, 198, 2), torch.bfloat16),
        "actions": ((32,), torch.bfloat16),
        "rewards": ((10,), torch.float32),
        "terminations": ((10,), torch.bool),
        "truncations": ((10,), torch.bool),
    }
)


def dsrl_replay_bytes_per_transition() -> int:
    """Return the exact tensor bytes occupied by one compact transition."""

    total = 0
    for shape, dtype in DSRL_REPLAY_FIELD_SPECS.values():
        numel = 1
        for dim in shape:
            numel *= dim
        total += numel * torch.empty((), dtype=dtype).element_size()
    return total


def _require_raw_rgb(
    name: str, image: object, *, batch_size: int | None
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
    if tuple(image.shape[1:]) != (256, 256, 3):
        raise ValueError(
            f"Tabero DSRL replay {name} must have trailing shape (256, 256, 3); "
            f"got {tuple(image.shape[1:])}."
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
) -> dict[str, torch.Tensor]:
    """Project one raw Tabero observation batch to the SAC-only replay schema.

    The source observation is not mutated. Main and wrist images are resized
    independently with the same bilinear operation used by the DSRL model, then
    stored in model-input BF16 to avoid retaining 256x256 raw images.
    """

    required = {
        "main_images",
        "wrist_images",
        "states",
        "tactile_marker_motion",
    }
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

    main = _require_raw_rgb("main_images", obs["main_images"], batch_size=batch_size)
    wrist = _require_raw_rgb("wrist_images", obs["wrist_images"], batch_size=batch_size)
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

    resized_views = []
    for image in (main, wrist):
        image_bchw = image.permute(0, 3, 1, 2).to(dtype=torch.float32) / 255.0
        resized = F.interpolate(
            image_bchw,
            size=(DSRL_REPLAY_IMAGE_SIZE, DSRL_REPLAY_IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        resized_views.append((resized * 2.0 - 1.0).to(dtype=torch.bfloat16))

    return {
        "dsrl_images": torch.stack(resized_views, dim=1).cpu().contiguous(),
        "states": states.to(device="cpu", dtype=torch.bfloat16).contiguous(),
        "tactile_marker_motion": tactile.to(
            device="cpu", dtype=torch.bfloat16
        ).contiguous(),
    }


def validate_compact_dsrl_observation(
    obs: Mapping[str, object],
    *,
    batch_size: int,
    prefix: str,
) -> None:
    """Validate one flattened or batched compact observation dictionary."""

    expected_keys = {"dsrl_images", "states", "tactile_marker_motion"}
    actual_keys = set(obs)
    if actual_keys != expected_keys:
        raise ValueError(
            f"Tabero DSRL compact replay {prefix} fields mismatch: expected "
            f"{sorted(expected_keys)}, got {sorted(actual_keys)}."
        )
    for key in sorted(expected_keys):
        shape, dtype = DSRL_REPLAY_FIELD_SPECS[f"{prefix}.{key}"]
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
