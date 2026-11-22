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

import dataclasses

import cv2
import numpy as np
from openpi import transforms
from openpi.models import model as _model

from rlinf.models.embodiment.openpi.policies.libero_policy import _parse_image

TABERO_CAMERA_TARGET_HW = (224, 224)


def stretch_camera_image_to_224(image: object) -> np.ndarray:
    """Match the RealWorld OpenPI client's no-padding INTER_AREA resize."""

    parsed = _parse_image(image)
    if parsed.ndim != 3 or parsed.shape[-1] != 3:
        raise ValueError(
            f"Tabero camera image must have HWC RGB shape; got {parsed.shape}."
        )
    if parsed.shape[:2] == TABERO_CAMERA_TARGET_HW:
        return parsed
    return cv2.resize(
        parsed,
        (TABERO_CAMERA_TARGET_HW[1], TABERO_CAMERA_TARGET_HW[0]),
        interpolation=cv2.INTER_AREA,
    )


@dataclasses.dataclass(frozen=True)
class TaberoVisionInputs(transforms.DataTransformFn):
    """Use only the two RGB cameras; retain state for action transforms."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = stretch_camera_image_to_224(data["image"])
        wrist_image = stretch_camera_image_to_224(data["wrist_image"])
        inputs = {
            "state": data["state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": (
                    np.True_
                    if self.model_type == _model.ModelType.PI0_FAST
                    else np.False_
                ),
            },
        }
        for key in ("actions", "prompt"):
            if key in data:
                inputs[key] = data[key]
        return inputs


@dataclasses.dataclass(frozen=True)
class TaberoTacImgInputs(transforms.DataTransformFn):
    """Map Tabero flat samples with tactile RGB into pi0 image inputs."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = stretch_camera_image_to_224(data["image"])
        wrist_image = stretch_camera_image_to_224(data["wrist_image"])
        tactile_image = stretch_camera_image_to_224(data["tactile_image"])

        inputs = {
            "state": data["state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": tactile_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class TaberoTacFieldInputs(transforms.DataTransformFn):
    """Map Tabero marker motion into the encoder-prefix tactile stream."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = stretch_camera_image_to_224(data["image"])
        wrist_image = stretch_camera_image_to_224(data["wrist_image"])

        right_image = np.zeros_like(base_image)
        right_mask = (
            np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_
        )
        inputs = {
            "state": data["state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": right_mask,
            },
        }

        if "tactile_marker_motion" not in data:
            raise KeyError("TaberoTacFieldInputs expects 'tactile_marker_motion'.")
        motion = np.asarray(data["tactile_marker_motion"])
        if motion.ndim != 3:
            raise ValueError(
                f"tactile_marker_motion must be 3D, got shape {motion.shape}."
            )
        n, markers, xy = motion.shape
        inputs["tactile_prefix"] = motion.reshape(n, markers * xy)

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class TaberoTacCoRLInputs(TaberoTacFieldInputs):
    """Add a contact gate from raw marker pixels before normalization.

    Frame zero is the reference; contact requires enough displaced markers
    in any one of the following eight frames. The pixel comparison is strict.
    """

    contact_threshold: float = 1.0
    contact_min_active: int = 5

    def __post_init__(self) -> None:
        if not np.isfinite(self.contact_threshold) or self.contact_threshold < 0:
            raise ValueError("contact_threshold must be finite and non-negative.")
        if (
            isinstance(self.contact_min_active, bool)
            or not isinstance(self.contact_min_active, int)
            or self.contact_min_active <= 0
        ):
            raise ValueError("contact_min_active must be a positive integer.")

    def __call__(self, data: dict) -> dict:
        inputs = super().__call__(data)
        # Cast before subtraction: unsigned pixel coordinates can wrap around.
        motion = np.asarray(data["tactile_marker_motion"], dtype=np.float64)
        if motion.shape[0] != 9 or motion.shape[1] == 0 or motion.shape[2] != 2:
            raise ValueError(
                "TacCoRL requires tactile_marker_motion shape (9, markers, 2); "
                f"got {motion.shape}."
            )
        displacement = np.linalg.norm(motion[1:] - motion[0], axis=-1)
        active = np.count_nonzero(displacement > self.contact_threshold, axis=-1)
        # Keep the scalar as a zero-dimensional ndarray so torch's default
        # collate treats a batch of contact gates as booleans rather than
        # interpreting ``np.bool_`` scalars as indices.
        inputs["tactile_contact"] = np.asarray(
            np.any(active >= self.contact_min_active), dtype=np.bool_
        )
        return inputs


@dataclasses.dataclass(frozen=True)
class TaberoTacForceInputs(transforms.DataTransformFn):
    """Map an 8x6 measured gripper-force history to one TCN prefix stream."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = stretch_camera_image_to_224(data["image"])
        wrist_image = stretch_camera_image_to_224(data["wrist_image"])

        right_image = np.zeros_like(base_image)
        right_mask = (
            np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_
        )
        inputs = {
            "state": data["state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": right_mask,
            },
        }

        if "tactile_gripper_force" not in data:
            raise KeyError("TaberoTacForceInputs expects 'tactile_gripper_force'.")
        force_history = np.asarray(data["tactile_gripper_force"])
        if force_history.shape != (8, 6):
            raise ValueError(
                "tactile_gripper_force must have shape (8, 6), got "
                f"{force_history.shape}."
            )
        inputs["tactile_prefix"] = force_history

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoForceOutputs(transforms.DataTransformFn):
    """Return the 13D Tabero action/force slice."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :13]}
