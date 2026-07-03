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

import numpy as np
from openpi import transforms
from openpi.models import model as _model

from rlinf.models.embodiment.openpi.policies.libero_policy import _parse_image


@dataclasses.dataclass(frozen=True)
class TaberoTacImgInputs(transforms.DataTransformFn):
    """Map Tabero flat samples with tactile RGB into pi0 image inputs."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["image"])
        wrist_image = _parse_image(data["wrist_image"])
        tactile_image = _parse_image(data["tactile_image"])

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
        base_image = _parse_image(data["image"])
        wrist_image = _parse_image(data["wrist_image"])

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
class LiberoForceOutputs(transforms.DataTransformFn):
    """Return the 13D Tabero action/force slice."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :13]}
