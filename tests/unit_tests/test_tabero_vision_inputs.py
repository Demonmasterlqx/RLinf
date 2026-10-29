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

"""Synthetic vision-only observation routing; no checkpoints or simulator."""

from types import SimpleNamespace

import numpy as np
import pytest

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0ForRLActionPrediction,
)
from rlinf.models.embodiment.openpi.policies.tabero_policy import (
    LiberoForceOutputs,
    TaberoVisionInputs,
)


@pytest.mark.parametrize("include_tactile", [False, True])
def test_vision_routing_excludes_all_tactile_inputs(include_tactile):
    config = get_openpi_config("pi05_lora_vision_realworld_replayed_task820")
    model = OpenPi0ForRLActionPrediction.__new__(OpenPi0ForRLActionPrediction)
    model.config = SimpleNamespace(config_name=config.name)
    env_obs = {
        "main_images": np.full((480, 640, 3), 20, dtype=np.uint8),
        "wrist_images": np.full((480, 640, 3), 80, dtype=np.uint8),
        "states": np.arange(7, dtype=np.float32),
        "task_descriptions": "pick up the Vitasoy and put it into the basket",
    }
    if include_tactile:
        # Poison values catch accidental use of any tactile channel.
        env_obs.update(
            tactile_images=object(),
            tactile_marker_motion=object(),
            tactile_gripper_force=object(),
        )
    processed = model.obs_processor(env_obs)
    assert set(processed) == {"image", "wrist_image", "state", "prompt"}
    processed.update(
        tactile_image=object(),
        tactile_marker_motion=object(),
        tactile_gripper_force=object(),
        actions=np.arange(50 * 13, dtype=np.float32).reshape(50, 13),
    )
    transformed = TaberoVisionInputs(model_type=config.model.model_type)(processed)
    assert set(transformed) == {"state", "image", "image_mask", "actions", "prompt"}
    assert tuple(transformed["image_mask"].values()) == (True, True, False)
    assert transformed["state"] is env_obs["states"]
    assert transformed["actions"] is processed["actions"]
    for key, value in (
        ("base_0_rgb", 20),
        ("left_wrist_0_rgb", 80),
        ("right_wrist_0_rgb", 0),
    ):
        image = transformed["image"][key]
        assert image.shape == (224, 224, 3)
        np.testing.assert_array_equal(image, np.full_like(image, value))

    # Removing force observations must not remove the six target-force outputs.
    padded = np.pad(transformed["actions"], ((0, 0), (0, 19)))
    np.testing.assert_array_equal(
        LiberoForceOutputs()({"actions": padded})["actions"], processed["actions"]
    )
