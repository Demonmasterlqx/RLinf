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

"""Contact gate boundaries using synthetic raw pixel coordinates only."""

import numpy as np
import pytest
import torch
from openpi.models import model
from torch.utils.data import default_collate

from rlinf.models.embodiment.openpi.policies.tabero_policy import TaberoTacCoRLInputs


def _sample(motion):
    return {
        "image": np.zeros((224, 224, 3), dtype=np.uint8),
        "wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "state": np.zeros(13, dtype=np.float32),
        "actions": np.zeros((50, 13), dtype=np.float32),
        "prompt": "grasp",
        "tactile_marker_motion": motion,
    }


@pytest.mark.parametrize(
    "displacement,count,expected",
    [(0.0, 6, False), (1.0, 5, False), (1.00001, 4, False), (1.00001, 5, True)],
)
def test_gate_strict_pixel_threshold_and_inclusive_marker_count(
    displacement, count, expected
):
    motion = np.zeros((9, 6, 2), dtype=np.float32)
    motion[8, :count, 0] = displacement
    result = TaberoTacCoRLInputs(model.ModelType.PI0)(_sample(motion))
    assert isinstance(result["tactile_contact"], np.ndarray)
    assert result["tactile_contact"].shape == ()
    assert result["tactile_contact"].dtype == np.bool_
    assert bool(result["tactile_contact"]) is expected
    np.testing.assert_array_equal(result["tactile_prefix"], motion.reshape(9, 12))
    assert result["actions"].shape == (50, 13)


def test_contact_zero_dim_arrays_default_collate_to_bool_batch():
    samples = [
        np.asarray(False, dtype=np.bool_),
        np.asarray(True, dtype=np.bool_),
        np.asarray(False, dtype=np.bool_),
    ]

    batch = default_collate(samples)

    assert batch.dtype == torch.bool
    assert batch.shape == (3,)
    torch.testing.assert_close(batch, torch.tensor([False, True, False]))


def test_contact_requires_five_markers_in_the_same_frame():
    motion = np.zeros((9, 6, 2), dtype=np.float32)
    motion[1, :3, 0] = 2.0
    motion[2, 3:, 0] = 2.0
    result = TaberoTacCoRLInputs(model.ModelType.PI0)(_sample(motion))
    assert not result["tactile_contact"]


def test_gate_uses_euclidean_norm_and_configurable_raw_pixel_threshold():
    motion = np.zeros((9, 6, 2), dtype=np.float32)
    motion[4, :2] = (3, 4)
    gate = TaberoTacCoRLInputs(
        model.ModelType.PI0, contact_threshold=5, contact_min_active=2
    )
    assert not gate(_sample(motion))["tactile_contact"]
    gate = TaberoTacCoRLInputs(
        model.ModelType.PI0, contact_threshold=4.9, contact_min_active=2
    )
    assert gate(_sample(motion))["tactile_contact"]


def test_gate_subtracts_reference_before_comparing_unsigned_pixels():
    motion = np.full((9, 6, 2), 100, dtype=np.uint8)
    motion[1, :5, 0] = 99
    gate = TaberoTacCoRLInputs(model.ModelType.PI0)
    assert not gate(_sample(motion))["tactile_contact"]
    motion[1, :5, 0] = 98
    assert gate(_sample(motion))["tactile_contact"]


@pytest.mark.parametrize("shape", [(8, 6, 2), (10, 6, 2), (9, 6, 3), (9, 0, 2)])
def test_gate_requires_reference_and_eight_xy_frames(shape):
    with pytest.raises(ValueError, match="shape"):
        TaberoTacCoRLInputs(model.ModelType.PI0)(_sample(np.zeros(shape)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"contact_threshold": -1},
        {"contact_threshold": float("nan")},
        {"contact_min_active": 0},
        {"contact_min_active": 1.5},
    ],
)
def test_invalid_gate_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        TaberoTacCoRLInputs(model.ModelType.PI0, **kwargs)
