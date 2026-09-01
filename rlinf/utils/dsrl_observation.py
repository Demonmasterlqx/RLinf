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

"""Versioned Tabero DSRL observation semantics."""

from types import MappingProxyType

DSRL_OBSERVATION_SEMANTICS = "main_wrist_shared_encoder_concat_v1"
DSRL_NUM_IMAGES = 2
DSRL_IMAGE_VIEW_ORDER = ("main", "wrist")

REALWORLD_TACIMG_DSRL_OBSERVATION_SEMANTICS = (
    "main_wrist_tactile_shared_encoder_concat_v1"
)
REALWORLD_TACIMG_DSRL_NUM_IMAGES = 3
REALWORLD_TACIMG_DSRL_IMAGE_VIEW_ORDER = ("main", "wrist", "tactile")

DSRL_OBSERVATION_CONTRACTS = MappingProxyType(
    {
        DSRL_OBSERVATION_SEMANTICS: {
            "num_images": DSRL_NUM_IMAGES,
            "view_order": DSRL_IMAGE_VIEW_ORDER,
            "use_tactile_marker_motion": True,
        },
        REALWORLD_TACIMG_DSRL_OBSERVATION_SEMANTICS: {
            "num_images": REALWORLD_TACIMG_DSRL_NUM_IMAGES,
            "view_order": REALWORLD_TACIMG_DSRL_IMAGE_VIEW_ORDER,
            "use_tactile_marker_motion": False,
        },
    }
)


def get_dsrl_observation_contract(semantics: str) -> dict[str, object]:
    """Return a copy of the requested DSRL observation contract."""

    contract = DSRL_OBSERVATION_CONTRACTS.get(semantics)
    if contract is None:
        raise ValueError(
            f"Unsupported OpenPI DSRL observation semantics {semantics!r}; "
            f"expected one of {sorted(DSRL_OBSERVATION_CONTRACTS)!r}."
        )
    return dict(contract)
