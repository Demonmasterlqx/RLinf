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

import numpy as np
import pytest
import torch
from openpi.models import model as _model

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0Config,
    OpenPi0ForRLActionPrediction,
)
from rlinf.models.embodiment.openpi.policies.tabero_policy import (
    LiberoForceOutputs,
    TaberoTacFieldInputs,
    TaberoTacImgInputs,
)
from rlinf.models.embodiment.openpi.tactile_encoder import TactileTCNEncoder
from rlinf.utils.ckpt_convertor.convert_tabero_openpi_jax_to_safetensors import (
    convert_tactile_prefix_encoder_params,
    fold_lora_into_base_weights,
)


def _image(value: int = 0) -> np.ndarray:
    return np.full((224, 224, 3), value, dtype=np.uint8)


def test_tabero_tacimg_inputs_map_three_images_and_force_outputs():
    transform = TaberoTacImgInputs(model_type=_model.ModelType.PI0)
    data = {
        "image": _image(1),
        "wrist_image": _image(2),
        "tactile_image": _image(3),
        "state": np.arange(16, dtype=np.float32),
        "actions": np.arange(50 * 16, dtype=np.float32).reshape(50, 16),
        "prompt": "stack the cube",
    }

    out = transform(data)

    assert out["image"]["base_0_rgb"][0, 0, 0] == 1
    assert out["image"]["left_wrist_0_rgb"][0, 0, 0] == 2
    assert out["image"]["right_wrist_0_rgb"][0, 0, 0] == 3
    assert out["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.True_,
        "right_wrist_0_rgb": np.True_,
    }
    assert out["state"].shape == (16,)
    assert out["actions"].shape == (50, 16)
    assert out["prompt"] == "stack the cube"

    action_batch = np.arange(2 * 50 * 16, dtype=np.float32).reshape(2, 50, 16)
    action_out = LiberoForceOutputs()({"actions": action_batch})
    assert action_out["actions"].shape == (2, 50, 13)
    np.testing.assert_array_equal(action_out["actions"], action_batch[:, :, :13])


def test_tabero_tacfield_inputs_require_marker_motion_and_build_prefix():
    transform = TaberoTacFieldInputs(model_type=_model.ModelType.PI0)
    data = {
        "image": _image(1),
        "wrist_image": _image(2),
        "state": np.arange(16, dtype=np.float32),
        "tactile_marker_motion": np.arange(9 * 198 * 2, dtype=np.float32).reshape(
            9, 198, 2
        ),
        "actions": np.zeros((50, 16), dtype=np.float32),
        "prompt": "lift the mug",
    }

    out = transform(data)

    assert out["image"]["right_wrist_0_rgb"].shape == (224, 224, 3)
    assert out["image_mask"]["right_wrist_0_rgb"] == np.False_
    assert out["tactile_prefix"].shape == (9, 396)
    np.testing.assert_array_equal(
        out["tactile_prefix"], data["tactile_marker_motion"].reshape(9, 396)
    )

    data_without_marker = {k: v for k, v in data.items() if k != "tactile_marker_motion"}
    with pytest.raises(KeyError, match="tactile_marker_motion"):
        transform(data_without_marker)


def test_tabero_openpi_configs_are_registered_with_extended_config():
    tacimg = get_openpi_config("pi0_lora_tacimg_tabero")
    tacfield = get_openpi_config("pi0_lora_tacfield_tabero")

    assert isinstance(tacimg.model, OpenPi0Config)
    assert tacimg.model.paligemma_variant == "gemma_2b_lora"
    assert tacimg.model.action_expert_variant == "gemma_300m_lora"
    assert tacimg.model.action_dim == 32
    assert tacimg.model.effective_action_dim == 13
    assert tacimg.model.tactile_dim_in == 0
    assert tacimg.data.repo_id == "NathanWu7/tabero"

    assert isinstance(tacfield.model, OpenPi0Config)
    assert tacfield.model.action_dim == 32
    assert tacfield.model.effective_action_dim == 13
    assert tacfield.model.tactile_dim_in == 0
    assert tacfield.model.tactile_prefix_dim_in == 9 * 198 * 2
    assert tacfield.model.tactile_prefix_history == 8
    assert tacfield.model.tactile_prefix_encoder_type == "tcn"
    assert tacfield.model.tactile_streams == ("tactile_prefix",)
    assert tacfield.data.repo_id == "NathanWu7/tabero_object_25"


def test_tactile_prefix_encoder_has_own_fsdp_wrap_name():
    model = OpenPi0ForRLActionPrediction.__new__(OpenPi0ForRLActionPrediction)

    assert "tactile_prefix_encoder" in model._no_split_names


def test_tabero_input_transform_keeps_flat_keys_without_prompt():
    model = OpenPi0ForRLActionPrediction.__new__(OpenPi0ForRLActionPrediction)
    tabero_transform = TaberoTacFieldInputs(model_type=_model.ModelType.PI0)
    model._input_transform = lambda data: {
        key: value for key, value in tabero_transform(data).items() if key != "prompt"
    }
    obs = {
        "image": torch.zeros(1, 224, 224, 3, dtype=torch.uint8),
        "wrist_image": torch.zeros(1, 224, 224, 3, dtype=torch.uint8),
        "state": torch.zeros(1, 16),
        "tactile_marker_motion": torch.zeros(1, 9, 198, 2),
        "chains": torch.zeros(1, 2, 5, 32),
        "denoise_inds": torch.zeros(1, 2, dtype=torch.long),
        "tokenized_prompt": torch.ones(1, 8, dtype=torch.long),
        "tokenized_prompt_mask": torch.ones(1, 8, dtype=torch.bool),
    }

    out = model.input_transform(obs, transpose=False)

    assert out["tactile_prefix"].shape == (1, 9, 396)
    assert "chains" not in out
    assert "denoise_inds" not in out
    torch.testing.assert_close(out["tokenized_prompt"], obs["tokenized_prompt"])
    torch.testing.assert_close(
        out["tokenized_prompt_mask"], obs["tokenized_prompt_mask"]
    )


def test_tactile_tcn_encoder_outputs_single_prefix_token():
    encoder = TactileTCNEncoder(
        input_dim=396,
        hidden_dim=32,
        output_dim=24,
        history_len=8,
        has_reference_frame=True,
        diff_from_reference=False,
    )
    tactile = torch.randn(3, 9, 396)

    out = encoder(tactile)

    assert out.shape == (3, 24)


def test_converter_folds_lora_weights_into_base_einsum():
    state_dict = {
        "llm/layers/attn/q_einsum/w": np.ones((2, 3, 4), dtype=np.float32),
        "llm/layers/attn/q_einsum/lora_a": np.full(
            (2, 3, 2), 2.0, dtype=np.float32
        ),
        "llm/layers/attn/q_einsum/lora_b": np.full(
            (2, 2, 4), 3.0, dtype=np.float32
        ),
        "llm/layers/mlp/linear": np.ones((5, 6), dtype=np.float32),
        "llm/layers/mlp/linear_lora_a": np.full((5, 2), 4.0, dtype=np.float32),
        "llm/layers/mlp/linear_lora_b": np.full((2, 6), 5.0, dtype=np.float32),
    }

    fold_lora_into_base_weights(state_dict)

    np.testing.assert_array_equal(
        state_dict["llm/layers/attn/q_einsum/w"],
        np.ones((2, 3, 4), dtype=np.float32) + 12.0,
    )
    np.testing.assert_array_equal(
        state_dict["llm/layers/mlp/linear"],
        np.ones((5, 6), dtype=np.float32) + 40.0,
    )
    assert not any("lora" in key for key in state_dict)


def test_converter_maps_tactile_prefix_encoder_params():
    params = {
        "tactile_prefix_encoder": {
            "blocks": {
                "block_0": {
                    "kernels": {
                        "kernel_0": {
                            "kernel": np.ones((2, 3), dtype=np.float32),
                            "bias": np.arange(3, dtype=np.float32),
                        }
                    },
                    "residual_proj": {
                        "kernel": np.ones((2, 3), dtype=np.float32) * 2,
                        "bias": np.arange(3, dtype=np.float32) + 1,
                    },
                },
                "block_1": {
                    "kernels": {
                        "kernel_0": {
                            "kernel": np.ones((3, 3), dtype=np.float32) * 3,
                            "bias": np.arange(3, dtype=np.float32) + 2,
                        }
                    }
                },
            },
            "out_proj": {
                "kernel": np.ones((3, 4), dtype=np.float32) * 4,
                "bias": np.arange(4, dtype=np.float32),
            },
        }
    }

    torch_params = convert_tactile_prefix_encoder_params(params)

    assert torch_params["tactile_prefix_encoder.blocks.0.kernels.0.weight"].shape == (
        3,
        2,
    )
    assert torch_params["tactile_prefix_encoder.blocks.0.residual_proj.weight"].shape == (
        3,
        2,
    )
    assert torch_params["tactile_prefix_encoder.blocks.1.kernels.0.weight"].shape == (
        3,
        3,
    )
    assert torch_params["tactile_prefix_encoder.out_proj.weight"].shape == (4, 3)
