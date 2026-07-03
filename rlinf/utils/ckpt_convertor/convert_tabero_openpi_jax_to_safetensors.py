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

"""Convert Tabero T2-VLA OpenPI Orbax checkpoints to RLinf safetensors."""

import dataclasses
import json
import os
import pathlib
import shutil
from typing import Literal

import numpy as np
import openpi.models.gemma
import safetensors.torch
import torch
import tyro

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0Config,
    OpenPi0ForRLActionPrediction,
)
from rlinf.utils.ckpt_convertor.convert_openpi_jax_to_python import (
    slice_gemma_state_dict,
    slice_initial_orbax_checkpoint,
    slice_paligemma_state_dict,
)


def _base_key_from_lora_a(key: str) -> tuple[str, str] | None:
    suffix = "/value" if key.endswith("/value") else ""
    stem = key[: -len(suffix)] if suffix else key
    if stem.endswith("_lora_a"):
        base_stem = stem[: -len("_lora_a")]
        return base_stem + suffix, base_stem + "_lora_b" + suffix
    if stem.endswith("/lora_a"):
        base_stem = stem[: -len("/lora_a")]
        return base_stem + "/w" + suffix, base_stem + "/lora_b" + suffix
    return None


def fold_lora_into_base_weights(state_dict: dict[str, np.ndarray]) -> None:
    """Fold JAX LoRA weights into their base einsum weights in-place."""
    for key in list(state_dict):
        if "lora_a" not in key:
            continue
        mapped = _base_key_from_lora_a(key)
        if mapped is None:
            continue
        base_key, lora_b_key = mapped
        if base_key not in state_dict:
            raise KeyError(f"Missing base weight '{base_key}' for LoRA key '{key}'.")
        if lora_b_key not in state_dict:
            raise KeyError(f"Missing LoRA B weight '{lora_b_key}' for '{key}'.")

        lora_a = np.asarray(state_dict.pop(key))
        lora_b = np.asarray(state_dict.pop(lora_b_key))
        update = np.einsum("...ar,...rb->...ab", lora_a, lora_b)
        if update.shape != np.asarray(state_dict[base_key]).shape:
            raise ValueError(
                f"LoRA update shape {update.shape} does not match base "
                f"{base_key} shape {np.asarray(state_dict[base_key]).shape}."
            )
        state_dict[base_key] = np.asarray(state_dict[base_key]) + update


def _as_array(value):
    return value["value"] if isinstance(value, dict) else value


def _linear_weight(kernel, bias, key: str) -> torch.Tensor:
    kernel = np.asarray(_as_array(kernel))
    bias = np.asarray(_as_array(bias))
    if kernel.ndim != 2:
        raise ValueError(f"{key}.kernel must be rank 2, got shape {kernel.shape}.")
    if kernel.shape[1] != bias.shape[0]:
        raise ValueError(
            f"{key}.kernel shape {kernel.shape} is incompatible with bias "
            f"shape {bias.shape}; expected kernel second dim == bias dim."
        )
    return torch.from_numpy(kernel).T


def _linear_bias(bias) -> torch.Tensor:
    return torch.from_numpy(np.asarray(_as_array(bias)))


def convert_tactile_prefix_encoder_params(params: dict) -> dict[str, torch.Tensor]:
    """Convert T2-VLA tactile_prefix_encoder params to RLinf PyTorch keys."""
    if "tactile_prefix_encoder" not in params:
        return {}

    encoder = params["tactile_prefix_encoder"]
    converted: dict[str, torch.Tensor] = {}
    for block_name in sorted(encoder["blocks"]):
        block_idx = int(block_name.split("_")[1])
        block = encoder["blocks"][block_name]
        for kernel_name in sorted(block["kernels"]):
            kernel_idx = int(kernel_name.split("_")[1])
            linear = block["kernels"][kernel_name]
            prefix = f"tactile_prefix_encoder.blocks.{block_idx}.kernels.{kernel_idx}"
            converted[f"{prefix}.weight"] = _linear_weight(
                linear["kernel"], linear["bias"], prefix
            )
            converted[f"{prefix}.bias"] = _linear_bias(linear["bias"])

        if "residual_proj" in block:
            prefix = f"tactile_prefix_encoder.blocks.{block_idx}.residual_proj"
            residual = block["residual_proj"]
            converted[f"{prefix}.weight"] = _linear_weight(
                residual["kernel"], residual["bias"], prefix
            )
            converted[f"{prefix}.bias"] = _linear_bias(residual["bias"])

    out_proj = encoder["out_proj"]
    converted["tactile_prefix_encoder.out_proj.weight"] = _linear_weight(
        out_proj["kernel"], out_proj["bias"], "tactile_prefix_encoder.out_proj"
    )
    converted["tactile_prefix_encoder.out_proj.bias"] = _linear_bias(
        out_proj["bias"]
    )
    return converted


def _projection_params(params: dict, model_config: OpenPi0Config) -> dict[str, torch.Tensor]:
    keys = (
        ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]
        if model_config.pi05
        else [
            "state_proj",
            "action_in_proj",
            "action_out_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ]
    )
    converted = {}
    for key in keys:
        converted[f"{key}.weight"] = _linear_weight(
            params[key]["kernel"], params[key]["bias"], key
        )
        converted[f"{key}.bias"] = _linear_bias(params[key]["bias"])
    return converted


class _PaliGemmaConfig:
    def __init__(self):
        self.vision_config = type(
            "obj",
            (object,),
            {
                "hidden_size": 1152,
                "num_hidden_layers": 27,
                "num_attention_heads": 16,
                "intermediate_size": 4304,
                "patch_size": 14,
                "projection_dim": 2048,
            },
        )()
        self.text_config = type(
            "obj",
            (object,),
            {
                "hidden_size": 2048,
                "num_hidden_layers": 18,
                "num_attention_heads": 8,
                "head_dim": 256,
                "intermediate_size": 16384,
            },
        )()


def _copy_assets(checkpoint_dir: str, output_path: str) -> None:
    checkpoint_path = pathlib.Path(checkpoint_dir)
    candidates = [checkpoint_path / "assets", checkpoint_path.parent / "assets"]
    assets_source = next((path for path in candidates if path.exists()), None)
    if assets_source is None:
        return

    assets_dest = pathlib.Path(output_path) / "assets"
    if assets_dest.exists():
        shutil.rmtree(assets_dest)
    shutil.copytree(assets_source, assets_dest, symlinks=True)

    # RLinf's OpenPI loader calls load_norm_stats(checkpoint_dir, asset_id), so
    # it expects checkpoint_dir/NathanWu7/... in addition to the OpenPI assets/
    # layout. Mirror the asset tree at the output root for that loader path.
    output_root = pathlib.Path(output_path)
    for child in assets_source.iterdir():
        mirrored_child = output_root / child.name
        if mirrored_child.exists() or mirrored_child.is_symlink():
            if mirrored_child.is_dir() and not mirrored_child.is_symlink():
                shutil.rmtree(mirrored_child)
            else:
                mirrored_child.unlink()
        if child.is_dir():
            shutil.copytree(child, mirrored_child, symlinks=True)
        elif child.is_symlink():
            os.symlink(os.readlink(child), mirrored_child)
        else:
            shutil.copy2(child, mirrored_child)


def convert_pi0_checkpoint(
    checkpoint_dir: str,
    config_name: str,
    precision: Literal["float32", "bfloat16"],
    output_path: str,
) -> None:
    train_config = get_openpi_config(config_name, model_path=output_path)
    model_config = train_config.model
    if not isinstance(model_config, OpenPi0Config):
        raise ValueError(f"Config {config_name} must use OpenPi0Config.")

    initial_params = slice_initial_orbax_checkpoint(
        checkpoint_dir=checkpoint_dir, restore_precision="float32"
    )
    paligemma_flat = initial_params["paligemma_params"]
    fold_lora_into_base_weights(paligemma_flat)

    paligemma_params, expert_params = slice_paligemma_state_dict(
        paligemma_flat, _PaliGemmaConfig()
    )
    action_expert_config = openpi.models.gemma.get_config(
        model_config.action_expert_variant
    )
    gemma_params = slice_gemma_state_dict(
        expert_params,
        action_expert_config,
        num_expert=1,
        checkpoint_dir=checkpoint_dir,
        pi05=model_config.pi05,
    )
    projection_params = _projection_params(
        initial_params["projection_params"], model_config
    )
    tactile_params = convert_tactile_prefix_encoder_params(
        initial_params["projection_params"]
    )

    model = OpenPi0ForRLActionPrediction(model_config)
    all_params = {
        **paligemma_params,
        **gemma_params,
        **projection_params,
        **tactile_params,
    }
    missing_keys, unexpected_keys = model.load_state_dict(all_params, strict=False)
    critical_missing = [
        key
        for key in missing_keys
        if key.startswith("tactile_prefix_encoder")
        or key
        in {
            "action_in_proj.weight",
            "action_in_proj.bias",
            "action_out_proj.weight",
            "action_out_proj.bias",
            "state_proj.weight",
            "state_proj.bias",
        }
    ]
    if critical_missing:
        raise RuntimeError(f"Critical keys were not loaded: {critical_missing}")
    if unexpected_keys:
        raise RuntimeError(f"Unexpected converted keys: {unexpected_keys[:20]}")

    if precision == "float32":
        model = model.to(torch.float32)
    elif precision == "bfloat16":
        model = model.to(torch.bfloat16)
    else:
        raise ValueError(f"Unsupported precision: {precision}")

    os.makedirs(output_path, exist_ok=True)
    safetensors.torch.save_model(model, os.path.join(output_path, "model.safetensors"))
    _copy_assets(checkpoint_dir, output_path)

    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(model_config), f, indent=2)

    print(f"Saved converted model to {output_path}")
    if missing_keys:
        print(f"Non-critical missing keys: {len(missing_keys)}")


def main(
    checkpoint_dir: str,
    config_name: str,
    output_path: str,
    precision: Literal["float32", "bfloat16"] = "bfloat16",
) -> None:
    convert_pi0_checkpoint(
        checkpoint_dir=checkpoint_dir,
        config_name=config_name,
        precision=precision,
        output_path=output_path,
    )


if __name__ == "__main__":
    tyro.cli(main)
