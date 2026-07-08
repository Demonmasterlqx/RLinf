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

"""Export an RLinf OpenPI PEFT-LoRA checkpoint for the T2-VLA OpenPI server.

RLinf can inject PEFT LoRA into a selected OpenPI submodule, while T2-VLA's
``scripts/serve_policy.py`` loads plain PyTorch safetensors. This utility loads
either an RLinf FSDP full-weights checkpoint or a lightweight
``trainable_weights.pt`` checkpoint, saves the trained LoRA adapter for
inspection, merges the adapter back into the selected submodule, drops RL-only
heads, and writes T2-VLA-compatible safetensors plus OpenPI assets.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from collections.abc import Mapping
from pathlib import Path
import shutil

import safetensors.torch
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from rlinf.models import get_model


RL_ONLY_PREFIXES = (
    "value_head.",
    "noise_head.",
    "dsrl_",
    "actor_image_encoder.",
    "actor_state_encoder.",
    "critic_image_encoder.",
    "critic_state_encoder.",
    "q_head.",
)

DROP_EXACT_KEYS = {
    # PEFT may materialize this tied embedding after merge, while the T2-VLA
    # PyTorch PI0 wrapper does not expose it as a loadable state_dict key.
    "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight",
}

ACTION_EXPERT_OVERLAY_PREFIXES = (
    "paligemma_with_expert.gemma_expert.",
    "action_in_proj.",
    "action_out_proj.",
    "state_proj.",
    "action_time_mlp",
    "time_mlp_",
    "tactile_prefix_encoder.",
)


def _load_model_cfg(train_config_path: str) -> DictConfig:
    train_cfg = OmegaConf.load(train_config_path)
    if "actor" not in train_cfg or "model" not in train_cfg.actor:
        raise KeyError(f"actor.model not found in {train_config_path}")
    return OmegaConf.create(OmegaConf.to_container(train_cfg.actor.model, resolve=True))


def _extract_state_dict(checkpoint) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"Expected mapping checkpoint, got {type(checkpoint).__name__}."
        )

    candidates = [
        ("fsdp_checkpoint", "model"),
        ("state_dict",),
        ("model_state_dict",),
        ("model",),
        ("module",),
    ]
    for path in candidates:
        current = checkpoint
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                break
            current = current[key]
        else:
            if isinstance(current, Mapping) and any(
                torch.is_tensor(v) for v in current.values()
            ):
                return current

    if any(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    raise KeyError("Could not locate tensor weights in checkpoint.")


def _checkpoint_metadata(checkpoint) -> Mapping:
    if isinstance(checkpoint, Mapping) and isinstance(
        checkpoint.get("metadata"), Mapping
    ):
        return checkpoint["metadata"]
    return {}


def _normalize_state_dict_keys(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        name = key
        for prefix in ("_orig_mod.", "module.", "_fsdp_wrapped_module."):
            while name.startswith(prefix):
                name = name[len(prefix) :]
        name = name.replace("._fsdp_wrapped_module.", ".")
        normalized[name] = value
    return normalized


def _copy_assets(model_path: str, output_dir: str) -> None:
    source_root = Path(model_path)
    output_root = Path(output_dir)

    assets_source = source_root / "assets"
    if assets_source.exists():
        assets_dest = output_root / "assets"
        if assets_dest.exists():
            shutil.rmtree(assets_dest)
        shutil.copytree(assets_source, assets_dest, symlinks=True)

    # RLinf converted OpenPI checkpoints also mirror asset_id directories at the
    # checkpoint root; T2-VLA expects stats under assets/, but mirroring keeps the
    # export usable by both loaders.
    for child in source_root.iterdir():
        if child.name in {"assets", "model.safetensors", "config.json"}:
            continue
        if not child.is_dir():
            continue
        dest = output_root / child.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(child, dest, symlinks=True)


def _json_safe(value):
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _save_filtered_safetensors(model: torch.nn.Module, output_path: str) -> None:
    state_dict = {}
    for key, value in model.state_dict().items():
        if key in DROP_EXACT_KEYS:
            continue
        if key.startswith(RL_ONLY_PREFIXES):
            continue
        if "lora_" in key:
            raise RuntimeError(f"LoRA key remained after merge: {key}")
        # Clone to avoid safetensors shared-storage errors from tied weights.
        state_dict[key] = value.detach().cpu().contiguous().clone()
    safetensors.torch.save_file(state_dict, output_path)


def _filter_action_expert_overlay_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().contiguous().clone()
        for key, value in state_dict.items()
        if torch.is_tensor(value) and key.startswith(ACTION_EXPERT_OVERLAY_PREFIXES)
    }


def _get_lora_module(model: torch.nn.Module, lora_target: str):
    if lora_target == "paligemma":
        return (
            model.paligemma_with_expert.paligemma,
            lambda module: setattr(model.paligemma_with_expert, "paligemma", module),
            "lora_adapter",
        )
    if lora_target == "action_expert":
        return (
            model.paligemma_with_expert.gemma_expert.model,
            lambda module: setattr(
                model.paligemma_with_expert.gemma_expert, "model", module
            ),
            "action_expert_lora_adapter",
        )
    raise ValueError(
        "Unsupported OpenPI lora_target "
        f"{lora_target!r}; expected 'paligemma' or 'action_expert'."
    )


def export_checkpoint(
    train_config_path: str,
    ckpt_path: str,
    output_dir: str,
    save_adapter: bool,
) -> None:
    model_cfg = _load_model_cfg(train_config_path)
    if not model_cfg.get("is_lora", False):
        raise ValueError("actor.model.is_lora must be true for LoRA export.")

    model_cfg.load_to_device = False
    model = get_model(model_cfg)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    checkpoint_meta = _checkpoint_metadata(checkpoint)
    is_trainable_checkpoint = checkpoint_meta.get("format") == "trainable_weights"
    state_dict = _normalize_state_dict_keys(_extract_state_dict(checkpoint))
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(
        "Loaded RLinf checkpoint with "
        f"{len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys."
    )
    if missing_keys:
        print(f"First missing keys: {missing_keys[:20]}")
        if not is_trainable_checkpoint:
            raise RuntimeError("Full checkpoint did not load all model keys.")
    if unexpected_keys:
        print(f"First unexpected keys: {unexpected_keys[:20]}")
        raise RuntimeError("Unexpected checkpoint keys were not loaded.")

    lora_target = str(model_cfg.get("lora_target", "paligemma"))
    lora_module, assign_lora_module, adapter_dir_name = _get_lora_module(
        model, lora_target
    )
    if not hasattr(lora_module, "merge_and_unload"):
        raise TypeError(
            f"OpenPI {lora_target} submodule is not a PEFT LoRA model."
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if save_adapter:
        adapter_dir = output_path / adapter_dir_name
        lora_module.save_pretrained(str(adapter_dir))
        print(f"Saved LoRA adapter to {adapter_dir}")

    assign_lora_module(lora_module.merge_and_unload())
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    _save_filtered_safetensors(model, str(output_path / "model.safetensors"))
    safetensors.torch.save_file(
        _filter_action_expert_overlay_state_dict(model.state_dict()),
        str(output_path / "action_expert_merged.safetensors"),
    )
    _copy_assets(str(model_cfg.model_path), str(output_path))

    model_config = getattr(model, "config", None)
    if dataclasses.is_dataclass(model_config):
        with (output_path / "config.json").open("w", encoding="utf-8") as f:
            json.dump(_json_safe(model_config), f, indent=2)

    with (output_path / "export_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_train_config": os.path.abspath(train_config_path),
                "source_ckpt": os.path.abspath(ckpt_path),
                "source_model_path": str(model_cfg.model_path),
                "source_ckpt_metadata": dict(checkpoint_meta),
                "format": "t2vla_openpi_pytorch_merged_lora",
                "lora_target": lora_target,
                "adapter_dir": adapter_dir_name if save_adapter else None,
                "action_expert_overlay": "action_expert_merged.safetensors",
            },
            f,
            indent=2,
        )

    print(f"Saved merged T2-VLA checkpoint to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config_path", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--no_save_adapter", action="store_true")
    args = parser.parse_args()

    export_checkpoint(
        train_config_path=args.train_config_path,
        ckpt_path=args.ckpt_path,
        output_dir=args.output_dir,
        save_adapter=not args.no_save_adapter,
    )


if __name__ == "__main__":
    main()
