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

"""Export Tabero RLT Stage 1/2 weights for T2-VLA inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

_ENCODER_PREFIX = "rlt_module.encoder."
_ACTOR_PREFIXES = ("backbone.", "actor_mean.")


def checkpoint_sha256(path: str | Path) -> str:
    """Hash the PyTorch checkpoint file used by a Tabero base model."""
    checkpoint_path = Path(path).expanduser()
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "model.safetensors"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Base model checkpoint file not found: {checkpoint_path}"
        )
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as checkpoint_file:
        while chunk := checkpoint_file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_encoder_shapes(
    *,
    input_dim: int,
    embed_dim: int,
    num_rl_tokens: int,
    prefix_seq_len: int,
    num_layers: int,
    mlp_ratio: float,
) -> dict[str, tuple[int, ...]]:
    mlp_dim = int(embed_dim * mlp_ratio)
    shapes = {
        "encoder.rl_token_embed": (num_rl_tokens, embed_dim),
        "encoder.prefix_pos_enc": (prefix_seq_len, embed_dim),
        "encoder.rl_token_pos_enc": (num_rl_tokens, embed_dim),
    }
    if input_dim != embed_dim:
        shapes.update(
            {
                "encoder.input_proj.weight": (embed_dim, input_dim),
                "encoder.input_proj.bias": (embed_dim,),
            }
        )
    for layer in range(num_layers):
        prefix = f"encoder.layers.{layer}"
        shapes.update(
            {
                f"{prefix}.self_norm.weight": (embed_dim,),
                f"{prefix}.self_norm.bias": (embed_dim,),
                f"{prefix}.self_attn.in_proj_weight": (3 * embed_dim, embed_dim),
                f"{prefix}.self_attn.in_proj_bias": (3 * embed_dim,),
                f"{prefix}.self_attn.out_proj.weight": (embed_dim, embed_dim),
                f"{prefix}.self_attn.out_proj.bias": (embed_dim,),
                f"{prefix}.mlp_norm.weight": (embed_dim,),
                f"{prefix}.mlp_norm.bias": (embed_dim,),
                f"{prefix}.mlp.0.weight": (mlp_dim, embed_dim),
                f"{prefix}.mlp.0.bias": (mlp_dim,),
                f"{prefix}.mlp.2.proj.weight": (2 * mlp_dim, mlp_dim),
                f"{prefix}.mlp.2.proj.bias": (2 * mlp_dim,),
                f"{prefix}.mlp.3.weight": (embed_dim, mlp_dim),
                f"{prefix}.mlp.3.bias": (embed_dim,),
            }
        )
    return shapes


def _expected_actor_shapes(
    *,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
) -> dict[str, tuple[int, ...]]:
    return {
        "backbone.0.weight": (hidden_dim, input_dim),
        "backbone.0.bias": (hidden_dim,),
        "backbone.2.weight": (hidden_dim, hidden_dim),
        "backbone.2.bias": (hidden_dim,),
        "backbone.4.weight": (hidden_dim, hidden_dim),
        "backbone.4.bias": (hidden_dim,),
        "actor_mean.weight": (output_dim, hidden_dim),
        "actor_mean.bias": (output_dim,),
    }


def _validate_state_shapes(
    name: str,
    state: dict[str, torch.Tensor],
    expected_shapes: dict[str, tuple[int, ...]],
) -> None:
    missing = sorted(set(expected_shapes) - set(state))
    unexpected = sorted(set(state) - set(expected_shapes))
    if missing or unexpected:
        raise ValueError(
            f"{name} tensor keys do not match the deployment architecture; "
            f"missing={missing}, unexpected={unexpected}."
        )
    for key, expected_shape in expected_shapes.items():
        tensor = state[key]
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{name} tensor {key} has shape {tuple(tensor.shape)}, "
                f"expected {expected_shape}."
            )
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name} tensor {key} contains NaN or Inf values.")


def _load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected a dictionary checkpoint at {path}.")
    return checkpoint


def _stage1_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, dict):
        raise ValueError(
            "Stage 1 checkpoint does not contain a model state dictionary."
        )
    return state


def _tensor_state(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().contiguous()
        for key, value in state.items()
        if torch.is_tensor(value)
    }


def export_tabero_rlt_bundle(
    *,
    stage1_checkpoint: str | Path,
    stage2_checkpoint: str | Path,
    output_dir: str | Path,
    base_model: str,
    stage2_global_step: int,
    normalized_action_bound: float = 4.0,
    proprio_dim: int = 7,
    action_dim: int = 13,
    num_action_chunks: int = 10,
    ref_num_action_chunks: int = 10,
    actor_hidden_dim: int = 256,
    rlt_input_dim: int = 2048,
    rlt_embed_dim: int = 2048,
    rlt_num_rl_tokens: int = 1,
    rlt_prefix_seq_len: int = 1024,
    rlt_num_layers: int = 2,
    rlt_num_heads: int = 8,
    rlt_mlp_ratio: float = 4.0,
    base_config_name: str = "pi0_lora_tacfield_tabero",
    base_action_horizon: int = 50,
    base_model_action_dim: int = 32,
    base_effective_action_dim: int = 13,
    base_prefix_hidden_dim: int = 2048,
    base_norm_asset_id: str = "NathanWu7/tabero_object_25",
    base_use_quantile_norm: bool = False,
    rlt_use_normalized_proprio: bool = True,
    state_indices: list[int] | None = None,
    reference_num_steps: int = 10,
    reference_sampling_method: str = "flow_ode",
    base_model_sha256: str | None = None,
    task_id: int | None = None,
    source_train_config: str | Path | None = None,
    target_global_step: int | None = None,
    is_final: bool | None = None,
) -> dict[str, Any]:
    """Write an encoder-only Stage 1 and actor-only Stage 2 deployment bundle."""
    stage1_checkpoint = Path(stage1_checkpoint).resolve()
    stage2_checkpoint = Path(stage2_checkpoint).resolve()
    output_dir = Path(output_dir).resolve()
    formal_values = (task_id, source_train_config, target_global_step, is_final)
    has_formal_provenance = any(value is not None for value in formal_values)
    if has_formal_provenance and any(value is None for value in formal_values):
        raise ValueError(
            "Formal RLT export provenance requires task_id, source_train_config, "
            "target_global_step, and is_final together."
        )
    resolved_train_config: Path | None = None
    if has_formal_provenance:
        if type(task_id) is not int or task_id not in {0, 5}:
            raise ValueError("Formal RLT export task_id must be 0 or 5.")
        if (
            type(target_global_step) is not int
            or target_global_step <= 0
            or target_global_step != stage2_global_step
        ):
            raise ValueError(
                "Formal RLT target_global_step must equal stage2_global_step."
            )
        if is_final is not True:
            raise ValueError("Formal RLT export must be the final target checkpoint.")
        resolved_train_config = Path(source_train_config).expanduser().resolve()
        if not resolved_train_config.is_file():
            raise FileNotFoundError(
                f"Formal RLT source training config not found: {resolved_train_config}"
            )
        expected_train_config_name = f"tabero_rlt_stage2_ac_task{task_id}_firm.yaml"
        if resolved_train_config.name != expected_train_config_name:
            raise ValueError(
                "Formal RLT source training config must be "
                f"{expected_train_config_name} for task{task_id}."
            )
    int_dimensions = {
        "proprio_dim": proprio_dim,
        "action_dim": action_dim,
        "num_action_chunks": num_action_chunks,
        "ref_num_action_chunks": ref_num_action_chunks,
        "actor_hidden_dim": actor_hidden_dim,
        "rlt_input_dim": rlt_input_dim,
        "rlt_embed_dim": rlt_embed_dim,
        "rlt_num_rl_tokens": rlt_num_rl_tokens,
        "rlt_prefix_seq_len": rlt_prefix_seq_len,
        "rlt_num_layers": rlt_num_layers,
        "rlt_num_heads": rlt_num_heads,
        "base_action_horizon": base_action_horizon,
        "base_model_action_dim": base_model_action_dim,
        "base_effective_action_dim": base_effective_action_dim,
        "base_prefix_hidden_dim": base_prefix_hidden_dim,
        "reference_num_steps": reference_num_steps,
    }
    if invalid := [name for name, value in int_dimensions.items() if int(value) <= 0]:
        raise ValueError(f"RLT dimensions must be positive; invalid={invalid}.")
    if ref_num_action_chunks < num_action_chunks:
        raise ValueError("ref_num_action_chunks must be >= num_action_chunks.")
    if base_action_horizon < ref_num_action_chunks:
        raise ValueError("base_action_horizon must be >= ref_num_action_chunks.")
    if base_effective_action_dim != action_dim:
        raise ValueError("base_effective_action_dim must match action_dim.")
    if proprio_dim > base_model_action_dim:
        raise ValueError("proprio_dim cannot exceed base_model_action_dim.")
    if base_prefix_hidden_dim != rlt_input_dim:
        raise ValueError("base_prefix_hidden_dim must match rlt_input_dim.")
    if not base_config_name or not base_norm_asset_id:
        raise ValueError("Base config name and norm asset id must be non-empty.")
    if rlt_embed_dim % rlt_num_heads != 0:
        raise ValueError("rlt_embed_dim must be divisible by rlt_num_heads.")
    if not math.isfinite(rlt_mlp_ratio) or rlt_mlp_ratio <= 0:
        raise ValueError("rlt_mlp_ratio must be finite and positive.")
    if not math.isfinite(normalized_action_bound) or normalized_action_bound <= 0:
        raise ValueError("normalized_action_bound must be finite and positive.")
    if not rlt_use_normalized_proprio:
        raise ValueError(
            "T2-VLA Tabero RLT deployment currently requires normalized proprio."
        )
    if state_indices is not None:
        state_indices = [int(index) for index in state_indices]
        if len(state_indices) != proprio_dim:
            raise ValueError("state_indices length must equal proprio_dim.")
        if any(index < 0 for index in state_indices) or len(set(state_indices)) != len(
            state_indices
        ):
            raise ValueError("state_indices must be unique non-negative integers.")
    if reference_sampling_method != "flow_ode":
        raise ValueError(
            "T2-VLA Tabero RLT deployment requires reference sampling method "
            "'flow_ode'."
        )
    actual_base_model_sha256 = checkpoint_sha256(base_model)
    if (
        base_model_sha256 is not None
        and base_model_sha256.lower() != actual_base_model_sha256
    ):
        raise ValueError(
            "Provided base_model_sha256 does not match the actual base checkpoint; "
            f"expected {base_model_sha256.lower()}, got {actual_base_model_sha256}."
        )
    base_model_sha256 = actual_base_model_sha256
    if len(base_model_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in base_model_sha256.lower()
    ):
        raise ValueError("base_model_sha256 must be a 64-character SHA-256 hex digest.")

    stage1 = _tensor_state(_stage1_state(_load_checkpoint(stage1_checkpoint)))
    encoder = {
        key.removeprefix("rlt_module."): value
        for key, value in stage1.items()
        if key.startswith(_ENCODER_PREFIX)
    }
    if not encoder:
        raise ValueError("Stage 1 checkpoint contains no RLT encoder weights.")

    stage2 = _tensor_state(_load_checkpoint(stage2_checkpoint))
    actor = {
        key: value for key, value in stage2.items() if key.startswith(_ACTOR_PREFIXES)
    }
    if not any(key.startswith("backbone.") for key in actor):
        raise ValueError("Stage 2 checkpoint contains no actor backbone weights.")
    if not any(key.startswith("actor_mean.") for key in actor):
        raise ValueError("Stage 2 checkpoint contains no actor output weights.")

    z_dim = rlt_embed_dim * rlt_num_rl_tokens
    _validate_state_shapes(
        "RLT encoder",
        encoder,
        _expected_encoder_shapes(
            input_dim=rlt_input_dim,
            embed_dim=rlt_embed_dim,
            num_rl_tokens=rlt_num_rl_tokens,
            prefix_seq_len=rlt_prefix_seq_len,
            num_layers=rlt_num_layers,
            mlp_ratio=rlt_mlp_ratio,
        ),
    )
    _validate_state_shapes(
        "RLT actor",
        actor,
        _expected_actor_shapes(
            input_dim=(num_action_chunks * action_dim + z_dim + proprio_dim),
            output_dim=num_action_chunks * action_dim,
            hidden_dim=actor_hidden_dim,
        ),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    encoder_name = "rlt_encoder.safetensors"
    actor_name = "rlt_actor.safetensors"
    save_file(encoder, output_dir / encoder_name)
    save_file(actor, output_dir / actor_name)

    manifest: dict[str, Any] = {
        "format": "tabero_rlt_t2vla",
        "format_version": 1,
        "action_space": "model_normalized",
        "normalized_action_bound": float(normalized_action_bound),
        "z_dim": z_dim,
        "proprio_dim": int(proprio_dim),
        "action_dim": int(action_dim),
        "num_action_chunks": int(num_action_chunks),
        "ref_num_action_chunks": int(ref_num_action_chunks),
        "actor_hidden_dim": int(actor_hidden_dim),
        "rlt_input_dim": int(rlt_input_dim),
        "rlt_embed_dim": int(rlt_embed_dim),
        "rlt_num_rl_tokens": int(rlt_num_rl_tokens),
        "rlt_prefix_seq_len": int(rlt_prefix_seq_len),
        "rlt_num_layers": int(rlt_num_layers),
        "rlt_num_heads": int(rlt_num_heads),
        "rlt_mlp_ratio": float(rlt_mlp_ratio),
        "rlt_image_only": False,
        "rlt_use_mask": True,
        "rlt_use_normalized_proprio": bool(rlt_use_normalized_proprio),
        "state_indices": state_indices,
        "reference_num_steps": int(reference_num_steps),
        "reference_sampling_method": reference_sampling_method,
        "base_model": str(base_model),
        "base_model_sha256": base_model_sha256.lower(),
        "base_config_name": str(base_config_name),
        "base_action_horizon": int(base_action_horizon),
        "base_model_action_dim": int(base_model_action_dim),
        "base_effective_action_dim": int(base_effective_action_dim),
        "base_prefix_hidden_dim": int(base_prefix_hidden_dim),
        "base_norm_asset_id": str(base_norm_asset_id),
        "base_use_quantile_norm": bool(base_use_quantile_norm),
        "stage1_checkpoint": str(stage1_checkpoint),
        "stage1_checkpoint_sha256": checkpoint_sha256(stage1_checkpoint),
        "stage2_checkpoint": str(stage2_checkpoint),
        "stage2_checkpoint_sha256": checkpoint_sha256(stage2_checkpoint),
        "stage2_global_step": int(stage2_global_step),
        "encoder_weights": encoder_name,
        "encoder_tensor_count": len(encoder),
        "actor_weights": actor_name,
        "actor_tensor_count": len(actor),
    }
    if has_formal_provenance:
        manifest.update(
            {
                "task_id": task_id,
                "source_train_config": str(resolved_train_config),
                "stage2_checkpoint_metadata": {
                    "format": "full_weights",
                    "method": "rlt",
                    "task_id": task_id,
                    "training_config": resolved_train_config.stem,
                    "step": int(stage2_global_step),
                    "global_step": int(stage2_global_step),
                    "target_global_step": target_global_step,
                    "is_final": is_final,
                },
            }
        )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--stage2-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--stage2-global-step", type=int, required=True)
    parser.add_argument("--normalized-action-bound", type=float, default=4.0)
    parser.add_argument("--proprio-dim", type=int, default=7)
    parser.add_argument("--action-dim", type=int, default=13)
    parser.add_argument("--num-action-chunks", type=int, default=10)
    parser.add_argument("--ref-num-action-chunks", type=int, default=10)
    parser.add_argument("--actor-hidden-dim", type=int, default=256)
    parser.add_argument("--rlt-input-dim", type=int, default=2048)
    parser.add_argument("--rlt-embed-dim", type=int, default=2048)
    parser.add_argument("--rlt-num-rl-tokens", type=int, default=1)
    parser.add_argument("--rlt-prefix-seq-len", type=int, default=1024)
    parser.add_argument("--rlt-num-layers", type=int, default=2)
    parser.add_argument("--rlt-num-heads", type=int, default=8)
    parser.add_argument("--rlt-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--base-config-name", default="pi0_lora_tacfield_tabero")
    parser.add_argument("--base-action-horizon", type=int, default=50)
    parser.add_argument("--base-model-action-dim", type=int, default=32)
    parser.add_argument("--base-effective-action-dim", type=int, default=13)
    parser.add_argument("--base-prefix-hidden-dim", type=int, default=2048)
    parser.add_argument("--base-norm-asset-id", default="NathanWu7/tabero_object_25")
    parser.add_argument(
        "--base-use-quantile-norm",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--rlt-use-normalized-proprio",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--state-indices", nargs="*", type=int, default=None)
    parser.add_argument("--reference-num-steps", type=int, default=10)
    parser.add_argument("--reference-sampling-method", default="flow_ode")
    parser.add_argument("--base-model-sha256", default=None)
    parser.add_argument("--task-id", type=int, choices=(0, 5), default=None)
    parser.add_argument("--source-train-config", type=Path, default=None)
    parser.add_argument("--target-global-step", type=int, default=None)
    parser.add_argument(
        "--is-final", action=argparse.BooleanOptionalAction, default=None
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = export_tabero_rlt_bundle(
        stage1_checkpoint=args.stage1_checkpoint,
        stage2_checkpoint=args.stage2_checkpoint,
        output_dir=args.output_dir,
        base_model=args.base_model,
        stage2_global_step=args.stage2_global_step,
        normalized_action_bound=args.normalized_action_bound,
        proprio_dim=args.proprio_dim,
        action_dim=args.action_dim,
        num_action_chunks=args.num_action_chunks,
        ref_num_action_chunks=args.ref_num_action_chunks,
        actor_hidden_dim=args.actor_hidden_dim,
        rlt_input_dim=args.rlt_input_dim,
        rlt_embed_dim=args.rlt_embed_dim,
        rlt_num_rl_tokens=args.rlt_num_rl_tokens,
        rlt_prefix_seq_len=args.rlt_prefix_seq_len,
        rlt_num_layers=args.rlt_num_layers,
        rlt_num_heads=args.rlt_num_heads,
        rlt_mlp_ratio=args.rlt_mlp_ratio,
        base_config_name=args.base_config_name,
        base_action_horizon=args.base_action_horizon,
        base_model_action_dim=args.base_model_action_dim,
        base_effective_action_dim=args.base_effective_action_dim,
        base_prefix_hidden_dim=args.base_prefix_hidden_dim,
        base_norm_asset_id=args.base_norm_asset_id,
        base_use_quantile_norm=args.base_use_quantile_norm,
        rlt_use_normalized_proprio=args.rlt_use_normalized_proprio,
        state_indices=args.state_indices,
        reference_num_steps=args.reference_num_steps,
        reference_sampling_method=args.reference_sampling_method,
        base_model_sha256=args.base_model_sha256,
        task_id=args.task_id,
        source_train_config=args.source_train_config,
        target_global_step=args.target_global_step,
        is_final=args.is_final,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
