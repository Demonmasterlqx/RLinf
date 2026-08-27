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
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping
from importlib import metadata as importlib_metadata
from pathlib import Path

import safetensors.torch
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from safetensors import safe_open

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

PIRL_ACTION_EXPERT_LORA_TENSOR_COUNT = 126
BUNDLE_FORMAT = "tabero_rlinf_openpi_lora_bundle"
BUNDLE_FORMAT_VERSION = 1
TABERO_SFT_METHODS = {
    "sft_full_lora_tacfield",
    "sft_full_lora_tacforce_tcn",
}


def _is_action_expert_lora_weight(key: str) -> bool:
    match = re.fullmatch(
        r"paligemma_with_expert\.gemma_expert\.model\.layers\.(\d+)\."
        r"(?:mlp\.(?:down_proj|gate_proj|up_proj)|"
        r"self_attn\.(?:k_proj|o_proj|q_proj|v_proj))\.weight",
        key,
    )
    return match is not None and 0 <= int(match.group(1)) < 18


def validate_pirl_action_expert_delta(model_path: Path, base_path: Path) -> None:
    """Require every action-expert LoRA target, and only those targets, to change."""
    try:
        with (
            safe_open(model_path, framework="pt", device="cpu") as model,
            safe_open(base_path, framework="pt", device="cpu") as base,
        ):
            model_keys = set(model.keys())
            base_keys = set(base.keys())
            missing_model_keys = sorted(base_keys - model_keys)
            unexpected_model_keys = sorted(model_keys - base_keys)
            if missing_model_keys or unexpected_model_keys:
                raise ValueError(
                    "piRL model/base keys differ while validating action-expert delta: "
                    f"missing_model_keys={missing_model_keys[:5]}, "
                    f"unexpected_model_keys={unexpected_model_keys[:5]}"
                )
            expected_changes = {
                key for key in base_keys if _is_action_expert_lora_weight(key)
            }
            actual_changes = {
                key
                for key in base_keys
                if not torch.equal(model.get_tensor(key), base.get_tensor(key))
            }
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(
            f"cannot compare piRL model with fixed base: {error}"
        ) from error
    missing = sorted(expected_changes - actual_changes)
    unexpected = sorted(actual_changes - expected_changes)
    if (
        len(expected_changes) != PIRL_ACTION_EXPERT_LORA_TENSOR_COUNT
        or actual_changes != expected_changes
    ):
        raise ValueError(
            "piRL action-expert delta mismatch: "
            f"expected_count={len(expected_changes)}, "
            f"actual_count={len(actual_changes)}, "
            "required_expected_count="
            f"{PIRL_ACTION_EXPERT_LORA_TENSOR_COUNT}, "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}; "
            "non-action-expert tensors must remain unchanged"
        )


@dataclasses.dataclass(frozen=True)
class LoraModuleSpec:
    module: torch.nn.Module
    assign_module: Callable[[torch.nn.Module], None]
    adapter_dir_name: str
    target_name: str


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


def _validate_trainable_checkpoint_keys(
    model: torch.nn.Module, state_dict: Mapping[str, torch.Tensor]
) -> None:
    """Reject incomplete sidecars and accidental frozen-base tensors."""
    expected_trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    checkpoint_keys = set(state_dict)
    missing_trainable = sorted(expected_trainable - checkpoint_keys)
    unexpected_trainable = sorted(checkpoint_keys - expected_trainable)
    if missing_trainable or unexpected_trainable:
        raise RuntimeError(
            "Trainable checkpoint must contain exactly the model's trainable "
            "parameters; "
            f"missing={missing_trainable[:20]}, "
            f"unexpected={unexpected_trainable[:20]}"
        )


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


def _copy_norm_stats(
    norm_stats_path: str | Path, output_dir: str | Path, asset_id: str
) -> None:
    """Install explicit dataset statistics into a deployable OpenPI export."""
    source = Path(norm_stats_path).expanduser().resolve()
    if source.is_dir():
        source = source / "norm_stats.json"
    if not source.is_file():
        raise FileNotFoundError(f"Normalization stats not found: {source}")
    output_root = Path(output_dir)
    destinations = (
        output_root / "assets" / asset_id / "norm_stats.json",
        output_root / asset_id / "norm_stats.json",
    )
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _resolve_norm_stats_source(
    model_cfg: DictConfig, checkpoint_meta: Mapping
) -> tuple[Path | None, str]:
    """Resolve normalization assets without weakening the checkpoint contract."""
    norm_asset_id = str(
        model_cfg.get("export_norm_asset_id")
        or checkpoint_meta.get("normalization_asset_id")
        or "NathanWu7/tabero_object_25"
    )
    openpi_data = model_cfg.get("openpi_data") or {}
    configured_source = openpi_data.get("norm_stats_path")
    if configured_source is not None:
        source = Path(configured_source).expanduser().resolve()
        if source.is_dir():
            source = source / "norm_stats.json"
    else:
        source = (
            Path(model_cfg.model_path).expanduser().resolve()
            / "assets"
            / norm_asset_id
            / "norm_stats.json"
        )
    if not source.is_file():
        return None, norm_asset_id

    expected_hash = checkpoint_meta.get("base_norm_stats_sha256")
    if expected_hash is not None and _sha256_file(source) != expected_hash:
        raise ValueError(
            "Normalization stats SHA256 differs from the piRL checkpoint "
            f"contract: path={source}, expected={expected_hash}, "
            f"actual={_sha256_file(source)}"
        )
    return source, norm_asset_id


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


def _checkpoint_path(path: str | Path) -> Path:
    checkpoint = Path(path).expanduser().resolve()
    if checkpoint.is_dir():
        checkpoint = checkpoint / "model.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
    return checkpoint


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _checkpoint_path(path).open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: str | Path) -> str:
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_peft_key(key: str) -> str:
    """Map a PEFT-wrapped parameter name to the merged OpenPI state key."""
    canonical = key
    marker = ".base_model.model."
    while marker in canonical:
        canonical = canonical.replace(marker, ".")
    return canonical


def _matches_module_prefix(key: str, module_name: str) -> bool:
    return key == module_name or key.startswith(f"{module_name}.")


def _collect_extra_trainable_keys(
    model: torch.nn.Module, configured_modules: tuple[str, ...]
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Collect every trainable non-LoRA, non-RL parameter for the bundle."""
    invalid_modules = [
        name for name in configured_modules if not isinstance(name, str) or not name
    ]
    if invalid_modules:
        raise ValueError(
            "extra_trainable_modules must contain non-empty strings; "
            f"invalid={invalid_modules!r}"
        )
    canonical_prefixes = tuple(
        dict.fromkeys(_canonical_peft_key(name) for name in configured_modules)
    )
    result: dict[str, str] = {}
    unowned: list[str] = []
    for source_key, parameter in model.named_parameters():
        if not parameter.requires_grad or "lora_" in source_key:
            continue
        canonical_key = _canonical_peft_key(source_key)
        if canonical_key.startswith(RL_ONLY_PREFIXES):
            continue
        if not any(
            _matches_module_prefix(source_key, module_name)
            for module_name in configured_modules
        ):
            unowned.append(source_key)
            continue
        previous = result.setdefault(canonical_key, source_key)
        if previous != source_key:
            raise ValueError(
                "PEFT key canonicalization collision for extra trainable parameters: "
                f"{previous!r} and {source_key!r} -> {canonical_key!r}"
            )
    if unowned:
        raise ValueError(
            "Trainable non-LoRA parameters must belong to actor.model."
            f"extra_trainable_modules; unowned={unowned[:20]}"
        )
    return result, canonical_prefixes


def _tensor_bytes(tensor: torch.Tensor) -> memoryview:
    value = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return memoryview(value)


def _schema_digest(schema: list[dict]) -> str:
    payload = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _safetensors_contract(path: str | Path) -> dict:
    checkpoint = _checkpoint_path(path)
    schema: list[dict] = []
    tensor_digest = hashlib.sha256()
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        for key in sorted(handle.keys()):
            tensor = handle.get_tensor(key)
            entry = {
                "key": key,
                "shape": list(tensor.shape),
                "dtype": str(handle.get_slice(key).get_dtype()),
            }
            schema.append(entry)
            tensor_digest.update(
                json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()
            )
            tensor_digest.update(_tensor_bytes(tensor))
    return {
        "tensor_count": len(schema),
        "schema": schema,
        "schema_sha256": _schema_digest(schema),
        "tensor_sha256": tensor_digest.hexdigest(),
    }


def _save_extra_trainable(
    model: torch.nn.Module,
    key_map: Mapping[str, str],
    output_path: Path,
    *,
    output_dtype: torch.dtype | None,
) -> None:
    merged_state = model.state_dict()
    missing = sorted(set(key_map) - set(merged_state))
    if missing:
        raise ValueError(
            f"Merged model is missing canonical extra-trainable keys: {missing[:20]}"
        )
    tensors = {}
    for key in sorted(key_map):
        value = merged_state[key].detach().cpu()
        if output_dtype is not None and value.is_floating_point():
            value = value.to(output_dtype)
        tensors[key] = value.contiguous().clone()
    safetensors.torch.save_file(tensors, output_path)


def _adapter_manifest(adapter_dir: Path, bundle_root: Path) -> dict:
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            f"PEFT adapter is incomplete under {adapter_dir}; expected "
            "adapter_config.json and adapter_model.safetensors"
        )
    with config_path.open(encoding="utf-8") as source:
        config = json.load(source)
    contract = _safetensors_contract(weights_path)
    return {
        "path": adapter_dir.relative_to(bundle_root).as_posix(),
        "config_sha256": _sha256_file(config_path),
        "weights_sha256": _sha256_file(weights_path),
        "rank": config.get("r"),
        "alpha": config.get("lora_alpha"),
        "target_modules": sorted(config.get("target_modules") or []),
        "exclude_modules": config.get("exclude_modules"),
        "tensor_count": contract["tensor_count"],
        "schema_sha256": contract["schema_sha256"],
    }


def _write_checksums(bundle_root: Path) -> None:
    checksums = {
        path.relative_to(bundle_root).as_posix(): _sha256_file(path)
        for path in sorted(bundle_root.rglob("*"))
        if path.is_file() and path.name != "checksums.json"
    }
    with (bundle_root / "checksums.json").open("w", encoding="utf-8") as output:
        json.dump(checksums, output, indent=2, sort_keys=True)


def _prepare_bundle_staging(bundle_dir: str | Path) -> tuple[Path, Path]:
    target = Path(bundle_dir).expanduser().resolve()
    if target.exists():
        if not target.is_dir():
            raise FileExistsError(f"LoRA bundle target is not a directory: {target}")
        if any(target.iterdir()):
            raise FileExistsError(f"LoRA bundle target must be new or empty: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent))
    )
    return target, staging


_SAFETENSORS_DTYPES = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}


def _save_filtered_safetensors(
    model: torch.nn.Module,
    output_path: str,
    *,
    base_model_path: str,
    allowed_extra_prefixes: tuple[str, ...] = (),
    output_dtype: torch.dtype | None = None,
) -> None:
    state_dict = {}
    for key, value in model.state_dict().items():
        if key in DROP_EXACT_KEYS:
            continue
        if key.startswith(RL_ONLY_PREFIXES):
            continue
        if "lora_" in key:
            raise RuntimeError(f"LoRA key remained after merge: {key}")
        state_dict[key] = value.detach().cpu()
    base_checkpoint = _checkpoint_path(base_model_path)
    with safe_open(base_checkpoint, framework="pt", device="cpu") as base:
        base_keys = set(base.keys())
        missing = sorted(base_keys - set(state_dict))
        unexpected = sorted(
            key
            for key in set(state_dict) - base_keys
            if not key.startswith(allowed_extra_prefixes)
        )
        shape_mismatches = sorted(
            (
                key,
                tuple(state_dict[key].shape),
                tuple(base.get_slice(key).get_shape()),
            )
            for key in base_keys & set(state_dict)
            if tuple(state_dict[key].shape) != tuple(base.get_slice(key).get_shape())
        )
        if missing or unexpected or shape_mismatches:
            raise ValueError(
                "Merged OpenPI state keys do not match the allowed architecture; "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}, "
                f"shape_mismatches={shape_mismatches[:10]}"
            )
        for key in sorted(state_dict):
            if output_dtype is not None and state_dict[key].is_floating_point():
                target_dtype = output_dtype
            elif key in base_keys:
                dtype_name = str(base.get_slice(key).get_dtype())
                if dtype_name not in _SAFETENSORS_DTYPES:
                    raise ValueError(
                        "Unsupported fixed-base safetensors dtype "
                        f"{dtype_name!r} for {key}"
                    )
                target_dtype = _SAFETENSORS_DTYPES[dtype_name]
            else:
                target_dtype = state_dict[key].dtype
            # Clone to avoid shared-storage errors from tied weights.
            state_dict[key] = (
                state_dict[key].to(dtype=target_dtype).contiguous().clone()
            )
    safetensors.torch.save_file(state_dict, output_path)


def _safetensor_count(path: str | Path) -> int:
    with safe_open(_checkpoint_path(path), framework="pt", device="cpu") as handle:
        return len(handle.keys())


def _validate_tabero_sft_export_schema(
    model_path: str | Path,
    base_model_path: str | Path,
    *,
    expected_tcn_tensor_count: int = 16,
) -> dict[str, int]:
    """Validate the base-plus-TacField schema used by Pi0 and Pi0.5 SFT."""
    with (
        safe_open(_checkpoint_path(model_path), framework="pt", device="cpu") as model,
        safe_open(
            _checkpoint_path(base_model_path), framework="pt", device="cpu"
        ) as base,
    ):
        model_keys = set(model.keys())
        base_keys = set(base.keys())
        missing = sorted(base_keys - model_keys)
        extra = sorted(model_keys - base_keys)
        invalid_extra = sorted(
            key for key in extra if not key.startswith("tactile_prefix_encoder.")
        )
        shape_mismatches = sorted(
            (
                key,
                tuple(model.get_slice(key).get_shape()),
                tuple(base.get_slice(key).get_shape()),
            )
            for key in model_keys & base_keys
            if model.get_slice(key).get_shape() != base.get_slice(key).get_shape()
        )
        dtypes = {str(model.get_slice(key).get_dtype()) for key in model_keys}
    if (
        missing
        or invalid_extra
        or shape_mismatches
        or len(extra) != expected_tcn_tensor_count
        or dtypes != {"BF16"}
    ):
        raise ValueError(
            "Tabero SFT export must preserve every fixed-base key and shape, "
            f"add exactly {expected_tcn_tensor_count} tactile-prefix tensors, "
            "and contain only BF16 tensors; "
            f"base_count={len(base_keys)}, model_count={len(model_keys)}, "
            f"extra_count={len(extra)}, missing={missing[:10]}, "
            f"invalid_extra={invalid_extra[:10]}, "
            f"shape_mismatches={shape_mismatches[:10]}, dtypes={sorted(dtypes)}"
        )
    return {
        "base_tensor_count": len(base_keys),
        "tactile_prefix_tensor_count": len(extra),
        "model_tensor_count": len(model_keys),
    }


def _validate_reference_schema(
    model_path: str | Path, reference_path: str | Path
) -> None:
    """Require an export to match an audited deployment checkpoint key-for-key."""
    with (
        safe_open(_checkpoint_path(model_path), framework="pt", device="cpu") as model,
        safe_open(
            _checkpoint_path(reference_path), framework="pt", device="cpu"
        ) as reference,
    ):
        model_keys = set(model.keys())
        reference_keys = set(reference.keys())
        missing = sorted(reference_keys - model_keys)
        unexpected = sorted(model_keys - reference_keys)
        shape_mismatches = sorted(
            (
                key,
                tuple(model.get_slice(key).get_shape()),
                tuple(reference.get_slice(key).get_shape()),
            )
            for key in model_keys & reference_keys
            if model.get_slice(key).get_shape() != reference.get_slice(key).get_shape()
        )
    if missing or unexpected or shape_mismatches:
        raise ValueError(
            "Tabero SFT export/reference schema mismatch: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}, "
            f"shape_mismatches={shape_mismatches[:10]}"
        )


def _build_export_metadata(
    *,
    train_config_path: str,
    ckpt_path: str,
    source_model_path: str,
    checkpoint_meta: Mapping,
    model_path: str | Path,
    lora_target: str,
    adapter_dirs: list[str],
    allow_non_final: bool = False,
) -> dict:
    train_config = Path(train_config_path).expanduser().resolve()
    source_checkpoint = _checkpoint_path(ckpt_path)
    base_checkpoint = _checkpoint_path(source_model_path)
    output_model = _checkpoint_path(model_path)
    expected_training_config = train_config.stem
    method = checkpoint_meta.get("method")
    required = {
        "format": "trainable_weights",
        "method": method,
        "training_config": expected_training_config,
    }
    if method not in {"pirl", *TABERO_SFT_METHODS}:
        raise ValueError(
            "OpenPI checkpoint provenance method must be 'pirl', "
            "'sft_full_lora_tacfield', or 'sft_full_lora_tacforce_tcn', "
            f"got {method!r}"
        )
    for field, expected in required.items():
        actual = checkpoint_meta.get(field)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                "OpenPI checkpoint provenance "
                f"{field} must be {expected!r}, got {actual!r}"
            )
    is_final = checkpoint_meta.get("is_final")
    if type(is_final) is not bool:
        raise ValueError(
            f"OpenPI checkpoint provenance is_final must be a boolean, got {is_final!r}"
        )
    if not is_final and not allow_non_final:
        raise ValueError(
            "OpenPI checkpoint provenance is_final must be True; pass "
            "--allow_non_final only when intentionally exporting an "
            "intermediate checkpoint for evaluation"
        )
    task_id = checkpoint_meta.get("task_id")
    if method == "pirl" and (type(task_id) is not int or task_id not in range(10)):
        raise ValueError("piRL checkpoint provenance task_id must be in [0, 9]")
    if method in TABERO_SFT_METHODS:
        dataset = checkpoint_meta.get("dataset")
        if dataset not in {
            "datas/tabero_firm",
            "datas/replay_firm_tabero",
            "datas/replay_firm_tabero_xarm_gripper",
            "datas/replay_firm_tabero_xarm_gripper_repaired_v1",
        }:
            raise ValueError(
                "Tabero SFT checkpoint provenance dataset is unsupported: "
                f"{dataset!r}."
            )
        if dataset == "datas/tabero_firm":
            if checkpoint_meta.get("training_precision") != "fp32":
                raise ValueError(
                    "Legacy Tabero SFT checkpoint provenance "
                    "training_precision must be 'fp32'."
                )
        else:
            required_precision = {
                "frozen_parameter_precision": "bf16",
                "trainable_parameter_precision": "fp32",
                "compute_precision": "bf16_amp",
                "export_precision": "bf16",
            }
            for field, expected in required_precision.items():
                if checkpoint_meta.get(field) != expected:
                    raise ValueError(
                        "Replay Firm checkpoint provenance "
                        f"{field} must be {expected!r}, got "
                        f"{checkpoint_meta.get(field)!r}"
                    )
    global_step = checkpoint_meta.get("global_step")
    if type(global_step) is not int or global_step <= 0:
        raise ValueError("OpenPI checkpoint provenance global_step must be positive")
    if checkpoint_meta.get("step") != global_step:
        raise ValueError(
            f"OpenPI checkpoint provenance step must equal global_step {global_step}"
        )
    target_global_step = checkpoint_meta.get("target_global_step")
    if type(target_global_step) is not int or target_global_step <= 0:
        raise ValueError(
            "OpenPI checkpoint provenance target_global_step must be positive"
        )
    if is_final and target_global_step != global_step:
        raise ValueError(
            "OpenPI final checkpoint provenance target_global_step must equal "
            f"global_step {global_step}"
        )
    if not is_final and global_step >= target_global_step:
        raise ValueError(
            "OpenPI non-final checkpoint provenance global_step must be less than "
            f"target_global_step {target_global_step}, got {global_step}"
        )
    if method == "pirl" and lora_target == "action_expert":
        validate_pirl_action_expert_delta(output_model, base_checkpoint)
    safe_metadata = _json_safe(dict(checkpoint_meta))
    base_tensor_count = _safetensor_count(base_checkpoint)
    model_tensor_count = _safetensor_count(output_model)
    return {
        "source_train_config": str(train_config),
        "source_ckpt": str(source_checkpoint),
        "source_ckpt_sha256": _sha256(source_checkpoint),
        "source_model_path": str(Path(source_model_path).expanduser().resolve()),
        "base_model_sha256": _sha256(base_checkpoint),
        "source_ckpt_metadata": safe_metadata,
        "format": "t2vla_openpi_pytorch_merged_lora",
        "method": method,
        "lora_target": lora_target,
        "task_id": task_id,
        "global_step": global_step,
        "target_global_step": target_global_step,
        "is_final": is_final,
        "model_sha256": _sha256(output_model),
        "model_tensor_count": model_tensor_count,
        "base_model_tensor_count": base_tensor_count,
        "extra_tensor_count": model_tensor_count - base_tensor_count,
        "adapter_dir": (
            adapter_dirs[0]
            if len(adapter_dirs) == 1
            else (adapter_dirs if adapter_dirs else None)
        ),
        "adapter_dirs": adapter_dirs,
        "dataset": checkpoint_meta.get("dataset"),
        "training_precision": checkpoint_meta.get("training_precision"),
        "export_precision": (
            checkpoint_meta.get("export_precision", "bf16")
            if method in TABERO_SFT_METHODS
            else None
        ),
        "frozen_parameter_precision": checkpoint_meta.get("frozen_parameter_precision"),
        "trainable_parameter_precision": checkpoint_meta.get(
            "trainable_parameter_precision"
        ),
        "compute_precision": checkpoint_meta.get("compute_precision"),
        "tactile_tcn_initialization": checkpoint_meta.get("tactile_tcn_initialization"),
    }


def _get_lora_module(model: torch.nn.Module, lora_target: str) -> LoraModuleSpec:
    if lora_target == "paligemma":
        return LoraModuleSpec(
            model.paligemma_with_expert.paligemma,
            lambda module: setattr(model.paligemma_with_expert, "paligemma", module),
            "lora_adapter",
            "paligemma",
        )
    if lora_target == "action_expert":
        return LoraModuleSpec(
            model.paligemma_with_expert.gemma_expert.model,
            lambda module: setattr(
                model.paligemma_with_expert.gemma_expert, "model", module
            ),
            "action_expert_lora_adapter",
            "action_expert",
        )
    raise ValueError(
        "Unsupported OpenPI lora_target "
        f"{lora_target!r}; expected 'paligemma', 'action_expert', or 'both'."
    )


def _get_lora_modules(model: torch.nn.Module, lora_target: str) -> list[LoraModuleSpec]:
    if lora_target == "both":
        return [
            _get_lora_module(model, "paligemma"),
            _get_lora_module(model, "action_expert"),
        ]
    return [_get_lora_module(model, lora_target)]


def _expected_lora_rank(model_cfg: DictConfig, target_name: str) -> int:
    rank = model_cfg.get(f"{target_name}_lora_rank")
    if rank is None:
        rank = model_cfg.get("lora_rank")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError(
            f"OpenPI {target_name} LoRA rank must be a positive integer, got {rank!r}"
        )
    return rank


def _build_lora_bundle_manifest(
    *,
    bundle_root: Path,
    model_cfg: DictConfig,
    checkpoint_meta: Mapping,
    export_metadata: Mapping,
    lora_target: str,
    adapter_specs: Mapping[str, dict],
    extra_key_map: Mapping[str, str],
    canonical_extra_prefixes: tuple[str, ...],
    extra_path: Path,
    norm_stats_path: Path,
    norm_asset_id: str,
    final_model_path: Path,
) -> dict:
    expected_targets = (
        {"paligemma", "action_expert"} if lora_target == "both" else {lora_target}
    )
    if set(adapter_specs) != expected_targets:
        raise ValueError(
            "LoRA bundle adapter targets do not match lora_target; "
            f"expected={sorted(expected_targets)}, actual={sorted(adapter_specs)}"
        )
    for target_name, adapter in adapter_specs.items():
        expected_rank = _expected_lora_rank(model_cfg, target_name)
        if adapter["rank"] != expected_rank or adapter["alpha"] != expected_rank:
            raise ValueError(
                f"LoRA bundle {target_name} adapter rank/alpha mismatch: "
                f"expected={expected_rank}, rank={adapter['rank']}, "
                f"alpha={adapter['alpha']}"
            )

    base_checkpoint = _checkpoint_path(str(model_cfg.model_path))
    base_config_path = base_checkpoint.parent / "config.json"
    openpi_cfg = model_cfg.get("openpi", {})
    config_name = checkpoint_meta.get("deployment_config_name") or openpi_cfg.get(
        "config_name"
    )
    if not isinstance(config_name, str) or not config_name:
        raise ValueError(
            "LoRA bundle requires checkpoint deployment_config_name or "
            "actor.model.openpi.config_name."
        )

    extra_contract = _safetensors_contract(extra_path)
    final_contract = _safetensors_contract(final_model_path)
    relative_norm_path = norm_stats_path.relative_to(bundle_root).as_posix()
    return {
        "format": BUNDLE_FORMAT,
        "format_version": BUNDLE_FORMAT_VERSION,
        "peft_version": importlib_metadata.version("peft"),
        "lora_target": lora_target,
        "base_model": {
            "source_path": str(Path(model_cfg.model_path).expanduser().resolve()),
            "model_file": "model.safetensors",
            "model_sha256": _sha256_file(base_checkpoint),
            "model_tensor_count": _safetensor_count(base_checkpoint),
            "config_file": "config.json" if base_config_path.is_file() else None,
            "config_sha256": (
                _sha256_file(base_config_path) if base_config_path.is_file() else None
            ),
        },
        "policy_contract": {
            "config_name": config_name,
            "norm_asset_id": norm_asset_id,
            "norm_stats_path": relative_norm_path,
            "norm_stats_sha256": _sha256_file(norm_stats_path),
            "dataset": checkpoint_meta.get("dataset"),
            "model_family": checkpoint_meta.get("model_family"),
            "action_horizon": checkpoint_meta.get(
                "action_horizon", openpi_cfg.get("action_horizon")
            ),
            "effective_action_dim": checkpoint_meta.get(
                "effective_action_dim", openpi_cfg.get("effective_action_dim")
            ),
            "state_dim": checkpoint_meta.get("state_dim", 7),
            "tactile_prefix_dim_in": checkpoint_meta.get(
                "tactile_prefix_dim_in", openpi_cfg.get("tactile_prefix_dim_in")
            ),
            "tactile_prefix_history": checkpoint_meta.get(
                "tactile_prefix_history", openpi_cfg.get("tactile_prefix_history")
            ),
            "gripper_coordinate": checkpoint_meta.get("gripper_coordinate"),
        },
        "checkpoint": {
            "source": export_metadata.get("source_ckpt"),
            "sha256": export_metadata.get("source_ckpt_sha256"),
            "method": export_metadata.get("method"),
            "global_step": export_metadata.get("global_step"),
            "target_global_step": export_metadata.get("target_global_step"),
            "is_final": export_metadata.get("is_final"),
            "metadata": _json_safe(dict(checkpoint_meta)),
        },
        "adapters": dict(adapter_specs),
        "extra_trainable": {
            "path": extra_path.relative_to(bundle_root).as_posix(),
            "configured_modules": list(
                model_cfg.get("extra_trainable_modules", ()) or ()
            ),
            "canonical_prefixes": list(canonical_extra_prefixes),
            "source_parameter_keys": [
                extra_key_map[key] for key in sorted(extra_key_map)
            ],
            "keys": sorted(extra_key_map),
            "file_sha256": _sha256_file(extra_path),
            **extra_contract,
        },
        "final_merged": {
            "reference_model_sha256": _sha256_file(final_model_path),
            **final_contract,
        },
        "exclusions": {
            "rl_only_prefixes": list(RL_ONLY_PREFIXES),
            "drop_exact_keys": sorted(DROP_EXACT_KEYS),
            "value_head_included": False,
        },
        "export_metadata": _json_safe(dict(export_metadata)),
    }


def export_checkpoint(
    train_config_path: str,
    ckpt_path: str,
    output_dir: str,
    save_adapter: bool,
    allow_non_final: bool = False,
    bundle_dir: str | None = None,
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
    if is_trainable_checkpoint:
        _validate_trainable_checkpoint_keys(model, state_dict)
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

    configured_extra_modules = tuple(model_cfg.get("extra_trainable_modules", ()) or ())
    extra_key_map, canonical_extra_prefixes = _collect_extra_trainable_keys(
        model, configured_extra_modules
    )

    lora_target = str(model_cfg.get("lora_target", "paligemma"))
    lora_specs = _get_lora_modules(model, lora_target)
    for lora_spec in lora_specs:
        if not hasattr(lora_spec.module, "merge_and_unload"):
            raise TypeError(
                f"OpenPI {lora_target} submodule "
                f"{lora_spec.adapter_dir_name} is not a PEFT LoRA model."
            )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    bundle_target = None
    bundle_staging = None
    if bundle_dir is not None:
        bundle_target, bundle_staging = _prepare_bundle_staging(bundle_dir)
        if bundle_target == output_path.expanduser().resolve():
            shutil.rmtree(bundle_staging)
            raise ValueError("--bundle_dir must differ from --output_dir.")

    try:
        adapter_dir_names = []
        bundle_adapter_specs = {}
        for lora_spec in lora_specs:
            if save_adapter:
                adapter_dir = output_path / lora_spec.adapter_dir_name
                lora_spec.module.save_pretrained(
                    str(adapter_dir), safe_serialization=True
                )
                adapter_dir_names.append(lora_spec.adapter_dir_name)
                print(f"Saved LoRA adapter to {adapter_dir}")
            if bundle_staging is not None:
                adapter_dir = bundle_staging / "adapters" / lora_spec.target_name
                lora_spec.module.save_pretrained(
                    str(adapter_dir), safe_serialization=True
                )
                bundle_adapter_specs[lora_spec.target_name] = _adapter_manifest(
                    adapter_dir, bundle_staging
                )

        for lora_spec in lora_specs:
            lora_spec.assign_module(lora_spec.module.merge_and_unload())
        is_tabero_sft = checkpoint_meta.get("method") in TABERO_SFT_METHODS
        if not is_tabero_sft:
            model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        model_path = output_path / "model.safetensors"
        _save_filtered_safetensors(
            model,
            str(model_path),
            base_model_path=str(model_cfg.model_path),
            allowed_extra_prefixes=("tactile_prefix_encoder.",)
            if is_tabero_sft
            else (),
            output_dtype=torch.bfloat16 if is_tabero_sft else None,
        )
        if is_tabero_sft:
            _validate_tabero_sft_export_schema(
                model_path,
                str(model_cfg.model_path),
            )
            reference_model_path = model_cfg.get("export_reference_model_path")
            if reference_model_path is not None:
                _validate_reference_schema(model_path, reference_model_path)
        _copy_assets(str(model_cfg.model_path), str(output_path))

        norm_stats_source, norm_asset_id = _resolve_norm_stats_source(
            model_cfg, checkpoint_meta
        )
        if is_tabero_sft:
            if norm_stats_source is None:
                raise ValueError(
                    "Tabero SFT export requires openpi_data.norm_stats_path."
                )
            _copy_norm_stats(norm_stats_source, output_path, norm_asset_id)
        if bundle_staging is not None and norm_stats_source is None:
            raise ValueError(
                "LoRA bundle export requires actor.model.openpi_data.norm_stats_path."
            )

        model_config = getattr(model, "config", None)
        if dataclasses.is_dataclass(model_config):
            with (output_path / "config.json").open("w", encoding="utf-8") as f:
                json.dump(_json_safe(model_config), f, indent=2)

        export_metadata = _build_export_metadata(
            train_config_path=os.path.abspath(train_config_path),
            ckpt_path=os.path.abspath(ckpt_path),
            source_model_path=str(model_cfg.model_path),
            checkpoint_meta=checkpoint_meta,
            model_path=model_path,
            lora_target=lora_target,
            adapter_dirs=adapter_dir_names if save_adapter else [],
            allow_non_final=allow_non_final,
        )
        with (output_path / "export_meta.json").open("w", encoding="utf-8") as f:
            json.dump(export_metadata, f, indent=2)

        if bundle_staging is not None:
            extra_path = bundle_staging / "extra_trainable.safetensors"
            _save_extra_trainable(
                model,
                extra_key_map,
                extra_path,
                output_dtype=torch.bfloat16 if is_tabero_sft else None,
            )
            bundle_norm_path = (
                bundle_staging / "assets" / norm_asset_id / "norm_stats.json"
            )
            bundle_norm_path.parent.mkdir(parents=True, exist_ok=True)
            norm_source_path = Path(norm_stats_source).expanduser().resolve()
            if norm_source_path.is_dir():
                norm_source_path = norm_source_path / "norm_stats.json"
            shutil.copy2(norm_source_path, bundle_norm_path)
            manifest = _build_lora_bundle_manifest(
                bundle_root=bundle_staging,
                model_cfg=model_cfg,
                checkpoint_meta=checkpoint_meta,
                export_metadata=export_metadata,
                lora_target=lora_target,
                adapter_specs=bundle_adapter_specs,
                extra_key_map=extra_key_map,
                canonical_extra_prefixes=canonical_extra_prefixes,
                extra_path=extra_path,
                norm_stats_path=bundle_norm_path,
                norm_asset_id=norm_asset_id,
                final_model_path=model_path,
            )
            with (bundle_staging / "manifest.json").open(
                "w", encoding="utf-8"
            ) as output:
                json.dump(manifest, output, indent=2, sort_keys=True)
            _write_checksums(bundle_staging)
            if bundle_target.exists():
                bundle_target.rmdir()
            bundle_staging.replace(bundle_target)
            bundle_staging = None
            print(f"Saved versioned LoRA bundle to {bundle_target}")

        print(f"Saved merged T2-VLA checkpoint to {output_path}")
    finally:
        if bundle_staging is not None and bundle_staging.exists():
            shutil.rmtree(bundle_staging)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config_path", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--bundle_dir",
        default=None,
        help=(
            "Optional new or empty directory for a versioned RLinf PEFT LoRA "
            "bundle loadable by T2-VLA."
        ),
    )
    parser.add_argument("--no_save_adapter", action="store_true")
    parser.add_argument(
        "--allow_non_final",
        action="store_true",
        help=(
            "Allow an explicitly non-final checkpoint to be exported for "
            "intermediate evaluation while preserving is_final=false provenance."
        ),
    )
    args = parser.parse_args()

    export_checkpoint(
        train_config_path=args.train_config_path,
        ckpt_path=args.ckpt_path,
        output_dir=args.output_dir,
        save_adapter=not args.no_save_adapter,
        allow_non_final=args.allow_non_final,
        bundle_dir=args.bundle_dir,
    )


if __name__ == "__main__":
    main()
