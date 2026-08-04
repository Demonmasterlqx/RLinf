# Copyright 2025 The RLinf Authors.
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
# openpi model configs

import logging
import os
import pathlib

import torch
from omegaconf import DictConfig

_TIED_STATE_DICT_MISSING_KEYS = {
    # This is a PyTorch tied-weight alias. The canonical 777/793 tensor OpenPI
    # safetensors schemas intentionally store only the owning embedding tensor.
    "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight",
}


def _validate_checkpoint_load_result(load_result, allowed_missing_prefixes) -> None:
    """Validate an intentionally non-strict OpenPI checkpoint load."""
    allowed_prefixes = tuple(allowed_missing_prefixes or ())
    disallowed_missing = [
        key
        for key in load_result.missing_keys
        if key not in _TIED_STATE_DICT_MISSING_KEYS
        and not any(key.startswith(prefix) for prefix in allowed_prefixes)
    ]
    if disallowed_missing or load_result.unexpected_keys:
        raise RuntimeError(
            "OpenPI checkpoint schema mismatch: "
            f"missing={disallowed_missing[:20]}, "
            f"unexpected={load_result.unexpected_keys[:20]}"
        )


def _apply_explicit_model_dtype(model, torch_dtype) -> None:
    """Honor an explicit actor precision without changing legacy null behavior."""
    if torch_dtype is None:
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        return
    if torch_dtype not in {torch.float32, torch.bfloat16, torch.float16}:
        raise ValueError(f"Unsupported OpenPI model dtype: {torch_dtype}.")
    model.to(dtype=torch_dtype)
    if torch_dtype == torch.float32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")


def _prepare_stage2_feature_model(model, actor_model_config) -> None:
    if not actor_model_config.rlt_stage2_encoder_only:
        return
    if not actor_model_config.use_rlt or not hasattr(model, "rlt_module"):
        raise ValueError("rlt_stage2_encoder_only=True requires an RLT module.")
    model.rlt_module.discard_decoder()


def get_model(cfg: DictConfig, torch_dtype=None):
    import glob

    import openpi.shared.download as download
    import openpi.transforms as transforms
    import safetensors
    from openpi.training import checkpoints as _checkpoints

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.openpi_action_model import (
        OpenPi0Config,
        OpenPi0ForRLActionPrediction,
    )

    # config
    config_name = getattr(cfg.openpi, "config_name", None)
    data_kwargs = getattr(cfg, "openpi_data", None)
    actor_train_config = get_openpi_config(
        config_name, model_path=cfg.model_path, data_kwargs=data_kwargs
    )

    actor_model_config = actor_train_config.model
    actor_model_config = OpenPi0Config(**actor_model_config.__dict__)
    override_model_config_kwargs = cfg.openpi
    if override_model_config_kwargs is not None:
        for key, val in override_model_config_kwargs.items():
            actor_model_config.__dict__[key] = val

    # load model
    checkpoint_dir = download.maybe_download(str(cfg.model_path))

    # Check if this is a checkpoint directory (saved by FSDP)
    # Check for model_state_dict/full_weights.pt (direct checkpoint) or actor/model_state_dict/full_weights.pt (from runner)
    full_weights_path = os.path.join(
        checkpoint_dir, "model_state_dict", "full_weights.pt"
    )
    actor_full_weights_path = os.path.join(
        checkpoint_dir, "actor", "model_state_dict", "full_weights.pt"
    )

    model: OpenPi0ForRLActionPrediction = OpenPi0ForRLActionPrediction(
        actor_model_config
    )
    # train expert only
    if actor_model_config.train_expert_only:
        model.freeze_vlm()

    # Load weights from checkpoint if it's a checkpoint directory, otherwise load from safetensors
    if os.path.exists(full_weights_path):
        # Direct checkpoint directory
        model_state_dict = torch.load(full_weights_path, map_location="cpu")
        load_result = model.load_state_dict(model_state_dict, strict=False)
    elif os.path.exists(actor_full_weights_path):
        # Checkpoint directory from runner
        model_state_dict = torch.load(actor_full_weights_path, map_location="cpu")
        load_result = model.load_state_dict(model_state_dict, strict=False)
    else:
        # Original model directory with safetensors files
        weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
        if not weight_paths:
            weight_paths = [os.path.join(checkpoint_dir, "model.safetensors")]
        all_state_dict = {}
        for weight_path in weight_paths:
            state_dict = safetensors.torch.load_file(weight_path, device="cpu")
            all_state_dict.update(state_dict)
        load_result = model.load_state_dict(all_state_dict, strict=False)

    allowed_missing_prefixes = cfg.get("checkpoint_load_allowed_missing_prefixes", None)
    if allowed_missing_prefixes is not None:
        _validate_checkpoint_load_result(load_result, allowed_missing_prefixes)

    _prepare_stage2_feature_model(model, actor_model_config)

    if actor_model_config.rlt_train_module_only:
        model.freeze_non_rlt_parameters()

    _apply_explicit_model_dtype(model, torch_dtype)
    dtype_counts = {}
    for parameter in model.parameters():
        dtype_counts[str(parameter.dtype)] = (
            dtype_counts.get(str(parameter.dtype), 0) + 1
        )
    logging.info(
        "OpenPI dtype audit: requested=%s parameter_tensor_counts=%s "
        "tf32_matmul=%s tf32_cudnn=%s",
        torch_dtype,
        dtype_counts,
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    # fsdp replace
    # model.paligemma_with_expert.replace_gemma_decoder_layers()
    # load data stats
    data_config = actor_train_config.data.create(
        actor_train_config.assets_dirs, actor_model_config
    )
    norm_stats_path = (
        data_kwargs.get("norm_stats_path") if data_kwargs is not None else None
    )
    if norm_stats_path is not None:
        norm_stats = data_config.norm_stats
        if norm_stats is None:
            norm_dir = pathlib.Path(norm_stats_path).expanduser()
            if norm_dir.is_file():
                norm_dir = norm_dir.parent
            norm_stats = _checkpoints.load_norm_stats(norm_dir.parent, norm_dir.name)
    else:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)
    # wrappers
    repack_transforms = transforms.Group()
    default_prompt = None
    model.setup_wrappers(
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
    )

    return model
