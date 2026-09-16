# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Export a configuration-driven DSRL actor with its raw observation contract."""

import argparse
import ctypes
import errno
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file
from torch import nn

from rlinf.models.embodiment.modules.compact_encoders import (
    CompactStateEncoder,
    LightweightImageEncoder64,
)
from rlinf.models.embodiment.modules.gaussian_policy import GaussianPolicy
from rlinf.models.embodiment.openpi.tactile_encoder import TactileTCNEncoder

ACTOR_PREFIXES = (
    "dsrl_action_noise_net.",
    "actor_image_encoder.",
    "actor_state_encoder.",
    "actor_tactile_encoder.",
)
AT_FDCWD = -100
RENAME_NOREPLACE = 1


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_actor(contract):
    """Construct the real training modules without allocating the frozen VLA."""
    c = contract
    actor = nn.Module()
    actor.dsrl_action_noise_net = GaussianPolicy(
        input_dim=(c["state_latent_dim"] if c["use_state"] else 0)
        + len(c["image_keys"]) * c["image_latent_dim"]
        + c["tactile_latent_dim"],
        output_dim=c["noise_dim"],
        hidden_dims=c["hidden_dims"],
        low=None,
        high=None,
        action_horizon=c["horizon"],
    )
    actor.actor_image_encoder = LightweightImageEncoder64(
        num_images=1, latent_dim=c["image_latent_dim"], image_size=64
    )
    if c["use_state"]:
        actor.actor_state_encoder = CompactStateEncoder(
            state_dim=c["state_dim"], hidden_dim=c["state_latent_dim"]
        )
    actor.actor_tactile_encoder = TactileTCNEncoder(
        input_dim=c["tactile_shape"][1] * 2,
        hidden_dim=c["tactile_latent_dim"],
        output_dim=c["tactile_latent_dim"],
        history_len=8,
        has_reference_frame=True,
        diff_from_reference=False,
    )
    return actor.to(dtype=getattr(torch, c["dtype"]))


def actor_contract(model_config, observation, state):
    """Derive architecture from training and spatial shapes from a captured observation."""
    c = model_config
    if c.get("use_dsrl") is not True or c.get("dsrl_use_tactile") is not True:
        raise ValueError("This exporter requires tactile DSRL training.")
    count = c["dsrl_num_images"]
    if type(count) is not int or count not in (1, 2, 3):
        raise ValueError("dsrl_num_images must be 1, 2, or 3.")
    image_keys = ["dsrl_raw_image", "dsrl_raw_wrist_image", "tactile_image"][:count]
    shapes = []
    for key in image_keys:
        value = observation[key]
        if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != torch.uint8:
            raise ValueError(f"{key} must be a raw HWC uint8 image.")
        shapes.append(list(value.shape))
    tactile = observation["tactile_marker_motion"]
    expected_tactile = (9, c["dsrl_tactile_input_dim"] // 2, 2)
    if c["dsrl_tactile_input_dim"] % 2 or tuple(tactile.shape) != expected_tactile:
        raise ValueError("Tactile observation and training input dimensions disagree.")
    use_state = c.get("dsrl_actor_use_state", True)
    if type(use_state) is not bool:
        raise ValueError("dsrl_actor_use_state must be boolean.")
    checked_keys = [*image_keys, "tactile_marker_motion"]
    if tactile.dtype != torch.float32:
        raise ValueError("Tactile observation must use float32.")
    if use_state:
        proprio = observation["state"]
        if (
            tuple(proprio.shape) != (c["dsrl_state_dim"],)
            or proprio.dtype != torch.float32
        ):
            raise ValueError("State observation shape or dtype mismatch.")
        checked_keys.append("state")
    if not all(torch.isfinite(observation[k]).all() for k in checked_keys):
        raise ValueError("Observation contains nonfinite values.")
    dtypes = {v.dtype for k, v in state.items() if k.startswith(ACTOR_PREFIXES)}
    if len(dtypes) != 1 or next(iter(dtypes)) not in (torch.bfloat16, torch.float32):
        raise ValueError("Actor weights must have one supported floating dtype.")
    return {
        "use_state": use_state,
        "image_keys": image_keys,
        "image_shapes": shapes,
        "state_key": "state",
        "state_dim": c["dsrl_state_dim"],
        "tactile_key": "tactile_marker_motion",
        "tactile_shape": list(expected_tactile),
        "image_latent_dim": c["dsrl_image_latent_dim"],
        "state_latent_dim": c["dsrl_state_latent_dim"],
        "tactile_latent_dim": c["dsrl_tactile_latent_dim"],
        "hidden_dims": list(c["dsrl_hidden_dims"]),
        "noise_dim": c["dsrl_action_noise_dim"],
        "horizon": c["action_horizon"],
        "num_steps": c["num_steps"],
        "dtype": str(next(iter(dtypes))).removeprefix("torch."),
        "image_preprocessing": "uint8_bilinear64_align_false_minus_one_one",
        "tactile_processing": "reference_plus_history8_no_difference_causal_tcn2_kernel3",
        "feature_order": "state_ordered_images_tactile"
        if use_state
        else "ordered_images_tactile",
    }


def validate_actor_state(state, contract):
    expected = build_actor(contract).state_dict()
    actor_state = {k: v for k, v in state.items() if k.startswith(ACTOR_PREFIXES)}
    if set(actor_state) != set(expected):
        raise ValueError(
            "Actor checkpoint keyspace differs from the training architecture."
        )
    for key, value in actor_state.items():
        if (
            value.shape != expected[key].shape
            or value.dtype != expected[key].dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Actor tensor shape/dtype/finite check failed: {key}")
    return {k: v.detach().cpu().contiguous() for k, v in actor_state.items()}


def _publish_directory_noreplace(source: Path, target: Path) -> None:
    if not sys.platform.startswith("linux"):
        raise RuntimeError(
            "atomic no-replace directory publish requires Linux renameat2"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as error:
        raise RuntimeError(
            "atomic no-replace directory publish requires libc renameat2"
        ) from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(target),
        RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(
            error_number,
            f"DSRL bundle output already exists: {target}",
            str(target),
        )
    if error_number in {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}:
        raise RuntimeError(
            "atomic no-replace directory publish is unsupported by this Linux "
            f"kernel/filesystem: {target}"
        )
    raise OSError(error_number, os.strerror(error_number), str(target))


def export_tabero_dsrl_bundle(
    *,
    trainable_checkpoint,
    train_config,
    observation_sample,
    output_dir,
    base_model,
    expected_base_model_sha256,
):
    """Export from a resolved training YAML and a raw single-observation tensor mapping."""
    checkpoint, config_path, obs_path = map(
        Path, (trainable_checkpoint, train_config, observation_sample)
    )
    base, output = Path(base_model).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    paths = [
        checkpoint,
        config_path,
        obs_path,
        base / "model.safetensors",
        base / "config.json",
        base / "export_meta.json",
    ]
    captured = {p: p.read_bytes() for p in paths if p != base / "model.safetensors"}
    hashes = {p: hashlib.sha256(data).hexdigest() for p, data in captured.items()}
    hashes[base / "model.safetensors"] = checkpoint_sha256(base / "model.safetensors")
    if hashes[base / "model.safetensors"] != expected_base_model_sha256:
        raise ValueError("Base model SHA-256 mismatch.")
    # Parse captured bytes and check their identities again before publishing.
    payload = torch.load(
        io.BytesIO(captured[checkpoint]), map_location="cpu", weights_only=True
    )
    cfg = OmegaConf.create(captured[config_path].decode())
    if "defaults" in cfg:
        raise ValueError(
            "Supply the resolved training YAML, not an uncomposed Hydra config."
        )
    c = OmegaConf.to_container(cfg.actor.model.openpi, resolve=True)
    obs = torch.load(
        io.BytesIO(captured[obs_path]), map_location="cpu", weights_only=True
    )
    state, metadata = payload["model"], payload["metadata"]
    task_id = cfg.env.train.init_params.task_id
    if type(task_id) is not int or task_id < 0 or metadata.get("method") != "dsrl":
        raise ValueError("Training task/method metadata mismatch.")
    step = metadata.get("global_step")
    if type(step) is not int or step < 0 or type(metadata.get("is_final")) is not bool:
        raise ValueError("Checkpoint step/finality metadata is invalid.")
    if "task_id" in metadata and metadata["task_id"] != task_id:
        raise ValueError("Checkpoint task_id disagrees with training config.")
    for key, value in cfg.actor.fsdp_config.get(
        "trainable_checkpoint_metadata", {}
    ).items():
        if metadata.get(key) != value:
            raise ValueError(f"Checkpoint metadata disagrees with config: {key}")
    for model_path in (cfg.actor.model.model_path, cfg.rollout.model.model_path):
        if Path(model_path).resolve() != base:
            raise ValueError("Training base path disagrees with export base.")
    contract = actor_contract(c, obs, state)
    actor_state = validate_actor_state(state, contract)
    base_config = json.loads(captured[base / "config.json"])
    base_meta = json.loads(captured[base / "export_meta.json"])
    if base_meta["model_sha256"] != expected_base_model_sha256:
        raise ValueError("Base export metadata SHA-256 mismatch.")
    fields = (
        "pi05",
        "discrete_state_input",
        "action_dim",
        "action_horizon",
        "max_token_len",
        "tactile_prefix_dim_in",
        "tactile_prefix_history",
        "tactile_prefix_encoder_type",
        "tactile_prefix_use_reference_frame",
        "tactile_prefix_diff_from_reference",
    )
    model = {k: base_config[k] for k in fields if k in base_config}
    for key, value in model.items():
        if key in c and c[key] != value:
            raise ValueError(f"Training/base model setting mismatch: {key}")
    if (
        c["config_name"] != base_config["config_name"]
        or contract["noise_dim"] != model["action_dim"]
    ):
        raise ValueError("Training/base config or noise dimension mismatch.")
    if (
        model.get("tactile_prefix_dim_in")
        != contract["tactile_shape"][0] * contract["tactile_shape"][1] * 2
    ):
        raise ValueError("Base VLA and DSRL tactile shapes disagree.")
    asset = base_meta["normalization_asset_id"]
    norm = base / "assets" / asset / "norm_stats.json"
    norm_hash = checkpoint_sha256(norm)
    hashes[norm] = norm_hash
    if base_meta.get("norm_stats_sha256") != norm_hash:
        raise ValueError("Base normalization metadata mismatch.")
    training_contract = cfg.actor.model.get("tabero_pi05_checkpoint_contract", {})
    for key, actual in [
        ("expected_norm_asset_id", asset),
        ("expected_norm_stats_sha256", norm_hash),
        ("expected_model_sha256", expected_base_model_sha256),
    ]:
        if key in training_contract and training_contract[key] != actual:
            raise ValueError(f"Training normalization/base contract mismatch: {key}")
    manifest = {
        "format": "tabero_dsrl_t2vla",
        "algorithm": "dsrl-sac",
        "task_id": task_id,
        "global_step": step,
        "is_final": metadata["is_final"],
        "source": {
            "checkpoint_sha256": hashes[checkpoint],
            "config_sha256": hashes[config_path],
            "observation_sha256": hashes[obs_path],
            "metadata": metadata,
            "semantics": {
                k: v
                for k, v in OmegaConf.to_container(cfg.algorithm, resolve=True).items()
                if k.startswith("dsrl_")
            },
        },
        "base": {
            "model_sha256": expected_base_model_sha256,
            "norm_sha256": norm_hash,
            "norm_asset_id": asset,
            "config_name": c["config_name"],
            "model": model,
            "use_quantile_norm": bool(model["pi05"]),
        },
        "actor_contract": contract,
        "actor_weights": "dsrl_actor.safetensors",
        "actor_shapes": {k: list(v.shape) for k, v in actor_state.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        save_file(actor_state, temporary / manifest["actor_weights"])
        manifest["actor_weights_sha256"] = checkpoint_sha256(
            temporary / manifest["actor_weights"]
        )
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        for path, digest in hashes.items():
            if checkpoint_sha256(path) != digest:
                raise ValueError(f"Export source changed: {path}")
        _publish_directory_noreplace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "trainable-checkpoint",
        "train-config",
        "observation-sample",
        "output-dir",
        "base-model",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-base-model-sha256", required=True)
    return parser.parse_args()


def main():
    print(json.dumps(export_tabero_dsrl_bundle(**vars(_parse_args())), indent=2))


if __name__ == "__main__":
    main()
