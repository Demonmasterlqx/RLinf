#!/usr/bin/env python3
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

"""Compare RLinf and T2-VLA Tabero DSRL inference numerically."""

import argparse
import hashlib
import json
import subprocess
import time
import traceback
import types
from pathlib import Path
from typing import Any

import torch
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.policies.tabero_dsrl_policy import TaberoDSRLActor
from torch import nn

from rlinf.models.embodiment.modules.compact_encoders import (
    CompactStateEncoder,
    LightweightImageEncoder64,
)
from rlinf.models.embodiment.modules.gaussian_policy import GaussianPolicy
from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0ForRLActionPrediction,
)
from rlinf.models.embodiment.openpi.tactile_encoder import TactileTCNEncoder

REPO_ROOT = Path(__file__).resolve().parents[2]
T2_REPO = REPO_ROOT.parent / "T2-VLA"
DEFAULT_BASE_CHECKPOINT = (
    REPO_ROOT.parent / "models" / "pi0_lora_tacfield_tabero_safetensors"
)
ACTOR_WEIGHT_KEY_COUNT = 48


class _RLinfActor(nn.Module):
    """The four real RLinf modules used by the DSRL actor."""

    def __init__(self) -> None:
        super().__init__()
        self.dsrl_action_noise_net = GaussianPolicy(
            input_dim=256,
            output_dim=32,
            hidden_dims=(128, 128, 128),
            low=None,
            high=None,
            action_horizon=50,
        )
        self.actor_image_encoder = LightweightImageEncoder64(
            num_images=1, latent_dim=64, image_size=64
        )
        self.actor_state_encoder = CompactStateEncoder(state_dim=7, hidden_dim=64)
        self.actor_tactile_encoder = TactileTCNEncoder(
            input_dim=396,
            hidden_dim=64,
            output_dim=64,
            history_len=8,
            has_reference_frame=True,
            diff_from_reference=False,
        )
        self.to(dtype=torch.bfloat16)


class _RLinfPreprocessor:
    """Minimal receiver for the real RLinf DSRL preprocessing methods."""

    _validate_dsrl_tactile = OpenPi0ForRLActionPrediction._validate_dsrl_tactile


def _git_sha(repository: Path, revision: str = "HEAD") -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", revision],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _fixed_raw_observation() -> dict[str, torch.Tensor]:
    image = torch.arange(256 * 256 * 3, dtype=torch.int64)
    image = image.remainder(256).to(torch.uint8).reshape(256, 256, 3)
    wrist_image = torch.flip(image, dims=(0, 1)).contiguous()
    state = torch.linspace(-0.75, 0.75, 7, dtype=torch.float32)
    tactile = torch.linspace(-0.5, 0.5, 9 * 198 * 2, dtype=torch.float32).reshape(
        9, 198, 2
    )
    return {
        "dsrl_raw_image": image,
        "dsrl_raw_wrist_image": wrist_image,
        "state": state,
        "tactile_marker_motion": tactile,
    }


def _deterministic_actor_weights(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], str]:
    digest = hashlib.sha256()
    result = {}
    for key_index, (key, target) in enumerate(state_dict.items()):
        values = torch.arange(target.numel(), dtype=torch.float32)
        values = ((values + key_index * 17).remainder(257) - 128) / 4096
        tensor = values.reshape(target.shape).to(torch.bfloat16)
        result[key] = tensor
        digest.update(key.encode())
        digest.update(tensor.contiguous().view(torch.uint16).numpy().tobytes())
    return result, digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
    }


def _compare(rlinf: torch.Tensor, t2: torch.Tensor, tolerance: float) -> dict[str, Any]:
    if rlinf.shape != t2.shape:
        max_abs = float("inf")
    else:
        difference = (rlinf.float() - t2.float()).abs()
        max_abs = float(difference.max().item()) if difference.numel() else 0.0
    return {
        "max_abs": max_abs,
        "passed": max_abs <= tolerance,
        "rlinf": _metadata(rlinf),
        "t2": _metadata(t2),
    }


@torch.no_grad()
def _run_actor_parity(
    device: torch.device, tolerance: float
) -> tuple[dict, dict, dict]:
    t2_actor = TaberoDSRLActor().to(device).eval()
    rlinf_actor = _RLinfActor().to(device).eval()
    if list(rlinf_actor.state_dict()) != list(t2_actor.state_dict()):
        raise RuntimeError("RLinf and T2 actor state_dict keyspaces differ.")
    if len(t2_actor.state_dict()) != ACTOR_WEIGHT_KEY_COUNT:
        raise RuntimeError(
            f"Expected {ACTOR_WEIGHT_KEY_COUNT} actor keys, "
            f"found {len(t2_actor.state_dict())}."
        )

    weights, weights_sha256 = _deterministic_actor_weights(t2_actor.state_dict())
    t2_actor.load_state_dict(weights, strict=True)
    rlinf_actor.load_state_dict(weights, strict=True)

    raw = _fixed_raw_observation()
    t2_image, t2_state, t2_tactile = t2_actor.preprocess(raw)
    preprocessor = _RLinfPreprocessor()
    batched = {
        "images": [
            raw["dsrl_raw_image"].unsqueeze(0).to(device),
            raw["dsrl_raw_wrist_image"].unsqueeze(0).to(device),
        ],
        "states": raw["state"].unsqueeze(0).to(device),
        "tactile_marker_motion": raw["tactile_marker_motion"].unsqueeze(0).to(device),
    }
    rlinf_image = OpenPi0ForRLActionPrediction._preprocess_dsrl_images(
        preprocessor, batched["images"], train=False
    ).to(device=device, dtype=torch.bfloat16)
    rlinf_state = OpenPi0ForRLActionPrediction._preprocess_states(
        preprocessor, batched["states"]
    ).to(device=device)
    rlinf_tactile = OpenPi0ForRLActionPrediction._prepare_dsrl_tactile(
        preprocessor,
        batched,
        batch_size=1,
        encoder=rlinf_actor.actor_tactile_encoder,
    )

    rlinf_state_features = rlinf_actor.actor_state_encoder(rlinf_state)
    rlinf_image_features = OpenPi0ForRLActionPrediction._encode_dsrl_image_views(
        rlinf_image, rlinf_actor.actor_image_encoder
    )
    rlinf_tactile_features = rlinf_actor.actor_tactile_encoder(rlinf_tactile)
    rlinf_features = torch.cat(
        [rlinf_state_features, rlinf_image_features, rlinf_tactile_features], dim=-1
    )
    rlinf_mean = rlinf_actor.dsrl_action_noise_net(rlinf_features).mean
    rlinf_deterministic = torch.tanh(rlinf_mean)
    rlinf_noise = rlinf_deterministic[:, None, :].expand(-1, 50, -1)

    t2_state_features = t2_actor.actor_state_encoder(t2_state)
    t2_image_features = t2_actor.actor_image_encoder(t2_image)
    t2_tactile_features = t2_actor.actor_tactile_encoder(t2_tactile)
    t2_mean = t2_actor.mean(raw)
    t2_deterministic = torch.tanh(t2_mean)
    t2_noise = t2_actor.noise(raw)

    pairs = {
        "image_input": (rlinf_image, t2_image),
        "main_image_input": (rlinf_image[:, 0], t2_image[:, 0]),
        "wrist_image_input": (rlinf_image[:, 1], t2_image[:, 1]),
        "state_input": (rlinf_state, t2_state),
        "tactile_input": (rlinf_tactile, t2_tactile),
        "state_features": (rlinf_state_features, t2_state_features),
        "image_features": (rlinf_image_features, t2_image_features),
        "main_image_features": (
            rlinf_image_features[:, :64],
            t2_image_features[:, :64],
        ),
        "wrist_image_features": (
            rlinf_image_features[:, 64:],
            t2_image_features[:, 64:],
        ),
        "tactile_features": (rlinf_tactile_features, t2_tactile_features),
        "gaussian_mean": (rlinf_mean, t2_mean),
        "deterministic_noise": (rlinf_deterministic, t2_deterministic),
        "broadcast_noise": (rlinf_noise, t2_noise),
    }
    stages = {
        name: _compare(rlinf_value, t2_value, tolerance)
        for name, (rlinf_value, t2_value) in pairs.items()
    }
    fixture = {
        "raw_image_shape": list(raw["dsrl_raw_image"].shape),
        "raw_wrist_image_shape": list(raw["dsrl_raw_wrist_image"].shape),
        "raw_state_shape": list(raw["state"].shape),
        "raw_tactile_shape": list(raw["tactile_marker_motion"].shape),
        "weight_key_count": len(weights),
        "weight_dtype": "bfloat16",
        "weights_sha256": weights_sha256,
    }
    artifacts = {
        "raw": raw,
        "weights": weights,
        "rlinf_noise": rlinf_noise,
        "t2_noise": t2_noise,
    }
    return stages, fixture, artifacts


def _legacy_rlinf_preprocess_adapter(self, observation, *, train=True):
    """Present T2's new six-item result to RLinf's five-item compatibility shim."""
    result = PI0Pytorch._preprocess_observation(self, observation, train=train)
    if len(result) != 6:
        raise RuntimeError(
            "Expected current T2 _preprocess_observation to return six items, "
            f"got {len(result)}."
        )
    return result[:5]


def _load_pi0_model(checkpoint: Path, device: torch.device):
    from safetensors.torch import load_model

    from rlinf.models.embodiment.openpi.openpi_action_model import OpenPi0Config

    config_values = json.loads((checkpoint / "config.json").read_text())
    config_values.update(
        {
            "num_steps": 10,
            "use_dsrl": True,
            "dsrl_use_tactile": True,
            "dsrl_num_images": 2,
            "dsrl_state_dim": 7,
            "dsrl_action_noise_dim": 32,
            "dsrl_num_q_heads": 10,
            "dsrl_agg_q": "mean",
            "dsrl_image_latent_dim": 64,
            "dsrl_state_latent_dim": 64,
            "dsrl_tactile_latent_dim": 64,
            "dsrl_hidden_dims": (128, 128, 128),
        }
    )
    model = OpenPi0ForRLActionPrediction(OpenPi0Config(**config_values))
    missing, unexpected = load_model(
        model, checkpoint / "model.safetensors", strict=False, device="cpu"
    )
    allowed_missing_prefixes = (
        "dsrl_action_noise_net.",
        "actor_image_encoder.",
        "actor_state_encoder.",
        "actor_tactile_encoder.",
        "critic_image_encoder.",
        "critic_state_encoder.",
        "critic_tactile_encoder.",
        "q_head.",
    )
    critical_missing = [
        key for key in missing if not key.startswith(allowed_missing_prefixes)
    ]
    if critical_missing or unexpected:
        raise RuntimeError(
            "Base checkpoint key mismatch: "
            f"critical_missing={critical_missing[:20]}, unexpected={unexpected[:20]}"
        )
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model.to(device=device)
    model.eval()
    return model, {"allowed_missing_keys": len(missing), "unexpected_keys": 0}


def _transformed_observation(model, raw: dict, device: torch.device):
    from openpi.models import model as model_api

    image = raw["dsrl_raw_image"][:224, :224]
    image = image.permute(2, 0, 1).to(device=device, dtype=torch.float32)
    image = (image / 255.0 * 2.0 - 1.0).unsqueeze(0)
    wrist_image = raw["dsrl_raw_wrist_image"][:224, :224]
    wrist_image = wrist_image.permute(2, 0, 1).to(device=device, dtype=torch.float32)
    wrist_image = (wrist_image / 255.0 * 2.0 - 1.0).unsqueeze(0)
    state = torch.zeros(1, 32, dtype=torch.float32, device=device)
    state[:, :7] = raw["state"].to(device)
    tactile_prefix = raw["tactile_marker_motion"].reshape(1, 9, 396).to(device)
    tokenized_prompt = torch.arange(1, 49, dtype=torch.int64, device=device)[None]
    tokenized_prompt_mask = torch.ones(1, 48, dtype=torch.bool, device=device)
    observation = model_api.Observation(
        images={
            "base_0_rgb": image,
            "left_wrist_0_rgb": wrist_image,
            "right_wrist_0_rgb": torch.zeros_like(image),
        },
        image_masks={
            "base_0_rgb": torch.ones(1, dtype=torch.bool, device=device),
            "left_wrist_0_rgb": torch.ones(1, dtype=torch.bool, device=device),
            "right_wrist_0_rgb": torch.zeros(1, dtype=torch.bool, device=device),
        },
        state=state,
        tactile_prefix=tactile_prefix,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )
    transformed = {
        "images/base_0_rgb": image,
        "images/left_wrist_0_rgb": wrist_image,
        "images/right_wrist_0_rgb": observation.images["right_wrist_0_rgb"],
        "image_masks/base_0_rgb": observation.image_masks["base_0_rgb"],
        "image_masks/left_wrist_0_rgb": observation.image_masks["left_wrist_0_rgb"],
        "image_masks/right_wrist_0_rgb": observation.image_masks["right_wrist_0_rgb"],
        "state": state,
        "tactile_prefix": tactile_prefix,
        "tokenized_prompt": tokenized_prompt,
        "tokenized_prompt_mask": tokenized_prompt_mask,
    }
    return observation, transformed


@torch.no_grad()
def _run_final_pi0_parity(
    checkpoint: Path,
    device: torch.device,
    tolerance: float,
    artifacts: dict,
) -> tuple[dict, dict]:
    weights_path = checkpoint / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Base PI0 weights do not exist: {weights_path}")
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"PI0 config does not exist: {config_path}")

    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, load_receipt = _load_pi0_model(checkpoint, device)
    model.load_state_dict(artifacts["weights"], strict=False)
    observation, transformed = _transformed_observation(model, artifacts["raw"], device)

    t2_steps = []
    original_denoise_step = model.denoise_step

    def capture_t2_step(self, state, prefix_masks, cache, x_t, timestep):
        velocity = original_denoise_step(state, prefix_masks, cache, x_t, timestep)
        t2_steps.append(
            (x_t.detach().cpu(), timestep.detach().cpu(), velocity.detach().cpu())
        )
        return velocity

    model.denoise_step = types.MethodType(capture_t2_step, model)
    t2_started = time.perf_counter()
    try:
        t2_action = PI0Pytorch.sample_actions(
            model,
            device,
            observation,
            noise=artifacts["t2_noise"],
            num_steps=10,
        )
    finally:
        del model.__dict__["denoise_step"]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t2_seconds = time.perf_counter() - t2_started

    # T2@21893e0 returns tactile_prefix as a sixth preprocessing item. RLinf's
    # compatibility helper still expects the five-item API from its pinned openpi.
    # The adapter only removes that duplicate return value; RLinf's real helper
    # reads the same tactile_prefix from `observation` immediately afterwards.
    had_instance_override = "_preprocess_observation" in model.__dict__
    prior_override = model.__dict__.get("_preprocess_observation")
    model._preprocess_observation = types.MethodType(  # noqa: SLF001
        _legacy_rlinf_preprocess_adapter, model
    )
    rlinf_steps = []
    original_get_velocity = model.get_velocity

    def capture_rlinf_step(self, state, x_t, timestep, prefix_masks, cache):
        velocity, suffix = original_get_velocity(
            state, x_t, timestep, prefix_masks, cache
        )
        rlinf_steps.append(
            (x_t.detach().cpu(), timestep.detach().cpu(), velocity.detach().cpu())
        )
        return velocity, suffix

    model.get_velocity = types.MethodType(capture_rlinf_step, model)
    try:
        rlinf_started = time.perf_counter()
        rlinf_result = OpenPi0ForRLActionPrediction.sample_actions(
            model,
            observation,
            noise=artifacts["rlinf_noise"],
            mode="eval",
            compute_values=False,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        rlinf_seconds = time.perf_counter() - rlinf_started
    finally:
        del model.__dict__["get_velocity"]
        if had_instance_override:
            model.__dict__["_preprocess_observation"] = prior_override
        else:
            del model.__dict__["_preprocess_observation"]

    rlinf_action = rlinf_result["actions"]
    stage = _compare(rlinf_action, t2_action, tolerance)
    stage.update(
        {
            "status": "passed" if stage["passed"] else "failed",
            "num_denoise_steps": 10,
            "t2_seconds": t2_seconds,
            "rlinf_seconds": rlinf_seconds,
            "same_model_instance": True,
            "t2_method": "PI0Pytorch.sample_actions",
            "rlinf_method": "OpenPi0ForRLActionPrediction.sample_actions",
            "rlinf_action_source": "return['actions']",
            "compatibility_adapter": (
                "T2 six-item preprocessing result adapted to RLinf's pinned "
                "five-item helper; tactile_prefix remains read from observation"
            ),
        }
    )
    stage["step_diagnostics"] = [
        {
            "step": index,
            "input_max_abs": _compare(rlinf[0], t2[0], tolerance)["max_abs"],
            "timestep_max_abs": _compare(rlinf[1], t2[1], tolerance)["max_abs"],
            "velocity_max_abs": _compare(rlinf[2], t2[2], tolerance)["max_abs"],
        }
        for index, (rlinf, t2) in enumerate(zip(rlinf_steps, t2_steps, strict=True))
    ]
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
    else:
        peak_allocated = None
        peak_reserved = None
    evidence = {
        "base_model_sha256": _sha256(weights_path),
        "load_receipt": load_receipt,
        "elapsed_seconds": time.perf_counter() - started,
        "cuda_peak_allocated_bytes": peak_allocated,
        "cuda_peak_reserved_bytes": peak_reserved,
        "transformed_observation": {
            key: _metadata(value)
            for key, value in transformed.items()
            if torch.is_tensor(value)
        },
    }
    return stage, evidence


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--with-final-pi0", action="store_true")
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    device = torch.device(args.device)
    stages, fixture, artifacts = _run_actor_parity(device, args.tolerance)
    final_evidence = None
    final_error = False
    if args.with_final_pi0:
        try:
            stages["final_pi0_action"], final_evidence = _run_final_pi0_parity(
                args.base_checkpoint.resolve(),
                device,
                args.tolerance,
                artifacts,
            )
        except Exception as error:  # noqa: BLE001 - persist diagnostic contract
            final_error = True
            stages["final_pi0_action"] = {
                "status": "error",
                "passed": False,
                "max_abs": None,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
    else:
        stages["final_pi0_action"] = {
            "status": "not_requested",
            "passed": None,
            "max_abs": None,
        }
    actor_passed = all(
        stage["passed"] for name, stage in stages.items() if name != "final_pi0_action"
    )
    final_passed = stages["final_pi0_action"]["passed"] if args.with_final_pi0 else True
    report = {
        "schema_version": 1,
        "passed": actor_passed and final_passed,
        "tolerance": args.tolerance,
        "device": str(args.device),
        "with_final_pi0": args.with_final_pi0,
        "revisions": {
            "base": {
                "git_sha": _git_sha(T2_REPO, "HEAD^"),
                "source": str(T2_REPO),
            },
            "t2": {"git_sha": _git_sha(T2_REPO), "source": str(T2_REPO)},
            "rlinf": {
                "git_sha": _git_sha(REPO_ROOT),
                "source": str(REPO_ROOT),
            },
        },
        "base_checkpoint": {"path": str(args.base_checkpoint.resolve())},
        "fixture": fixture,
        "stages": stages,
    }
    if final_evidence is not None:
        report["final_pi0_evidence"] = final_evidence
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if final_error:
        return 2
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
