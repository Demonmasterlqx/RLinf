#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Compare deployed DSRL tensors with the actual RLinf modules and preprocessing."""

import argparse
import json
from pathlib import Path

import torch
from openpi.policies.tabero_dsrl_policy import TaberoDSRLBundle

from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0ForRLActionPrediction,
)
from rlinf.utils.ckpt_convertor.export_tabero_dsrl_for_t2vla import build_actor


@torch.no_grad()
def compare_actor(actor, observation, *, tolerance=0.0):
    """Compare preprocessing, features, means and noise with identical weights."""
    from dataclasses import asdict

    c = actor.contract
    rlinf = build_actor(asdict(c)).to(actor.device).eval()
    rlinf.load_state_dict(actor.state_dict(), strict=True)
    images = OpenPi0ForRLActionPrediction._preprocess_dsrl_images(
        None, [observation[k].to(actor.device)[None] for k in c.image_keys]
    )
    dtype = getattr(torch, c.dtype)
    images = images.to(dtype)
    state = (
        observation[c.state_key].to(actor.device, dtype)[None] if c.use_state else None
    )
    tactile = (
        observation[c.tactile_key]
        .to(actor.device, dtype)
        .reshape(1, 9, c.tactile_input_dim)
    )
    t_images, t_state, t_tactile = actor.preprocess(observation)
    image_features = OpenPi0ForRLActionPrediction._encode_dsrl_image_views(
        images, rlinf.actor_image_encoder
    )
    state_features = rlinf.actor_state_encoder(state) if c.use_state else None
    tactile_features = rlinf.actor_tactile_encoder(tactile)
    parts = [image_features, tactile_features]
    if c.use_state:
        parts.insert(0, state_features)
    features = torch.cat(parts, dim=-1)
    mean = rlinf.dsrl_action_noise_net.mean_layer(
        rlinf.dsrl_action_noise_net.shared_net(features)
    )
    noise, _ = rlinf.dsrl_action_noise_net.sample(features, deterministic=True)
    stages = {}
    pairs = {
        "image_input": (images, t_images),
        "tactile_input": (tactile, t_tactile),
        "image_features": (image_features, actor.actor_image_encoder(t_images)),
        "tactile_features": (tactile_features, actor.actor_tactile_encoder(t_tactile)),
        "features": (features, actor.features(observation)),
        "mean": (mean, actor.mean(observation)),
        "noise": (noise, actor.noise(observation)),
    }
    if c.use_state:
        pairs["state_input"] = (state, t_state)
        pairs["state_features"] = (state_features, actor.actor_state_encoder(t_state))
    for key, (left, right) in pairs.items():
        error = (
            (left.float() - right.float()).abs().max().item()
            if left.shape == right.shape
            else float("inf")
        )
        stages[key] = {
            "shape": list(left.shape),
            "dtype": str(left.dtype),
            "max_abs": error,
            "passed": error <= tolerance and left.dtype == right.dtype,
        }
    return {
        "passed": all(x["passed"] for x in stages.values()),
        "tolerance": tolerance,
        "stages": stages,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = TaberoDSRLBundle.load(
        args.bundle, base_checkpoint_dir=args.base_checkpoint
    )
    observation = torch.load(args.observation, map_location="cpu", weights_only=True)
    result = compare_actor(
        bundle.actor.to(args.device), observation, tolerance=args.tolerance
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
