# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Synchronous DSRL task weighting without changing SAC targets or observations."""

import torch
from omegaconf import OmegaConf

from rlinf.utils.dsrl_transition import TABERO_DSRL_CHUNK_BOUNDARY_MODE
from rlinf.utils.multi_task import options, validate_environments


def validate_config(cfg) -> None:
    """Gate multi-task SAC to the existing terminal-safe DSRL adapters."""
    options(cfg)
    required = {
        "runner.task_type": "embodied",
        "actor.training_backend": "fsdp",
        "actor.model.model_type": "openpi",
        "actor.model.openpi.use_dsrl": True,
        "algorithm.adv_type": "embodied_sac",
        "algorithm.loss_type": "embodied_sac",
        "algorithm.reward_type": "chunk_level",
        "rollout.collect_transitions": True,
    }
    for key, expected in required.items():
        if OmegaConf.select(cfg, key) != expected:
            raise ValueError(f"DSRL multi_task requires {key}={expected!r}.")
    for key in (
        "runner.use_training_pipeline",
        "runner.only_eval",
        "actor.enable_sft_co_train",
        "reward.use_reward_model",
        "algorithm.replay_buffer.enable_preload",
    ):
        if OmegaConf.select(cfg, key, default=False):
            raise ValueError(f"DSRL multi_task does not support {key}.")
    if OmegaConf.select(cfg, "algorithm.demo_buffer") is not None:
        raise ValueError("DSRL multi_task does not support demo_buffer.")
    if OmegaConf.select(cfg, "algorithm.q_head_type", default="default") != "default":
        raise ValueError("DSRL multi_task requires the standard SAC Q head.")
    if OmegaConf.select(cfg, "algorithm.dsrl_transition_boundary_semantics") is None:
        raise ValueError("DSRL multi_task requires explicit transition semantics.")
    validate_environments(cfg)
    for split in ("train", "eval"):
        if split == "eval" and cfg.runner.get("val_check_interval", -1) <= 0:
            continue
        env = cfg.env[split]
        boundary = (
            TABERO_DSRL_CHUNK_BOUNDARY_MODE
            if env.init_params.id == "Isaac-Libero-Franka-Hybrid-Tactile-v0"
            else "terminal_safe_v1"
        )
        if env.init_params.get("chunk_boundary_mode") != boundary:
            raise ValueError(f"DSRL multi_task requires {boundary} boundaries.")


def weighted_sac_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weight each transition, averaging its Q heads before the batch reduction."""
    if values.ndim < 1 or weights.shape != (values.shape[0],):
        raise ValueError("SAC task weights must be one scalar per transition.")
    per_sample = values.float().reshape(values.shape[0], -1).mean(dim=1)
    return (per_sample * weights.detach().to(per_sample)).mean()
