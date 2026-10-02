# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Validation for synchronous RLT task weighting using the shared controller."""

from omegaconf import OmegaConf

from rlinf.utils.multi_task import options, validate_environments


def validate_config(cfg) -> None:
    """Keep v1 on the complete-episode, transition-replay RLT path."""
    options(cfg)
    required = {
        "runner.task_type": "embodied",
        "actor.training_backend": "fsdp",
        "actor.model.model_type": "rlt_mlp_policy",
        "algorithm.loss_type": "rlt_ac",
        "algorithm.adv_type": "embodied_sac",
        "algorithm.rlt_schedule.enable": True,
        "algorithm.rlt_schedule.transition_replay": True,
        "algorithm.rlt_route.actor_scope": "full_task",
        "rollout.collect_transitions": True,
        "rollout.rlt_feature_model.model_type": "openpi",
        "rollout.rlt_feature_model.openpi.use_rlt": True,
        "algorithm.entropy_tuning.alpha_type": "fixed_alpha",
        "algorithm.entropy_tuning.initial_alpha": 0.0,
    }
    for key, expected in required.items():
        if OmegaConf.select(cfg, key) != expected:
            raise ValueError(f"RLT multi_task requires {key}={expected!r}.")
    for key in (
        "runner.use_training_pipeline",
        "runner.only_eval",
        "actor.enable_sft_co_train",
        "reward.use_reward_model",
        "algorithm.replay_buffer.enable_preload",
    ):
        if OmegaConf.select(cfg, key, default=False):
            raise ValueError(f"RLT multi_task does not support {key}.")
    for key in ("algorithm.demo_buffer", "rollout.expert_model"):
        if OmegaConf.select(cfg, key) is not None:
            raise ValueError(f"RLT multi_task does not support {key}.")
    for key in ("algorithm.q_head_type", "actor.model.q_head_type"):
        if OmegaConf.select(cfg, key, default="default") != "default":
            raise ValueError("RLT multi_task requires the standard twin-Q head.")
    if (
        OmegaConf.select(cfg, "algorithm.bootstrap_type", default="standard")
        != "standard"
    ):
        raise ValueError("RLT multi_task requires standard terminal bootstrapping.")
    validate_environments(cfg)
