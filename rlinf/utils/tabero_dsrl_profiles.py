# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Strict training profiles accepted by Tabero DSRL export and evaluation."""

from dataclasses import dataclass
from typing import Any

FORMAL_8GPU_50STEP_PROFILE = "formal_8gpu_50step"
TASK5_4GPU_40STEP_SMALL_PROFILE = "task5_4gpu_40step_small"


@dataclass(frozen=True)
class TaberoDSRLTrainingProfile:
    """Immutable final-checkpoint and resolved-config contract."""

    name: str
    allowed_task_ids: tuple[int, ...]
    global_step: int
    actor_world_size: int
    training_config_template: str
    config_requirements: tuple[tuple[str, Any], ...]

    def training_config(self, task_id: int) -> str:
        """Return the exact checkpoint config identifier for ``task_id``."""
        return self.training_config_template.format(task_id=task_id)


_PROFILES = {
    FORMAL_8GPU_50STEP_PROFILE: TaberoDSRLTrainingProfile(
        name=FORMAL_8GPU_50STEP_PROFILE,
        allowed_task_ids=(0, 5),
        global_step=50,
        actor_world_size=4,
        training_config_template=(
            "isaaclab_pi0_dsrl_tacfield_tabero_task{task_id}_firm_8gpu_50step"
        ),
        config_requirements=(
            ("runner.max_epochs", 50),
            ("runner.save_interval", 10),
            ("env.train.total_num_envs", 84),
            ("env.train.rollout_epoch", 2),
            ("algorithm.update_epoch", 200),
            ("algorithm.gamma", 0.999),
            ("algorithm.tau", 0.005),
        ),
    ),
    TASK5_4GPU_40STEP_SMALL_PROFILE: TaberoDSRLTrainingProfile(
        name=TASK5_4GPU_40STEP_SMALL_PROFILE,
        allowed_task_ids=(5,),
        global_step=40,
        actor_world_size=2,
        training_config_template=(
            "isaaclab_pi0_dsrl_tacfield_tabero_task5_firm_4gpu_40step_small"
        ),
        config_requirements=(
            ("cluster.component_placement.actor", "2-3"),
            ("cluster.component_placement.rollout", "0-1"),
            ("cluster.component_placement.env", "0-1"),
            ("runner.max_epochs", 40),
            ("runner.save_interval", 10),
            ("env.train.total_num_envs", 20),
            ("env.train.rollout_epoch", 1),
            ("env.train.max_steps_per_rollout_epoch", 360),
            ("env.train.max_episode_steps", 360),
            ("env.train.init_params.max_episode_steps", 360),
            ("env.train.init_params.marker_history_len", 8),
            ("env.train.init_params.combined_marker_count", 198),
            ("env.train.init_params.main_image_key", "agentview_rgb"),
            ("env.train.init_params.wrist_image_key", "eye_in_hand_rgb"),
            (
                "env.train.init_params.marker_motion_key",
                "gripper_marker_motion",
            ),
            ("algorithm.update_epoch", 20),
            ("algorithm.gamma", 0.999),
            ("algorithm.tau", 0.005),
            ("algorithm.replay_buffer.min_buffer_size", 5),
            ("algorithm.train_actor_steps", 10),
            ("actor.global_batch_size", 20),
            ("actor.micro_batch_size", 2),
        ),
    ),
}
TABERO_DSRL_TRAINING_PROFILE_CHOICES = tuple(_PROFILES)


def resolve_tabero_dsrl_training_profile(
    name: str,
    task_id: int,
) -> TaberoDSRLTrainingProfile:
    """Resolve one explicit profile and reject unsupported task/profile pairs."""
    if not isinstance(name, str) or name not in _PROFILES:
        raise ValueError(
            "DSRL training profile must be one of "
            f"{list(TABERO_DSRL_TRAINING_PROFILE_CHOICES)}; got {name!r}"
        )
    profile = _PROFILES[name]
    if task_id not in profile.allowed_task_ids:
        allowed = ", ".join(f"Task {value}" for value in profile.allowed_task_ids)
        raise ValueError(
            f"DSRL training profile {name!r} supports only {allowed}; "
            f"got task_id={task_id!r}"
        )
    return profile
