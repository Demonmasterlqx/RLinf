# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Reward semantics and audits for Tabero OpenPI DSRL training."""

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

DSRL_REWARD_SEMANTICS = "discounted_alive_masked_v1"

DSRL_REWARD_AUDIT_FIELDS = (
    "transition_count",
    "primitive_count",
    "nonzero_primitive_reward_count",
    "nonzero_macro_reward_count",
    "reward_sum",
    "reward_max",
    "termination_count",
    "truncation_count",
    "termination_truncation_overlap_count",
    "done_count",
    "nonfinite_reward_count",
    "post_done_nonzero_reward_count",
    "reward_without_termination_count",
    "termination_without_positive_reward_count",
)
_DSRL_REWARD_AUDIT_MAX_FIELDS = frozenset({"reward_max"})


def empty_dsrl_reward_audit() -> dict[str, float]:
    """Return a zero-valued audit suitable for an empty insertion."""

    return dict.fromkeys(DSRL_REWARD_AUDIT_FIELDS, 0.0)


def _validate_num_action_chunks(num_action_chunks: int) -> None:
    if type(num_action_chunks) is not int or num_action_chunks <= 0:
        raise ValueError(
            "OpenPI DSRL num_action_chunks must be a positive integer; "
            f"got {num_action_chunks!r}."
        )


def _gamma_f32(gamma: Any, *, device: torch.device) -> torch.Tensor:
    gamma_f32 = torch.as_tensor(gamma, device=device, dtype=torch.float32)
    if gamma_f32.numel() != 1:
        raise ValueError(
            f"OpenPI DSRL gamma must be a scalar; got shape {tuple(gamma_f32.shape)}."
        )
    return gamma_f32.reshape(())


def _validate_chunk_tensors(
    rewards: torch.Tensor,
    terminations: torch.Tensor,
    truncations: torch.Tensor,
) -> None:
    tensors = {
        "rewards": rewards,
        "terminations": terminations,
        "truncations": truncations,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"OpenPI DSRL {name} must be a tensor; got {type(tensor).__name__}."
            )
        if tensor.ndim != 2:
            raise ValueError(
                f"OpenPI DSRL {name} must have shape [batch, chunk]; "
                f"got {tuple(tensor.shape)}."
            )

    expected_shape = tuple(rewards.shape)
    shape_mismatches = {
        name: tuple(tensor.shape)
        for name, tensor in tensors.items()
        if tuple(tensor.shape) != expected_shape
    }
    if shape_mismatches:
        raise ValueError(
            "OpenPI DSRL rewards, terminations, and truncations must have "
            f"identical shapes; expected {expected_shape}, got {shape_mismatches}."
        )

    device_mismatches = {
        name: str(tensor.device)
        for name, tensor in tensors.items()
        if tensor.device != rewards.device
    }
    if device_mismatches:
        raise ValueError(
            "OpenPI DSRL rewards, terminations, and truncations must share one "
            f"device; rewards are on {rewards.device}, got {device_mismatches}."
        )


def summarize_dsrl_chunk_rewards(
    rewards: torch.Tensor,
    terminations: torch.Tensor,
    truncations: torch.Tensor,
) -> dict[str, float]:
    """Return sufficient statistics for one batch of primitive action chunks."""

    _validate_chunk_tensors(rewards, terminations, truncations)
    rewards_f32 = rewards.detach().to(torch.float32)
    finite = torch.isfinite(rewards_f32)
    finite_rewards = torch.where(finite, rewards_f32, 0.0)
    nonzero = finite & finite_rewards.ne(0)
    positive = finite & finite_rewards.gt(0)
    terminations_bool = terminations.detach().bool()
    truncations_bool = truncations.detach().bool()
    done = terminations_bool | truncations_bool
    done_before = torch.cat(
        (
            torch.zeros_like(done[:, :1], dtype=torch.int64),
            done[:, :-1].to(torch.int64).cumsum(dim=-1),
        ),
        dim=-1,
    )
    post_done = done_before.gt(0)

    scalar_tensors = (
        torch.as_tensor(rewards.shape[0], device=rewards.device),
        torch.as_tensor(rewards.numel(), device=rewards.device),
        nonzero.sum(),
        nonzero.any(dim=-1).sum(),
        finite_rewards.sum(),
        finite_rewards.max()
        if finite_rewards.numel()
        else torch.zeros((), device=rewards.device),
        terminations_bool.sum(),
        truncations_bool.sum(),
        (terminations_bool & truncations_bool).sum(),
        done.sum(),
        (~finite).sum(),
        (post_done & nonzero).sum(),
        (nonzero & ~terminations_bool).sum(),
        (terminations_bool & ~positive).sum(),
    )
    # One host transfer avoids synchronizing the simulator once per statistic.
    values = (
        torch.stack([value.to(dtype=torch.float64) for value in scalar_tensors])
        .cpu()
        .tolist()
    )
    return dict(zip(DSRL_REWARD_AUDIT_FIELDS, values, strict=True))


def combine_dsrl_reward_audits(
    audits: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    """Combine disjoint reward-audit shards using sum/max sufficient stats."""

    combined = empty_dsrl_reward_audit()
    for audit in audits:
        missing = [field for field in DSRL_REWARD_AUDIT_FIELDS if field not in audit]
        if missing:
            raise ValueError(f"DSRL reward audit is missing fields {missing}.")
        for field in DSRL_REWARD_AUDIT_FIELDS:
            value = float(audit[field])
            if not math.isfinite(value):
                raise ValueError(
                    f"DSRL reward audit field {field!r} must be finite; got {value}."
                )
            if field in _DSRL_REWARD_AUDIT_MAX_FIELDS:
                combined[field] = max(combined[field], value)
            else:
                combined[field] += value
    return combined


def compare_tabero_dsrl_reward_audits(
    env_metrics: Mapping[str, float],
    replay_metrics: Mapping[str, float],
    *,
    absolute_tolerance: float = 1.0e-5,
) -> tuple[dict[str, float], tuple[str, ...]]:
    """Compare one formal Firm rollout against its compact-replay insertion.

    The input mappings use the public runner metric names: environment audit
    values are under ``reward_audit/`` and replay insertion values are under
    ``replay_buffer/last_insert_``. Missing or non-finite values are reported as
    mismatches so the runner can log the evidence before stopping.
    """

    mismatches: list[str] = []

    def read(metrics: Mapping[str, float], key: str) -> float | None:
        if key not in metrics:
            mismatches.append(f"missing metric {key}")
            return None
        try:
            value = float(metrics[key])
        except (TypeError, ValueError):
            mismatches.append(f"metric {key} is not numeric: {metrics[key]!r}")
            return None
        if not math.isfinite(value):
            mismatches.append(f"metric {key} is not finite: {value}")
            return None
        return value

    def equal(label: str, values: Sequence[float | None]) -> None:
        finite_values = [value for value in values if value is not None]
        if len(finite_values) != len(values):
            return
        reference = finite_values[0]
        if any(
            not math.isclose(
                value,
                reference,
                rel_tol=0.0,
                abs_tol=absolute_tolerance,
            )
            for value in finite_values[1:]
        ):
            mismatches.append(f"{label} mismatch: {finite_values}")

    env_audit = {
        field: read(env_metrics, f"reward_audit/{field}")
        for field in DSRL_REWARD_AUDIT_FIELDS
    }
    replay_audit = {
        field: read(replay_metrics, f"replay_buffer/last_insert_{field}")
        for field in DSRL_REWARD_AUDIT_FIELDS
    }
    firm_success_count = read(env_metrics, "firm_success_count")
    firm_episode_count = read(env_metrics, "firm_episode_count")
    completed_episode_count = read(env_metrics, "completed_episode_count")
    nonfirm_episode_count = read(env_metrics, "nonfirm_episode_count")
    gentle_episode_count = read(env_metrics, "gentle_episode_count")
    episode_reward_sum = read(env_metrics, "reward_sum")
    terminal_step_reward_sum = read(env_metrics, "terminal_step_reward_sum")
    episode_termination_count = read(env_metrics, "termination_count")
    episode_truncation_count = read(env_metrics, "truncation_count")

    equal(
        "Firm success/reward count",
        (
            firm_success_count,
            env_audit["nonzero_primitive_reward_count"],
            env_audit["nonzero_macro_reward_count"],
            replay_audit["nonzero_primitive_reward_count"],
            replay_audit["nonzero_macro_reward_count"],
        ),
    )
    equal(
        "env/replay failure termination count",
        (
            env_audit["termination_without_positive_reward_count"],
            replay_audit["termination_without_positive_reward_count"],
        ),
    )
    equal(
        "env/replay transition count",
        (env_audit["transition_count"], replay_audit["transition_count"]),
    )
    equal(
        "env/replay primitive count",
        (env_audit["primitive_count"], replay_audit["primitive_count"]),
    )
    equal(
        "termination count",
        (
            episode_termination_count,
            env_audit["termination_count"],
            replay_audit["termination_count"],
        ),
    )
    equal(
        "truncation count",
        (
            episode_truncation_count,
            env_audit["truncation_count"],
            replay_audit["truncation_count"],
        ),
    )
    equal(
        "done count",
        (
            firm_episode_count,
            completed_episode_count,
            env_audit["done_count"],
            replay_audit["done_count"],
        ),
    )
    if (
        firm_episode_count is not None
        and episode_termination_count is not None
        and episode_truncation_count is not None
        and not math.isclose(
            firm_episode_count,
            episode_termination_count + episode_truncation_count,
            rel_tol=0.0,
            abs_tol=absolute_tolerance,
        )
    ):
        mismatches.append(
            "Firm episode count does not equal termination plus truncation count: "
            f"{firm_episode_count} != {episode_termination_count} + "
            f"{episode_truncation_count}"
        )
    equal(
        "reward sum",
        (
            firm_success_count,
            episode_reward_sum,
            terminal_step_reward_sum,
            env_audit["reward_sum"],
            replay_audit["reward_sum"],
        ),
    )
    equal(
        "reward max",
        (env_audit["reward_max"], replay_audit["reward_max"]),
    )

    # IsaacLab terminations include both task success and legitimate failure
    # boundaries such as ``object_1_dropped``.  A failure termination carries
    # no positive reward, but it must still be preserved as a replay ``done``.
    # Keep this distinction explicit instead of treating every termination as
    # a success reward.
    for source_name, audit in (("env", env_audit), ("replay", replay_audit)):
        termination_count = audit["termination_count"]
        positive_reward_count = audit["nonzero_primitive_reward_count"]
        failure_termination_count = audit["termination_without_positive_reward_count"]
        if (
            termination_count is not None
            and positive_reward_count is not None
            and failure_termination_count is not None
            and not math.isclose(
                termination_count,
                positive_reward_count + failure_termination_count,
                rel_tol=0.0,
                abs_tol=absolute_tolerance,
            )
        ):
            mismatches.append(
                f"{source_name} termination count does not equal positive-reward "
                "plus failure termination count: "
                f"{termination_count} != {positive_reward_count} + "
                f"{failure_termination_count}"
            )

    zero_fields = (
        "nonfinite_reward_count",
        "post_done_nonzero_reward_count",
        "reward_without_termination_count",
        "termination_truncation_overlap_count",
    )
    for source_name, audit in (("env", env_audit), ("replay", replay_audit)):
        for field in zero_fields:
            value = audit[field]
            if value is not None and not math.isclose(
                value, 0.0, rel_tol=0.0, abs_tol=absolute_tolerance
            ):
                mismatches.append(f"{source_name} {field} must be zero; got {value}")

    for label, value in (
        ("nonfirm_episode_count", nonfirm_episode_count),
        ("gentle_episode_count", gentle_episode_count),
    ):
        if value is not None and not math.isclose(
            value, 0.0, rel_tol=0.0, abs_tol=absolute_tolerance
        ):
            mismatches.append(f"{label} must be zero; got {value}")

    audit_metrics = {
        "audit/dsrl_reward_match": float(not mismatches),
        "audit/dsrl_reward_mismatch_count": float(len(mismatches)),
    }
    return audit_metrics, tuple(mismatches)


def chunk_bootstrap_discount(
    gamma: Any,
    *,
    num_action_chunks: int,
    device: torch.device,
) -> torch.Tensor:
    """Return ``gamma**num_action_chunks`` as a scalar float32 tensor."""
    _validate_num_action_chunks(num_action_chunks)
    return _gamma_f32(gamma, device=device).pow(num_action_chunks)


def discounted_alive_masked_chunk_rewards(
    rewards: torch.Tensor,
    terminations: torch.Tensor,
    truncations: torch.Tensor,
    *,
    gamma: Any,
    num_action_chunks: int,
) -> torch.Tensor:
    """Aggregate primitive rewards through the first done, inclusively.

    The done step's reward is retained. Rewards after the first termination or
    truncation are masked because IsaacLab may already have reset that row into
    a new episode while finishing the action chunk.
    """
    _validate_chunk_tensors(rewards, terminations, truncations)

    _validate_num_action_chunks(num_action_chunks)
    chunk_length = rewards.shape[-1]
    if chunk_length != num_action_chunks:
        raise ValueError(
            "OpenPI DSRL reward chunk length must equal num_action_chunks; "
            f"got chunk_length={chunk_length}, num_action_chunks={num_action_chunks}."
        )

    done = terminations.bool() | truncations.bool()
    done_before = torch.cat(
        (
            torch.zeros_like(done[:, :1], dtype=torch.int64),
            done[:, :-1].to(torch.int64).cumsum(dim=-1),
        ),
        dim=-1,
    )
    alive_f32 = done_before.eq(0).to(torch.float32)
    gamma_f32 = _gamma_f32(gamma, device=rewards.device)
    discounts_f32 = gamma_f32.pow(
        torch.arange(chunk_length, device=rewards.device, dtype=torch.float32)
    )
    return torch.sum(
        rewards.to(torch.float32) * alive_f32 * discounts_f32,
        dim=-1,
        keepdim=True,
    )
