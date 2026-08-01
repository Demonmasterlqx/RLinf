# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Reward semantics for Tabero OpenPI DSRL training."""

from typing import Any

import torch

DSRL_REWARD_SEMANTICS = "discounted_alive_masked_v1"


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
