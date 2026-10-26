# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""RLT action semantics shared by training and deployment export."""

import math
from collections.abc import Mapping
from pathlib import Path

import torch


def build_action_contract(bound: float, parameterization: str = "residual") -> dict:
    """Describe explicitly whether the bounded actor output is absolute or residual."""
    if parameterization not in ("absolute", "residual"):
        raise ValueError(
            f"Unsupported RLT action_parameterization: {parameterization!r}."
        )
    bound = float(bound)
    if not math.isfinite(bound) or bound <= 0:
        raise ValueError("RLT residual_action_bound must be finite and positive.")
    if parameterization == "absolute":
        return {
            "action_parameterization": "absolute",
            "action_composition": "direct",
            "normalized_action_bound": bound,
        }
    return {
        "action_parameterization": "residual",
        "action_composition": "reference_plus_bounded_delta",
        "residual_action_bound": bound,
    }


def validate_action_contract(
    actual, bound: float, parameterization: str = "residual"
) -> dict:
    """Reject missing, legacy or differently scaled action semantics."""
    expected = build_action_contract(bound, parameterization)
    if not isinstance(actual, Mapping) or any(
        actual.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(
            "RLT action contract mismatch: configured policy requires "
            f"{expected}, got {actual}."
        )
    return expected


def checkpoint_action_contract(
    path, payload, *, bound: float, step=None, parameterization: str | None = None
) -> dict:
    """Resolve embedded/sidecar semantics or an explicit unlabelled-weight declaration.

    Explicit declarations cannot override existing checkpoint metadata. A source
    without metadata needs a caller-supplied parameterization; no type is inferred.
    """
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("Invalid RLT action contract metadata.")
    if "rlt_action_contract" not in metadata:
        path = Path(path)
        sidecar = (
            path.parent.parent / "sac_components/rlt_schedule/checkpoint_rank_0.pt"
        )
        if path.parent.name == "model_state_dict" and sidecar.is_file():
            metadata = torch.load(sidecar, map_location="cpu", weights_only=True)
    if "rlt_action_contract" in metadata:
        actual = metadata["rlt_action_contract"]
        if not isinstance(actual, Mapping):
            raise ValueError("Invalid RLT action contract metadata.")
        declared = actual.get("action_parameterization")
        contract = validate_action_contract(actual, bound, parameterization or declared)
    elif parameterization is not None:
        contract = build_action_contract(bound, parameterization)
    else:
        raise ValueError(
            "Missing RLT action contract: explicitly specify action_parameterization "
            "as absolute or residual for unlabelled weights."
        )
    saved_step = metadata.get("global_step", metadata.get("step"))
    if step is not None and saved_step is not None and saved_step != step:
        raise ValueError("RLT action contract checkpoint global step mismatch.")
    return contract
