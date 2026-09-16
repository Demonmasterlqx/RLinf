# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Export tensor validation and atomic publication; no local model dependencies."""

import pytest
import torch

from rlinf.utils.ckpt_convertor.export_tabero_dsrl_for_t2vla import (
    _publish_directory_noreplace,
    build_actor,
    validate_actor_state,
)


@pytest.fixture
def contract():
    return {
        "use_state": True,
        "state_latent_dim": 8,
        "image_keys": ["image", "wrist"],
        "image_latent_dim": 8,
        "tactile_latent_dim": 8,
        "noise_dim": 13,
        "hidden_dims": [17, 11],
        "horizon": 5,
        "state_dim": 7,
        "tactile_shape": [9, 5, 2],
        "dtype": "bfloat16",
    }


@pytest.mark.parametrize(
    "corruption", ["missing", "unexpected", "shape", "dtype", "nan"]
)
def test_actor_export_rejects_corrupt_weights(contract, corruption):
    state = build_actor(contract).state_dict()
    key = next(iter(state))
    if corruption == "missing":
        state.pop(key)
    if corruption == "unexpected":
        state["actor_image_encoder.extra"] = torch.zeros(1)
    if corruption == "shape":
        state[key] = state[key].flatten()
    if corruption == "dtype":
        state[key] = state[key].float()
    if corruption == "nan":
        state[key].flatten()[0] = float("nan")
    with pytest.raises(ValueError):
        validate_actor_state(state, contract)


def test_export_selects_only_actor_and_preserves_dtype(contract):
    state = build_actor(contract).state_dict()
    state["q_head.example"] = torch.zeros(1)
    exported = validate_actor_state(state, contract)
    assert "q_head.example" not in exported
    assert all(t.dtype == torch.bfloat16 for t in exported.values())


def test_atomic_export_never_overwrites_existing_directory(tmp_path):
    staging, output = tmp_path / "staging", tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    (output / "keep").write_text("original")
    with pytest.raises(FileExistsError):
        _publish_directory_noreplace(staging, output)
    assert (output / "keep").read_text() == "original"
