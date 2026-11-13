# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Export tensor validation and atomic publication; no local model dependencies."""

import pytest
import torch

from rlinf.utils.ckpt_convertor.export_tabero_dsrl_for_t2vla import (
    _publish_directory_noreplace,
    actor_contract,
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


def test_three_image_actor_exports_without_state_or_marker_encoder(contract):
    contract.update(
        use_state=False,
        use_tactile=False,
        tactile_latent_dim=0,
        tactile_shape=[],
        image_keys=["dsrl_raw_image", "dsrl_raw_wrist_image", "tactile_image"],
    )
    state = build_actor(contract).state_dict()
    assert state["dsrl_action_noise_net.shared_net.0.weight"].shape[1] == 24
    assert not any(
        k.startswith(("actor_state_encoder.", "actor_tactile_encoder.")) for k in state
    )
    cfg = {
        "use_dsrl": True,
        "dsrl_use_tactile": False,
        "dsrl_num_images": 3,
        "dsrl_actor_use_state": False,
        "dsrl_state_dim": 7,
        "dsrl_image_latent_dim": 8,
        "dsrl_state_latent_dim": 8,
        "dsrl_hidden_dims": [17, 11],
        "dsrl_action_noise_dim": 13,
        "action_horizon": 5,
        "num_steps": 10,
    }
    observation = {
        k: torch.zeros(24, 32, 3, dtype=torch.uint8) for k in contract["image_keys"]
    }
    derived = actor_contract(cfg, observation, state)
    assert derived["feature_order"] == "ordered_images"
    assert derived["use_tactile"] is False
    assert derived["tactile_shape"] == []
    assert set(validate_actor_state(state, derived)) == set(state)
    observation.pop("tactile_image")
    with pytest.raises(KeyError, match="tactile_image"):
        actor_contract(cfg, observation, state)
