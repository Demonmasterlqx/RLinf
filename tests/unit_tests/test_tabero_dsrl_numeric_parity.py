# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Numerical parity across independent implementations, with synthetic tensors."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

T2_SRC = Path(__file__).resolve().parents[3] / "T2-VLA" / "src"
if not T2_SRC.exists():
    pytest.skip(
        "T2-VLA checkout required for cross-repository parity", allow_module_level=True
    )
import openpi.policies  # noqa: E402

openpi.policies.__path__.append(str(T2_SRC / "openpi" / "policies"))
from openpi.policies.tabero_dsrl_policy import (  # noqa: E402
    ActorContract,
    TaberoDSRLActor,
)

runner_path = (
    Path(__file__).resolve().parents[2]
    / "examples/embodiment/run_tabero_dsrl_parity.py"
)
spec = importlib.util.spec_from_file_location("dsrl_parity_runner", runner_path)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


@pytest.mark.parametrize("use_state", [True, False])
@pytest.mark.parametrize(
    "markers,views,width,dtype",
    [(5, 1, 8, "float32"), (11, 2, 16, "bfloat16"), (7, 3, 12, "bfloat16")],
)
def test_independent_actor_numerics(markers, views, width, dtype, use_state):
    torch.set_num_threads(2)
    torch.manual_seed(41)
    c = ActorContract(
        use_state=use_state,
        image_keys=[f"image{i}" for i in range(views)],
        image_shapes=[[37, 53, 3]] * views,
        state_key="state",
        state_dim=7,
        tactile_key="tactile",
        tactile_shape=[9, markers, 2],
        image_latent_dim=width,
        state_latent_dim=width,
        tactile_latent_dim=width,
        hidden_dims=[23, 19],
        noise_dim=13,
        horizon=6,
        num_steps=4,
        dtype=dtype,
        image_preprocessing="uint8_bilinear64_align_false_minus_one_one",
        tactile_processing="reference_plus_history8_no_difference_causal_tcn2_kernel3",
        feature_order="state_ordered_images_tactile"
        if use_state
        else "ordered_images_tactile",
    )
    actor = TaberoDSRLActor(c).eval()
    obs = {k: torch.randint(256, (37, 53, 3), dtype=torch.uint8) for k in c.image_keys}
    obs.update(state=torch.randn(7), tactile=torch.randn(9, markers, 2))
    result = runner.compare_actor(actor, obs)
    assert result["passed"], result
