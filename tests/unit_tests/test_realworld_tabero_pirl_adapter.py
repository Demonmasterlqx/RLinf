# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.config import (
    _validate_tabero_realworld_action_filter_contract,
    _validate_tabero_realworld_gripper_checkpoint_metadata,
    _validate_tabero_realworld_pi05_pirl_contract,
)
from rlinf.envs.isaaclab.isaaclab_env import IsaaclabBaseEnv
from rlinf.envs.isaaclab.tasks.realworld_tabero_tacfield import (
    IsaaclabRealWorldTaberoTacFieldEnv,
    _build_state,
    _load_task_contract,
    _RealWorldActionChunkFilter,
    _RealWorldMarkerHistory,
    _RealWorldTactileImageHistory,
    map_model_actions_to_xarm_sim,
    map_model_gripper_unit_to_xarm_sim,
    map_xarm_sim_gripper_observation_to_model,
)
from rlinf.envs.isaaclab.tasks.tabero_force_reward import (
    make_trajectory_force_success_reward_term,
)
from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0ForRLActionPrediction,
)
from rlinf.models.embodiment.openpi.policies.tabero_policy import (
    stretch_camera_image_to_224,
)
from rlinf.utils.tabero_ppo_boundary import (
    validate_tabero_pi05_pirl_deployment_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
GENTLE_GRASP_CONFIG_DIR = REPO_ROOT / "Tabero_X/benchmarks/datasets/realworld/config"
RLINF_CONFIG_DIR = REPO_ROOT / "RLinf/examples/embodiment/config"
LOCAL_MODEL_PATH = REPO_ROOT / "models/pi05_realworld_replayed_task820_23000_lora"


def _load_client_gripper_mapping_module():
    source = REPO_ROOT / "Tabero_X/benchmarks/openpi/gripper_action_mapping.py"
    spec = importlib.util.spec_from_file_location(
        "tabero_x_gripper_action_mapping_golden", source
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_client_action_filter_class():
    return _load_client_gripper_mapping_module().XarmSimActionChunkFilter


def _make_torch_filter(num_envs: int) -> _RealWorldActionChunkFilter:
    return _RealWorldActionChunkFilter(
        num_envs,
        transition_steps=3,
        max_position_step_m=0.008,
        max_position_delta_change_m=0.006,
        max_orientation_step_deg=2.0,
        max_orientation_delta_change_deg=1.5,
    )


def _action_filter_contract_cfg(train_cfg, eval_cfg):
    return OmegaConf.create(
        {
            "env": {
                "train": {"init_params": {"action_filter": train_cfg}},
                "eval": {"init_params": {"action_filter": eval_cfg}},
            }
        }
    )


def _enabled_action_filter_cfg():
    return {
        "enabled": True,
        "transition_steps": 3,
        "max_position_step_m": 0.008,
        "max_position_delta_change_m": 0.006,
        "max_orientation_step_deg": 2.0,
        "max_orientation_delta_change_deg": 1.5,
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_xarm_gripper_mapping_endpoints_clamp_roundtrip_and_dtype(dtype):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0], dtype=dtype, device=device)
    sim = map_model_gripper_unit_to_xarm_sim(policy)

    assert sim.device.type == device.type
    assert sim.dtype == dtype
    torch.testing.assert_close(
        sim,
        torch.tensor([0.045, 0.045, 0.0225, 0.0, 0.0], dtype=dtype, device=device),
    )
    torch.testing.assert_close(
        map_xarm_sim_gripper_observation_to_model(sim),
        torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0], dtype=dtype, device=device),
    )

    in_range = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0]], dtype=dtype, device=device)
    torch.testing.assert_close(
        map_xarm_sim_gripper_observation_to_model(
            map_model_gripper_unit_to_xarm_sim(in_range)
        ),
        in_range,
    )


def test_xarm_gripper_mapping_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="finite floating point"):
        map_model_gripper_unit_to_xarm_sim(torch.tensor([float("nan")]))
    with pytest.raises(ValueError, match="finite floating point"):
        map_xarm_sim_gripper_observation_to_model(torch.tensor([float("inf")]))


def test_action_mapping_changes_only_gripper_without_mutating_input():
    actions = torch.arange(2 * 10 * 13, dtype=torch.float32).reshape(2, 10, 13)
    actions[..., 6] = torch.tensor([0.0, 0.5] * 10).reshape(2, 10)
    original = actions.clone()

    mapped = map_model_actions_to_xarm_sim(actions)

    assert torch.equal(actions, original)
    torch.testing.assert_close(
        mapped[..., 6],
        (1.0 - original[..., 6]) * 0.045,
    )
    assert torch.equal(mapped[..., :6], original[..., :6])
    assert torch.equal(mapped[..., 7:], original[..., 7:])


def test_torch_gripper_mapping_matches_tabero_x_numpy_mapping():
    client_mapping = _load_client_gripper_mapping_module()
    policy = np.array([-0.5, 0.0, 0.25, 0.5, 1.0, 1.5], dtype=np.float32)
    expected_sim = client_mapping.map_model_gripper_unit_to_xarm_sim(policy)
    actual_sim = map_model_gripper_unit_to_xarm_sim(torch.from_numpy(policy)).numpy()
    np.testing.assert_array_equal(actual_sim, expected_sim)

    expected_policy = client_mapping.map_xarm_sim_gripper_observation_to_model(
        expected_sim
    )
    actual_policy = map_xarm_sim_gripper_observation_to_model(
        torch.from_numpy(expected_sim)
    ).numpy()
    np.testing.assert_array_equal(actual_policy, expected_policy)


def test_build_state_reports_open_reset_gripper_in_model_coordinates():
    policy_obs = {
        "eef_pose": torch.tensor([[0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]]),
        "gripper_pos": torch.tensor([[0.0, 0.0]]),
    }

    state = _build_state(policy_obs)

    assert state.shape == (1, 7)
    assert state[0, 6].item() == pytest.approx(1.0)


def test_action_filter_matches_client_for_two_consecutive_chunks():
    client_filter_cls = _load_client_action_filter_class()
    client_filter = client_filter_cls(
        executed_steps=10,
        transition_steps=3,
        max_position_step_m=0.008,
        max_position_delta_change_m=0.006,
        max_orientation_step_deg=2.0,
        max_orientation_delta_change_deg=1.5,
    )
    torch_filter = _make_torch_filter(1)

    anchor = np.array([0.42, -0.07, 0.31, 0.15, -0.08, 0.03, 0.045], dtype=np.float32)
    client_filter.reset(anchor)
    torch_filter.reset(torch.from_numpy(anchor[None].copy()))

    rng = np.random.default_rng(20260823)
    for _ in range(2):
        chunk = rng.normal(size=(10, 13)).astype(np.float32)
        chunk[:, :3] = anchor[:3] + 0.03 * chunk[:, :3]
        chunk[:, 3:6] = anchor[3:6] + 0.2 * chunk[:, 3:6]
        chunk[:, 6] = rng.uniform(0.0, 0.045, size=10)
        expected = client_filter(chunk)
        actual = torch_filter.filter(torch.from_numpy(chunk[None].copy()))[0].numpy()
        np.testing.assert_allclose(actual[:, :6], expected[:, :6], rtol=0, atol=2e-6)
        np.testing.assert_array_equal(actual[:, 6:], chunk[:, 6:])


def test_action_filter_reset_is_per_environment():
    action_filter = _make_torch_filter(2)
    anchors = torch.tensor(
        [
            [0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.1, 0.2, 0.1, 0.0, 0.0, 0.045],
        ],
        dtype=torch.float32,
    )
    action_filter.reset(anchors)
    first_chunk = anchors[:, None, :].expand(-1, 10, -1)
    first_chunk = torch.cat(
        [first_chunk, torch.zeros(2, 10, 6, dtype=torch.float32)], dim=-1
    ).clone()
    first_chunk[:, :, 0] += 0.03
    action_filter.filter(first_chunk)

    reset_anchors = anchors.clone()
    reset_anchors[1, 0] = 0.2
    action_filter.reset(reset_anchors, env_ids=torch.tensor([1]))
    assert torch.isclose(action_filter._last_action[0, 0], torch.tensor(0.43))
    assert torch.isclose(action_filter._last_action[1, 0], torch.tensor(0.2))
    torch.testing.assert_close(
        action_filter._last_position_delta[1], torch.zeros(3, dtype=torch.float64)
    )


@pytest.mark.parametrize("remove_action_filter", [False, True])
def test_realworld_adapter_does_not_construct_disabled_action_filter(
    monkeypatch,
    remove_action_filter,
):
    monkeypatch.setenv("REALWORLD_TABERO_TACIMG_PIRL_RUN_ID", "unit_test")
    with initialize_config_dir(version_base="1.1", config_dir=str(RLINF_CONFIG_DIR)):
        cfg = compose(
            config_name=(
                "isaaclab_pi05_pirl_realworld_tabero_tacimg_task6_step23000_"
                "fixed_gripper_shared_sim_smoke"
            )
        ).env.train
    if remove_action_filter:
        del cfg.init_params.action_filter
    cfg.init_params.extension_path = str(REPO_ROOT / "Tabero_X/source/tac_manip")
    cfg.init_params.realworld_config_dir = str(
        REPO_ROOT / "Tabero_X/benchmarks/datasets/realworld/config"
    )
    cfg.init_params.realworld_assets_dir = str(
        REPO_ROOT / "Tabero_X/benchmarks/datasets/realworld/USD"
    )

    monkeypatch.setattr(IsaaclabBaseEnv, "__init__", lambda self, *args: None)
    env = IsaaclabRealWorldTaberoTacFieldEnv(
        cfg,
        num_envs=2,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )

    assert env._action_filter is None


def test_camera_stretch_is_pixel_exact_with_client_inter_area_contract():
    image = np.random.default_rng(7).integers(
        0, 256, size=(480, 640, 3), dtype=np.uint8
    )
    expected = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    actual = stretch_camera_image_to_224(image)
    np.testing.assert_array_equal(actual, expected)


def test_task_contract_selects_only_vitasoy_success_branch():
    prompt, goals = _load_task_contract(
        GENTLE_GRASP_CONFIG_DIR,
        target_object="target_object_1",
        task_description="pick up the Vitasoy and put it into the basket",
    )
    assert prompt == "pick up the Vitasoy and put it into the basket"
    assert len(goals) == 1
    assert len(goals[0]["any_of"]) == 1
    assert goals[0]["any_of"][0]["ref_obj"] == "target_object_1"
    assert goals[0]["any_of"][0]["target"] == "target_object_4"


def test_marker_history_builds_reference_plus_eight_current_frames():
    history = _RealWorldMarkerHistory(num_envs=2)
    marker_motion = torch.arange(2 * 2 * 2 * 220 * 2, dtype=torch.float32).reshape(
        2, 2, 2, 220, 2
    )
    first = history.update(marker_motion)
    assert first.shape == (2, 9, 440, 2)
    torch.testing.assert_close(first[:, 0], marker_motion[:, :, 0].reshape(2, 440, 2))
    for index in range(1, 9):
        torch.testing.assert_close(
            first[:, index], marker_motion[:, :, 1].reshape(2, 440, 2)
        )


def test_tactile_image_history_builds_left_right_4x4_mosaic():
    history = _RealWorldTactileImageHistory(num_envs=1)
    mosaic = None
    for frame_index in range(8):
        tactile_rgb = torch.zeros(1, 2, 56, 56, 3, dtype=torch.uint8)
        tactile_rgb[:, 0] = frame_index
        tactile_rgb[:, 1] = 100 + frame_index
        mosaic = history.update(tactile_rgb)

    assert mosaic is not None
    assert mosaic.shape == (1, 224, 224, 3)
    assert mosaic.dtype == torch.uint8
    for frame_index in range(8):
        row = frame_index // 2
        left_column = frame_index % 2
        right_column = left_column + 2
        assert (
            torch.unique(
                mosaic[
                    0,
                    row * 56 : (row + 1) * 56,
                    left_column * 56 : (left_column + 1) * 56,
                ]
            ).item()
            == frame_index
        )
        assert (
            torch.unique(
                mosaic[
                    0,
                    row * 56 : (row + 1) * 56,
                    right_column * 56 : (right_column + 1) * 56,
                ]
            ).item()
            == 100 + frame_index
        )


def test_tactile_image_history_resets_selected_environment_only():
    history = _RealWorldTactileImageHistory(num_envs=2)
    first = torch.zeros(2, 2, 56, 56, 3, dtype=torch.uint8)
    first[0] = 10
    first[1] = 20
    history.update(first)
    history.reset(torch.tensor([0]))

    second = torch.zeros_like(first)
    second[0] = 30
    second[1] = 40
    mosaic = history.update(second)

    assert torch.unique(mosaic[0]).item() == 30
    assert 20 in torch.unique(mosaic[1]).tolist()
    assert 40 in torch.unique(mosaic[1]).tolist()


def test_tactile_image_resize_matches_opencv_inter_area_within_rounding():
    rng = np.random.default_rng(20260829)
    tactile_rgb = rng.integers(0, 256, size=(1, 2, 700, 400, 3), dtype=np.uint8)
    mosaic = (
        _RealWorldTactileImageHistory(num_envs=1)
        .update(torch.from_numpy(tactile_rgb))[0]
        .numpy()
    )

    for finger_index in range(2):
        expected = cv2.resize(
            tactile_rgb[0, finger_index], (56, 56), interpolation=cv2.INTER_AREA
        )
        column = 2 * finger_index
        actual = mosaic[0:56, column * 56 : (column + 1) * 56]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1)


def test_vlm_value_mask_includes_appended_tactile_token():
    class _FirstCoordinateValueHead(torch.nn.Module):
        def forward(self, inputs):
            return inputs[:, :1]

    model = SimpleNamespace(
        config=SimpleNamespace(
            config_name="pi05_lora_tacfield_tabero_xarm_gripper",
            value_vlm_mode="mean_token",
            num_images_in_input=2,
        ),
        value_head=_FirstCoordinateValueHead(),
    )
    prefix_output = torch.zeros(1, 969, 2048)
    prefix_output[:, -1, 0] = 1.0

    value = OpenPi0ForRLActionPrediction.get_value_from_vlm(model, prefix_output)

    torch.testing.assert_close(value, torch.tensor([1.0 / 713.0]))


def test_rollout_inference_uses_synced_bfloat16_projection_autocast(monkeypatch):
    autocast_calls = []
    expected = object()

    class _AutocastContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    def fake_autocast(**kwargs):
        autocast_calls.append(kwargs)
        return _AutocastContext()

    monkeypatch.setattr(torch, "autocast", fake_autocast)
    model = SimpleNamespace(
        action_in_proj=SimpleNamespace(
            weight=SimpleNamespace(
                dtype=torch.bfloat16,
                device=SimpleNamespace(type="cuda"),
            )
        ),
        sample_actions=lambda *args, **kwargs: expected,
    )

    actual = OpenPi0ForRLActionPrediction._sample_actions_with_inference_autocast(
        model, "observation", mode="train"
    )

    assert actual is expected
    assert autocast_calls == [
        {"device_type": "cuda", "dtype": torch.bfloat16, "enabled": True}
    ]


def test_chunk_step_masks_policy_actions_after_early_done():
    env = IsaaclabRealWorldTaberoTacFieldEnv.__new__(IsaaclabRealWorldTaberoTacFieldEnv)
    env.num_envs = 1
    env.device = torch.device("cpu")
    env._action_filter = _make_torch_filter(1)
    anchor = torch.tensor([[0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    env._action_filter.reset(anchor)
    call_index = 0

    def terminal_safe_step(actions, *, active_mask):
        nonlocal call_index
        done = active_mask & (call_index == 2)
        call_index += 1
        infos = {
            "_tabero_executed_action": map_model_actions_to_xarm_sim(actions),
            "_realworld_hold_state": anchor.clone(),
            "_realworld_terminal_capture": done.clone(),
        }
        return (
            {"states": anchor.clone()},
            torch.zeros(1),
            done,
            torch.zeros(1, dtype=torch.bool),
            infos,
        )

    def reset(*, env_ids):
        env._action_filter.reset(anchor, env_ids=env_ids)
        return {"states": anchor.clone()}, {}

    env._terminal_safe_step = terminal_safe_step
    env.reset = reset
    raw = torch.zeros(1, 10, 13, dtype=torch.float32)
    raw[:, :, :7] = anchor[:, None]
    raw[:, :, 0] = 0.5
    raw[:, :, 7:] = 3.0
    _, _, terminations, _, infos_list = env.chunk_step(raw)

    assert terminations[0, 2]
    executed = infos_list[-1]["_tabero_executed_chunk_actions"]
    recorded_raw = infos_list[-1]["_tabero_raw_chunk_actions"]
    torch.testing.assert_close(recorded_raw, raw)
    expected_hold = torch.zeros(1, 7, 13)
    expected_hold[:, :, :7] = anchor[:, None]
    expected_hold = map_model_actions_to_xarm_sim(expected_hold)
    torch.testing.assert_close(executed[:, 3:, :7], expected_hold[:, :, :7])
    torch.testing.assert_close(executed[:, 3:, 7:], torch.zeros(1, 7, 6))
    metrics = infos_list[-1]["chunk_boundary_metrics"]
    assert metrics["post_done_policy_actions"].item() == 0
    assert metrics["post_done_hold_steps"].item() == 7


def test_chunk_step_keeps_raw_policy_actions_and_records_mapped_execution():
    env = IsaaclabRealWorldTaberoTacFieldEnv.__new__(IsaaclabRealWorldTaberoTacFieldEnv)
    env.num_envs = 2
    env.device = torch.device("cpu")
    env._action_filter = None
    seen_actions = []

    def terminal_safe_step(actions, *, active_mask):
        seen_actions.append(actions.clone())
        infos = {
            "_tabero_executed_action": map_model_actions_to_xarm_sim(actions),
            "_realworld_hold_state": torch.zeros(2, 7),
            "_realworld_terminal_capture": torch.zeros(2, dtype=torch.bool),
        }
        return (
            {"states": torch.zeros(2, 7)},
            torch.zeros(2),
            torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
            infos,
        )

    env._terminal_safe_step = terminal_safe_step
    raw = torch.randn(2, 10, 13, generator=torch.Generator().manual_seed(20260831))
    original = raw.clone()
    _, _, _, _, infos_list = env.chunk_step(raw)

    executed = infos_list[-1]["_tabero_executed_chunk_actions"]
    recorded_raw = infos_list[-1]["_tabero_raw_chunk_actions"]
    assert torch.equal(raw, original)
    assert torch.equal(recorded_raw, original)
    assert torch.equal(torch.stack(seen_actions, dim=1), original)
    torch.testing.assert_close(executed, map_model_actions_to_xarm_sim(original))
    assert torch.equal(executed[..., :6], original[..., :6])
    assert torch.equal(executed[..., 7:], original[..., 7:])


def test_terminal_safe_step_maps_gripper_once_and_keeps_hold_state_in_model_units():
    env = IsaaclabRealWorldTaberoTacFieldEnv.__new__(IsaaclabRealWorldTaberoTacFieldEnv)
    env.num_envs = 1
    env.device = torch.device("cpu")
    env.cfg = SimpleNamespace(max_episode_steps=300)
    env._elapsed_steps = torch.zeros(1, dtype=torch.long)
    env._gripper_diagnostics_path = None
    env._wrap_obs = lambda raw_obs, marker_update_mask=None: {
        "states": _build_state(raw_obs["policy"])
    }
    env._record_metrics = lambda reward, terminations, infos: {"episode": {}}
    seen_actions = []

    class FakeEnv:
        def step(self, actions):
            seen_actions.append(actions.clone())
            raw_obs = {
                "policy": {
                    "eef_pose": torch.tensor(
                        [[0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]]
                    ),
                    "gripper_pos": actions[:, 6:7].clone(),
                }
            }
            return (
                raw_obs,
                torch.zeros(1),
                torch.zeros(1, dtype=torch.bool),
                torch.zeros(1, dtype=torch.bool),
                {},
            )

    env.env = FakeEnv()
    policy_action = torch.zeros(1, 13)
    policy_action[:, 6] = 0.5
    original = policy_action.clone()

    _, _, _, _, infos = env._terminal_safe_step(
        policy_action,
        active_mask=torch.ones(1, dtype=torch.bool),
    )

    assert torch.equal(policy_action, original)
    assert seen_actions[0][0, 6].item() == pytest.approx(0.0225)
    assert infos["_tabero_executed_action"][0, 6].item() == pytest.approx(0.0225)
    assert infos["_realworld_hold_state"][0, 6].item() == pytest.approx(0.5)


def test_reset_skips_disabled_action_filter_for_full_and_selected_resets():
    env = IsaaclabRealWorldTaberoTacFieldEnv.__new__(IsaaclabRealWorldTaberoTacFieldEnv)
    env.num_envs = 2
    env.device = torch.device("cpu")
    env._action_filter = None
    env._marker_history = SimpleNamespace(reset=lambda env_ids: None)
    env._tactile_image_history = SimpleNamespace(reset=lambda env_ids: None)
    reset_calls = []

    class FakeEnv:
        def reset(self, seed=None, env_ids=None):
            reset_calls.append((seed, env_ids))
            return {}, {}

    env.env = FakeEnv()
    env._wrap_obs = lambda raw_obs, marker_update_mask=None: {
        "states": torch.zeros(2, 7)
    }
    env._reset_metrics = lambda env_ids: None

    env.reset(seed=17)
    selected = torch.tensor([1])
    env.reset(env_ids=selected)

    assert reset_calls[0] == (17, None)
    assert reset_calls[1][0] is None
    assert torch.equal(reset_calls[1][1], selected)


def test_fixed_gripper_config_disables_action_filter_by_default():
    config_name = (
        "isaaclab_pi05_pirl_realworld_tabero_tacimg_task6_step23000_"
        "fixed_gripper_shared_sim_smoke"
    )
    with initialize_config_dir(version_base="1.1", config_dir=str(RLINF_CONFIG_DIR)):
        cfg = compose(config_name=config_name)

    for split_name in ("train", "eval"):
        assert cfg.env[split_name].init_params.action_filter.enabled is False
        assert list(cfg.env[split_name].init_params.action_filter.keys()) == ["enabled"]
        assert "gripper_mapping" not in cfg.env[split_name].init_params
    metadata = cfg.actor.fsdp_config.trainable_checkpoint_metadata
    assert metadata.action_filter == "disabled"
    assert metadata.gripper_mapping == "xarm_unit_inverse_v1"
    assert metadata.policy_gripper_coordinate == "unit_0_closed_1_open"
    assert metadata.sim_gripper_coordinate == "meters_0_open_0045_close"
    assert metadata.gripper_travel_m == pytest.approx(0.045)
    assert (
        _validate_tabero_realworld_action_filter_contract(cfg, metadata) == "disabled"
    )


def test_action_filter_contract_keeps_explicit_enable_as_opt_in():
    cfg = _action_filter_contract_cfg(
        _enabled_action_filter_cfg(),
        _enabled_action_filter_cfg(),
    )
    assert (
        _validate_tabero_realworld_action_filter_contract(
            cfg,
            {"action_filter": "xarm_sim_action_chunk_filter_v1"},
        )
        == "xarm_sim_action_chunk_filter_v1"
    )


def test_fixed_gripper_checkpoint_metadata_rejects_drift():
    metadata = {
        "gripper_mapping": "xarm_unit_inverse_v1",
        "policy_gripper_coordinate": "unit_0_closed_1_open",
        "sim_gripper_coordinate": "meters_0_open_0045_close",
        "gripper_travel_m": 0.045,
    }
    assert _validate_tabero_realworld_gripper_checkpoint_metadata(metadata) == {
        "gripper_mapping": "xarm_unit_inverse_v1",
        "policy_gripper_coordinate": "unit_0_closed_1_open",
        "sim_gripper_coordinate": "meters_0_open_0045_close",
        "gripper_travel_m": 0.045,
    }

    drifted_metadata = dict(metadata)
    drifted_metadata["gripper_mapping"] = "legacy_direct"
    with pytest.raises(ValueError, match="gripper_mapping='xarm_unit_inverse_v1'"):
        _validate_tabero_realworld_gripper_checkpoint_metadata(drifted_metadata)


def test_fixed_gripper_shared_sim_smoke_contract(monkeypatch):
    config_name = (
        "isaaclab_pi05_pirl_realworld_tabero_tacimg_task6_"
        "step23000_fixed_gripper_shared_sim_smoke"
    )
    monkeypatch.setenv("REALWORLD_TABERO_TACIMG_PIRL_RUN_ID", "unit_test")
    with initialize_config_dir(version_base="1.1", config_dir=str(RLINF_CONFIG_DIR)):
        cfg = compose(config_name=config_name)

    assert cfg.actor.fsdp_config.gradient_checkpointing is True
    assert cfg.actor.fsdp_config.gradient_checkpointing_use_reentrant is False
    assert cfg.cluster.component_placement.actor.placement == 0
    assert cfg.cluster.component_placement.rollout.placement == 2
    assert cfg.cluster.component_placement.env.placement == 3
    assert cfg.env.train.total_num_envs == 1
    assert cfg.env.train.max_steps_per_rollout_epoch == 300
    assert cfg.env.train.max_episode_steps == 300
    assert cfg.runner.max_epochs == 1
    assert cfg.runner.max_steps == 1
    assert cfg.env.train.video_cfg.save_video is True
    assert cfg.env.eval.video_cfg.save_video is True
    assert list(cfg.env.train.video_cfg.image_names) == ["agentview", "eye_in_hand"]
    assert cfg.env.train.video_cfg.composite_name == "combined"
    assert cfg.actor.model.model_path == str(LOCAL_MODEL_PATH)
    assert cfg.rollout.model.model_path == str(LOCAL_MODEL_PATH)
    metadata = cfg.actor.fsdp_config.trainable_checkpoint_metadata
    assert metadata.gripper_mapping == "xarm_unit_inverse_v1"
    assert metadata.gradient_checkpointing is True
    assert metadata.gradient_checkpointing_use_reentrant is False
    assert (
        _validate_tabero_realworld_gripper_checkpoint_metadata(metadata)[
            "gripper_mapping"
        ]
        == "xarm_unit_inverse_v1"
    )
    assert cfg.actor.model.openpi.action_horizon == 50
    assert cfg.actor.model.openpi.action_chunk == 10
    assert cfg.actor.model.openpi.num_images_in_input == 3
    assert cfg.env.train.init_params.tactile_image_history_len == 8
    _validate_tabero_realworld_pi05_pirl_contract(cfg, cfg.actor.model)


def test_openpi_gradient_checkpointing_bridge_is_non_reentrant(monkeypatch):
    base_class = OpenPi0ForRLActionPrediction.__mro__[1]
    enable_calls = []
    monkeypatch.setattr(
        base_class,
        "gradient_checkpointing_enable",
        lambda self: enable_calls.append(self),
    )
    policy = object.__new__(OpenPi0ForRLActionPrediction)

    policy.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    assert enable_calls == [policy]

    with pytest.raises(ValueError, match="only supports use_reentrant=false"):
        policy.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": True}
        )

    with pytest.raises(ValueError, match="unsupported options"):
        policy.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"preserve_rng_state": True}
        )


def test_action_filter_contract_rejects_state_and_metadata_drift():
    disabled_cfg = _action_filter_contract_cfg(
        {"enabled": False},
        {"enabled": False},
    )
    with pytest.raises(ValueError, match="metadata.action_filter='disabled'"):
        _validate_tabero_realworld_action_filter_contract(
            disabled_cfg,
            {"action_filter": "xarm_sim_action_chunk_filter_v1"},
        )

    mismatched_cfg = _action_filter_contract_cfg(
        {"enabled": False},
        _enabled_action_filter_cfg(),
    )
    with pytest.raises(ValueError, match="same action_filter.enabled state"):
        _validate_tabero_realworld_action_filter_contract(
            mismatched_cfg,
            {"action_filter": "disabled"},
        )

    non_boolean_cfg = _action_filter_contract_cfg(
        {"enabled": "false"},
        {"enabled": False},
    )
    with pytest.raises(ValueError, match="enabled to be boolean"):
        _validate_tabero_realworld_action_filter_contract(
            non_boolean_cfg,
            {"action_filter": "disabled"},
        )

    missing_parameter_cfg = _action_filter_contract_cfg(
        {"enabled": True},
        {"enabled": True},
    )
    with pytest.raises(ValueError, match="enabled action filter requires"):
        _validate_tabero_realworld_action_filter_contract(
            missing_parameter_cfg,
            {"action_filter": "xarm_sim_action_chunk_filter_v1"},
        )


def test_realworld_direct_force_reward_uses_contact_only_mean():
    class FakeManagerTermBase:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self.env = env

    class FakeTerminationManager:
        def __init__(self):
            self.time_outs = torch.tensor([False])
            self.success = torch.tensor([False])

        def get_term(self, name):
            assert name == "success"
            return self.success

    termination_manager = FakeTerminationManager()
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        step_dt=0.05,
        termination_manager=termination_manager,
    )
    squeeze_schedule = [0.0, 4.0, 2.0, 0.0]
    force_step = {"value": 0}

    def direct_force_reader(_env, sensor_name_list, history_length):
        assert _env is env
        assert sensor_name_list == ["xense_left", "xense_right"]
        assert history_length == 1
        squeeze = squeeze_schedule[force_step["value"]]
        force_step["value"] += 1
        force = torch.zeros((1, 1, 2, 3), dtype=torch.float32)
        force[:, :, :, 2] = squeeze / 2.0
        return force

    params = {
        "success_term_name": "success",
        "failure_term_names": (),
        "direct_force_reader": direct_force_reader,
        "direct_force_params": {
            "sensor_name_list": ["xense_left", "xense_right"],
            "history_length": 1,
        },
        "terminal_reward": 1.0,
        "coefficient": 0.3,
        "epsilon": 0.1,
        "max_bonus": 0.2,
        "min_valid_samples": 2,
        "contact_epsilon": 1.0e-4,
    }
    reward_cls = make_trajectory_force_success_reward_term(FakeManagerTermBase)
    reward_term = reward_cls(SimpleNamespace(params=params), env)

    rewards = []
    for step in range(4):
        termination_manager.success = torch.tensor([step == 3])
        rewards.append(reward_term(env, **params))

    assert reward_term.valid_sample_count.tolist() == [2]
    torch.testing.assert_close(reward_term.trajectory_mean_force, torch.tensor([3.0]))
    torch.testing.assert_close(reward_term.current_force_bonus, torch.tensor([0.1]))
    assert rewards[-1].item() == pytest.approx(1.1 / 0.05)


def _write_checkpoint_contract_fixture(checkpoint_dir: Path) -> tuple[str, str]:
    model_file = checkpoint_dir / "model.safetensors"
    model_file.write_bytes(b"small model fixture")
    model_sha = hashlib.sha256(model_file.read_bytes()).hexdigest()
    config_name = "pi05_lora_tacfield_tabero_xarm_gripper"
    dataset = "datas/replay_firm_tabero_xarm_gripper"
    asset_id = "replay_firm_tabero_xarm_gripper"
    source_metadata = {
        "dataset": dataset,
        "model_family": "pi05",
        "openpi_config_name": config_name,
        "deployment_config_name": config_name,
        "action_horizon": 10,
        "effective_action_dim": 13,
        "tactile_prefix_dim_in": 7920,
        "tactile_prefix_history": 8,
        "gripper_coordinate": "xarm_positive_open",
    }
    export_meta = {
        "format": "t2vla_openpi_pytorch_merged_lora",
        "method": "sft_full_lora_tacfield",
        "dataset": dataset,
        "model_sha256": model_sha,
        "source_ckpt_metadata": source_metadata,
        "is_final": False,
        "global_step": 10000,
        "target_global_step": 20000,
    }
    (checkpoint_dir / "export_meta.json").write_text(json.dumps(export_meta))
    model_config = {
        "action_dim": 32,
        "action_horizon": 10,
        "pi05": True,
        "discrete_state_input": True,
        "config_name": config_name,
        "num_images_in_input": 2,
        "action_chunk": 10,
        "action_env_dim": 13,
        "num_steps": 10,
        "tactile_type": "expert_his_c_fut",
        "tactile_dim": 6,
        "tactile_dim_in": 0,
        "effective_action_dim": 13,
        "tactile_prefix_dim_in": 7920,
        "tactile_prefix_history": 8,
        "tactile_prefix_encoder_type": "tcn",
        "tactile_prefix_use_reference_frame": True,
        "tactile_prefix_diff_from_reference": False,
        "tactile_streams": ["tactile_prefix"],
    }
    (checkpoint_dir / "config.json").write_text(json.dumps(model_config))
    norm_dir = checkpoint_dir / asset_id
    norm_dir.mkdir()
    norm_stats = {
        "norm_stats": {
            name: {
                statistic: [0.0] * dim for statistic in ("mean", "std", "q01", "q99")
            }
            for name, dim in {"state": 7, "actions": 13, "tactile_prefix": 880}.items()
        }
    }
    norm_file = norm_dir / "norm_stats.json"
    norm_file.write_text(json.dumps(norm_stats))
    norm_sha = hashlib.sha256(norm_file.read_bytes()).hexdigest()
    return model_sha, norm_sha


def test_checkpoint_contract_is_parameterized_for_xarm_asset(tmp_path: Path):
    model_sha, norm_sha = _write_checkpoint_contract_fixture(tmp_path)
    result = validate_tabero_pi05_pirl_deployment_checkpoint(
        tmp_path,
        expected_model_sha256=model_sha,
        expected_norm_stats_sha256=norm_sha,
        expected_config_name="pi05_lora_tacfield_tabero_xarm_gripper",
        expected_norm_asset_id="replay_firm_tabero_xarm_gripper",
        expected_dataset="datas/replay_firm_tabero_xarm_gripper",
        expected_gripper_coordinate="xarm_positive_open",
        require_final=False,
    )
    assert result["global_step"] == 10000
    assert result["norm_asset_id"] == "replay_firm_tabero_xarm_gripper"


def test_checkpoint_contract_accepts_tacimg_horizon50_export(tmp_path: Path):
    model_file = tmp_path / "model.safetensors"
    model_file.write_bytes(b"small tacimg model fixture")
    model_sha = hashlib.sha256(model_file.read_bytes()).hexdigest()
    config_name = "pi05_lora_tacimg_realworld_replayed_task820_force"
    dataset = "datas/realworld_replayed_task820_firm"
    asset_id = "pi05_horizon50_tacimg_task820_firm"
    source_metadata = {
        "dataset": dataset,
        "model_family": "pi05",
        "openpi_config_name": config_name,
        "deployment_config_name": config_name,
        "action_horizon": 50,
        "execution_steps": 10,
        "effective_action_dim": 13,
        "tactile_input": "tactile_image",
        "num_images_in_input": 3,
        "excluded_tactile_inputs": [
            "tactile_gripper_force",
            "tactile_marker_motion",
        ],
        "gripper_coordinate": "xarm_positive_open",
    }
    export_meta = {
        "format": "t2vla_openpi_pytorch_merged_lora",
        "method": "sft_full_lora_tacimg",
        "dataset": dataset,
        "model_sha256": model_sha,
        "source_ckpt_metadata": source_metadata,
        "is_final": False,
        "global_step": 23000,
        "target_global_step": 30000,
    }
    (tmp_path / "export_meta.json").write_text(json.dumps(export_meta))
    model_config = {
        "action_dim": 32,
        "action_horizon": 50,
        "pi05": True,
        "discrete_state_input": True,
        "config_name": config_name,
        "num_images_in_input": 3,
        "action_chunk": 50,
        "action_env_dim": 13,
        "num_steps": 10,
        "tactile_type": "expert_his_c_fut",
        "tactile_dim": 6,
        "tactile_dim_in": 0,
        "effective_action_dim": 13,
        "tactile_prefix_dim_in": None,
        "tactile_prefix_history": None,
        "tactile_prefix_encoder_type": None,
        "tactile_prefix_use_reference_frame": None,
        "tactile_prefix_diff_from_reference": None,
        "tactile_streams": [],
    }
    (tmp_path / "config.json").write_text(json.dumps(model_config))
    norm_dir = tmp_path / asset_id
    norm_dir.mkdir()
    norm_stats = {
        "norm_stats": {
            name: {
                statistic: [0.0] * dim for statistic in ("mean", "std", "q01", "q99")
            }
            for name, dim in {"state": 7, "actions": 13}.items()
        }
    }
    norm_file = norm_dir / "norm_stats.json"
    norm_file.write_text(json.dumps(norm_stats))
    norm_sha = hashlib.sha256(norm_file.read_bytes()).hexdigest()

    result = validate_tabero_pi05_pirl_deployment_checkpoint(
        tmp_path,
        expected_model_sha256=model_sha,
        expected_norm_stats_sha256=norm_sha,
        expected_config_name=config_name,
        expected_norm_asset_id=asset_id,
        expected_dataset=dataset,
        expected_gripper_coordinate="xarm_positive_open",
        require_final=False,
    )

    assert result["global_step"] == 23000
    assert result["tactile_input"] == "tactile_image"
