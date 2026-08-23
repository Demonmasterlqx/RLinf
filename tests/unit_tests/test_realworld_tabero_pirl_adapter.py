from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from hydra import compose, initialize_config_dir

from rlinf.envs.isaaclab.tasks.realworld_tabero_tacfield import (
    IsaaclabRealWorldTaberoTacFieldEnv,
    _load_task_contract,
    _RealWorldActionChunkFilter,
    _RealWorldMarkerHistory,
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


def _load_client_action_filter_class():
    source = REPO_ROOT / "Tabero_X/benchmarks/openpi/gripper_action_mapping.py"
    spec = importlib.util.spec_from_file_location(
        "tabero_x_gripper_action_mapping_golden", source
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.XarmSimActionChunkFilter


def _make_torch_filter(num_envs: int) -> _RealWorldActionChunkFilter:
    return _RealWorldActionChunkFilter(
        num_envs,
        transition_steps=3,
        max_position_step_m=0.008,
        max_position_delta_change_m=0.006,
        max_orientation_step_deg=2.0,
        max_orientation_delta_change_deg=1.5,
    )


def test_action_filter_matches_client_for_two_consecutive_chunks():
    client_filter_cls = _load_client_action_filter_class()
    client_filter = client_filter_cls(
        executed_steps=10,
        transition_steps=3,
        max_position_step_m=0.008,
        max_position_delta_change_m=0.006,
        max_orientation_step_deg=2.0,
        max_orientation_delta_change_deg=1.5,
        invert_gripper_output=False,
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
    anchor = torch.tensor([[0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 0.045]], dtype=torch.float32)
    env._action_filter.reset(anchor)
    call_index = 0

    def terminal_safe_step(actions, *, active_mask):
        nonlocal call_index
        del actions
        done = active_mask & (call_index == 2)
        call_index += 1
        infos = {
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
    torch.testing.assert_close(executed[:, 3:, :7], anchor[:, None].expand(-1, 7, -1))
    torch.testing.assert_close(executed[:, 3:, 7:], torch.zeros(1, 7, 6))
    metrics = infos_list[-1]["chunk_boundary_metrics"]
    assert metrics["post_done_policy_actions"].item() == 0
    assert metrics["post_done_hold_steps"].item() == 7


def test_smoke_config_pins_xarm_checkpoint_and_bridge_false():
    with initialize_config_dir(version_base="1.1", config_dir=str(RLINF_CONFIG_DIR)):
        cfg = compose(
            config_name="isaaclab_pi05_pirl_realworld_tabero_task6_2gpu_smoke"
        )
    assert cfg.env.train.init_params.policy_gripper_sign_bridge is False
    assert cfg.env.train.init_params.target_object == "target_object_1"
    assert cfg.env.train.init_params.reset_source == "task_config_default_reset"
    assert cfg.env.train.max_episode_steps == 300
    assert cfg.actor.model.openpi.config_name.endswith("_xarm_gripper")
    assert (
        cfg.actor.model.tabero_pi05_checkpoint_contract.expected_norm_asset_id
        == "replay_firm_tabero_xarm_gripper"
    )
    assert cfg.actor.model.tabero_pi05_checkpoint_contract.require_final is False


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
