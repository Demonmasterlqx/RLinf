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

import inspect
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

import rlinf.models as model_registry
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0Config,
    OpenPi0ForRLActionPrediction,
)
from rlinf.models.embodiment.openpi.tactile_encoder import TactileTCNEncoder
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.utils.drq import apply_drq
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class _TinyImageEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.proj = nn.Linear(3, output_dim)

    def forward(self, images):
        features = images.mean(dim=(1, 3, 4)).to(dtype=self.proj.weight.dtype)
        return self.proj(features)


def _bare_model(config) -> OpenPi0ForRLActionPrediction:
    model = OpenPi0ForRLActionPrediction.__new__(OpenPi0ForRLActionPrediction)
    nn.Module.__init__(model)
    model.config = config
    return model


def _runtime_model(
    *, use_tactile: bool, num_images: int = 1, tactile_input_dim: int = 396
) -> OpenPi0ForRLActionPrediction:
    config = SimpleNamespace(
        use_dsrl=True,
        dsrl_use_tactile=use_tactile,
        dsrl_tactile_input_dim=tactile_input_dim,
        dsrl_tactile_latent_dim=64,
        dsrl_state_dim=7,
        dsrl_action_noise_dim=32,
        dsrl_num_q_heads=10,
        dsrl_image_latent_dim=64,
        dsrl_num_images=num_images,
        dsrl_state_latent_dim=64,
        dsrl_hidden_dims=(32, 32),
        action_horizon=2,
    )
    model = _bare_model(config)
    model._init_dsrl_components()
    model.actor_image_encoder = _TinyImageEncoder(64)
    model.critic_image_encoder = _TinyImageEncoder(64)
    return model.float()


def _obs(
    *,
    include_tactile: bool = True,
    include_wrist: bool = False,
    tactile_shape=(2, 9, 198, 2),
):
    obs = {
        "main_images": torch.randint(0, 256, (2, 32, 32, 3), dtype=torch.uint8),
        "states": torch.randn(2, 7),
    }
    if include_wrist:
        obs["wrist_images"] = torch.randint(0, 256, (2, 32, 32, 3), dtype=torch.uint8)
    if include_tactile:
        obs["tactile_marker_motion"] = torch.randn(*tactile_shape)
    return obs


def _has_grad(module: nn.Module) -> bool:
    return any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in module.parameters()
    )


def test_dsrl_tactile_config_defaults_are_opt_in():
    config = OpenPi0Config()

    assert config.dsrl_use_tactile is False
    assert config.dsrl_tactile_latent_dim == 64
    assert config.dsrl_num_images == 1


def test_dsrl_tactile_components_use_independent_tcn_encoders():
    model = _runtime_model(use_tactile=True)

    assert isinstance(model.actor_tactile_encoder, TactileTCNEncoder)
    assert isinstance(model.critic_tactile_encoder, TactileTCNEncoder)
    assert model.actor_tactile_encoder is not model.critic_tactile_encoder
    for encoder in (model.actor_tactile_encoder, model.critic_tactile_encoder):
        assert encoder.input_dim == 396
        assert encoder.history_len == 8
        assert encoder.has_reference_frame is True
        assert encoder.diff_from_reference is False
        assert encoder.out_proj.out_features == 64
    assert model.dsrl_action_noise_net.input_dim == 64 + 64 + 64
    assert model.q_head.q_heads[0].net[0].in_features == 64 + 64 + 64 + 32


def test_dsrl_dual_camera_components_share_each_side_encoder_and_expand_features():
    model = _runtime_model(use_tactile=True, num_images=2)

    assert model.actor_image_encoder is not model.critic_image_encoder
    assert model.dsrl_action_noise_net.input_dim == 64 + 64 + 64 + 64
    assert model.q_head.q_heads[0].net[0].in_features == 64 + 64 + 64 + 64 + 32


def test_dsrl_dual_camera_preprocessing_preserves_main_wrist_order_and_range():
    model = _runtime_model(use_tactile=False, num_images=2)
    main = torch.zeros(2, 48, 32, 3, dtype=torch.uint8)
    wrist = torch.full((2, 24, 40, 3), 255, dtype=torch.uint8)

    normalized = model._normalize_dsrl_obs(
        {"main_images": main, "wrist_images": wrist, "states": torch.zeros(2, 7)}
    )
    images = model._preprocess_dsrl_images(normalized["images"])

    assert normalized["images"][0] is main
    assert normalized["images"][1] is wrist
    assert images.shape == (2, 2, 3, 64, 64)
    assert images.dtype == torch.float32
    assert torch.equal(images[:, 0], torch.full_like(images[:, 0], -1.0))
    assert torch.equal(images[:, 1], torch.full_like(images[:, 1], 1.0))


def test_dsrl_compact_replay_observation_matches_raw_actor_and_critic_inputs():
    torch.manual_seed(19)
    model = _runtime_model(use_tactile=True, num_images=2)
    raw_obs = _obs(include_wrist=True)
    normalized = model._normalize_dsrl_obs(raw_obs)
    compact_obs = {
        "dsrl_images": model._preprocess_dsrl_images(normalized["images"]).to(
            torch.bfloat16
        ),
        "states": raw_obs["states"].to(torch.bfloat16),
        "tactile_marker_motion": raw_obs["tactile_marker_motion"].to(torch.bfloat16),
    }

    raw_actions, _, _ = model.sac_forward(raw_obs, mode="eval")
    compact_actions, _, _ = model.sac_forward(compact_obs, mode="eval")
    raw_q = model.sac_q_forward(raw_obs, actions=raw_actions)
    compact_q = model.sac_q_forward(compact_obs, actions=raw_actions[:, 0, :])

    torch.testing.assert_close(raw_actions, compact_actions, rtol=2e-3, atol=2e-5)
    torch.testing.assert_close(raw_q, compact_q, rtol=2e-2, atol=2e-4)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda obs: obs.pop("wrist_images"), "requires 'wrist_images'"),
        (
            lambda obs: obs.__setitem__("wrist_images", obs["wrist_images"].float()),
            "wrist image must use uint8",
        ),
        (
            lambda obs: obs.__setitem__(
                "wrist_images", torch.zeros(2, 255, 256, 3, dtype=torch.uint8)
            ),
            "wrist image expected shape",
        ),
        (
            lambda obs: obs.__setitem__(
                "main_images", torch.zeros(2, 256, 256, dtype=torch.uint8)
            ),
            "main image must be a rank-4 tensor",
        ),
        (
            lambda obs: obs.__setitem__(
                "wrist_images", torch.zeros(1, 256, 256, 3, dtype=torch.uint8)
            ),
            "wrist image batch mismatch",
        ),
    ],
)
def test_tabero_dsrl_dual_camera_rejects_invalid_raw_view_contract(
    mutation,
    message,
):
    model = _bare_model(
        SimpleNamespace(
            dsrl_num_images=2,
            config_name="pi0_lora_tacfield_tabero",
        )
    )
    obs = {
        "main_images": torch.zeros(2, 256, 256, 3, dtype=torch.uint8),
        "wrist_images": torch.ones(2, 256, 256, 3, dtype=torch.uint8),
        "states": torch.zeros(2, 7),
    }
    mutation(obs)

    with pytest.raises(ValueError, match=message):
        model._normalize_dsrl_obs(obs)


def test_dsrl_dual_camera_actor_and_critic_depend_on_wrist_and_share_view_gradients():
    torch.manual_seed(7)
    model = _runtime_model(use_tactile=True, num_images=2)
    main = torch.rand(2, 32, 32, 3, requires_grad=True)
    wrist = torch.rand(2, 32, 32, 3, requires_grad=True)
    obs = _obs(include_wrist=True)
    obs["main_images"] = main
    obs["wrist_images"] = wrist

    actions, _, _ = model.sac_forward(obs, mode="eval")
    actions.sum().backward()

    assert main.grad is not None and torch.count_nonzero(main.grad) > 0
    assert wrist.grad is not None and torch.count_nonzero(wrist.grad) > 0
    assert _has_grad(model.actor_image_encoder)
    assert not _has_grad(model.critic_image_encoder)

    model.zero_grad(set_to_none=True)
    main.grad = None
    wrist.grad = None
    q_values = model.sac_q_forward(obs, actions=actions.detach())
    q_values.sum().backward()

    assert main.grad is not None and torch.count_nonzero(main.grad) > 0
    assert wrist.grad is not None and torch.count_nonzero(wrist.grad) > 0
    assert _has_grad(model.critic_image_encoder)
    assert not _has_grad(model.actor_image_encoder)

    changed_obs = dict(obs)
    changed_obs["wrist_images"] = torch.ones_like(wrist)
    with torch.no_grad():
        changed_actions, _, _ = model.sac_forward(changed_obs, mode="eval")
        changed_q = model.sac_q_forward(changed_obs, actions=actions.detach())
    assert not torch.allclose(actions, changed_actions)
    assert not torch.allclose(q_values, changed_q)


def test_dsrl_drq_uses_independent_crop_offsets_for_main_and_wrist(monkeypatch):
    offsets = iter((0, 0, 8, 8))

    def fake_randint(_low, _high, size, *, device):
        return torch.full(size, next(offsets), dtype=torch.long, device=device)

    monkeypatch.setattr(torch, "randint", fake_randint)
    pixels = torch.arange(8 * 8, dtype=torch.float32).reshape(1, 8, 8, 1)
    pixels = pixels.expand(-1, -1, -1, 3).contiguous()

    augmented = apply_drq(
        {"main_images": pixels.clone(), "wrist_images": pixels.clone()},
        pad=4,
    )

    assert augmented["main_images"].shape == pixels.shape
    assert augmented["wrist_images"].shape == pixels.shape
    assert not torch.equal(augmented["main_images"], augmented["wrist_images"])


def test_dsrl_drq_compact_views_use_independent_crop_offsets(monkeypatch):
    offsets = iter(((0, 8), (0, 8)))

    def fake_randint(_low, _high, size, *, device):
        return torch.tensor(next(offsets), dtype=torch.long, device=device).reshape(
            size
        )

    monkeypatch.setattr(torch, "randint", fake_randint)
    pixels = torch.arange(8 * 8, dtype=torch.bfloat16).reshape(1, 1, 1, 8, 8)
    pixels = pixels.expand(-1, 2, 3, -1, -1).contiguous()

    augmented = apply_drq({"dsrl_images": pixels.clone()}, pad=4)

    assert augmented["dsrl_images"].shape == pixels.shape
    assert not torch.equal(
        augmented["dsrl_images"][:, 0], augmented["dsrl_images"][:, 1]
    )


def test_dsrl_replay_round_trip_preserves_wrist_and_feeds_actor_and_critic():
    values = torch.tensor([[10, 20], [30, 40]], dtype=torch.uint8)

    def images(offset):
        return (values + offset)[..., None, None, None].expand(2, 2, 32, 32, 3)

    trajectory = Trajectory(
        max_episode_length=20,
        model_weights_id="dual-camera-test",
        rewards=torch.zeros(2, 2, 10),
        curr_obs={
            "main_images": images(0),
            "wrist_images": images(100),
            "states": torch.randn(2, 2, 7),
            "tactile_marker_motion": torch.randn(2, 2, 9, 198, 2),
        },
        next_obs={
            "main_images": images(1),
            "wrist_images": images(101),
            "states": torch.randn(2, 2, 7),
            "tactile_marker_motion": torch.randn(2, 2, 9, 198, 2),
        },
    )
    replay = TrajectoryReplayBuffer(seed=3, auto_save=False, sample_window_size=4)
    try:
        replay.add_trajectories([trajectory])
        batch = replay.sample_chunks(4)
    finally:
        replay.close()

    for obs_key in ("curr_obs", "next_obs"):
        sampled_obs = batch[obs_key]
        assert "wrist_images" in sampled_obs
        assert sampled_obs["wrist_images"].shape == (4, 32, 32, 3)
        delta = sampled_obs["wrist_images"][:, 0, 0, 0].to(torch.int16)
        delta -= sampled_obs["main_images"][:, 0, 0, 0].to(torch.int16)
        assert torch.equal(delta, torch.full_like(delta, 100))

    model = _runtime_model(use_tactile=True, num_images=2)
    actions, _, _ = model.sac_forward(batch["curr_obs"], mode="eval")
    q_values = model.sac_q_forward(
        batch["next_obs"],
        actions=actions.detach(),
    )
    assert actions.shape == (4, 2, 32)
    assert q_values.shape == (4, 10)


@pytest.mark.parametrize("method_name", ["sac_forward", "sac_q_forward"])
def test_dsrl_tactile_requires_marker_motion_key(method_name):
    model = _runtime_model(use_tactile=True)
    kwargs = (
        {"actions": torch.randn(2, 2, 32)} if method_name == "sac_q_forward" else {}
    )

    with pytest.raises(ValueError, match="tactile_marker_motion.*keys"):
        getattr(model, method_name)(_obs(include_tactile=False), **kwargs)


@pytest.mark.parametrize(
    "actual_shape",
    [(2, 8, 198, 2), (2, 9, 197, 2), (2, 9, 198), (2, 9, 198, 3)],
)
@pytest.mark.parametrize("method_name", ["sac_forward", "sac_q_forward"])
def test_dsrl_tactile_rejects_wrong_shape_with_actual_shape(method_name, actual_shape):
    model = _runtime_model(use_tactile=True)
    kwargs = (
        {"actions": torch.randn(2, 2, 32)} if method_name == "sac_q_forward" else {}
    )

    with pytest.raises(
        ValueError, match=str(actual_shape).replace("(", r"\(").replace(")", r"\)")
    ):
        getattr(model, method_name)(_obs(tactile_shape=actual_shape), **kwargs)


@pytest.mark.parametrize("marker_count", [198, 440])
def test_dsrl_tactile_actor_and_q_shapes_and_gradients_are_independent(marker_count):
    model = _runtime_model(use_tactile=True, tactile_input_dim=marker_count * 2)
    obs = _obs(tactile_shape=(2, 9, marker_count, 2))

    actions, logprobs, _ = model.sac_forward(obs, train=False)

    assert actions.shape == (2, 2, 32)
    assert logprobs.shape == (2,)
    actions.sum().backward()
    assert _has_grad(model.actor_tactile_encoder)
    assert not _has_grad(model.critic_tactile_encoder)

    model.zero_grad(set_to_none=True)
    q_values = model.sac_q_forward(obs, actions=actions.detach(), train=False)

    assert q_values.shape == (2, 10)
    q_values.sum().backward()
    assert _has_grad(model.critic_tactile_encoder)
    assert not _has_grad(model.actor_tactile_encoder)


def test_standard_dsrl_does_not_require_or_initialize_tactile():
    model = _runtime_model(use_tactile=False)

    actions, logprobs, _ = model.sac_forward(_obs(include_tactile=False))
    q_values = model.sac_q_forward(
        _obs(include_tactile=False), actions=actions.detach()
    )

    assert not hasattr(model, "actor_tactile_encoder")
    assert not hasattr(model, "critic_tactile_encoder")
    assert actions.shape == (2, 2, 32)
    assert logprobs.shape == (2,)
    assert q_values.shape == (2, 10)


def test_predict_action_batch_sends_tacfield_to_prefix_and_dsrl_paths():
    model = _bare_model(
        SimpleNamespace(
            use_dsrl=True,
            dsrl_use_tactile=True,
            dsrl_num_images=2,
            is_nft=False,
        )
    )
    tactile = torch.randn(2, 9, 198, 2)
    env_obs = {
        "main_images": torch.zeros(2, 16, 16, 3, dtype=torch.uint8),
        "wrist_images": torch.ones(2, 16, 16, 3, dtype=torch.uint8),
        "states": torch.zeros(2, 7),
        "tactile_marker_motion": tactile,
    }
    captured = {}

    def input_transform(self, obs, transpose):
        captured["prefix_obs"] = obs
        return {
            **obs,
            "tokenized_prompt": torch.ones(2, 2, dtype=torch.long),
            "tokenized_prompt_mask": torch.ones(2, 2, dtype=torch.bool),
        }

    def sac_forward(self, obs, **kwargs):
        captured["dsrl_obs"] = obs
        return torch.zeros(2, 2, 32), torch.zeros(2), None

    def sample_actions(self, observation, **kwargs):
        captured["sample_kwargs"] = kwargs
        return {
            "actions": torch.zeros(2, 2, 13),
            "chains": torch.zeros(2, 2, 2, 32),
            "denoise_inds": torch.zeros(2, 2, dtype=torch.long),
            "prev_values": torch.zeros(2, 1),
        }

    model.obs_processor = MethodType(lambda self, obs: obs, model)
    model.input_transform = MethodType(input_transform, model)
    model.precision_processor = MethodType(lambda self, obs: obs, model)
    model._observation_from_dict = MethodType(lambda self, obs: object(), model)
    model.sac_forward = MethodType(sac_forward, model)
    model.sample_actions = MethodType(sample_actions, model)
    model._make_output_transform_input = MethodType(
        lambda self, actions, observation: {"actions": actions}, model
    )
    model.output_transform = MethodType(lambda self, outputs: outputs, model)

    _, result = model.predict_action_batch(env_obs)

    assert captured["prefix_obs"]["tactile_marker_motion"] is tactile
    assert captured["dsrl_obs"]["tactile_marker_motion"] is tactile
    assert captured["dsrl_obs"]["images"][0] is env_obs["main_images"]
    assert captured["dsrl_obs"]["images"][1] is env_obs["wrist_images"]
    assert captured["sample_kwargs"]["collect_forward_metadata"] is False
    assert captured["sample_kwargs"]["compute_values"] is False
    assert set(result["forward_inputs"]) == {"action"}
    assert result["forward_inputs"]["action"].shape == (2, 32)
    assert result["prev_values"] is None


@pytest.mark.parametrize(
    ("env_obs", "message"),
    [
        (
            {
                "main_images": torch.zeros(2, 16, 16, 3, dtype=torch.uint8),
                "states": torch.zeros(2, 7),
            },
            "tactile_marker_motion.*keys",
        ),
        (
            {
                "main_images": torch.zeros(2, 16, 16, 3, dtype=torch.uint8),
                "states": torch.zeros(2, 7),
                "tactile_marker_motion": torch.zeros(2, 8, 198, 2),
            },
            r"\(2, 8, 198, 2\)",
        ),
    ],
)
def test_predict_action_batch_validates_raw_dsrl_tactile_before_transforms(
    env_obs, message
):
    model = _bare_model(
        SimpleNamespace(use_dsrl=True, dsrl_use_tactile=True, is_nft=False)
    )
    model.obs_processor = MethodType(
        lambda self, obs: pytest.fail("obs_processor must not run before validation"),
        model,
    )

    with pytest.raises(ValueError, match=message):
        model.predict_action_batch(env_obs)


def test_freeze_non_dsrl_parameters_matches_exact_whitelist():
    model = _runtime_model(use_tactile=True)
    model.backbone = nn.Linear(2, 2)
    model.tactile_prefix_encoder = nn.Linear(2, 2)
    model.projection = nn.Linear(2, 2)
    model.value_head = nn.Linear(2, 1)
    model.lora_adapter = nn.Linear(2, 2)

    model.freeze_non_dsrl_parameters()

    allowed_prefixes = (
        "dsrl_action_noise_net.",
        "actor_image_encoder.",
        "actor_state_encoder.",
        "actor_tactile_encoder.",
        "critic_image_encoder.",
        "critic_state_encoder.",
        "critic_tactile_encoder.",
        "q_head.",
    )
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad is name.startswith(allowed_prefixes), name


class _FactoryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.dsrl_action_noise_net = nn.Linear(2, 2)
        self.freeze_calls = 0

    def freeze_non_dsrl_parameters(self):
        self.freeze_calls += 1


def _factory_config(*, is_lora: bool):
    return OmegaConf.create(
        {
            "model_type": "openpi",
            "precision": "bf16",
            "load_to_device": False,
            "is_lora": is_lora,
            "openpi": {"use_dsrl": True},
        }
    )


def test_model_factory_freezes_dsrl_after_model_builder(monkeypatch):
    model = _FactoryModel()
    monkeypatch.setitem(
        model_registry._MODEL_REGISTRY, "openpi", lambda cfg, dtype: model
    )

    result = model_registry.get_model(_factory_config(is_lora=False))

    assert result is model
    assert model.freeze_calls == 1


def test_model_factory_rejects_dsrl_with_lora_before_building(monkeypatch):
    built = False

    def builder(cfg, dtype):
        nonlocal built
        built = True
        return _FactoryModel()

    monkeypatch.setitem(model_registry._MODEL_REGISTRY, "openpi", builder)

    with pytest.raises(ValueError, match="DSRL.*is_lora=false"):
        model_registry.get_model(_factory_config(is_lora=True))

    assert built is False


def test_dsrl_critic_optimizer_filter_includes_tactile_encoder():
    assert EmbodiedSACFSDPPolicy._dsrl_critic_param_filters() == [
        "critic_image_encoder",
        "critic_state_encoder",
        "critic_tactile_encoder",
        "q_head",
    ]


class _ActorForwardModel:
    def __call__(self, *, forward_type, **kwargs):
        if forward_type.value == "sac":
            return torch.zeros(2, 2, 32), torch.zeros(2), None
        if forward_type.value == "sac_q":
            return torch.arange(20, dtype=torch.float32).reshape(2, 10)
        raise AssertionError(forward_type)


def test_dsrl_actor_metrics_use_nested_openpi_q_head_count():
    worker = EmbodiedSACFSDPPolicy.__new__(EmbodiedSACFSDPPolicy)
    worker.cfg = OmegaConf.create(
        {
            "actor": {
                "model": {
                    "model_type": "openpi",
                    "openpi": {"dsrl_num_q_heads": 10},
                }
            },
            "algorithm": {"q_head_type": "default", "actor_agg_q": "mean"},
        }
    )
    worker.use_dsrl = True
    worker.model = _ActorForwardModel()
    worker.entropy_temp = SimpleNamespace(alpha=torch.tensor(0.1))

    _, _, metrics = inspect.unwrap(EmbodiedSACFSDPPolicy.forward_actor)(
        worker, {"curr_obs": {}}
    )

    assert set(metrics) == {"q_pi", *(f"q_value_{idx}" for idx in range(10))}


class _CheckpointStrategy:
    def __init__(self):
        self.save_calls = []
        self.load_calls = []

    def save_checkpoint(self, **kwargs):
        self.save_calls.append(kwargs)

    def load_checkpoint(self, **kwargs):
        self.load_calls.append(kwargs)

    def get_model_state_dict(self, model, **kwargs):
        return model.state_dict()

    def load_model_with_state_dict(self, model, state_dict, **kwargs):
        model.load_state_dict(state_dict)


class _CheckpointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.actor_image_encoder = nn.Linear(2, 2)
        self.critic_image_encoder = nn.Linear(2, 2)
        self.critic_state_encoder = nn.Linear(2, 2)
        self.critic_tactile_encoder = nn.Linear(2, 2)
        self.q_head = nn.Linear(2, 2)


def _checkpoint_worker(tmp_path):
    worker = EmbodiedSACFSDPPolicy.__new__(EmbodiedSACFSDPPolicy)
    worker.cfg = OmegaConf.create(
        {
            "actor": {
                "fsdp_config": {
                    "use_orig_params": True,
                    "checkpoint_format": "local_shard",
                    "save_full_model_weights": False,
                    "save_trainable_model_weights": True,
                }
            },
            "algorithm": {"tau": 0.005},
        }
    )
    worker._cfg = worker.cfg.actor
    worker._rank = 0
    worker._strategy = _CheckpointStrategy()
    worker.model = _CheckpointModel()
    worker.target_model = _CheckpointModel()
    worker.target_model_initialized = True
    worker.target_update_type = "all"
    worker.optimizer = object()
    worker.qf_optimizer = object()
    worker.lr_scheduler = object()
    worker.qf_lr_scheduler = object()
    worker.alpha_optimizer = None
    worker.is_weight_offloaded = False
    worker.is_optimizer_offloaded = False
    worker.use_dsrl = True
    worker._logger = SimpleNamespace(info=lambda *_args: None)
    worker.replay_buffer = SimpleNamespace(
        size=0,
        total_samples=0,
        save_checkpoint=lambda path: None,
        load_checkpoint=lambda path: None,
    )
    worker._init_target_shadow()
    return worker


def test_dsrl_checkpoint_honors_fsdp_config_and_exports_trainable_weights(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    worker = _checkpoint_worker(tmp_path)
    trainable_exports = []
    worker._save_trainable_model_weights = MethodType(
        lambda self, save_path, step: trainable_exports.append((save_path, step)),
        worker,
    )

    worker.save_checkpoint(str(tmp_path), step=1)

    [main_save] = worker._strategy.save_calls
    assert main_save["checkpoint_format"] == "local_shard"
    assert main_save["save_full_model_weights"] is False
    assert trainable_exports == [(str(tmp_path), 1)]


def test_dsrl_checkpoint_load_rebuilds_target_shadow_from_restored_weights(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    worker = _checkpoint_worker(tmp_path)
    with torch.no_grad():
        for parameter in worker.target_model.parameters():
            parameter.fill_(-1.0)
    worker._init_target_shadow()

    restored_target = _CheckpointModel()
    with torch.no_grad():
        for parameter in restored_target.parameters():
            parameter.fill_(3.0)
    target_dir = tmp_path / "sac_components" / "target_model"
    target_dir.mkdir(parents=True)
    torch.save(restored_target.state_dict(), target_dir / "checkpoint_rank_0.pt")

    worker.load_checkpoint(str(tmp_path))

    [main_load] = worker._strategy.load_calls
    assert main_load["checkpoint_format"] == "local_shard"
    expected_shadow_names = {
        name
        for name, _ in worker.target_model.named_parameters()
        if name.split(".")[0]
        in {
            "critic_image_encoder",
            "critic_state_encoder",
            "critic_tactile_encoder",
            "q_head",
        }
    }
    assert set(worker._target_shadow_f32) == expected_shadow_names
    for name, parameter in worker.target_model.named_parameters():
        if name in expected_shadow_names:
            torch.testing.assert_close(parameter, torch.full_like(parameter, 3.0))
            torch.testing.assert_close(
                worker._target_shadow_f32[name],
                torch.full_like(worker._target_shadow_f32[name], 3.0),
            )
        else:
            torch.testing.assert_close(parameter, torch.full_like(parameter, 3.0))


def test_dsrl_checkpoint_load_returns_component_receipt(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    worker = _checkpoint_worker(tmp_path)
    worker.entropy_temp = object()
    worker.alpha_optimizer = object()
    worker.alpha_lr_scheduler = object()
    replay_loads = []
    worker.replay_buffer = SimpleNamespace(
        size=1,
        total_samples=1512,
        load_checkpoint=replay_loads.append,
    )
    target_dir = tmp_path / "sac_components" / "target_model"
    target_dir.mkdir(parents=True)
    restored_target = _CheckpointModel()
    with torch.no_grad():
        for parameter in restored_target.parameters():
            parameter.fill_(4.0)
    torch.save(
        restored_target.state_dict(),
        target_dir / "checkpoint_rank_0.pt",
    )

    receipt = worker.load_checkpoint(str(tmp_path))

    assert len(worker._strategy.load_calls) == 2
    assert worker._strategy.load_calls[0]["optimizers"] == [
        worker.optimizer,
        worker.qf_optimizer,
    ]
    assert worker._strategy.load_calls[1]["optimizers"] is worker.alpha_optimizer
    assert replay_loads == [str(tmp_path / "sac_components/replay_buffer/rank_0")]
    for name, parameter in worker.target_model.named_parameters():
        if name.split(".")[0] in {
            "critic_image_encoder",
            "critic_state_encoder",
            "critic_tactile_encoder",
            "q_head",
        }:
            torch.testing.assert_close(parameter, torch.full_like(parameter, 4.0))
    assert receipt == {
        "rank": 0,
        "checkpoint_format": "local_shard",
        "model": "loaded",
        "optimizers": ["actor", "critic"],
        "alpha": "loaded",
        "target_model": "loaded",
        "replay_buffer": {"size": 1, "total_samples": 1512},
    }


class _CompletedHandle:
    def __init__(self, result=None):
        self.result = result

    def wait(self):
        return self.result


class _InitOnlyGroup:
    def init_worker(self):
        return _CompletedHandle()


class _ResumeActor(_InitOnlyGroup):
    def __init__(self, receipts):
        self.receipts = receipts
        self.loaded_path = None

    def load_checkpoint(self, path):
        self.loaded_path = path
        return _CompletedHandle(self.receipts)


def test_embodied_runner_logs_checkpoint_component_receipts(tmp_path):
    checkpoint = tmp_path / "global_step_1"
    (checkpoint / "actor").mkdir(parents=True)
    receipts = [
        {
            "rank": rank,
            "model": "loaded",
            "optimizers": ["actor", "critic"],
            "alpha": "loaded",
            "target_model": "loaded",
            "replay_buffer": {"size": 1, "total_samples": 1512},
        }
        for rank in range(4)
    ]
    messages = []
    runner = EmbodiedRunner.__new__(EmbodiedRunner)
    runner.cfg = OmegaConf.create({"runner": {"resume_dir": str(checkpoint)}})
    runner.actor = _ResumeActor(receipts)
    runner.rollout = _InitOnlyGroup()
    runner.env = _InitOnlyGroup()
    runner.reward = None
    runner.logger = SimpleNamespace(info=messages.append)
    runner.global_step = 0

    runner.init_workers()

    assert runner.actor.loaded_path == str(checkpoint / "actor")
    assert runner.global_step == 1
    assert any("Checkpoint restore receipts" in message for message in messages)
    assert any("'rank': 3" in message for message in messages)


def test_embodied_runner_preserves_none_checkpoint_receipt_compatibility(tmp_path):
    checkpoint = tmp_path / "global_step_1"
    (checkpoint / "actor").mkdir(parents=True)
    messages = []
    runner = EmbodiedRunner.__new__(EmbodiedRunner)
    runner.cfg = OmegaConf.create({"runner": {"resume_dir": str(checkpoint)}})
    runner.actor = _ResumeActor([None, None, None, None])
    runner.rollout = _InitOnlyGroup()
    runner.env = _InitOnlyGroup()
    runner.reward = None
    runner.logger = SimpleNamespace(info=messages.append)
    runner.global_step = 0

    runner.init_workers()

    assert runner.global_step == 1
    assert not any("Checkpoint restore receipts" in message for message in messages)


def test_dsrl_target_shadow_and_ema_only_track_target_critic_parameters(tmp_path):
    worker = _checkpoint_worker(tmp_path)
    with torch.no_grad():
        for parameter in worker.model.parameters():
            parameter.fill_(3.0)
        for parameter in worker.target_model.parameters():
            parameter.fill_(1.0)

    worker._init_target_shadow()
    worker.soft_update_target_model(tau=0.5)

    expected_modules = {
        "critic_image_encoder",
        "critic_state_encoder",
        "critic_tactile_encoder",
        "q_head",
    }
    assert {
        name.split(".")[0] for name in worker._target_shadow_f32
    } == expected_modules
    for name, parameter in worker.target_model.named_parameters():
        expected = 2.0 if name.split(".")[0] in expected_modules else 1.0
        torch.testing.assert_close(parameter, torch.full_like(parameter, expected))
