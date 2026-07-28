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

from copy import deepcopy
from math import prod
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0ForRLActionPrediction,
)
from rlinf.utils.dsrl_checkpoint import (
    DSRL_TRAINABLE_MANIFEST_V1,
    restore_target_payload,
    select_dsrl_trainable_state,
)
from rlinf.utils.dsrl_rollout_sync import DSRL_ROLLOUT_SYNC_MANIFEST_V1
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy

CRITIC_PREFIXES = (
    "critic_image_encoder.",
    "critic_state_encoder.",
    "critic_tactile_encoder.",
    "q_head.",
)
TRAINABLE_PREFIXES = (
    "dsrl_action_noise_net.",
    "actor_image_encoder.",
    "actor_state_encoder.",
    "actor_tactile_encoder.",
    *CRITIC_PREFIXES,
)


class _CheckpointStrategy:
    def __init__(self):
        self.save_calls = []
        self.load_calls = []
        self.full_state_dict_calls = 0
        self.full_load_calls = 0

    def save_checkpoint(self, **kwargs):
        self.save_calls.append(kwargs)

    def load_checkpoint(self, **kwargs):
        self.load_calls.append(kwargs)

    def get_model_state_dict(self, *_args, **_kwargs):
        self.full_state_dict_calls += 1
        raise AssertionError(
            "DSRL compact checkpoint must not gather a full state dict"
        )

    def load_model_with_state_dict(self, *_args, **_kwargs):
        self.full_load_calls += 1
        raise AssertionError("DSRL compact checkpoint must not use a full load API")


class _CheckpointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.dsrl_action_noise_net = nn.Linear(2, 2)
        self.actor_image_encoder = nn.Linear(2, 2)
        self.actor_state_encoder = nn.Linear(2, 2)
        self.actor_tactile_encoder = nn.Linear(2, 2)
        self.critic_image_encoder = nn.Linear(2, 2)
        self.critic_state_encoder = nn.Linear(2, 2)
        self.critic_tactile_encoder = nn.Linear(2, 2)
        self.q_head = nn.Linear(2, 2)


def _worker(*, use_dsrl=True, save_trainable=False):
    worker = EmbodiedSACFSDPPolicy.__new__(EmbodiedSACFSDPPolicy)
    worker.cfg = OmegaConf.create(
        {
            "actor": {
                "fsdp_config": {
                    "use_orig_params": True,
                    "sharding_strategy": "no_shard",
                    "checkpoint_format": "local_shard",
                    "save_full_model_weights": False,
                    "save_trainable_model_weights": save_trainable,
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
    worker.use_dsrl = use_dsrl
    worker.replay_buffer = SimpleNamespace(
        size=0,
        total_samples=0,
        save_checkpoint=lambda _path: None,
        load_checkpoint=lambda _path: None,
    )
    worker._logger = SimpleNamespace(info=lambda *_args: None)
    return worker


@pytest.fixture
def distributed(monkeypatch):
    barriers = []
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: barriers.append(True))
    return barriers


def _target_path(base_path):
    return base_path / "sac_components" / "target_model" / "checkpoint_rank_0.pt"


def _critic_named_parameters(model):
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.startswith(CRITIC_PREFIXES)
    }


def _save_compact_target(tmp_path, distributed):
    worker = _worker()
    with torch.no_grad():
        for index, parameter in enumerate(worker.target_model.parameters(), start=1):
            parameter.fill_(index)
    worker._init_target_shadow()
    for index, shadow in enumerate(worker._target_shadow_f32.values(), start=1):
        shadow.add_(index / 1000)

    worker.save_checkpoint(str(tmp_path), step=17)
    payload = torch.load(_target_path(tmp_path), map_location="cpu", weights_only=True)
    return worker, payload


def test_compact_target_save_contains_only_critic_and_exact_shadow(
    tmp_path, distributed
):
    worker, payload = _save_compact_target(tmp_path, distributed)
    runtime = _critic_named_parameters(worker.target_model)

    assert payload["format"] == "tabero_dsrl_target"
    assert payload["version"] == 1
    assert payload["metadata"] == {
        "step": 17,
        "rank": 0,
        "world_size": 4,
        "tensor_count": len(runtime),
        "parameter_count": sum(param.numel() for param in runtime.values()),
        "shadow_tensor_count": len(runtime),
        "shadow_parameter_count": sum(param.numel() for param in runtime.values()),
    }
    assert set(payload["model"]) == set(runtime)
    assert set(payload["target_shadow_f32"]) == set(runtime)
    assert all(name.startswith(CRITIC_PREFIXES) for name in payload["model"])
    assert not any("actor" in name or "backbone" in name for name in payload["model"])
    for name, parameter in runtime.items():
        assert torch.equal(payload["model"][name], parameter.cpu())
        assert payload["model"][name].is_contiguous()
        assert payload["target_shadow_f32"][name].dtype == torch.float32
        assert torch.equal(
            payload["target_shadow_f32"][name], worker._target_shadow_f32[name].cpu()
        )
    assert worker._strategy.full_state_dict_calls == 0


def test_compact_target_round_trip_restores_parameters_and_shadow_exactly(
    tmp_path, distributed
):
    worker, payload = _save_compact_target(tmp_path, distributed)
    expected_model = deepcopy(payload["model"])
    expected_shadow = deepcopy(payload["target_shadow_f32"])
    with torch.no_grad():
        for parameter in worker.target_model.parameters():
            parameter.fill_(-9)
        for shadow in worker._target_shadow_f32.values():
            shadow.fill_(-11)

    receipt = worker.load_checkpoint(str(tmp_path))

    for name, parameter in _critic_named_parameters(worker.target_model).items():
        assert torch.equal(parameter.cpu(), expected_model[name])
        assert torch.equal(worker._target_shadow_f32[name].cpu(), expected_shadow[name])
    assert worker.target_model.backbone.weight.eq(-9).all()
    assert worker.target_model.actor_image_encoder.weight.eq(-9).all()
    assert receipt["target_model"] == {
        "format": "compact_v1",
        "tensor_count": len(expected_model),
        "parameter_count": sum(tensor.numel() for tensor in expected_model.values()),
        "shadow_tensor_count": len(expected_shadow),
    }
    assert worker._strategy.full_load_calls == 0


def _valid_compact_payload(worker):
    worker._init_target_shadow()
    model = {
        name: parameter.detach().cpu().contiguous().clone()
        for name, parameter in _critic_named_parameters(worker.target_model).items()
    }
    shadow = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in worker._target_shadow_f32.items()
    }
    parameter_count = sum(tensor.numel() for tensor in model.values())
    return {
        "format": "tabero_dsrl_target",
        "version": 1,
        "metadata": {
            "step": 1,
            "rank": 0,
            "world_size": 4,
            "tensor_count": len(model),
            "parameter_count": parameter_count,
            "shadow_tensor_count": len(shadow),
            "shadow_parameter_count": parameter_count,
        },
        "model": model,
        "target_shadow_f32": shadow,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda p: p["model"].pop("q_head.bias"), "missing keys"),
        (
            lambda p: p["model"].__setitem__("q_head.extra", torch.ones(1)),
            "unexpected keys",
        ),
        (
            lambda p: p["model"].__setitem__("q_head.weight", torch.ones(1)),
            "shape mismatches",
        ),
        (
            lambda p: p["model"]["q_head.weight"].fill_(torch.nan),
            "non-finite",
        ),
        (
            lambda p: p["model"].__setitem__(
                "q_head.weight", p["model"]["q_head.weight"].double()
            ),
            "dtype mismatches",
        ),
        (lambda p: p.__setitem__("format", "wrong_format"), "wrong format"),
        (lambda p: p.__setitem__("version", 2), "version"),
        (lambda p: p.__setitem__("version", True), "version.*integer"),
        (lambda p: p["metadata"].__setitem__("rank", 3), "rank"),
        (lambda p: p["metadata"].__setitem__("rank", False), "rank.*integer"),
        (
            lambda p: p["metadata"].__setitem__("world_size", 4.0),
            "world_size.*integer",
        ),
        (
            lambda p: p["metadata"].__setitem__(
                "tensor_count", float(p["metadata"]["tensor_count"])
            ),
            "tensor_count.*integer",
        ),
        (lambda p: p["metadata"].__setitem__("step", True), "step.*integer"),
        (
            lambda p: p["metadata"].__setitem__("tensor_count", 999),
            "count mismatch",
        ),
        (lambda p: p["target_shadow_f32"].pop("q_head.bias"), "shadow.*missing"),
        (
            lambda p: p["target_shadow_f32"].__setitem__(
                "q_head.weight", p["target_shadow_f32"]["q_head.weight"].double()
            ),
            "shadow.*dtype mismatches",
        ),
    ],
)
def test_compact_target_load_rejects_invalid_payload(
    tmp_path, distributed, mutation, message
):
    worker = _worker()
    payload = _valid_compact_payload(worker)
    mutation(payload)
    path = _target_path(tmp_path)
    path.parent.mkdir(parents=True)
    torch.save(payload, path)

    with pytest.raises(ValueError, match=message):
        restore_target_payload(
            payload,
            worker.target_model,
            rank=0,
            world_size=4,
        )


def _legacy_state(worker):
    state = {}
    for name, parameter in worker.target_model.named_parameters():
        state[f"module._fsdp_wrapped_module.{name}"] = torch.full_like(parameter, 7)
    return state


def test_legacy_full_target_filters_critic_and_rebuilds_shadow(tmp_path, distributed):
    worker = _worker()
    with torch.no_grad():
        for parameter in worker.target_model.parameters():
            parameter.fill_(-3)
    shadow, receipt = restore_target_payload(
        _legacy_state(worker),
        worker.target_model,
        rank=0,
        world_size=4,
    )
    worker._target_shadow_f32 = shadow

    for name, parameter in worker.target_model.named_parameters():
        expected = 7 if name.startswith(CRITIC_PREFIXES) else -3
        assert parameter.eq(expected).all(), name
    assert set(worker._target_shadow_f32) == set(
        _critic_named_parameters(worker.target_model)
    )
    assert all(
        shadow.dtype == torch.float32 for shadow in worker._target_shadow_f32.values()
    )
    assert all(shadow.eq(7).all() for shadow in worker._target_shadow_f32.values())
    assert receipt["format"] == "legacy_full"
    assert receipt["tensor_count"] == 8
    assert worker._strategy.full_load_calls == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda state: state.pop("module._fsdp_wrapped_module.q_head.bias"),
            "missing keys",
        ),
        (
            lambda state: state.__setitem__(
                "module._fsdp_wrapped_module.q_head.extra", torch.ones(1)
            ),
            "unexpected keys",
        ),
        (
            lambda state: state.__setitem__(
                "module._fsdp_wrapped_module.q_head.weight", torch.ones(1)
            ),
            "shape mismatches",
        ),
        (
            lambda state: state["module._fsdp_wrapped_module.q_head.weight"].fill_(
                torch.inf
            ),
            "non-finite",
        ),
    ],
)
def test_legacy_target_load_rejects_invalid_critic_state(
    tmp_path, distributed, mutation, message
):
    worker = _worker()
    state = _legacy_state(worker)
    mutation(state)
    with pytest.raises(ValueError, match=message):
        restore_target_payload(
            state,
            worker.target_model,
            rank=0,
            world_size=4,
        )


def _canonical_trainable_shapes():
    shapes = dict(DSRL_ROLLOUT_SYNC_MANIFEST_V1)
    shapes.update(
        {
            name.replace("actor_", "critic_", 1): shape
            for name, shape in DSRL_ROLLOUT_SYNC_MANIFEST_V1.items()
            if name.startswith("actor_")
        }
    )
    q_layer_shapes = {
        "net.0.weight": (128, 224),
        "net.0.bias": (128,),
        "net.1.weight": (128,),
        "net.1.bias": (128,),
        "net.3.weight": (128, 128),
        "net.3.bias": (128,),
        "net.4.weight": (128,),
        "net.4.bias": (128,),
        "net.6.weight": (128, 128),
        "net.6.bias": (128,),
        "net.7.weight": (128,),
        "net.7.bias": (128,),
        "net.9.weight": (1, 128),
        "net.9.bias": (1,),
    }
    for head_index in range(10):
        shapes.update(
            {
                f"q_head.q_heads.{head_index}.{suffix}": shape
                for suffix, shape in q_layer_shapes.items()
            }
        )
    assert len(shapes) == 220
    assert sum(prod(shape) for shape in shapes.values()) == 5_183_754
    return shapes


class _ManyParameters(nn.Module):
    def __init__(self):
        super().__init__()
        self.checkpoint_parameters = [
            (name, nn.Parameter(torch.ones(shape)))
            for name, shape in _canonical_trainable_shapes().items()
        ]
        for index, (_, parameter) in enumerate(self.checkpoint_parameters):
            self.register_parameter(f"registered_parameter_{index:03d}", parameter)

    def named_parameters(self, *args, **kwargs):
        del args, kwargs
        yield from self.checkpoint_parameters

    def replace_name(self, old_name, new_name):
        self.checkpoint_parameters = [
            (new_name if name == old_name else name, parameter)
            for name, parameter in self.checkpoint_parameters
        ]

    def replace_shape(self, name, shape):
        for index, (parameter_name, parameter) in enumerate(self.checkpoint_parameters):
            if parameter_name == name:
                replacement = nn.Parameter(torch.ones(shape))
                self.checkpoint_parameters[index] = (parameter_name, replacement)
                setattr(self, f"registered_parameter_{index:03d}", replacement)
                return
        raise AssertionError(name)

    def parameter_by_name(self, expected_name):
        return next(
            parameter
            for name, parameter in self.checkpoint_parameters
            if name == expected_name
        )


def _sidecar_worker(model=None):
    worker = _worker(save_trainable=True)
    worker.model = model or _ManyParameters()
    worker._cfg.fsdp_config.trainable_checkpoint_metadata = {
        "method": "dsrl",
        "target_global_step": 50,
        "global_step": -1,
        "is_final": False,
    }
    return worker


def test_dsrl_sidecar_saves_exact_direct_trainable_manifest(tmp_path, distributed):
    worker = _sidecar_worker()

    worker._save_trainable_model_weights(str(tmp_path), step=50)

    payload = torch.load(
        tmp_path / "model_state_dict" / "trainable_weights.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert len(payload["model"]) == 220
    assert sum(tensor.numel() for tensor in payload["model"].values()) == 5_183_754
    assert all(name.startswith(TRAINABLE_PREFIXES) for name in payload["model"])
    assert all(tensor.device.type == "cpu" for tensor in payload["model"].values())
    assert all(tensor.is_contiguous() for tensor in payload["model"].values())
    assert payload["metadata"]["parameter_count"] == 220
    assert payload["metadata"]["global_step"] == 50
    assert payload["metadata"]["is_final"] is True
    assert distributed == [True]
    assert worker._strategy.full_state_dict_calls == 0


def test_dsrl_trainable_manifest_matches_representative_runtime_components():
    model = OpenPi0ForRLActionPrediction.__new__(OpenPi0ForRLActionPrediction)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        use_dsrl=True,
        dsrl_use_tactile=True,
        dsrl_tactile_latent_dim=64,
        dsrl_state_dim=7,
        dsrl_action_noise_dim=32,
        dsrl_num_q_heads=10,
        dsrl_image_latent_dim=64,
        dsrl_state_latent_dim=64,
        dsrl_hidden_dims=(128, 128, 128),
        action_horizon=2,
    )
    model._init_dsrl_components()

    state = select_dsrl_trainable_state(model)

    assert set(state) == set(DSRL_TRAINABLE_MANIFEST_V1)
    assert {name: tuple(tensor.shape) for name, tensor in state.items()} == dict(
        DSRL_TRAINABLE_MANIFEST_V1
    )


def test_dsrl_sidecar_nonzero_rank_only_synchronizes(monkeypatch, tmp_path):
    barriers = []
    worker = _sidecar_worker(nn.Linear(1, 1))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: barriers.append(True))

    worker._save_trainable_model_weights(str(tmp_path), step=1)

    assert barriers == [True]
    assert not (tmp_path / "model_state_dict").exists()


@pytest.mark.parametrize(
    ("break_model", "message"),
    [
        (
            lambda model: model.replace_name(
                "actor_image_encoder.encoder.0.weight", "backbone.replacement"
            ),
            "allowed prefixes",
        ),
        (
            lambda model: setattr(
                model.parameter_by_name("actor_image_encoder.encoder.0.weight"),
                "requires_grad",
                False,
            ),
            "exactly 220 tensors",
        ),
        (
            lambda model: model.parameter_by_name(
                "q_head.q_heads.0.net.0.weight"
            ).data.fill_(torch.nan),
            "non-finite",
        ),
    ],
)
def test_dsrl_sidecar_rejects_bad_key_count_or_finite(
    tmp_path, distributed, break_model, message
):
    model = _ManyParameters()
    break_model(model)
    worker = _sidecar_worker(model)

    with pytest.raises(ValueError, match=message):
        select_dsrl_trainable_state(worker.model)


def test_dsrl_sidecar_rejects_same_numel_wrong_shape():
    model = _ManyParameters()
    model.replace_shape("q_head.q_heads.0.net.0.weight", (224, 128))

    with pytest.raises(ValueError, match="shape mismatches"):
        select_dsrl_trainable_state(model)


def test_dsrl_sidecar_rejects_compensating_allowed_prefix_key_substitution():
    model = _ManyParameters()
    model.replace_name(
        "q_head.q_heads.0.net.0.weight",
        "q_head.q_heads.0.net.replacement.weight",
    )

    with pytest.raises(ValueError, match="missing keys.*unexpected keys"):
        select_dsrl_trainable_state(model)


def test_non_dsrl_sidecar_routes_to_inherited_implementation(monkeypatch, tmp_path):
    worker = _worker(use_dsrl=False, save_trainable=True)
    calls = []
    monkeypatch.setattr(
        EmbodiedFSDPActor,
        "_save_trainable_model_weights",
        lambda self, path, step: calls.append((self, path, step)),
    )

    worker._save_trainable_model_weights(str(tmp_path), step=9)

    assert calls == [(worker, str(tmp_path), 9)]


def test_non_dsrl_target_uses_prior_full_state_apis(tmp_path, monkeypatch):
    worker = _worker(use_dsrl=False)
    state = worker.target_model.state_dict()
    worker._strategy.get_model_state_dict = lambda *_args, **_kwargs: state
    worker._strategy.load_model_with_state_dict = lambda *_args, **_kwargs: setattr(
        worker._strategy, "full_load_calls", 1
    )

    worker.save_checkpoint(str(tmp_path), step=2)
    worker.load_checkpoint(str(tmp_path))

    assert _target_path(tmp_path).exists()
    assert worker._strategy.full_load_calls == 1


@pytest.mark.parametrize("rank", range(4))
def test_high_level_checkpoint_preserves_component_calls(tmp_path, distributed, rank):
    worker = _worker()
    worker._rank = rank
    worker.entropy_temp = object()
    worker.alpha_optimizer = object()
    worker.alpha_lr_scheduler = object()
    replay_saves = []
    replay_loads = []
    worker.replay_buffer = SimpleNamespace(
        size=4,
        total_samples=99,
        save_checkpoint=replay_saves.append,
        load_checkpoint=replay_loads.append,
    )
    worker._init_target_shadow()

    worker.save_checkpoint(str(tmp_path), step=3)
    receipt = worker.load_checkpoint(str(tmp_path))

    assert len(worker._strategy.save_calls) == 2
    assert worker._strategy.save_calls[0]["optimizers"] == [
        worker.optimizer,
        worker.qf_optimizer,
    ]
    assert worker._strategy.save_calls[1]["optimizers"] is worker.alpha_optimizer
    assert len(worker._strategy.load_calls) == 2
    assert worker._strategy.load_calls[0]["optimizers"] == [
        worker.optimizer,
        worker.qf_optimizer,
    ]
    assert worker._strategy.load_calls[1]["optimizers"] is worker.alpha_optimizer
    replay_path = str(tmp_path / f"sac_components/replay_buffer/rank_{rank}")
    assert replay_saves == [replay_path]
    assert replay_loads == [replay_path]
    assert receipt["alpha"] == "loaded"
