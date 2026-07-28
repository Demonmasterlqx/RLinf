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

import asyncio
import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from rlinf.hybrid_engines.weight_syncer import PatchWeightSyncer
from rlinf.hybrid_engines.weight_syncer.patch_syncer import WeightPatch
from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0ForRLActionPrediction,
)
from rlinf.scheduler import Worker
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

ROLLOUT_SYNC_PREFIXES = [
    "dsrl_action_noise_net.",
    "actor_image_encoder.",
    "actor_state_encoder.",
    "actor_tactile_encoder.",
]


def _representative_dsrl_model() -> OpenPi0ForRLActionPrediction:
    config = SimpleNamespace(
        use_dsrl=True,
        dsrl_use_tactile=True,
        dsrl_tactile_latent_dim=64,
        dsrl_state_dim=7,
        dsrl_action_noise_dim=32,
        dsrl_num_q_heads=10,
        dsrl_image_latent_dim=64,
        dsrl_state_latent_dim=64,
        dsrl_hidden_dims=(128, 128, 128),
        action_horizon=10,
    )
    model = OpenPi0ForRLActionPrediction.__new__(OpenPi0ForRLActionPrediction)
    nn.Module.__init__(model)
    model.config = config
    model._init_dsrl_components()
    model.frozen_pi0 = nn.Linear(4, 4)
    model.alpha = nn.Parameter(torch.tensor(0.1), requires_grad=False)
    return model


class _FSDPNameWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self._fsdp_wrapped_module = module


def _actor_cfg(**fsdp_overrides):
    fsdp_config = {"sharding_strategy": "no_shard", "use_orig_params": True}
    fsdp_config.update(fsdp_overrides)
    return OmegaConf.create(
        {
            "training_backend": "fsdp",
            "rollout_sync_prefixes": ROLLOUT_SYNC_PREFIXES,
            "fsdp_config": fsdp_config,
        }
    )


def test_dsrl_sender_selects_exact_actor_keyspace_and_normalizes_fsdp_names():
    model = _FSDPNameWrapper(_representative_dsrl_model())

    selected = EmbodiedSACFSDPPolicy._select_dsrl_rollout_state_dict(
        model, ROLLOUT_SYNC_PREFIXES
    )
    EmbodiedSACFSDPPolicy._validate_dsrl_rollout_state_dict(selected)

    assert len(selected) == 48
    assert sum(parameter.numel() for parameter in selected.values()) == 2_311_648
    assert all(key.startswith(tuple(ROLLOUT_SYNC_PREFIXES)) for key in selected)
    assert not any("_fsdp_wrapped_module" in key for key in selected)


def test_dsrl_sender_excludes_critic_q_alpha_target_and_frozen_pi0_keys():
    selected = EmbodiedSACFSDPPolicy._select_dsrl_rollout_state_dict(
        _representative_dsrl_model(), ROLLOUT_SYNC_PREFIXES
    )

    excluded_fragments = ("critic", "q_head", "alpha", "target", "frozen_pi0")
    assert not any(
        fragment in key for key in selected for fragment in excluded_fragments
    )


def test_dsrl_sender_does_not_call_full_model_state_dict_api():
    worker = EmbodiedSACFSDPPolicy.__new__(EmbodiedSACFSDPPolicy)
    worker.use_dsrl = True
    worker.cfg = OmegaConf.create({"actor": _actor_cfg()})
    worker.model = _representative_dsrl_model()
    worker.param_names_need_sync = list(
        EmbodiedSACFSDPPolicy._select_dsrl_rollout_state_dict(
            worker.model, ROLLOUT_SYNC_PREFIXES
        )
    )

    def fail_full_state_dict(*args, **kwargs):
        raise AssertionError("full model state-dict retrieval must not be called")

    worker.get_model_state_dict = fail_full_state_dict
    worker.model.state_dict = fail_full_state_dict

    selected = worker.get_rollout_state_dict()

    assert list(selected) == worker.param_names_need_sync


class _SmallSyncModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dsrl_action_noise_net = nn.Linear(2, 2)
        self.actor_image_encoder = nn.Linear(2, 2)
        self.actor_state_encoder = nn.Linear(2, 2)
        self.actor_tactile_encoder = nn.Linear(2, 2)
        self.critic_image_encoder = nn.Linear(2, 2)
        self.q_head = nn.Linear(2, 1)
        self.frozen_pi0 = nn.Linear(2, 2)


class _DuplexTransport:
    def __init__(self):
        self.sender_to_receiver = asyncio.Queue()
        self.receiver_to_sender = asyncio.Queue()

    async def sender_send(self, data):
        await self.sender_to_receiver.put(data)

    async def sender_recv(self):
        return await self.receiver_to_sender.get()

    async def receiver_send(self, data):
        await self.receiver_to_sender.put(data)

    async def receiver_recv(self):
        return await self.sender_to_receiver.get()


def test_dsrl_sender_receiver_filters_match_and_patch_init_apply(monkeypatch):
    sender_model = _SmallSyncModel()
    receiver_model = copy.deepcopy(sender_model)
    sender_state = EmbodiedSACFSDPPolicy._select_dsrl_rollout_state_dict(
        sender_model, ROLLOUT_SYNC_PREFIXES
    )
    receiver_state = MultiStepRolloutWorker._filter_dsrl_rollout_state_dict(
        receiver_model.state_dict(), ROLLOUT_SYNC_PREFIXES
    )
    assert list(sender_state) == list(receiver_state)

    transport = _DuplexTransport()
    sender_syncer = PatchWeightSyncer(
        snapshot_device="cpu", transport_device="cpu", delta_encoding=False
    )
    receiver_syncer = PatchWeightSyncer(
        snapshot_device="cpu", transport_device="cpu", delta_encoding=False
    )

    async def run_sync():
        original_device_type = Worker.torch_device_type
        monkeypatch.setattr(Worker, "torch_device_type", "cpu")
        await asyncio.gather(
            sender_syncer.init_sender(
                state_dict=sender_state,
                param_names_need_sync=list(sender_state),
                send=transport.sender_send,
                recv=transport.sender_recv,
            ),
            receiver_syncer.init_receiver(
                state_dict=receiver_state,
                recv=transport.receiver_recv,
                send=transport.receiver_send,
            ),
        )
        monkeypatch.setattr(Worker, "torch_device_type", original_device_type)

        key = "actor_image_encoder.weight"
        ordinal = receiver_syncer.ordered_keys.index(key)
        new_value = torch.tensor([17.0], dtype=receiver_state[key].dtype)
        payload = WeightPatch(
            version=torch.tensor(3, dtype=torch.int64),
            ordinals=torch.tensor([ordinal], dtype=torch.int32),
            nnz_per_tensor=torch.tensor([1], dtype=torch.int32),
            rows=torch.tensor([0], dtype=torch.uint8),
            cols=torch.tensor([0], dtype=torch.uint8),
            values=new_value.view(torch.uint8),
        )
        await transport.sender_send(payload)
        return await receiver_syncer.apply(receiver_model, transport.receiver_recv)

    critic_before = receiver_model.critic_image_encoder.weight.detach().clone()
    version = asyncio.run(run_sync())

    assert version == 3
    assert receiver_model.actor_image_encoder.weight[0, 0].item() == 17.0
    torch.testing.assert_close(
        receiver_model.critic_image_encoder.weight, critic_before
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"training_backend": "megatron"}, "training_backend.*fsdp"),
        ({"sharding_strategy": "full_shard"}, "sharding_strategy.*no_shard"),
        ({"use_orig_params": False}, "use_orig_params.*true"),
    ],
)
def test_dsrl_rollout_sync_rejects_invalid_fsdp_config(overrides, message):
    overrides = dict(overrides)
    training_backend = overrides.pop("training_backend", "fsdp")
    actor_cfg = _actor_cfg(**overrides)
    actor_cfg.training_backend = training_backend

    with pytest.raises(ValueError, match=message):
        EmbodiedSACFSDPPolicy._validate_dsrl_rollout_sync_config(actor_cfg)


@pytest.mark.parametrize(
    "prefixes",
    [ROLLOUT_SYNC_PREFIXES[:-1], [*ROLLOUT_SYNC_PREFIXES, "critic_image_encoder."]],
)
def test_dsrl_rollout_sync_rejects_missing_or_extra_configured_prefix(prefixes):
    actor_cfg = _actor_cfg()
    actor_cfg.rollout_sync_prefixes = prefixes

    with pytest.raises(ValueError, match="rollout_sync_prefixes.*exactly"):
        EmbodiedSACFSDPPolicy._validate_dsrl_rollout_sync_config(actor_cfg)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_dsrl_rollout_sync_rejects_wrong_model_keyspace(mutation):
    selected = EmbodiedSACFSDPPolicy._select_dsrl_rollout_state_dict(
        _representative_dsrl_model(), ROLLOUT_SYNC_PREFIXES
    )
    if mutation == "missing":
        selected.pop(next(iter(selected)))
    else:
        selected["actor_state_encoder.unexpected"] = torch.ones(1)

    with pytest.raises(ValueError, match="48 tensors.*2,311,648 parameters"):
        EmbodiedSACFSDPPolicy._validate_dsrl_rollout_state_dict(selected)


def test_non_dsrl_sender_keeps_existing_state_dict_path():
    worker = EmbodiedSACFSDPPolicy.__new__(EmbodiedSACFSDPPolicy)
    worker.use_dsrl = False
    expected = {"full.weight": torch.ones(1)}
    calls = []

    def get_model_state_dict(**kwargs):
        calls.append(kwargs)
        return expected

    worker.get_model_state_dict = get_model_state_dict

    assert worker.get_rollout_state_dict() is expected
    assert calls == [{"cpu_offload": False, "full_state_dict": False}]


def test_non_dsrl_receiver_keeps_existing_state_dict_path():
    worker = MultiStepRolloutWorker.__new__(MultiStepRolloutWorker)
    worker.model_cfg = OmegaConf.create({"openpi": {"use_dsrl": False}})
    expected = {"full.weight": torch.ones(1)}
    worker.hf_model = SimpleNamespace(state_dict=lambda: expected)

    assert worker._get_rollout_sync_state_dict() is expected
