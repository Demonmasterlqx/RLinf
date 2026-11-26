# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only SFT worker seams: payload, metrics and consumed-batch resume."""

import contextlib
import dataclasses
import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.data.openpi_cotrain import (
    CoTrainConfig,
    CoTrainDatasetSpec,
    MixtureDataset,
    StratifiedBatchSampler,
)
from rlinf.workers.sft import fsdp_vla_sft_worker as module


@dataclasses.dataclass(frozen=True)
class SyntheticModelConfig:
    discrete_state_input: bool = True


@dataclasses.dataclass(frozen=True)
class SyntheticTrainConfig:
    model: SyntheticModelConfig


class ObservationDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 40

    def __getitem__(self, index):
        return {
            "image": {"base_0_rgb": np.zeros((4, 4, 3), dtype=np.uint8)},
            "image_mask": {"base_0_rgb": np.bool_(True)},
            "state": np.zeros(3, dtype=np.float32),
            "actions": np.full((2, 3), index, dtype=np.float32),
            "tactile_prefix": np.zeros((9, 12), dtype=np.float32),
            "tactile_contact": np.bool_(index % 2),
        }


def _loader(seed=42, rank=1):
    sampler = StratifiedBatchSampler(
        40, 40, 8, seed=seed, rank=rank, world_size=2, num_batches=10
    )
    raw = torch.utils.data.DataLoader(
        MixtureDataset(ObservationDataset(), ObservationDataset()),
        batch_sampler=sampler,
        generator=torch.Generator().manual_seed(123),
    )
    cotrain = CoTrainConfig(
        CoTrainDatasetSpec("synthetic-real"), CoTrainDatasetSpec("synthetic-sim")
    )
    return module._OpenPiTactileDataLoader(
        raw, data_config="synthetic", cotrain=cotrain
    )


def test_wrapper_preserves_tactile_contact_and_domain():
    loader = _loader()
    batch = next(iter(loader))
    assert loader.data_config() == "synthetic"
    assert batch["tactile_contact"].dtype == torch.bool
    assert batch["tactile_prefix"].shape == (8, 9, 12)
    torch.testing.assert_close(
        batch["observation"].tactile_contact, batch["tactile_contact"]
    )
    assert batch["domain"].tolist().count(1) == 4
    assert not hasattr(batch["observation"], "domain")


def test_worker_routes_cotrain_and_ignores_legacy_paths(monkeypatch):
    from rlinf.models.embodiment.openpi import dataconfig

    calls = []

    def get_config(*args, **kwargs):
        assert kwargs["repo_id"] == "synthetic-real"
        assert kwargs["batch_size"] == 16
        assert kwargs["data_kwargs"]["norm_stats_path"] == "offline-union"
        return SyntheticTrainConfig(SyntheticModelConfig())

    def build(config, cotrain, **kwargs):
        assert config.model.discrete_state_input is False
        assert cotrain.real.dataset_path == "synthetic-real"
        assert cotrain.sim.dataset_path == "synthetic-sim"
        calls.append(kwargs)
        return _loader()._pytorch_data_loader, "synthetic"

    monkeypatch.setattr(dataconfig, "get_openpi_config", get_config)
    monkeypatch.setattr(module, "create_cotrain_data_loader", build)
    cfg = OmegaConf.create(
        {
            "data": {
                "cotrain": {
                    "real": {"dataset_path": "synthetic-real"},
                    "sim": {"dataset_path": "synthetic-sim"},
                }
            },
            "actor": {
                "micro_batch_size": 8,
                "seed": 42,
                "model": {
                    "model_type": "openpi",
                    "model_path": "unused",
                    "openpi": {"config_name": "unused", "discrete_state_input": False},
                    "openpi_data": {"norm_stats_path": "offline-union"},
                },
            },
        }
    )
    worker = SimpleNamespace(
        cfg=cfg, _rank=1, _world_size=2, _logger=logging.getLogger(__name__)
    )
    loader, data = module.FSDPVlaSftWorker.build_dataloader(worker, "ignored-path")
    assert data == "synthetic"
    assert calls == [{"micro_batch_size": 8, "rank": 1, "world_size": 2, "seed": 42}]
    assert next(iter(loader))["domain"].tolist().count(1) == 4


@pytest.mark.parametrize("with_sample_loss", [True, False])
@pytest.mark.parametrize("domains", [[0, 1, 0, 1], [0, 0, 0, 0], [1, 1, 1, 1]])
def test_worker_strips_domain_and_aggregates_optional_sample_losses(
    with_sample_loss, domains
):
    loss = torch.tensor(4.0, requires_grad=True)
    output = {"loss": loss, "action_loss": torch.tensor(3.0)}
    if with_sample_loss:
        output["per_sample_loss"] = torch.tensor(
            [1.0, 3.0, 5.0, 7.0], requires_grad=True
        )

    def forward(**kwargs):
        assert "domain" not in kwargs["data"]
        assert kwargs["data"]["tactile_contact"] is True
        return output

    worker = SimpleNamespace(
        model=forward,
        amp_context=contextlib.nullcontext(),
        _released_cuda_cache_before_first_forward=True,
    )
    batch = {"domain": domains, "tactile_contact": True}
    actual_loss, metrics = module.FSDPVlaSftWorker.get_train_model_output(worker, batch)
    assert actual_loss is loss
    assert batch["domain"] is domains
    assert metrics["loss"] == 4.0 and metrics["action_loss"] == 3.0
    assert "per_sample_loss" not in metrics
    for domain, name in ((0, "loss_real"), (1, "loss_sim")):
        if with_sample_loss and domain in domains:
            selected = [
                value
                for value, tag in zip([1.0, 3.0, 5.0, 7.0], domains, strict=True)
                if tag == domain
            ]
            assert metrics[name] == sum(selected) / len(selected)
        else:
            assert name not in metrics


def test_worker_accepts_tensor_only_model_without_domain_metrics():
    loss = torch.tensor(2.0, requires_grad=True)
    worker = SimpleNamespace(
        model=lambda **kwargs: loss,
        amp_context=contextlib.nullcontext(),
        _released_cuda_cache_before_first_forward=True,
    )
    _, metrics = module.FSDPVlaSftWorker.get_train_model_output(
        worker, {"domain": [0, 1]}
    )
    assert metrics == {"loss": 2.0}


@pytest.mark.parametrize("offset", [0, 3, 10])
def test_worker_data_state_restores_epoch_and_consumed_offset(
    monkeypatch, tmp_path, offset
):
    monkeypatch.setattr(module, "get_rng_state", torch.get_rng_state)
    monkeypatch.setattr(module, "set_rng_state", torch.set_rng_state)
    original = _loader()
    original.set_epoch(2)
    iterator = iter(original)
    for _ in range(offset):
        next(iterator)
    expected = list(iterator)
    worker = SimpleNamespace(
        data_loader=original, _data_epoch=2, _data_iter_offset=offset
    )
    module.FSDPVlaSftWorker._save_openpi_data_state(worker, str(tmp_path))
    saved = json.loads((tmp_path / "data_state.json").read_text())
    assert saved["data_iter_offset"] == offset
    resumed = SimpleNamespace(data_loader=_loader())
    rng_before = torch.get_rng_state()
    module.FSDPVlaSftWorker._load_openpi_data_state(resumed, str(tmp_path))
    torch.testing.assert_close(torch.get_rng_state(), rng_before)
    actual = list(resumed.data_iter)
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a["actions"], b["actions"])
        torch.testing.assert_close(a["domain"], b["domain"])
        torch.testing.assert_close(a["tactile_contact"], b["tactile_contact"])


def test_resume_rejects_changed_seed_and_accepts_rank_independent_signature(tmp_path):
    worker = SimpleNamespace(
        data_loader=_loader(rank=0), _data_epoch=0, _data_iter_offset=2
    )
    module.FSDPVlaSftWorker._save_openpi_data_state(worker, str(tmp_path))
    assert _loader(rank=0).resume_signature() == _loader(rank=1).resume_signature()
    changed = SimpleNamespace(data_loader=_loader(seed=7))
    with pytest.raises(ValueError, match="stream differs"):
        module.FSDPVlaSftWorker._load_openpi_data_state(changed, str(tmp_path))


def test_resume_keeps_legacy_same_fps_and_rejects_changed_temporal_mode(tmp_path):
    original = _loader()
    assert "temporal_mode" not in original.resume_signature()["datasets"]
    worker = SimpleNamespace(data_loader=original, _data_epoch=0, _data_iter_offset=2)
    module.FSDPVlaSftWorker._save_openpi_data_state(worker, str(tmp_path))
    changed = _loader()
    changed._cotrain = dataclasses.replace(
        changed._cotrain, temporal_mode="native_frames"
    )
    assert changed.resume_signature()["datasets"]["temporal_mode"] == "native_frames"
    with pytest.raises(ValueError, match="stream differs"):
        module.FSDPVlaSftWorker._load_openpi_data_state(
            SimpleNamespace(data_loader=changed), str(tmp_path)
        )
