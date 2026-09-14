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

"""SFT must tokenize with the same state-input setting as the actor."""

import dataclasses
import logging
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from rlinf.workers.sft import fsdp_vla_sft_worker as worker_module


@dataclasses.dataclass(frozen=True)
class _Model:
    discrete_state_input: bool


@dataclasses.dataclass(frozen=True)
class _Config:
    model: _Model


@pytest.mark.parametrize("default", [True, False])
@pytest.mark.parametrize("override", [True, False, None])
def test_sft_state_override_reaches_loader_without_mutating_registry(
    monkeypatch, default, override
):
    import openpi.training.data_loader as loader_module

    from rlinf.models.embodiment.openpi import dataconfig

    registered = _Config(_Model(default))
    received = []
    loader = SimpleNamespace(data_config=lambda: "data")
    monkeypatch.setattr(dataconfig, "get_openpi_config", lambda *a, **kw: registered)
    monkeypatch.setattr(worker_module, "resolve_lerobot_repo_id", lambda _: "synthetic")
    monkeypatch.setattr(worker_module, "_OpenPiTactileDataLoader", lambda value: value)

    def create(config, **kwargs):
        received.append(config)
        return loader

    monkeypatch.setattr(loader_module, "create_data_loader", create)
    openpi = {"config_name": "synthetic"}
    if override is not None:
        openpi["discrete_state_input"] = override
    worker = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "actor": {
                    "micro_batch_size": 1,
                    "model": {
                        "model_type": "openpi",
                        "model_path": "unused",
                        "openpi": openpi,
                    },
                }
            }
        ),
        _world_size=1,
        _logger=logging.getLogger(__name__),
    )
    result = worker_module.FSDPVlaSftWorker.build_dataloader(worker, [])
    assert result == (loader, "data")
    assert received[0].model.discrete_state_input is (
        default if override is None else override
    )
    assert registered.model.discrete_state_input is default
    assert received[0] is not registered
    assert received[0].model is not registered.model
