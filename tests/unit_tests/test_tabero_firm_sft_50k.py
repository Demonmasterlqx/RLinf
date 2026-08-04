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

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DistributedSampler

from rlinf.hybrid_engines.fsdp.utils import get_lr_scheduler
from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0Config,
    _reduce_sft_action_loss,
)
from rlinf.runners.sft_runner import SFTRunner
from rlinf.workers.sft.fsdp_vla_sft_worker import (
    FSDPVlaSftWorker,
    _OpenPiTactileDataLoader,
)
from toolkits.checkpoint.audit_tabero_firm_sft_smoke import audit

CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/sft/config/tabero_firm_sft_full_lora_tacfield_fp32_50k.yaml"
)


def _raw_batch(value: float) -> dict:
    return {
        "image": {
            "base_0_rgb": torch.zeros(1, 224, 224, 3, dtype=torch.uint8),
            "left_wrist_0_rgb": torch.zeros(1, 224, 224, 3, dtype=torch.uint8),
        },
        "image_mask": {
            "base_0_rgb": torch.ones(1, dtype=torch.bool),
            "left_wrist_0_rgb": torch.ones(1, dtype=torch.bool),
        },
        "state": torch.zeros(1, 32),
        "tokenized_prompt": torch.ones(1, 8, dtype=torch.long),
        "tokenized_prompt_mask": torch.ones(1, 8, dtype=torch.bool),
        "tactile_prefix": torch.full((1, 9, 396), value),
        "actions": torch.full((1, 50, 32), value),
    }


class _TorchLoader:
    def __init__(self, batches, sampler=None):
        self._batches = batches
        self.sampler = sampler
        self.dataset = None

    def __iter__(self):
        yield from self._batches

    def __len__(self):
        return len(self._batches)


class _OpenPiTorchLoader:
    def __init__(self, batches, sampler=None):
        self._data_loader = _TorchLoader(batches, sampler=sampler)

    def __iter__(self):
        while True:
            yield from self._data_loader


class _Delegate:
    def __init__(self, batches, sampler=None):
        self._data_loader = _OpenPiTorchLoader(batches, sampler=sampler)

    def data_config(self):
        return "tabero-firm"


def test_weighted_full_loss_matches_reference_action_layout():
    elementwise = torch.empty(2, 50, 32)
    elementwise[..., :7] = 1.0
    elementwise[..., 7:13] = 4.0
    elementwise[..., 13:] = 9.0
    config = OpenPi0Config(
        action_dim=32,
        effective_action_dim=13,
        tactile_type="expert_his_c_fut",
        tactile_dim=6,
        tactile_loss_weight=0.1,
        padding_loss_weight=1.0,
        expert_his_c_fut_loss_mode="weighted_full",
    )

    output = _reduce_sft_action_loss(elementwise, config)

    expected = (7 * 1.0 + 6 * 0.1 * 4.0 + 19 * 9.0) / 32
    torch.testing.assert_close(output["loss"], torch.tensor(expected))
    torch.testing.assert_close(output["action_loss"], torch.tensor(1.0))
    torch.testing.assert_close(output["tactile_loss"], torch.tensor(4.0))
    torch.testing.assert_close(output["padding_loss"], torch.tensor(9.0))


def test_openpi_schedule_reaches_floor_at_independent_decay_step():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=2.5e-5)
    scheduler = get_lr_scheduler(
        "openpi_cosine",
        optimizer,
        num_warmup_steps=1000,
        num_training_steps=50000,
        num_decay_steps=30000,
        min_lr=2.5e-6,
    )
    multiplier = scheduler.lr_lambdas[0]

    assert multiplier(0) * 2.5e-5 == pytest.approx(2.5e-5 / 1001)
    assert multiplier(1000) * 2.5e-5 == pytest.approx(2.5e-5)
    assert multiplier(30000) * 2.5e-5 == pytest.approx(2.5e-6)
    assert multiplier(50000) * 2.5e-5 == pytest.approx(2.5e-6)


def test_openpi_loader_is_finite_and_forwards_sampler_epoch():
    sampler = DistributedSampler(
        list(range(32)), num_replicas=1, rank=0, shuffle=True, seed=0
    )
    loader = _OpenPiTactileDataLoader(
        _Delegate([_raw_batch(0.0), _raw_batch(1.0)], sampler=sampler)
    )

    first_epoch = list(iter(sampler))
    loader.set_epoch(1)
    second_epoch = list(iter(sampler))
    payloads = list(loader)

    assert len(loader) == 2
    assert len(payloads) == 2
    assert first_epoch != second_epoch
    assert sampler.epoch == 1
    assert payloads[1]["actions"][0, 0, 0].item() == 1.0


def test_openpi_data_state_restores_epoch_offset_and_model_rng(tmp_path):
    sampler = DistributedSampler(
        list(range(32)), num_replicas=1, rank=0, shuffle=True, seed=0
    )
    loader = _OpenPiTactileDataLoader(
        _Delegate(
            [_raw_batch(0.0), _raw_batch(1.0), _raw_batch(2.0)],
            sampler=sampler,
        )
    )
    worker = object.__new__(FSDPVlaSftWorker)
    worker.data_loader = loader
    worker._data_epoch = 3
    worker._data_iter_offset = 2
    worker._save_openpi_data_state(str(tmp_path))

    worker._data_epoch = 0
    worker._data_iter_offset = 0
    torch.manual_seed(123)
    rng_before = torch.get_rng_state().clone()
    worker._load_openpi_data_state(str(tmp_path))
    rng_after = torch.get_rng_state()

    assert worker._data_epoch == 3
    assert worker._data_iter_offset == 2
    assert sampler.epoch == 3
    torch.testing.assert_close(rng_before, rng_after)
    next_payload = next(worker.data_iter)
    assert next_payload["actions"][0, 0, 0].item() == 2.0


def test_formal_config_encodes_50k_fp32_contract(monkeypatch, tmp_path):
    monkeypatch.setenv("EMBODIED_PATH", str(CONFIG_PATH.parents[1]))
    monkeypatch.setenv("TABERO_FIRM_50K_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("TABERO_FIRM_50K_RUN_NAME", "formal-test")
    monkeypatch.setenv("TABERO_FIRM_50K_CHECKPOINT_ROOT", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("TABERO_FIRM_NORM_STATS", str(tmp_path / "norm_stats.json"))
    cfg = OmegaConf.load(CONFIG_PATH)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    assert cfg.runner.max_steps == 50000
    assert cfg.runner.save_interval == 5000
    assert cfg.cluster.component_placement.actor == "0-1,3-7"
    assert cfg.actor.micro_batch_size == 1
    assert cfg.actor.global_batch_size == 28
    assert cfg.actor.model.precision == "fp32"
    assert cfg.actor.model.lora_target == "both"
    assert cfg.actor.model.extra_trainable_modules == ["tactile_prefix_encoder"]
    assert cfg.actor.model.openpi.effective_action_dim == 13
    assert cfg.actor.model.openpi.tactile_dim == 6
    assert cfg.actor.model.openpi.expert_his_c_fut_loss_mode == "weighted_full"
    assert cfg.actor.optim.lr_warmup_steps == 1000
    assert cfg.actor.optim.lr_decay_steps == 30000
    assert cfg.actor.optim.min_lr == pytest.approx(2.5e-6)
    assert cfg.actor.fsdp_config.checkpoint_format == "dcp"
    assert cfg.actor.fsdp_config.mixed_precision.param_dtype == "fp32"
    assert cfg.actor.fsdp_config.amp_autocast.enabled is False
    assert cfg.actor.fsdp_config.grad_scaler.enabled is False


def test_sft_runner_uses_explicit_checkpoint_root(tmp_path):
    class _Handle:
        def wait(self):
            return None

    class _Actor:
        def __init__(self):
            self.calls = []

        def save_checkpoint(self, path, step):
            self.calls.append((path, step))
            return _Handle()

    runner = object.__new__(SFTRunner)
    runner.cfg = OmegaConf.create(
        {
            "runner": {
                "checkpoint_root": str(tmp_path / "checkpoints"),
                "logger": {"log_path": "unused", "experiment_name": "unused"},
            }
        }
    )
    runner.actor = _Actor()
    runner.global_step = 5000
    runner.early_stop = None

    runner._save_checkpoint()

    assert runner.actor.calls == [
        (str(tmp_path / "checkpoints/global_step_5000/actor"), 5000)
    ]


def test_preflight_sidecar_audit_accepts_resume_step_pair(tmp_path):
    paths = [tmp_path / "step2.pt", tmp_path / "step4.pt"]
    names = (
        "paligemma_with_expert.paligemma.layer.lora_A.weight",
        "paligemma_with_expert.gemma_expert.model.layer.lora_A.weight",
        "tactile_prefix_encoder.weight",
    )
    for index, (path, step) in enumerate(zip(paths, (2, 4), strict=True)):
        torch.save(
            {
                "model": {
                    name: torch.full((1,), float(index), dtype=torch.float32)
                    for name in names
                },
                "metadata": {
                    "global_step": step,
                    "is_final": step == 4,
                    "parameter_count": len(names),
                },
            },
            path,
        )

    result = audit(paths[0], paths[1], expected_steps=(2, 4))

    assert result["tensor_count"] == 3
    assert result["groups"]["vlm_lora"]["changed_tensor_count"] == 1
    assert result["groups"]["action_expert_lora"]["changed_tensor_count"] == 1
    assert result["groups"]["tcn"]["changed_tensor_count"] == 1
