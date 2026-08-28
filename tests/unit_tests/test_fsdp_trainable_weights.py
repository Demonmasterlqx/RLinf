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

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

import rlinf.hybrid_engines.fsdp.fsdp_model_manager as fsdp_model_manager
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.hybrid_engines.fsdp.model_weight_ema import ModelWeightEMA


class _Logger:
    def info(self, *_args, **_kwargs):
        pass


def test_model_weight_ema_updates_in_fp32_and_temporarily_applies_weights():
    parameter = nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    ema = ModelWeightEMA([parameter], 0.99, parameter_names=["weight"])

    parameter.data.fill_(3.0)
    ema.update()

    assert ema.shadows[0].dtype == torch.float32
    torch.testing.assert_close(ema.shadows[0], torch.tensor([1.02]))
    assert ema.num_updates == 1

    with ema.apply_to_parameters():
        torch.testing.assert_close(
            parameter.float(), torch.tensor([1.02]), atol=0.01, rtol=0
        )
    torch.testing.assert_close(parameter.float(), torch.tensor([3.0]))


def test_model_weight_ema_cpu_shadows_preserve_state_and_live_weights():
    parameter = nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.bfloat16))
    ema = ModelWeightEMA(
        [parameter],
        0.5,
        parameter_names=["weight"],
        shadow_device="cpu",
    )

    assert ema.shadow_device == torch.device("cpu")
    assert ema.shadows[0].device.type == "cpu"

    parameter.data.copy_(torch.tensor([3.0, 6.0], dtype=torch.bfloat16))
    ema.update()
    torch.testing.assert_close(ema.shadows[0], torch.tensor([2.0, 4.0]))

    state = ema.state_dict()
    restored_parameter = nn.Parameter(torch.tensor([7.0, 8.0]))
    restored = ModelWeightEMA(
        [restored_parameter],
        0.5,
        parameter_names=["weight"],
        shadow_device="cpu",
    )
    restored.load_state_dict(state)

    assert restored.num_updates == 1
    torch.testing.assert_close(restored.shadows[0], torch.tensor([2.0, 4.0]))
    with restored.apply_to_parameters():
        torch.testing.assert_close(restored_parameter, torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(restored_parameter, torch.tensor([7.0, 8.0]))


def test_model_weight_ema_state_restore_is_strict():
    source_parameter = nn.Parameter(torch.tensor([1.0, 2.0]))
    source = ModelWeightEMA([source_parameter], 0.9, parameter_names=["flat_parameter"])
    source_parameter.data.add_(2.0)
    source.update()

    target_parameter = nn.Parameter(torch.zeros(2))
    target = ModelWeightEMA([target_parameter], 0.9, parameter_names=["flat_parameter"])
    target.load_state_dict(source.state_dict())

    torch.testing.assert_close(target.shadows[0], source.shadows[0])
    assert target.num_updates == 1

    wrong_decay = ModelWeightEMA(
        [target_parameter], 0.99, parameter_names=["flat_parameter"]
    )
    with pytest.raises(ValueError, match="decay mismatch"):
        wrong_decay.load_state_dict(source.state_dict())

    wrong_name = ModelWeightEMA(
        [target_parameter], 0.9, parameter_names=["different_parameter"]
    )
    with pytest.raises(ValueError, match="topology"):
        wrong_name.load_state_dict(source.state_dict())


def test_fsdp_model_manager_initializes_cpu_ema_and_releases_cuda_cache(monkeypatch):
    manager = FSDPModelManager.__new__(FSDPModelManager)
    manager.model = nn.Linear(2, 1)
    manager.optimizer = torch.optim.SGD(manager.model.parameters(), lr=0.1)
    manager._cfg = OmegaConf.create({"model_weight_ema_decay": 0.99})
    manager.critic_warmup_steps = 0
    manager._logger = _Logger()
    empty_cache_calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empty_cache_calls.append(1))

    manager._initialize_model_weight_ema()

    assert manager._model_weight_ema is not None
    assert all(
        shadow.device.type == "cpu" for shadow in manager._model_weight_ema.shadows
    )
    assert empty_cache_calls == [1]


def test_normalize_fsdp_param_name_strips_wrappers():
    assert (
        FSDPModelManager._normalize_fsdp_param_name(
            "module._fsdp_wrapped_module.foo._fsdp_wrapped_module.weight"
        )
        == "foo.weight"
    )


def test_save_trainable_model_weights_fails_when_no_trainable_params(
    monkeypatch, tmp_path
):
    model = nn.Linear(2, 2)
    for param in model.parameters():
        param.requires_grad = False

    manager = FSDPModelManager.__new__(FSDPModelManager)
    manager.model = model
    manager._cfg = SimpleNamespace(fsdp_config={"save_trainable_model_weights": True})
    manager._logger = _Logger()

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    with pytest.raises(RuntimeError, match="no trainable parameters"):
        manager._save_trainable_model_weights(str(tmp_path), step=0)


class _FlatParamWrappedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._flat_param = nn.Parameter(torch.ones(4))


def test_save_trainable_model_weights_uses_pre_wrap_trainable_names_for_fsdp_flat_params(
    monkeypatch, tmp_path
):
    model = _FlatParamWrappedModel()

    manager = FSDPModelManager.__new__(FSDPModelManager)
    manager.model = model
    manager._cfg = SimpleNamespace(fsdp_config={"save_trainable_model_weights": True})
    manager._logger = _Logger()
    manager.trainable_param_names = [
        "paligemma.adapter.lora_A.weight",
        "gemma_expert.adapter.lora_B.weight",
    ]

    def fake_full_state_dict(*_args, **_kwargs):
        return {
            "_fsdp_wrapped_module.paligemma.adapter.lora_A.weight": torch.tensor(
                [[1.0, 2.0]]
            ),
            "_fsdp_wrapped_module.gemma_expert.adapter.lora_B.weight": torch.tensor(
                [[3.0], [4.0]]
            ),
            "_fsdp_wrapped_module.frozen.weight": torch.tensor([5.0]),
            "_fsdp_wrapped_module.persistent_buffer": torch.tensor([6.0]),
        }

    manager.get_model_state_dict = fake_full_state_dict

    monkeypatch.setattr(fsdp_model_manager, "FSDP", _FlatParamWrappedModel)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    manager._save_trainable_model_weights(str(tmp_path), step=7)

    checkpoint = torch.load(
        tmp_path / "model_state_dict" / "trainable_weights.pt",
        map_location="cpu",
        weights_only=False,
    )

    assert checkpoint["metadata"]["step"] == 7
    assert checkpoint["metadata"]["parameter_count"] == 2
    assert list(checkpoint["model"].keys()) == [
        "paligemma.adapter.lora_A.weight",
        "gemma_expert.adapter.lora_B.weight",
    ]
    torch.testing.assert_close(
        checkpoint["model"]["paligemma.adapter.lora_A.weight"],
        torch.tensor([[1.0, 2.0]]),
    )
    torch.testing.assert_close(
        checkpoint["model"]["gemma_expert.adapter.lora_B.weight"],
        torch.tensor([[3.0], [4.0]]),
    )


@pytest.mark.parametrize(("step", "is_final"), [(40, False), (50, True)])
def test_save_trainable_model_weights_merges_configured_provenance(
    monkeypatch, tmp_path, step, is_final
):
    manager = FSDPModelManager.__new__(FSDPModelManager)
    manager.model = nn.Linear(2, 1)
    manager._cfg = SimpleNamespace(
        fsdp_config={
            "save_trainable_model_weights": True,
            "trainable_checkpoint_metadata": {
                "method": "pirl",
                "task_id": 5,
                "training_config": (
                    "isaaclab_pi0_peft_lora_tacfield_tabero_task5_firm_8gpu_50step"
                ),
                "target_global_step": 50,
                "global_step": -1,
                "is_final": False,
            },
        }
    )
    manager._logger = _Logger()

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    manager._save_trainable_model_weights(str(tmp_path), step=step)

    checkpoint = torch.load(
        tmp_path / "model_state_dict" / "trainable_weights.pt",
        map_location="cpu",
        weights_only=True,
    )
    metadata = checkpoint["metadata"]
    assert metadata["method"] == "pirl"
    assert metadata["task_id"] == 5
    assert metadata["training_config"].endswith("task5_firm_8gpu_50step")
    assert metadata["target_global_step"] == 50
    assert metadata["step"] == step
    assert metadata["global_step"] == step
    assert metadata["is_final"] is is_final


def test_save_trainable_model_weights_serializes_omegaconf_metadata_safely(
    monkeypatch, tmp_path
):
    manager = FSDPModelManager.__new__(FSDPModelManager)
    manager.model = nn.Linear(2, 1)
    manager._cfg = SimpleNamespace(
        fsdp_config=OmegaConf.create(
            {
                "save_trainable_model_weights": True,
                "trainable_checkpoint_metadata": {
                    "target_global_step": 1,
                    "excluded_tactile_inputs": [
                        "tactile_image",
                        "tactile_marker_motion",
                    ],
                },
            }
        )
    )
    manager._logger = _Logger()

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    manager._save_trainable_model_weights(str(tmp_path), step=1)

    checkpoint = torch.load(
        tmp_path / "model_state_dict" / "trainable_weights.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["metadata"]["excluded_tactile_inputs"] == [
        "tactile_image",
        "tactile_marker_motion",
    ]


def test_save_trainable_model_weights_exports_ema_and_restores_live_weights(
    monkeypatch, tmp_path
):
    model = nn.Linear(2, 1, bias=False)
    model.weight.data.fill_(1.0)
    ema = ModelWeightEMA(model.parameters(), 0.5, parameter_names=["weight"])
    model.weight.data.fill_(3.0)
    ema.update()

    manager = FSDPModelManager.__new__(FSDPModelManager)
    manager.model = model
    manager.trainable_param_names = ["weight"]
    manager._model_weight_ema = ema
    manager._cfg = SimpleNamespace(fsdp_config={"save_trainable_model_weights": True})
    manager._logger = _Logger()

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    manager._save_trainable_model_weights(str(tmp_path), step=4)

    checkpoint = torch.load(
        tmp_path / "model_state_dict" / "trainable_weights.pt",
        map_location="cpu",
        weights_only=True,
    )
    torch.testing.assert_close(checkpoint["model"]["weight"], torch.full((1, 2), 2.0))
    torch.testing.assert_close(model.weight, torch.full((1, 2), 3.0))
    assert checkpoint["metadata"]["weight_variant"] == "ema"
    assert checkpoint["metadata"]["model_weight_ema_decay"] == 0.5
    assert checkpoint["metadata"]["model_weight_ema_num_updates"] == 1


def test_model_weight_ema_rank_checkpoint_round_trip(monkeypatch, tmp_path):
    parameter = nn.Parameter(torch.tensor([2.0]))
    source_ema = ModelWeightEMA([parameter], 0.99, parameter_names=["flat_parameter"])
    parameter.data.fill_(4.0)
    source_ema.update()

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    source_manager = FSDPModelManager.__new__(FSDPModelManager)
    source_manager._model_weight_ema = source_ema
    source_manager._save_model_weight_ema(str(tmp_path))

    target_parameter = nn.Parameter(torch.tensor([0.0]))
    target_manager = FSDPModelManager.__new__(FSDPModelManager)
    target_manager._model_weight_ema = ModelWeightEMA(
        [target_parameter], 0.99, parameter_names=["flat_parameter"]
    )
    target_manager._load_model_weight_ema(str(tmp_path))

    torch.testing.assert_close(
        target_manager._model_weight_ema.shadows[0], source_ema.shadows[0]
    )
    assert target_manager._model_weight_ema.num_updates == 1
