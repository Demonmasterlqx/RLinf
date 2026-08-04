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
import itertools
import json
import os
from typing import Any

import torch
from omegaconf import DictConfig
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.config import SupportedModel
from rlinf.data.lerobot_paths import resolve_lerobot_repo_id
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.utils import get_rng_state, set_rng_state
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


class _OpenPiTactileDataLoader:
    """Expose finite OpenPI epochs and preserve RLinf-local tactile fields."""

    def __init__(self, delegate):
        self._delegate = delegate
        self._data_loader = delegate._data_loader
        self._pytorch_data_loader = self._resolve_pytorch_data_loader()

    def _resolve_pytorch_data_loader(self):
        return getattr(self._data_loader, "_data_loader", None) or getattr(
            self._data_loader, "torch_loader", None
        )

    @property
    def sampler(self):
        if self._pytorch_data_loader is None:
            return None
        return getattr(self._pytorch_data_loader, "sampler", None)

    @property
    def dataset(self):
        if self._pytorch_data_loader is None:
            return None
        return getattr(self._pytorch_data_loader, "dataset", None)

    def __len__(self):
        if self._pytorch_data_loader is None:
            raise TypeError("The wrapped OpenPI loader does not expose __len__.")
        return len(self._pytorch_data_loader)

    def set_epoch(self, epoch: int) -> None:
        sampler = self.sampler
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        dataset = self.dataset
        if dataset is not None and hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

    def data_config(self):
        return self._delegate.data_config()

    def __iter__(self):
        from openpi.models import model as _model

        batches = iter(self._data_loader)
        if self._pytorch_data_loader is not None:
            batches = itertools.islice(batches, len(self))
        for batch in batches:
            observation = _model.Observation.from_dict(batch)
            payload = {
                "observation": observation,
                "actions": batch["actions"],
            }
            if "tactile_prefix" in batch:
                # Keep tactile as an explicit payload field. Extra attributes on
                # OpenPI's Observation do not survive every FSDP argument path.
                payload["tactile_prefix"] = batch["tactile_prefix"]
            yield payload


class FSDPVlaSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def build_dataloader(self, data_paths: Any, eval_dataset: bool = False):
        if (
            SupportedModel(self.cfg.actor.model.model_type)
            == SupportedModel.OPENPI_PYTORCH
        ):
            from rlinf.data.datasets.openpi_pytorch import (
                build_openpi_pytorch_sft_dataloader,
            )

            return build_openpi_pytorch_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.OPENPI]:
            repo_id = resolve_lerobot_repo_id(data_paths)
            if repo_id is None:
                raise ValueError(
                    "OpenPI SFT requires data.train_data_paths to be set to a local "
                    "dataset path or LeRobot repo id."
                )

            import openpi.training.data_loader as openpi_data_loader

            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

            config = get_openpi_config(
                self.cfg.actor.model.openpi.config_name,
                model_path=self.cfg.actor.model.model_path,
                batch_size=self.cfg.actor.micro_batch_size * self._world_size,
                repo_id=repo_id,
                data_kwargs=getattr(self.cfg.actor.model, "openpi_data", None),
            )
            data_loader = openpi_data_loader.create_data_loader(
                config, framework="pytorch", shuffle=True
            )
            data_loader = _OpenPiTactileDataLoader(data_loader)
            return data_loader, data_loader.data_config()
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.LINGBOTVLA
        ]:
            from rlinf.models.embodiment.lingbotvla.sft_builder import (
                build_lingbot_sft_dataloader,
            )

            return build_lingbot_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths
            )
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.DREAMZERO
        ]:
            from rlinf.data.datasets.dreamzero import (
                build_dreamzero_sft_dataloader,
            )

            return build_dreamzero_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def get_eval_model_output(self, batch: dict[str, Any]):
        # now the eval is not supported for embodied sft
        raise NotImplementedError("eval is not supported for embodied sft right now.")

    def get_train_model_output(self, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        with self.amp_context:
            output = self.model(forward_type=ForwardType.SFT, data=batch)

        if isinstance(output, torch.Tensor):
            loss = output
        else:
            loss = output["loss"]

        step_metrics = {"loss": loss.detach().item()}
        if isinstance(output, dict):
            for key, value in output.items():
                if key == "loss":
                    continue
                if torch.is_tensor(value):
                    if value.numel() == 1:
                        step_metrics[key] = value.detach().item()
                elif isinstance(value, (float, int)):
                    step_metrics[key] = value
        return loss, step_metrics

    def _save_openpi_data_state(self, save_path: str) -> None:
        with open(
            os.path.join(save_path, "data_state.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                {
                    "data_epoch": self._data_epoch,
                    "data_iter_offset": self._data_iter_offset,
                },
                handle,
            )

    def _load_openpi_data_state(self, load_path: str) -> None:
        state_path = os.path.join(load_path, "data_state.json")
        if not os.path.isfile(state_path):
            raise FileNotFoundError(
                f"OpenPI SFT checkpoint is missing data_state.json: {state_path}"
            )
        with open(state_path, encoding="utf-8") as handle:
            state = json.load(handle)
        self._data_epoch = int(state["data_epoch"])
        self._data_iter_offset = int(state["data_iter_offset"])
        if self._data_epoch < 0 or self._data_iter_offset < 0:
            raise ValueError(
                f"OpenPI data checkpoint has negative epoch or offset: {state}."
            )
        if self._data_iter_offset > len(self.data_loader):
            raise ValueError(
                "OpenPI data checkpoint offset exceeds the epoch length: "
                f"{self._data_iter_offset} > {len(self.data_loader)}."
            )

        # Loading the FSDP checkpoint restores the model RNG. Advancing the
        # loader to its saved cursor must not consume that model RNG stream.
        restored_rng_state = get_rng_state()
        self.data_loader.set_epoch(self._data_epoch)
        self.data_iter = iter(self.data_loader)
        for _ in range(self._data_iter_offset):
            try:
                next(self.data_iter)
            except StopIteration as error:
                raise RuntimeError(
                    "OpenPI data checkpoint offset could not be restored."
                ) from error
        set_rng_state(restored_rng_state)

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        super().save_checkpoint(save_path, step)

        if isinstance(self.data_loader, StatefulDataLoader):
            state = self.data_loader.state_dict()

            all_states = [None] * self._world_size
            torch.distributed.all_gather_object(all_states, state)

            if self._rank == 0:
                torch.save(all_states, os.path.join(save_path, "data.pt"))

            torch.distributed.barrier()

            rng_state = get_rng_state()
            all_rng_states = [None] * self._world_size
            torch.distributed.all_gather_object(all_rng_states, rng_state)
            if self._rank == 0:
                torch.save(all_rng_states, os.path.join(save_path, "rng.pt"))

            torch.distributed.barrier()
        elif isinstance(self.data_loader, _OpenPiTactileDataLoader):
            if self._rank == 0:
                self._save_openpi_data_state(save_path)
            torch.distributed.barrier()

    def load_checkpoint(self, load_path: str) -> None:
        super().load_checkpoint(load_path)

        if isinstance(self.data_loader, StatefulDataLoader):
            all_states = torch.load(
                os.path.join(load_path, "data.pt"), weights_only=False
            )
            state = all_states[self._rank]
            self.data_loader.load_state_dict(state)
            self.data_iter = iter(self.data_loader)

            rng_path = os.path.join(load_path, "rng.pt")
            if os.path.exists(rng_path):
                all_rng_states = torch.load(rng_path, weights_only=False)
                set_rng_state(all_rng_states[self._rank])

            torch.distributed.barrier()
        elif isinstance(self.data_loader, _OpenPiTactileDataLoader):
            self._load_openpi_data_state(load_path)
            torch.distributed.barrier()

    def get_max_steps_per_epoch(self):
        if self.data_loader is None:
            return 0
        model_type = SupportedModel(self.cfg.actor.model.model_type)
        if model_type == SupportedModel.OPENPI_PYTORCH:
            return max(1, len(self.data_loader) // self.gradient_accumulation)
        if model_type == SupportedModel.OPENPI:
            num_batches = len(self.data_loader)
            return max(1, num_batches // self.gradient_accumulation)
        return super().get_max_steps_per_epoch()

    @staticmethod
    def _openpi_pytorch_dataloader(openpi_dataloader: Any):
        """Unwrap OpenPI `DataLoaderImpl` to the inner PyTorch DataLoader.

        OpenPI torch path:
          DataLoaderImpl._data_loader -> TorchDataLoader
          TorchDataLoader._data_loader / .torch_loader -> torch.utils.data.DataLoader

        """
        torch_data_loader = getattr(openpi_dataloader, "_data_loader", None)
        pytorch_dl = getattr(torch_data_loader, "_data_loader", None) or getattr(
            torch_data_loader, "torch_loader", None
        )
        if pytorch_dl is None:
            raise TypeError(
                "OpenPI dataloader does not expose an inner torch DataLoader; cannot infer steps per epoch from len()."
            )
        return pytorch_dl
