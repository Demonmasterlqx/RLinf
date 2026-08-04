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

import math
import random
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from openpi import transforms as _transforms
from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
from torch.utils._pytree import tree_map

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.explore_noise_net import ExploreNoiseNet
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.models.embodiment.openpi.tactile_encoder import TactileTCNEncoder
from rlinf.utils.logging import get_logger
from rlinf.utils.nested_dict_process import copy_dict_tensor
from rlinf.utils.pytree import register_pytree_dataclasses


def _to_numpy(x):
    return np.asarray(x.detach().cpu()) if torch.is_tensor(x) else x


def _uses_expert_future_tactile(config: Any) -> bool:
    """Return whether actions contain control followed by future force slots."""
    tactile_type = str(getattr(config, "tactile_type", "no")).lower()
    return tactile_type.endswith("expert_his_c_fut")


def _reduce_sft_action_loss(
    elementwise_loss: torch.Tensor, config: Any
) -> dict[str, torch.Tensor]:
    """Reduce PI0 elementwise flow loss using the configured action layout.

    Tabero ``expert_his_c_fut`` actions contain control, force, and optional
    padding dimensions. This mirrors the reference OpenPI weighted loss while
    retaining scalar components for training diagnostics.
    """
    if not _uses_expert_future_tactile(config):
        return {"loss": elementwise_loss.mean()}

    action_dim = int(getattr(config, "action_dim", elementwise_loss.shape[-1]))
    effective_action_dim = int(
        getattr(config, "effective_action_dim", action_dim) or action_dim
    )
    tactile_dim = int(getattr(config, "tactile_dim", 0))
    control_dim = effective_action_dim - tactile_dim
    if elementwise_loss.shape[-1] != action_dim:
        raise ValueError(
            "PI0 SFT loss width does not match action_dim: "
            f"{elementwise_loss.shape[-1]} != {action_dim}."
        )
    if not 0 < control_dim < effective_action_dim <= action_dim:
        raise ValueError(
            "expert_his_c_fut requires 0 < control_dim < "
            "effective_action_dim <= action_dim; got "
            f"control_dim={control_dim}, effective_action_dim="
            f"{effective_action_dim}, action_dim={action_dim}."
        )

    tactile_slice = slice(control_dim, effective_action_dim)
    action_loss = elementwise_loss[..., :control_dim].mean()
    tactile_loss = elementwise_loss[..., tactile_slice].mean()
    components = {
        "action_loss": action_loss,
        "tactile_loss": tactile_loss,
    }

    padding_loss = None
    if effective_action_dim < action_dim:
        padding_loss = elementwise_loss[..., effective_action_dim:].mean()
        components["padding_loss"] = padding_loss

    tactile_weight = float(getattr(config, "tactile_loss_weight", 0.1))
    padding_weight = float(getattr(config, "padding_loss_weight", 0.0))
    loss_mode = str(getattr(config, "expert_his_c_fut_loss_mode", "weighted_full"))
    if loss_mode == "weighted_full":
        weights = torch.ones(
            action_dim,
            device=elementwise_loss.device,
            dtype=elementwise_loss.dtype,
        )
        weights[tactile_slice] = tactile_weight
        if effective_action_dim < action_dim:
            weights[effective_action_dim:] = padding_weight
        total_loss = (elementwise_loss * weights).mean()
    elif loss_mode == "split":
        total_loss = action_loss + tactile_weight * tactile_loss
        if padding_loss is not None:
            total_loss = total_loss + padding_weight * padding_loss
    else:
        raise ValueError(
            "Unsupported expert_his_c_fut_loss_mode "
            f"{loss_mode!r}; expected 'weighted_full' or 'split'."
        )

    return {"loss": total_loss, **components}


@dataclass(frozen=True)
class OpenPi0Config(Pi0Config):
    # config for rl
    config_name: str = "pi0_libero"  # pi0_libero, pi05_libero, pi0_maniskill, pi05_maniskill, pi0_metaworld, pi05_metaworld
    num_images_in_input: int = 2  # number of images in input
    noise_method: str = "flow_sde"  # flow_ode, flow_sde, flow_noise, flow_cps
    # noise config for flow-sde
    noise_level: float = 0.5
    noise_anneal: bool = False
    noise_params: list = field(
        default_factory=lambda: [0.7, 0.3, 400]
    )  # noise_start, noise_end, noise_anneal_steps
    # noise config for flow-noise
    noise_logvar_range: list = field(
        default_factory=lambda: [0.08, 0.16]
    )  # [min_std, max_std]
    # hyper-parameters
    action_chunk: int = 5  # action chunk
    action_env_dim: int = 7  # for environment action dim
    num_steps: int = 10  # denoise steps

    # Tabero / T2-VLA tactile-compatible parameters. The upstream OpenPI package
    # used by RLinf does not expose these fields, so RLinf carries them locally.
    tactile_type: str = "no"
    tactile_dim: int = 14
    tactile_dim_in: int | None = None
    tactile_history: int | None = None
    effective_action_dim: int | None = None
    tactile_loss_weight: float = 0.1
    padding_loss_weight: float = 1.0
    expert_his_c_fut_loss_mode: str = "weighted_full"
    tactile_encoder_type: str = "mlp"
    tactile_use_reference_frame: bool = False
    tactile_diff_from_reference: bool = True
    tactile_prefix_dim_in: int | None = None
    tactile_prefix_history: int | None = None
    tactile_prefix_encoder_type: str | None = None
    tactile_prefix_use_reference_frame: bool | None = None
    tactile_prefix_diff_from_reference: bool | None = None
    tactile_streams: tuple[str, ...] = field(default_factory=tuple)
    tactile_suffix_placement: str = "suffix"

    # training config
    train_expert_only: bool = False
    safe_get_logprob: bool = False
    joint_logprob: bool = False  # designed for flow-noise
    double_layer: bool = False  # designed for flow-sde without acceleration
    ignore_last: bool = False  # ignore the last action for noise injection
    # critic
    detach_critic_input: bool = False  # detach critic input with the action expert
    chunk_critic_input: bool = False  # use only the action chunk for critic estimation
    add_value_head: bool = False  # add value head for ppo
    value_after_vlm: bool = False  # value after vlm, pi05 mode
    value_vlm_mode: str = "mean_token"  # last_token, mean_token, first_token

    # ===== DSRL-specific parameters =====
    use_dsrl: bool = False  # Enable DSRL algorithm
    dsrl_state_dim: int = 8  # Raw state dimension for DSRL encoders
    dsrl_action_noise_dim: int = 32  # Noise dimension output by GaussianPolicy
    dsrl_num_q_heads: int = 10  # Number of Q-networks
    dsrl_agg_q: str = "mean"  # Q aggregation method: 'mean' | 'min'
    dsrl_image_latent_dim: int = 64  # Latent dim for lightweight image encoder
    dsrl_num_images: int = 1  # Number of ordered DSRL image views
    dsrl_state_latent_dim: int = 64  # Hidden dim for state encoder
    dsrl_use_tactile: bool = False  # Include TacField history in DSRL steering
    dsrl_tactile_latent_dim: int = 64  # Latent dim for each tactile encoder
    dsrl_hidden_dims: tuple = field(
        default_factory=lambda: (128, 128, 128)
    )  # Hidden dims for Q-head and GaussianPolicy

    # ===== NFT-specific parameters =====
    is_nft: bool = False

    # ===== RLT SFT parameters =====
    use_rlt: bool = False
    rlt_train_module_only: bool = False
    rlt_alpha: float = 1.0
    rlt_input_dim: int = 2048
    rlt_embed_dim: int = 2048
    rlt_num_rl_tokens: int = 1
    rlt_prefix_seq_len: int = 768
    rlt_num_layers: int = 2
    rlt_num_heads: int = 8
    rlt_mlp_ratio: float = 4.0
    rlt_image_only: bool = True
    rlt_use_mask: bool = False
    rlt_action_space: str = "environment"
    rlt_use_normalized_proprio: bool = False
    rlt_stage2_encoder_only: bool = False
    state_indices: list[int] | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.rlt_train_module_only and not self.use_rlt:
            raise ValueError("rlt_train_module_only=True requires use_rlt=True.")
        if self.tactile_dim_in is None:
            object.__setattr__(self, "tactile_dim_in", self.tactile_dim)
        if self.effective_action_dim is None:
            object.__setattr__(self, "effective_action_dim", self.action_dim)


class OpenPi0ForRLActionPrediction(PI0Pytorch, BasePolicy):
    """
    Pi0 model for reinforcement learning action prediction.
    """

    config: OpenPi0Config

    @property
    def _no_split_modules(self) -> list[str]:
        if self.config.train_expert_only:
            no_split_modules = [
                "GemmaDecoderLayer",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
            ]
        else:
            no_split_modules = [
                "GemmaMLP",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
            ]
        if self.config.noise_method == "flow_noise":
            no_split_modules.append("ExploreNoiseNet")
        if self.config.use_rlt:
            no_split_modules.append("RLTSelfAttentionLayer")
        return no_split_modules

    @property
    def _no_split_names(self) -> list[str]:
        return [
            "action_in_proj",
            "action_out_proj",
            "lm_head",
            # --pi0 only--
            "state_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
            # --pi05 only--
            "time_mlp_in",
            "time_mlp_out",
            # --Tabero tacfield--
            "tactile_prefix_encoder",
        ]

    def __init__(
        self,
        config: OpenPi0Config,
    ):
        # Override `sample_actions` to prevent parent class polymorphic call
        sample_actions_func = self.sample_actions
        super().__init__(config)
        self.sample_actions = sample_actions_func
        self._replace_projection_layers_for_config()
        self._init_tactile_prefix_encoder()
        self.logger = get_logger()
        self.global_step = 0
        # assert
        assert not (self.config.double_layer and self.config.joint_logprob), (
            "double_layer and joint_logprob can not be set at the same time"
        )

        # rl model init
        if self.config.value_after_vlm:
            proj_width = 2048
        else:
            proj_width = 1024
        # value head
        if self.config.add_value_head:
            if self.config.config_name in [
                "pi05_maniskill",
                "pi05_libero",
                "pi05_droid_polaris",
            ]:
                value_head_hidden_sizes = (1024, 512, 256)
            else:
                value_head_hidden_sizes = (512, 256, 128)
            value_head_activation = "relu"
            self.value_head = ValueHead(
                input_dim=proj_width,
                hidden_sizes=value_head_hidden_sizes,
                output_dim=1,
                activation=value_head_activation,
                bias_last=True,
            )
        self.use_vlm_value = getattr(self.config, "value_after_vlm", False) and getattr(
            self.config, "add_value_head", False
        )
        # noise head for flow-noise
        if self.config.noise_method == "flow_noise":
            self.noise_head = ExploreNoiseNet(
                in_dim=1024,
                out_dim=self.config.action_dim,
                hidden_dims=[128, 64],
                activation_type="tanh",
                noise_logvar_range=self.config.noise_logvar_range,
                noise_scheduler_type="learn",
            )

        if self.config.use_rlt:
            from rlinf.models.embodiment.modules.rlt_token_transformer import (
                RLTTokenTransformer,
            )

            self.rlt_module = RLTTokenTransformer(
                input_dim=self.config.rlt_input_dim,
                embed_dim=self.config.rlt_embed_dim,
                num_rl_tokens=self.config.rlt_num_rl_tokens,
                prefix_seq_len=self.config.rlt_prefix_seq_len,
                num_layers=self.config.rlt_num_layers,
                num_heads=self.config.rlt_num_heads,
                mlp_ratio=self.config.rlt_mlp_ratio,
            ).to(dtype=torch.bfloat16)

        # ===== DSRL components initialization =====
        if self.config.use_dsrl:
            self._init_dsrl_components()

        for name, module in self.named_modules():
            # Set _fsdp_wrap_name to the last part of the path (e.g., "model.action_in_proj" -> "action_in_proj")
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

        self.torch_compile_enabled = False

    def freeze_non_rlt_parameters(self):
        """Freeze the VLA backbone and keep only the Stage 1 RLT module trainable."""
        if not hasattr(self, "rlt_module"):
            raise ValueError("RLT-only training requires an initialized rlt_module.")
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name.startswith("rlt_module.")

    def freeze_non_dsrl_parameters(self):
        """Freeze the Pi0 backbone and keep only DSRL steering modules trainable."""
        if not self.config.use_dsrl or not hasattr(self, "dsrl_action_noise_net"):
            raise ValueError("DSRL-only training requires initialized DSRL modules.")
        trainable_prefixes = (
            "dsrl_action_noise_net.",
            "actor_image_encoder.",
            "actor_state_encoder.",
            "actor_tactile_encoder.",
            "critic_image_encoder.",
            "critic_state_encoder.",
            "critic_tactile_encoder.",
            "q_head.",
        )
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name.startswith(trainable_prefixes)

    def _init_dsrl_components(self):
        from rlinf.models.embodiment.modules.compact_encoders import (
            CompactMultiQHead,
            CompactStateEncoder,
            LightweightImageEncoder64,
        )
        from rlinf.models.embodiment.modules.gaussian_policy import GaussianPolicy

        # Match the checkpoint backbone dtype before safetensors loading so FSDP
        # observes a single parameter dtype when it creates FlatParameters.
        dsrl_dtype = torch.bfloat16
        tactile_latent_dim = (
            self.config.dsrl_tactile_latent_dim if self.config.dsrl_use_tactile else 0
        )
        dsrl_num_images = int(getattr(self.config, "dsrl_num_images", 1))
        if dsrl_num_images not in {1, 2}:
            raise ValueError(
                f"OpenPI DSRL dsrl_num_images must be 1 or 2; got {dsrl_num_images}."
            )
        state_side_dim = self.config.dsrl_state_latent_dim + tactile_latent_dim
        image_side_dim = self.config.dsrl_image_latent_dim * dsrl_num_images
        dsrl_input_dim = state_side_dim + image_side_dim

        self.dsrl_action_noise_net = GaussianPolicy(
            input_dim=dsrl_input_dim,
            output_dim=self.config.dsrl_action_noise_dim,
            hidden_dims=self.config.dsrl_hidden_dims,
            low=None,
            high=None,
            action_horizon=self.config.action_horizon,
        ).to(dtype=dsrl_dtype)
        self.actor_image_encoder = LightweightImageEncoder64(
            num_images=1,
            latent_dim=self.config.dsrl_image_latent_dim,
            image_size=64,
        ).to(dtype=dsrl_dtype)
        self.actor_state_encoder = CompactStateEncoder(
            state_dim=self.config.dsrl_state_dim,
            hidden_dim=self.config.dsrl_state_latent_dim,
        ).to(dtype=dsrl_dtype)
        self.critic_image_encoder = LightweightImageEncoder64(
            num_images=1,
            latent_dim=self.config.dsrl_image_latent_dim,
            image_size=64,
        ).to(dtype=dsrl_dtype)
        self.critic_state_encoder = CompactStateEncoder(
            state_dim=self.config.dsrl_state_dim,
            hidden_dim=self.config.dsrl_state_latent_dim,
        ).to(dtype=dsrl_dtype)
        if self.config.dsrl_use_tactile:
            tactile_encoder_kwargs = {
                "input_dim": 396,
                "hidden_dim": self.config.dsrl_tactile_latent_dim,
                "output_dim": self.config.dsrl_tactile_latent_dim,
                "history_len": 8,
                "has_reference_frame": True,
                "diff_from_reference": False,
            }
            self.actor_tactile_encoder = TactileTCNEncoder(**tactile_encoder_kwargs).to(
                dtype=dsrl_dtype
            )
            self.critic_tactile_encoder = TactileTCNEncoder(
                **tactile_encoder_kwargs
            ).to(dtype=dsrl_dtype)
        self.q_head = CompactMultiQHead(
            state_dim=state_side_dim,
            image_dim=image_side_dim,
            action_dim=self.config.dsrl_action_noise_dim,
            hidden_dims=self.config.dsrl_hidden_dims,
            num_q_heads=self.config.dsrl_num_q_heads,
            output_dim=1,
        ).to(dtype=dsrl_dtype)

    def _replace_projection_layers_for_config(self):
        """Align PyTorch projection layers with checkpoint action dimension."""
        expert_width = self.action_in_proj.out_features
        device = self.action_in_proj.weight.device
        dtype = self.action_in_proj.weight.dtype

        if self.action_in_proj.in_features == self.config.action_dim:
            return

        self.action_in_proj = nn.Linear(self.config.action_dim, expert_width).to(
            device=device, dtype=dtype
        )
        self.action_out_proj = nn.Linear(expert_width, self.config.action_dim).to(
            device=device, dtype=dtype
        )
        if not self.pi05:
            self.state_proj = nn.Linear(self.config.action_dim, expert_width).to(
                device=device, dtype=dtype
            )
            self.action_time_mlp_in = nn.Linear(2 * expert_width, expert_width).to(
                device=device, dtype=dtype
            )
            self.action_time_mlp_out = nn.Linear(expert_width, expert_width).to(
                device=device, dtype=dtype
            )

    def _init_tactile_prefix_encoder(self):
        """Create the optional Tabero tactile prefix encoder."""
        tactile_streams = tuple(self.config.tactile_streams or ())
        if (
            "tactile_prefix" not in tactile_streams
            or not self.config.tactile_prefix_dim_in
            or self.config.tactile_prefix_dim_in <= 0
        ):
            self.tactile_prefix_encoder = None
            return

        if self.config.tactile_prefix_encoder_type != "tcn":
            raise ValueError(
                "Only tactile_prefix_encoder_type='tcn' is supported for Tabero."
            )
        if self.config.tactile_prefix_history is None:
            raise ValueError("tactile_prefix_history is required for TCN tactile.")
        if self.config.tactile_prefix_use_reference_frame is None:
            raise ValueError(
                "tactile_prefix_use_reference_frame is required for TCN tactile."
            )
        if self.config.tactile_prefix_diff_from_reference is None:
            raise ValueError(
                "tactile_prefix_diff_from_reference is required for TCN tactile."
            )

        steps = (
            self.config.tactile_prefix_history + 1
            if self.config.tactile_prefix_use_reference_frame
            else self.config.tactile_prefix_history
        )
        if self.config.tactile_prefix_dim_in % steps != 0:
            raise ValueError(
                "tactile_prefix_dim_in must be divisible by the effective "
                f"history length; got dim={self.config.tactile_prefix_dim_in}, "
                f"steps={steps}."
            )

        prefix_width = _gemma.get_config(self.config.paligemma_variant).width
        self.tactile_prefix_encoder = TactileTCNEncoder(
            input_dim=self.config.tactile_prefix_dim_in // steps,
            hidden_dim=2 * prefix_width,
            output_dim=prefix_width,
            history_len=self.config.tactile_prefix_history,
            has_reference_frame=self.config.tactile_prefix_use_reference_frame,
            diff_from_reference=self.config.tactile_prefix_diff_from_reference,
        )

    def set_global_step(self, global_step):
        self.global_step = global_step

    def setup_wrappers(
        self,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
    ):
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)

    def _observation_from_dict(self, data: dict):
        """Create an OpenPI observation while preserving RLinf-local tactile fields."""
        observation = _model.Observation.from_dict(data)
        if "tactile_prefix" in data:
            object.__setattr__(observation, "tactile_prefix", data["tactile_prefix"])
        return observation

    def _preprocess_observation_with_tactile(self, observation, *, train=True):
        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=train)
        )
        tactile_prefix = getattr(observation, "tactile_prefix", None)
        return images, img_masks, lang_tokens, lang_masks, state, tactile_prefix

    def input_transform(self, obs: dict, transpose=True):
        inputs = tree_map(lambda x: x, obs)
        # process input
        first_process = "prompt" in inputs.keys()
        if first_process:
            inputs.pop("prompt")
        else:
            slash_key_inputs = {key: inputs[key] for key in inputs.keys() if "/" in key}
            if slash_key_inputs:
                inputs = slash_key_inputs
            else:
                tabero_input_keys = {
                    "image",
                    "wrist_image",
                    "tactile_image",
                    "state",
                    "actions",
                    "tactile_marker_motion",
                }
                inputs = {
                    key: inputs[key]
                    for key in inputs.keys()
                    if key in tabero_input_keys
                }

        # tensor -> numpy
        inputs = tree_map(_to_numpy, inputs)
        shaped_values = [v for v in inputs.values() if hasattr(v, "shape")]
        if not shaped_values:
            raise ValueError(
                "input_transform received no batched observation tensors after "
                f"filtering keys: {list(obs.keys())}"
            )
        batch_size = shaped_values[0].shape[0]
        # split
        batch_samples = []
        for i in range(batch_size):
            sample = tree_map(lambda x: x[i], inputs)
            if transpose:
                # convert from [3,256,256] -> [256,256,3]
                sample = tree_map(
                    lambda x: (
                        x.transpose(1, 2, 0) if len(x.shape) == 3 and transpose else x
                    ),
                    sample,
                )
            else:
                sample = tree_map(lambda x: x if len(x.shape) == 3 else x, sample)
            if first_process:
                sample["prompt"] = obs["prompt"][i]
            else:
                sample["prompt"] = "xxxx"
            batch_samples.append(sample)
        # transform
        with ThreadPoolExecutor(max_workers=min(len(batch_samples), 8)) as ex:
            transformed_samples = list(ex.map(self._input_transform, batch_samples))
        # recombine
        inputs = tree_map(
            lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
            *transformed_samples,
        )
        # inputs = tree_map(lambda *x: torch.stack(x, axis=0), inputs)
        if not first_process:
            inputs["tokenized_prompt"] = obs["tokenized_prompt"]
            inputs["tokenized_prompt_mask"] = obs["tokenized_prompt_mask"]
        return inputs

    def output_transform(self, outputs):
        # split & transform
        batch_size = outputs["actions"].shape[0]
        transformed_samples = []
        for i in range(batch_size):
            sample = tree_map(lambda x: np.asarray(x[i].detach().cpu()), outputs)
            sample = self._output_transform(sample)
            transformed_samples.append(sample)
        # recombine
        outputs = tree_map(
            lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
            *transformed_samples,
        )
        outputs["actions"] = outputs["actions"][:, : self.config.action_chunk]
        return outputs

    def _make_output_transform_input(self, actions, observation):
        output_data = {"actions": actions, "state": observation.state}
        tactile_prefix = getattr(observation, "tactile_prefix", None)
        if tactile_prefix is not None:
            output_data["tactile_prefix"] = tactile_prefix
        return output_data

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        elif forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        elif forward_type == ForwardType.NFT:
            return self.nft_forward(**kwargs)
        elif forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)
        elif forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)
        else:
            raise NotImplementedError

    def sft_forward(self, data, use_action_chunk_loss: bool = False, **kwargs):
        gradient_checkpointing_enabled = getattr(
            self, "_rlinf_gradient_checkpointing_enabled", False
        )
        if not gradient_checkpointing_enabled and hasattr(
            self, "gradient_checkpointing_disable"
        ):
            self.gradient_checkpointing_disable()

        if isinstance(data, tuple):
            observation, actions = data
            payload_tactile_prefix = None
        else:
            observation = data["observation"]
            actions = data["actions"]
            payload_tactile_prefix = data.get("tactile_prefix")

        if payload_tactile_prefix is not None:
            object.__setattr__(observation, "tactile_prefix", payload_tactile_prefix)

        device = next(self.parameters()).device
        tactile_prefix = getattr(observation, "tactile_prefix", None)
        if (
            getattr(self, "tactile_prefix_encoder", None) is not None
            and tactile_prefix is None
        ):
            raise RuntimeError(
                "Tabero SFT requires tactile_prefix when tactile_prefix_encoder "
                "is configured."
            )
        register_pytree_dataclasses(observation)
        observation = tree_map(
            lambda x: (
                torch.as_tensor(x, device=device).contiguous().clone()
                if x is not None
                else x
            ),
            observation,
        )
        if tactile_prefix is not None:
            object.__setattr__(
                observation,
                "tactile_prefix",
                torch.as_tensor(tactile_prefix, device=device).contiguous().clone(),
            )

        if self.config.use_rlt and self.config.rlt_train_module_only:
            prefix_output, prefix_mask = self._extract_rlt_prefix_embeddings(
                observation, train=True
            )
            rlt_param = next(self.rlt_module.parameters())
            prefix_output = prefix_output.to(
                device=rlt_param.device, dtype=rlt_param.dtype
            )
            rlt_mask = prefix_mask if self.config.rlt_use_mask else None
            rlt_loss, _ = self.rlt_module(prefix_output, rlt_mask)
            return {"loss": rlt_loss, "rlt_loss": rlt_loss}

        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=device)
        else:
            actions = actions.to(device=device)
        actions = actions.to(dtype=torch.float32)

        # PI0Pytorch.forward returns per-element MSE (reduction="none").
        use_prefix_aware_forward = self.config.use_rlt or (
            self.tactile_prefix_encoder is not None and tactile_prefix is not None
        )
        if use_prefix_aware_forward:
            loss, prefix_output, prefix_mask = self._sft_forward_with_rlt_prefix(
                observation, actions
            )
        else:
            loss = super().forward(observation, actions)
        if use_action_chunk_loss:
            loss = loss[:, : self.config.action_chunk, : self.config.action_env_dim]
            loss_output = {"loss": loss.mean()}
        else:
            loss_output = _reduce_sft_action_loss(loss, self.config)
        vla_loss = loss_output["loss"]
        if not self.config.use_rlt:
            if len(loss_output) == 1:
                return vla_loss
            return loss_output

        rlt_param = next(self.rlt_module.parameters())
        prefix_output = prefix_output.to(device=rlt_param.device, dtype=rlt_param.dtype)
        rlt_mask = prefix_mask if self.config.rlt_use_mask else None
        rlt_loss, _ = self.rlt_module(prefix_output, rlt_mask)
        total_loss = rlt_loss + self.config.rlt_alpha * vla_loss
        return {
            "loss": total_loss,
            "vla_loss": vla_loss,
            "rlt_loss": rlt_loss,
            **{name: value for name, value in loss_output.items() if name != "loss"},
        }

    def _sft_forward_with_rlt_prefix(self, observation, actions):
        images, img_masks, lang_tokens, lang_masks, state, tactile_prefix = (
            self._preprocess_observation_with_tactile(observation, train=True)
        )

        noise = self.sample_noise(actions.shape, actions.device)
        time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, tactile_prefix
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(state, x_t, time)
        )
        backbone_dtype = self.paligemma_with_expert.paligemma.language_model.layers[
            0
        ].self_attn.q_proj.weight.dtype
        if prefix_embs.dtype != backbone_dtype:
            prefix_embs = prefix_embs.to(dtype=backbone_dtype)
        if suffix_embs.dtype != backbone_dtype:
            suffix_embs = suffix_embs.to(dtype=backbone_dtype)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(
            prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        ):
            paired_model = self.paligemma_with_expert
            paired_training = paired_model.training
            if not getattr(self, "_rlinf_gradient_checkpointing_enabled", False):
                # RLinf/openpi's paired forward force-enables expert checkpointing
                # solely from its own `training` flag.  Clear only that wrapper
                # flag for this call; all child modules remain in train mode.
                paired_model.training = False
            try:
                (prefix_output, suffix_out), _ = paired_model.forward(
                    attention_mask=att_2d_masks_4d,
                    position_ids=position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, suffix_embs],
                    use_cache=False,
                    adarms_cond=[None, adarms_cond],
                )
            finally:
                paired_model.training = paired_training
            return prefix_output, suffix_out

        prefix_output, suffix_out = self._apply_checkpoint(
            forward_func,
            prefix_embs,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        loss = F.mse_loss(u_t, v_t, reduction="none")

        prefix_output, prefix_pad_masks = self._select_rlt_prefix_embeddings(
            prefix_output.detach(),
            prefix_pad_masks,
            lang_tokens,
            tactile_token_count=int(
                self.tactile_prefix_encoder is not None and tactile_prefix is not None
            ),
        )
        return loss, prefix_output, prefix_pad_masks

    def _build_rlt_prefix_cache(self, observation, *, train: bool):
        images, img_masks, lang_tokens, lang_masks, state, tactile_prefix = (
            self._preprocess_observation_with_tactile(observation, train=train)
        )
        device = next(self.parameters()).device
        images = [img.to(device) for img in images]
        img_masks = [img_mask.to(device) for img_mask in img_masks]
        if lang_tokens is not None:
            lang_tokens = lang_tokens.to(device)
        if lang_masks is not None:
            lang_masks = lang_masks.to(device)
        state = state.to(device)
        if tactile_prefix is not None:
            tactile_prefix = tactile_prefix.to(device)

        prefix_output, prefix_pad_masks, past_key_values = self._build_prefix_cache(
            images, img_masks, lang_tokens, lang_masks, tactile_prefix
        )
        tactile_token_count = int(
            getattr(self, "tactile_prefix_encoder", None) is not None
            and tactile_prefix is not None
        )
        return (
            prefix_output,
            prefix_pad_masks,
            past_key_values,
            lang_tokens,
            state,
            tactile_token_count,
        )

    def _select_rlt_prefix_embeddings(
        self,
        prefix_output,
        prefix_pad_masks,
        lang_tokens,
        tactile_token_count: int = 0,
    ):
        if self.config.rlt_image_only and lang_tokens is not None:
            num_image_tokens = (
                prefix_output.shape[1] - lang_tokens.shape[1] - tactile_token_count
            )
            image_output = prefix_output[:, :num_image_tokens]
            image_masks = prefix_pad_masks[:, :num_image_tokens]
            if tactile_token_count:
                prefix_output = torch.cat(
                    [image_output, prefix_output[:, -tactile_token_count:]], dim=1
                )
                prefix_pad_masks = torch.cat(
                    [image_masks, prefix_pad_masks[:, -tactile_token_count:]], dim=1
                )
            else:
                prefix_output = image_output
                prefix_pad_masks = image_masks
        return prefix_output, prefix_pad_masks

    def _extract_rlt_prefix_embeddings(self, observation, *, train: bool):
        with torch.no_grad():
            (
                prefix_output,
                prefix_pad_masks,
                _,
                lang_tokens,
                _,
                tactile_token_count,
            ) = self._build_rlt_prefix_cache(observation, train=train)

        return self._select_rlt_prefix_embeddings(
            prefix_output,
            prefix_pad_masks,
            lang_tokens,
            tactile_token_count=tactile_token_count,
        )

    def _select_configured_state(self, states):
        indices = self.config.state_indices
        if not indices:
            return states
        indices = list(indices)

        if hasattr(states, "shape"):
            state_dim = states.shape[-1]
        else:
            state_dim = np.asarray(states).shape[-1]
        if state_dim == len(indices):
            return states
        if state_dim <= max(indices):
            raise ValueError(
                f"Cannot select state_indices={indices} from state dim {state_dim}."
            )

        if torch.is_tensor(states):
            index_tensor = torch.as_tensor(indices, device=states.device)
            return states.index_select(-1, index_tensor)
        return np.asarray(states)[..., indices]

    def _prepare_rlt_reference_chunk(self, outputs, observation):
        if self.config.rlt_action_space == "model_normalized":
            return outputs["actions"][
                :, : self.config.action_chunk, : self.config.action_env_dim
            ]
        return self.output_transform(
            self._make_output_transform_input(outputs["actions"], observation)
        )["actions"]

    def _prepare_rlt_proprio(self, raw_states, normalized_state):
        if not self.config.rlt_use_normalized_proprio:
            return self._select_configured_state(raw_states)

        raw_state_dim = (
            raw_states.shape[-1]
            if hasattr(raw_states, "shape")
            else np.asarray(raw_states).shape[-1]
        )
        return self._select_configured_state(normalized_state[..., :raw_state_dim])

    @torch.no_grad()
    def decode_rlt_actions(
        self,
        normalized_actions: torch.Tensor,
        env_obs: dict[str, Any],
    ) -> torch.Tensor:
        model_action_dim = int(self.config.action_dim)
        if normalized_actions.shape[-1] > model_action_dim:
            raise ValueError(
                "RLT action width exceeds the OpenPI model action width: "
                f"{normalized_actions.shape[-1]} > {model_action_dim}."
            )

        to_process_obs = self.obs_processor(env_obs)
        processed_obs = self.input_transform(to_process_obs, transpose=False)
        processed_obs = self.precision_processor(processed_obs)
        observation = self._observation_from_dict(processed_obs)

        padded_actions = torch.zeros(
            *normalized_actions.shape[:-1],
            model_action_dim,
            device=normalized_actions.device,
            dtype=normalized_actions.dtype,
        )
        padded_actions[..., : normalized_actions.shape[-1]] = normalized_actions
        decoded = self.output_transform(
            self._make_output_transform_input(padded_actions, observation)
        )["actions"]
        return decoded[..., : self.config.action_env_dim]

    @torch.no_grad()
    def extract_rlt_obs(
        self,
        env_obs: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        if not self.config.use_rlt or not hasattr(self, "rlt_module"):
            raise ValueError("extract_rlt_obs requires openpi.use_rlt=True.")

        to_process_obs = self.obs_processor(env_obs)
        processed_obs = self.input_transform(to_process_obs, transpose=False)
        processed_obs = self.precision_processor(processed_obs)
        observation = self._observation_from_dict(processed_obs)

        (
            prefix_output,
            prefix_pad_masks,
            past_key_values,
            lang_tokens,
            state,
            tactile_token_count,
        ) = self._build_rlt_prefix_cache(observation, train=False)
        rlt_prefix_output, rlt_prefix_mask = self._select_rlt_prefix_embeddings(
            prefix_output,
            prefix_pad_masks,
            lang_tokens,
            tactile_token_count=tactile_token_count,
        )
        rlt_param = next(self.rlt_module.parameters())
        rlt_prefix_output = rlt_prefix_output.to(
            device=rlt_param.device, dtype=rlt_param.dtype
        )
        rlt_mask = rlt_prefix_mask if self.config.rlt_use_mask else None
        z_rl = self.rlt_module.encode_flat(rlt_prefix_output, rlt_mask).to(
            dtype=torch.float32
        )

        outputs = self._sample_actions_with_prefix_cache(
            state,
            prefix_output,
            prefix_pad_masks,
            past_key_values,
            mode="eval",
            compute_values=False,
        )
        ref_chunk = self._prepare_rlt_reference_chunk(outputs, observation)
        use_legacy_maniskill_normalized_proprio = (
            isinstance(self.config.config_name, str)
            and "maniskill" in self.config.config_name.lower()
        )
        if use_legacy_maniskill_normalized_proprio:
            raw_proprio = self._select_configured_state(env_obs["states"])
            state_dim = (
                raw_proprio.shape[-1]
                if hasattr(raw_proprio, "shape")
                else np.asarray(raw_proprio).shape[-1]
            )
            proprio = observation.state[..., :state_dim]
        else:
            proprio = self._prepare_rlt_proprio(env_obs["states"], observation.state)
        if not torch.is_tensor(proprio):
            proprio = torch.as_tensor(proprio)

        return {
            "z_rl": z_rl,
            "proprio": proprio.to(device=z_rl.device, dtype=torch.float32),
            "ref_chunk": ref_chunk.to(device=z_rl.device, dtype=torch.float32),
        }

    def prepare_dagger_sft_batch(self, batch):
        """Prepare replay-buffer samples for DAgger SFT updates."""
        device = next(self.parameters()).device
        obs_dict = {}
        obs_prefix_keys = [k for k in batch.keys() if k.startswith("observation/")]
        for key in obs_prefix_keys:
            obs_dict[key] = batch[key]
        if "tokenized_prompt" in batch:
            obs_dict["tokenized_prompt"] = batch["tokenized_prompt"]
        if "tokenized_prompt_mask" in batch:
            obs_dict["tokenized_prompt_mask"] = batch["tokenized_prompt_mask"]

        bsz = batch["action"].shape[0]
        if "model_action" in batch:
            actions = (
                batch["model_action"]
                .reshape(bsz, self.config.action_horizon, self.config.action_dim)
                .clone()
            )
            processed_obs = self.input_transform(obs_dict, transpose=False)
            processed_obs = self.precision_processor(processed_obs)
            observation = self._observation_from_dict(processed_obs)
        else:
            obs_dict["actions"] = batch["action"].reshape(
                bsz, self.config.action_chunk, -1
            )
            obs_dict["prompt"] = ["empty" for _ in range(bsz)]
            processed_obs = self.input_transform(obs_dict, transpose=False)
            if "tokenized_prompt" in batch:
                processed_obs["tokenized_prompt"] = batch["tokenized_prompt"]
            if "tokenized_prompt_mask" in batch:
                processed_obs["tokenized_prompt_mask"] = batch["tokenized_prompt_mask"]
            processed_obs = self.precision_processor(processed_obs)
            observation = self._observation_from_dict(processed_obs)
            actions = processed_obs["actions"].clone()
            processed_obs.pop("actions")

        register_pytree_dataclasses(observation)
        observation = tree_map(
            lambda x: torch.as_tensor(x, device=device).contiguous().clone(),
            observation,
        )
        return {
            "observation": observation,
            "actions": actions.to(torch.float32).to(device),
        }

    def prepare_lerobot_sft_batch(self, batch):
        """Prepare replay-buffer samples for DAgger SFT updates."""
        device = next(self.parameters()).device
        obs_dict = {}
        raw_obs_keys = [
            k
            for k in batch.keys()
            if k
            in [
                "image",
                "wrist_image",
                "extra_view_image",
                "extra_view_image-0",
                "extra_view_image-1",
                "state",
            ]
        ]
        _merge_keys = ["extra_view_image-0", "extra_view_image-1"]
        merge_extra = "extra_view_image" not in raw_obs_keys and all(
            k in raw_obs_keys for k in _merge_keys
        )
        for key in raw_obs_keys:
            # process other keys
            if merge_extra and key in _merge_keys:
                continue
            else:
                obs_dict[f"observation/{key}"] = batch[key]
        if merge_extra:
            obs_dict["observation/extra_view_image"] = []
            for key in _merge_keys:
                obs_dict["observation/extra_view_image"].append(batch[key])
            obs_dict["observation/extra_view_image"] = torch.stack(
                obs_dict["observation/extra_view_image"], dim=1
            )

        bsz = batch["actions"].shape[0]
        obs_dict["actions"] = batch["actions"].reshape(
            bsz, self.config.action_chunk, -1
        )
        obs_dict["prompt"] = batch["task"]
        processed_obs = self.input_transform(obs_dict, transpose=False)
        processed_obs = self.precision_processor(processed_obs)
        observation = _model.Observation.from_dict(processed_obs)
        actions = processed_obs["actions"].clone()
        processed_obs.pop("actions")
        register_pytree_dataclasses(observation)
        observation = tree_map(
            lambda x: torch.as_tensor(x, device=device).contiguous().clone(),
            observation,
        )
        return {
            "observation": observation,
            "actions": actions.to(torch.float32).to(device),
        }

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        # get kwargs
        compute_values = kwargs.get("compute_values", False)
        chains = forward_inputs["chains"]
        denoise_inds = forward_inputs["denoise_inds"]
        # input transform
        observation = self.input_transform(forward_inputs, transpose=False)
        observation = self._observation_from_dict(observation)
        images, img_masks, lang_tokens, lang_masks, state, tactile_prefix = (
            self._preprocess_observation_with_tactile(observation, train=False)
        )
        # transfer to device
        device = chains.device
        images = [img.to(device) for img in images]
        img_masks = [img_mask.to(device) for img_mask in img_masks]
        state = state.to(device)
        # get log prob
        log_probs, value_t, entropy = self.get_log_prob_value(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            tactile_prefix,
            chains,
            denoise_inds,
            compute_values,
        )
        log_probs = log_probs[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        entropy = entropy[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        # post process
        log_probs = log_probs.mean(dim=1)
        entropy = entropy.mean(dim=[1, 2, 3], keepdim=False)[
            :, None
        ]  # [:,None] to align with loss-mask shape
        value_t = value_t.mean(dim=-1, keepdim=False)
        return {
            "logprobs": log_probs,
            "values": value_t,
            "entropy": entropy,
        }

    def nft_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        """Compute velocity v_theta at explicit (x_t, timesteps) for NFT loss."""
        # obs process
        observation = self.input_transform(forward_inputs, transpose=False)
        observation = self._observation_from_dict(observation)
        images, img_masks, lang_tokens, lang_masks, state, tactile_prefix = (
            self._preprocess_observation_with_tactile(observation, train=False)
        )
        # move device
        device = next(self.parameters()).device
        images = [img.to(device) for img in images]
        img_masks = [m.to(device) for m in img_masks]
        state = state.to(device)
        # nft inputs
        nft_inputs = kwargs["nft_inputs"]
        x_t = nft_inputs["x_t"].to(device)
        t = nft_inputs["timesteps"].to(device)
        # get v_theta
        _, prefix_pad_masks, past_key_values = self._build_prefix_cache(
            images, img_masks, lang_tokens, lang_masks, tactile_prefix
        )
        compute_values = kwargs.get("compute_values", False)
        v_theta, suffix_out = self.get_velocity(
            state, x_t, t, prefix_pad_masks, past_key_values
        )
        v_theta = v_theta[:, : self.config.action_chunk, :]
        # result
        result: dict[str, Any] = {"v_theta": v_theta, "x_t": x_t, "timesteps": t}
        if compute_values and self.config.add_value_head:
            result["values"] = self._compute_value_from_suffix(suffix_out)[:, None]
        return result

    def obs_processor(self, env_obs):
        if "tabero" in self.config.config_name:
            processed_obs = {
                "image": env_obs["main_images"],
                "wrist_image": env_obs["wrist_images"],
                "state": env_obs["states"],
                "prompt": env_obs["task_descriptions"],
            }
            if "tacimg" in self.config.config_name:
                tactile_image = env_obs.get("tactile_images")
                if tactile_image is None:
                    tactile_image = env_obs.get("extra_view_images")
                if tactile_image is None:
                    raise KeyError(
                        "Tabero tacimg expects 'tactile_images' or "
                        "'extra_view_images' in env_obs."
                    )
                processed_obs["tactile_image"] = tactile_image
            if "tacfield" in self.config.config_name:
                if "tactile_marker_motion" not in env_obs:
                    raise KeyError(
                        "Tabero tacfield expects 'tactile_marker_motion' in env_obs."
                    )
                processed_obs["tactile_marker_motion"] = env_obs[
                    "tactile_marker_motion"
                ]
            return processed_obs

        env_states = self._select_configured_state(env_obs["states"])
        processed_obs = {
            "observation/image": env_obs["main_images"],
            "prompt": env_obs["task_descriptions"],
        }
        if "calvin" in self.config.config_name:
            state = env_states
            processed_obs["observation/state_ee_pos"] = state[:, :3]
            processed_obs["observation/state_ee_rot"] = state[:, 3:6]
            processed_obs["observation/state_gripper"] = state[:, 6:7]
        else:
            processed_obs["observation/state"] = env_states
        if env_obs["wrist_images"] is not None:
            processed_obs["observation/wrist_image"] = env_obs["wrist_images"]
        if env_obs["extra_view_images"] is not None:
            processed_obs["observation/extra_view_image"] = env_obs["extra_view_images"]
        return processed_obs

    def precision_processor(self, processed_obs):
        device = next(self.parameters()).device
        for key, value in processed_obs.items():
            if isinstance(value, list):
                processed_obs[key] = [
                    item.to(device=device).contiguous()
                    if torch.is_tensor(item)
                    else item
                    for item in value
                ]
            elif torch.is_tensor(value):
                processed_obs[key] = value.to(device=device).contiguous()
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    processed_obs[key][sub_key] = sub_value.to(
                        device=device
                    ).contiguous()
        return processed_obs

    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        compute_values=True,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        dsrl_obs = None
        if self.config.use_dsrl:
            dsrl_obs = self._normalize_dsrl_obs(env_obs)
        if self.config.use_dsrl and self.config.dsrl_use_tactile:
            main_images = env_obs.get("main_images")
            batch_size = (
                main_images.shape[0]
                if torch.is_tensor(main_images) and main_images.ndim > 0
                else None
            )
            self._validate_dsrl_tactile(env_obs, batch_size=batch_size)

        to_process_obs = self.obs_processor(env_obs)  # env obs -> policy input obs
        processed_obs = self.input_transform(
            to_process_obs, transpose=False
        )  # policy input obs -> model input obs
        processed_obs = self.precision_processor(
            processed_obs
        )  # obs precision processor
        observation = self._observation_from_dict(processed_obs)

        is_dsrl_active = self.config.use_dsrl
        if is_dsrl_active:
            # DSRL mode (both train and eval)

            # Step 1: SAC agent outputs noise
            assert dsrl_obs is not None
            noise_actions, noise_logprob, _ = self.sac_forward(
                dsrl_obs, train=False, mode=mode
            )

            # Step 2: Use noise to sample actual actions from diffusion model
            outputs = self.sample_actions(
                observation,
                noise=noise_actions,
                mode="eval",
                compute_values=False,
                collect_forward_metadata=False,
            )

            # Step 3: Extract actual actions for environment interaction
            real_actions = self.output_transform(
                self._make_output_transform_input(outputs["actions"], observation)
            )["actions"]

            # Return actual actions to environment, but forward_inputs stores noise.
            actions = real_actions
            prev_logprobs = noise_logprob  # SAC noise logprob
            # SAC only needs one 32D latent. GaussianPolicy repeats that latent
            # over the 50-step Pi0 horizon for diffusion, but replay must not
            # retain 50 identical copies or any frozen-Pi0 intermediates.
            forward_inputs = {"action": noise_actions[:, 0, :].detach().contiguous()}
            result = {
                "prev_logprobs": prev_logprobs,
                "prev_values": None,
                "forward_inputs": forward_inputs,
            }
            return actions, result

        else:
            # Non-DSRL or eval mode
            outputs = self.sample_actions(
                observation, mode=mode, compute_values=compute_values
            )
            actions = self.output_transform(
                self._make_output_transform_input(outputs["actions"], observation)
            )["actions"]
            prev_logprobs = outputs["prev_logprobs"]
            prev_values = outputs["prev_values"]

        forward_inputs = {
            "chains": outputs["chains"],
            "denoise_inds": outputs["denoise_inds"],
            "tokenized_prompt": processed_obs["tokenized_prompt"],
            "tokenized_prompt_mask": processed_obs["tokenized_prompt_mask"],
            # "action" is the env-executed action, and "model_action" is the original output by the model.
            # For small models, they are consistent. For large models (like pi), "action" is the result after output_transform.
            # For realworld human-in-the-loop training, only "action" can be provided by human.
            "action": actions.reshape(actions.shape[0], -1).contiguous(),
            "model_action": outputs["actions"]
            .reshape(outputs["actions"].shape[0], -1)
            .contiguous(),
        }
        if self.config.is_nft:
            nft_outputs = {
                key: value for key, value in outputs.items() if key.startswith("nft_")
            }
            forward_inputs.update(nft_outputs)

        # Clone observations to avoid cross-step reference issues.
        cloned_obs = copy_dict_tensor(
            {k: v for k, v in to_process_obs.items() if k != "prompt"}
        )
        forward_inputs.update(cloned_obs)

        result = {
            "prev_logprobs": prev_logprobs,
            "prev_values": prev_values,
            "forward_inputs": forward_inputs,
        }
        return actions, result

    @torch.no_grad()
    def sample_actions(
        self,
        observation: _model.Observation,
        noise=None,
        mode="train",
        compute_values=True,
        collect_forward_metadata=True,
    ) -> torch.Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        device = observation.state.device
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)
        else:
            # DSRL: SAC provides noise, convert dtype to match action_in_proj
            noise = noise.to(self.action_in_proj.weight.dtype)

        images, img_masks, lang_tokens, lang_masks, state, tactile_prefix = (
            self._preprocess_observation_with_tactile(observation, train=False)
        )

        prefix_output, prefix_pad_masks, past_key_values = self._build_prefix_cache(
            images, img_masks, lang_tokens, lang_masks, tactile_prefix
        )

        return self._sample_actions_with_prefix_cache(
            state,
            prefix_output,
            prefix_pad_masks,
            past_key_values,
            noise=noise,
            mode=mode,
            compute_values=compute_values,
            collect_forward_metadata=collect_forward_metadata,
        )

    def _sample_actions_with_prefix_cache(
        self,
        state,
        prefix_output,
        prefix_pad_masks,
        past_key_values,
        noise=None,
        mode="train",
        compute_values=True,
        collect_forward_metadata=True,
    ) -> torch.Tensor:
        bsize = state.shape[0]
        device = state.device
        num_steps = self.config.num_steps
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)
        else:
            # DSRL: SAC provides noise, convert dtype to match action_in_proj
            noise = noise.to(self.action_in_proj.weight.dtype)

        x_t = noise
        # add sde sample and traj collect
        chains = []
        log_probs = []
        values = []
        if collect_forward_metadata:
            chains.append(x_t)

        # add value based on the vlm for pi05, expert for pi0
        if collect_forward_metadata and self.use_vlm_value:
            values_vlm = self.get_value_from_vlm(prefix_output)
        if collect_forward_metadata and self.config.joint_logprob:
            initial_log_prob = self.get_logprob_norm(
                x_t, torch.zeros_like(noise), torch.ones_like(noise)
            )
            log_probs.append(initial_log_prob)

        # In the joint logprob mode, we need to sample the logprob for each denoise step
        # In the non-joint logprob mode, only one denoise step is sampled and ode-sde mix sampling is used
        # denoise index
        collect_nft_state = self.config.is_nft and mode == "train"
        if mode == "train":
            if self.config.joint_logprob or collect_nft_state:
                denoise_inds = torch.arange(num_steps)
            else:
                if self.config.ignore_last:
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 2)] * num_steps
                    )
                else:
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 1)] * num_steps
                    )
        else:
            denoise_inds = torch.tensor([-1] * num_steps)
        denoise_inds = denoise_inds[None].repeat(bsize, 1)

        # collect nft states for nft algorithm
        nft_state = self._init_nft_state(collect_nft_state, x_t, num_steps, device)

        # denoise step
        for idx in range(num_steps):
            # sample mean var val
            if idx == denoise_inds[0][idx]:
                sample_method = self.config.noise_method
            else:
                sample_method = "flow_ode"
            x_t_prev = x_t
            x_t_mean, x_t_std, value_t, v_t = self.sample_mean_var_val(
                x_t,
                idx,
                state,
                prefix_pad_masks,
                past_key_values,
                sample_method,
                num_steps,
                compute_values,
            )
            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
            self._update_nft_state(nft_state, idx, x_t_prev, v_t, x_t, sample_method)
            if collect_forward_metadata:
                log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)
                values.append(value_t)
                chains.append(x_t)
                log_probs.append(log_prob)
        x_0 = x_t
        if not collect_forward_metadata:
            return {"actions": x_0}
        chains = torch.stack(chains, dim=1)
        # post process for logprob
        log_probs = torch.stack(log_probs, dim=1)[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        if self.config.joint_logprob:
            log_probs = log_probs.mean(dim=1)
        else:
            log_probs = log_probs[
                torch.arange(log_probs.shape[0], device=device),
                denoise_inds[:, 0],
            ]
        # post process for value
        if self.use_vlm_value:
            values = values_vlm[:, None]
        else:
            values = torch.stack(values, dim=1).mean(dim=-1, keepdim=True)
        result = {
            "actions": x_0,
            "chains": chains,
            "prev_logprobs": log_probs,
            "prev_values": values,
            "denoise_inds": denoise_inds,
        }
        if collect_nft_state:
            result.update(nft_state)
            result["nft_x0"] = x_0.detach()
        return result

    def _get_timesteps(self, denoise_steps, device):
        timesteps = torch.linspace(1, 1 / denoise_steps, denoise_steps, device=device)
        timesteps = torch.cat([timesteps, torch.zeros((1), device=device)])
        return timesteps

    def sample_mean_var_val(
        self,
        x_t,
        idx,
        state,
        prefix_pad_masks,
        past_key_values,
        sample_method,
        denoise_steps,
        compute_values=True,
    ):
        """
        Sample the mean, variance and value of the action at a given timestep.
        Rollout sample (idx is int) and actor get_log_prob_value (idx is tensor)
        will load this function. `sample_method` is one of flow_ode/flow_sde/
        flow_cps/flow_noise.
        """
        # expand the shape
        bsize = state.shape[0]
        device = state.device
        if isinstance(idx, int):
            idx = torch.full((), idx, device=device).expand(bsize)
        # build parameters
        noise_level = self._get_noise_level(device=device, dtype=x_t.dtype)
        timesteps = self._get_timesteps(denoise_steps, device)
        # input parameters
        t_input = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]
        # velocity prediction
        v_t, suffix_out = self.get_velocity(
            state, x_t, t_input, prefix_pad_masks, past_key_values
        )
        # value prediction
        if (
            self.config.add_value_head
            and compute_values
            and not self.config.value_after_vlm
        ):
            value_t = self._compute_value_from_suffix(suffix_out)
        else:
            value_t = torch.zeros((bsize), device=device)
        # sample mean and variance
        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)

        if sample_method == "flow_ode":
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = torch.zeros_like(t_input)
        elif sample_method == "flow_sde":
            denom_timesteps = torch.where(timesteps == 1, timesteps[1], timesteps)
            sigma_ratio = timesteps / (1 - denom_timesteps)
            sigmas = noise_level * torch.sqrt(sigma_ratio)[:-1]
            sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
            x0_weight = torch.ones_like(t_input) - (t_input - delta)
            x1_weight = t_input - delta - sigma_i**2 * delta / (2 * t_input)
            x_t_std = torch.sqrt(delta) * sigma_i
        elif sample_method == "flow_cps":
            pi = torch.pi
            cos_term = torch.cos(pi * noise_level / 2).to(device)
            sin_term = torch.sin(pi * noise_level / 2).to(device)
            x0_weight = torch.ones_like(t_input) - (t_input - delta)
            x1_weight = (t_input - delta) * cos_term
            x_t_std = (t_input - delta) * sin_term
        elif sample_method == "flow_noise":
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = self.noise_head(suffix_out)
        else:
            raise ValueError(f"Invalid noise method: {sample_method}")
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std, value_t, v_t

    def get_suffix_out(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(state, x_t, timestep)
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = (
            "eager"  # noqa: SLF001
        )

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return suffix_out

    def get_velocity(self, state, x_t, timestep, prefix_pad_masks, past_key_values):
        """Compute velocity prediction v_t and raw suffix_out at a given timestep."""
        suffix_out = self.get_suffix_out(
            state, prefix_pad_masks, past_key_values, x_t, timestep
        )
        v_t = self.action_out_proj(suffix_out)
        return v_t, suffix_out

    def embed_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        tactile_prefix=None,
    ):
        """Embed image/language prefix tokens and optional Tabero tactile token."""
        prefix_embs, prefix_pad_masks, prefix_att_masks = super().embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        if self.tactile_prefix_encoder is None or tactile_prefix is None:
            return prefix_embs, prefix_pad_masks, prefix_att_masks

        tactile_prefix = tactile_prefix.to(
            device=prefix_embs.device,
            dtype=self.tactile_prefix_encoder.out_proj.weight.dtype,
        )

        def tactile_embed_func(tactile):
            return self.tactile_prefix_encoder(tactile)

        tactile_emb = self._apply_checkpoint(tactile_embed_func, tactile_prefix)
        tcn_is_trainable = self.training and any(
            parameter.requires_grad
            for parameter in self.tactile_prefix_encoder.parameters()
        )
        if tcn_is_trainable and not tactile_emb.requires_grad:
            raise RuntimeError(
                "Trainable tactile_prefix_encoder produced a detached tactile token."
            )
        if tcn_is_trainable and not getattr(
            self, "_tactile_gradient_audit_registered", False
        ):
            tactile_emb.register_hook(
                lambda grad: get_logger().info(
                    "Tabero tactile token gradient audit: norm=%s max_abs=%s finite=%s",
                    float(grad.float().norm()),
                    float(grad.float().abs().max()),
                    bool(torch.isfinite(grad).all()),
                )
            )
            self._tactile_gradient_audit_registered = True
        tactile_emb = tactile_emb.to(dtype=prefix_embs.dtype)[:, None, :]
        tactile_pad_mask = torch.ones(
            tactile_emb.shape[:2],
            dtype=torch.bool,
            device=prefix_pad_masks.device,
        )
        tactile_att_mask = torch.zeros(
            tactile_emb.shape[:2],
            dtype=torch.bool,
            device=prefix_att_masks.device,
        )

        prefix_embs = torch.cat([prefix_embs, tactile_emb], dim=1)
        prefix_pad_masks = torch.cat([prefix_pad_masks, tactile_pad_mask], dim=1)
        prefix_att_masks = torch.cat([prefix_att_masks, tactile_att_mask], dim=1)
        return prefix_embs, prefix_pad_masks, prefix_att_masks

    def _build_prefix_cache(
        self, images, img_masks, lang_tokens, lang_masks, tactile_prefix=None
    ):
        """Embed prefix tokens and compute KV cache for efficient suffix generation."""
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, tactile_prefix
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        (prefix_output, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return prefix_output, prefix_pad_masks, past_key_values

    def _compute_value_from_suffix(self, suffix_out):
        """Compute value from suffix output using value head."""
        if self.config.chunk_critic_input:
            suffix_out_value = torch.mean(
                suffix_out[:, : self.config.action_chunk], dim=1, keepdim=False
            )
        else:
            suffix_out_value = torch.mean(suffix_out, dim=1, keepdim=False)
        if self.config.detach_critic_input:
            suffix_out_value = suffix_out_value.detach()
        return self.value_head(suffix_out_value)[:, 0]

    # TODO: to check potential nan here
    def get_logprob_norm(self, sample, mu, sigma):
        # logprob = log p(x|mu,sigma) = -log(sigma) - 0.5 * log(2 * pi) - 0.5 * ((x - mu) / sigma) ** 2
        if self.config.safe_get_logprob:
            log_prob = -torch.pow((sample - mu), 2)
        else:
            mask = sigma == 0
            sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
            constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
                2 * torch.pi * torch.ones_like(sample)
            )
            exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
            log_prob = constant_term + exponent_term
            log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
        return log_prob

    def preprocess_for_train(self, data):
        return data

    def get_log_prob_value(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        tactile_prefix,
        chains,
        denoise_inds,
        compute_values=False,
    ):
        bsize = state.shape[0]
        batch_indices = torch.arange(bsize)
        prefix_output, prefix_pad_masks, past_key_values = self._build_prefix_cache(
            images, img_masks, lang_tokens, lang_masks, tactile_prefix
        )
        chains_log_probs = []
        chains_values = []
        chains_entropy = []

        # get log prob
        if self.config.joint_logprob:
            num_steps = self.config.num_steps
            initial_log_prob = self.get_logprob_norm(
                chains[:, 0],
                torch.zeros_like(chains[:, 0]),
                torch.ones_like(chains[:, 0]),
            )
            initial_entropy = self.gaussian_entropy(torch.ones_like(chains[:, 0]))
            chains_log_probs.append(initial_log_prob)
            chains_entropy.append(initial_entropy)
        else:
            num_steps = 1
        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[batch_indices, denoise_ind]
            chains_next = chains[batch_indices, denoise_ind + 1]
            x_t_mean, x_t_std, value_t, _ = self.sample_mean_var_val(
                chains_pre,
                denoise_ind,
                state,
                prefix_pad_masks,
                past_key_values,
                self.config.noise_method,
                self.config.num_steps,
                compute_values,
            )
            log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            entropy = self.gaussian_entropy(x_t_std)
            chains_log_probs.append(log_probs)
            chains_entropy.append(entropy)
            if not self.use_vlm_value:
                chains_values.append(value_t)
        if self.use_vlm_value:
            chains_values.append(self.get_value_from_vlm(prefix_output))
        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        chains_values = torch.stack(chains_values, dim=1)

        # entropy is only available for flow-noise method
        if self.config.noise_method == "flow_noise":
            chains_entropy = torch.stack(chains_entropy, dim=1)
        else:
            chains_entropy = torch.zeros_like(chains_log_probs)
        return chains_log_probs, chains_values, chains_entropy

    def get_value_from_vlm(self, prefix_output):
        # prefix_output:
        # pi05: [bs, (256 * 3 + 200) = 968, 2048]
        # pi0: [bs, (256 * 3 + 48) = 816, 1024]
        # token length
        if "pi05_" in self.config.config_name:
            lang_token_len = 200
            all_token_length = 968
        elif "pi0_" in self.config.config_name:
            lang_token_len = 48
            all_token_length = 816

        if self.config.value_vlm_mode == "mean_token":
            prefix_mask = (
                [True] * 256 * self.config.num_images_in_input
                + [False] * 256 * (3 - self.config.num_images_in_input)
                + [True] * lang_token_len
            )
        elif self.config.value_vlm_mode == "last_token":
            prefix_mask = [False] * (all_token_length - 1) + [True] * 1
        elif self.config.value_vlm_mode == "first_token":
            prefix_mask = [True] * 1 + [False] * (all_token_length - 1)
        prefix_out_value = prefix_output[:, prefix_mask, :]
        prefix_out_value = prefix_out_value.mean(dim=1, keepdim=False)
        prefix_out_value = prefix_out_value.to(dtype=torch.float32)
        values_vlm = self.value_head(prefix_out_value)[:, 0]
        return values_vlm

    def gaussian_entropy(self, sigma):
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        entropy = 0.5 * torch.log(2 * math.pi * math.e * (sigma_safe**2))
        return entropy

    def freeze_vlm(self):
        if self.config.train_expert_only:
            # Base freeze: paligemma (SigLIP vision encoder + Gemma)
            self.paligemma_with_expert.paligemma.eval()
            for params in self.paligemma_with_expert.paligemma.parameters():
                params.requires_grad = False

            # ========== DSRL additional freezing ==========
            if self.config.use_dsrl:
                self.logger.info(
                    "[FREEZE_VLM] DSRL mode: freezing gemma_expert parameters"
                )
                self.paligemma_with_expert.gemma_expert.eval()
                for params in self.paligemma_with_expert.gemma_expert.parameters():
                    params.requires_grad = False

                # Freeze projection layers (used in rollout/eval but not optimized).
                # Pi0 has: action_in_proj, action_out_proj, state_proj, action_time_mlp_in/out
                # Pi0.5 has: action_in_proj, action_out_proj, time_mlp_in/out (no state_proj)
                self.logger.info(
                    "[FREEZE_VLM] DSRL mode: freezing projection layers (used in rollout/eval but not optimized)"
                )
                if self.pi05:
                    projection_names = [
                        "action_in_proj",
                        "action_out_proj",
                        "time_mlp_in",
                        "time_mlp_out",
                    ]
                else:
                    projection_names = [
                        "action_in_proj",
                        "action_out_proj",
                        "state_proj",
                        "action_time_mlp",
                    ]
                frozen_count = 0
                for name, param in self.named_parameters():
                    if any(proj_name in name for proj_name in projection_names):
                        param.requires_grad = False
                        frozen_count += 1
                        if frozen_count <= 10:  # Print first 10 for brevity
                            self.logger.info(f"  Froze: {name}")
                if frozen_count > 10:
                    self.logger.info(
                        f"  ... and {frozen_count - 10} more projection layer parameters"
                    )

                # Freeze reinflow_explore_noise_net (only used in reinflow diffuser sampling)
                if hasattr(self, "reinflow_explore_noise_net"):
                    self.logger.info(
                        "[FREEZE_VLM] DSRL mode: freezing reinflow_explore_noise_net (used in non-DSRL rollout but not optimized)"
                    )
                    self.reinflow_explore_noise_net.eval()
                    noise_net_params = 0
                    for params in self.reinflow_explore_noise_net.parameters():
                        params.requires_grad = False
                        noise_net_params += params.numel()
                    self.logger.info(
                        f"  Froze {noise_net_params:,} parameters in reinflow_explore_noise_net"
                    )

    # ===== DSRL-specific methods =====

    def _normalize_dsrl_obs(self, obs):
        """Normalize and validate the ordered DSRL image-view contract."""
        num_images = int(getattr(self.config, "dsrl_num_images", 1))
        if num_images not in {1, 2}:
            raise ValueError(
                f"OpenPI DSRL dsrl_num_images must be 1 or 2; got {num_images}."
            )
        if "dsrl_images" in obs:
            if "images" in obs or "main_images" in obs or "wrist_images" in obs:
                raise ValueError(
                    "OpenPI DSRL compact replay observation cannot mix "
                    "'dsrl_images' with raw image fields."
                )
            compact_images = obs["dsrl_images"]
            expected_shape = (
                int(obs["states"].shape[0]),
                num_images,
                3,
                64,
                64,
            )
            if (
                not torch.is_tensor(compact_images)
                or tuple(compact_images.shape) != expected_shape
            ):
                actual = (
                    tuple(compact_images.shape)
                    if hasattr(compact_images, "shape")
                    else None
                )
                raise ValueError(
                    "OpenPI DSRL compact replay images expected shape "
                    f"{expected_shape}; got {actual}."
                )
            if compact_images.dtype != torch.bfloat16:
                raise ValueError(
                    "OpenPI DSRL compact replay images must use bfloat16; "
                    f"got {compact_images.dtype}."
                )
            normalized = {
                "dsrl_images": compact_images,
                "states": obs["states"],
            }
        elif "images" in obs:
            normalized = dict(obs)
        else:
            if "main_images" not in obs:
                raise ValueError(
                    f"Invalid obs format: {obs.keys()}. Expected 'images' or "
                    "'main_images' key."
                )
            images = [obs["main_images"]]
            if num_images == 2:
                if "wrist_images" not in obs or obs["wrist_images"] is None:
                    raise ValueError(
                        "OpenPI DSRL dual-camera mode requires 'wrist_images'; "
                        f"available keys={list(obs.keys())}."
                    )
                images.append(obs["wrist_images"])
            normalized = {
                "images": images,
                "states": obs["states"],
            }
        if "tactile_marker_motion" in obs:
            normalized["tactile_marker_motion"] = obs["tactile_marker_motion"]
        if "dsrl_images" not in normalized:
            self._validate_dsrl_image_views(
                normalized["images"], normalized.get("states")
            )
        return normalized

    def _prepare_dsrl_images(self, obs, *, train=False):
        """Return raw or replay-preprocessed image views in model input format."""

        if "dsrl_images" in obs:
            return obs["dsrl_images"]
        return self._preprocess_dsrl_images(obs["images"], train=train)

    def _validate_dsrl_image_views(self, images, states=None):
        num_images = int(getattr(self.config, "dsrl_num_images", 1))
        if not isinstance(images, (list, tuple)):
            raise ValueError(
                "OpenPI DSRL 'images' must be an ordered list/tuple in "
                "main-to-wrist order."
            )
        if len(images) != num_images:
            raise ValueError(
                f"OpenPI DSRL expected {num_images} image views in main-to-wrist "
                f"order; got {len(images)}."
            )

        expected_batch = (
            int(states.shape[0])
            if torch.is_tensor(states) and states.ndim > 0
            else None
        )
        is_tabero_dual_camera = (
            getattr(self.config, "config_name", None) == "pi0_lora_tacfield_tabero"
            and num_images == 2
        )
        view_names = ("main", "wrist") if num_images == 2 else ("main",)
        for view_name, image in zip(view_names, images, strict=True):
            if not torch.is_tensor(image) or image.ndim != 4:
                actual = tuple(image.shape) if hasattr(image, "shape") else None
                raise ValueError(
                    f"OpenPI DSRL {view_name} image must be a rank-4 tensor; "
                    f"got {actual}."
                )
            if expected_batch is not None and image.shape[0] != expected_batch:
                raise ValueError(
                    f"OpenPI DSRL {view_name} image batch mismatch: expected "
                    f"{expected_batch}, got {image.shape[0]}."
                )
            is_nhwc = image.shape[-1] == 3
            is_nchw = image.shape[1] == 3
            if not is_nhwc and not is_nchw:
                raise ValueError(
                    f"OpenPI DSRL {view_name} image expected RGB in NHWC or NCHW; "
                    f"got {tuple(image.shape)}."
                )
            if is_tabero_dual_camera:
                expected_shape = (expected_batch, 256, 256, 3)
                if tuple(image.shape) != expected_shape:
                    raise ValueError(
                        f"Tabero DSRL {view_name} image expected shape "
                        f"{expected_shape}; got {tuple(image.shape)}."
                    )
                if image.dtype != torch.uint8:
                    raise ValueError(
                        f"Tabero DSRL {view_name} image must use uint8; "
                        f"got {image.dtype}."
                    )
        return images

    def _validate_dsrl_tactile(self, obs, *, batch_size):
        key = "tactile_marker_motion"
        if key not in obs:
            raise ValueError(
                f"DSRL tactile mode requires '{key}'; available keys={list(obs.keys())}."
            )
        tactile = obs[key]
        actual_shape = tuple(tactile.shape) if hasattr(tactile, "shape") else None
        expected_shape = (batch_size, 9, 198, 2)
        if (
            batch_size is None
            or not torch.is_tensor(tactile)
            or actual_shape != expected_shape
        ):
            raise ValueError(
                f"DSRL tactile '{key}' expected shape {expected_shape}, "
                f"got {actual_shape}."
            )
        return tactile

    def _prepare_dsrl_tactile(self, obs, *, batch_size, encoder):
        tactile = self._validate_dsrl_tactile(obs, batch_size=batch_size)
        parameter = next(encoder.parameters())
        return tactile.reshape(batch_size, 9, 396).to(
            device=parameter.device, dtype=parameter.dtype
        )

    def sac_forward(
        self, obs=None, data=None, train=False, return_dist_params=False, **kwargs
    ):
        """SAC forward pass for DSRL.

        Args:
            obs: Observation dict (preferred, matches sac_dsrl).
                 Supports two formats:
                   1. {"images": list of tensors, "states": tensor} - internal format
                   2. {"main_images": tensor, "wrist_images": tensor, "states": tensor} - env format
            data: Dictionary containing observations (legacy, for backward compatibility).
            train: Whether to use data augmentation.
            return_dist_params: Whether to return distribution parameters for logging.

        Returns:
            actions: [B, action_horizon, output_dim] - noise or actual actions
            logprobs: [B] - log probabilities
            dist_params: (mean, std) or None - distribution parameters for logging
        """
        if not self.config.use_dsrl:
            raise ValueError("sac_forward called but use_dsrl=False")

        # Support both call styles: obs (new, from sac_dsrl) or data (legacy)
        if obs is None:
            obs = data.get("obs", data) if data is not None else kwargs.get("obs", {})

        obs = self._normalize_dsrl_obs(obs)

        # Preprocess ordered image views independently.
        # Returns [B, N, C, 64, 64] in [-1, 1] range (float32).
        images = self._prepare_dsrl_images(obs, train=train)
        states = self._preprocess_states(obs["states"])

        # Move to the same device as actor encoders, convert to bfloat16
        device = next(self.actor_image_encoder.parameters()).device
        images = images.to(device=device, dtype=torch.bfloat16)
        states = states.to(device=device, dtype=torch.bfloat16)
        tactile = None
        if self.config.dsrl_use_tactile:
            tactile = self._prepare_dsrl_tactile(
                obs,
                batch_size=states.shape[0],
                encoder=self.actor_tactile_encoder,
            )

        # Extract features (using actor's independent encoder)
        image_features = self._encode_dsrl_image_views(images, self.actor_image_encoder)
        state_features = self.actor_state_encoder(states)  # [B, 64]
        features = [state_features, image_features]
        if tactile is not None:
            features.append(self.actor_tactile_encoder(tactile))
        features = torch.cat(features, dim=-1)

        # Sample from GaussianPolicy
        mode = kwargs.get("mode", "train")
        deterministic = mode == "eval"

        action_noise, logprobs = self.dsrl_action_noise_net.sample(
            features, deterministic=deterministic
        )

        # Optional: return distribution parameters for logging
        dist_params = None
        if return_dist_params:
            dist = self.dsrl_action_noise_net.forward(features)
            dist_params = (dist.mean, dist.stddev)

        return action_noise, logprobs, dist_params

    def sac_q_forward(
        self,
        obs=None,
        data=None,
        actions=None,
        detach_encoder=False,
        train=False,
        **kwargs,
    ):
        """Q-value forward pass for DSRL.

        Args:
            obs: Observation dict (preferred, matches sac_dsrl).
                 Supports two formats:
                   1. {"images": list of tensors, "states": tensor} - internal format
                   2. {"main_images": tensor, "wrist_images": tensor, "states": tensor} - env format
            data: Dictionary containing observations (legacy, for backward compatibility).
            actions: [B, action_dim] or [B, action_horizon, action_dim]
            detach_encoder: Whether to detach encoder gradients.
            train: Whether to use data augmentation.

        Returns:
            q_values: [B, num_q_heads] - Q-values from all Q-networks.
        """
        if not self.config.use_dsrl:
            raise ValueError("sac_q_forward called but use_dsrl=False")

        # Support both call styles: obs (new, from sac_dsrl) or data (legacy)
        if obs is None:
            obs = data.get("obs", data) if data is not None else kwargs.get("obs", {})
        if actions is None:
            actions = kwargs.get("actions")

        obs = self._normalize_dsrl_obs(obs)

        # Preprocess ordered image views independently.
        # Returns [B, N, C, 64, 64] in [-1, 1] range (float32).
        images = self._prepare_dsrl_images(obs, train=train)
        states = self._preprocess_states(obs["states"])

        # Move to the same device as critic encoders, convert to bfloat16
        device = next(self.critic_image_encoder.parameters()).device
        images = images.to(device=device, dtype=torch.bfloat16)
        states = states.to(device=device, dtype=torch.bfloat16)
        actions = actions.to(device=device, dtype=torch.bfloat16)
        tactile = None
        if self.config.dsrl_use_tactile:
            tactile = self._prepare_dsrl_tactile(
                obs,
                batch_size=states.shape[0],
                encoder=self.critic_tactile_encoder,
            )

        # Extract features (using critic's independent encoder)
        image_features = self._encode_dsrl_image_views(
            images, self.critic_image_encoder
        )
        state_features = self.critic_state_encoder(states)
        tactile_features = None
        if tactile is not None:
            tactile_features = self.critic_tactile_encoder(tactile)

        # Optionally detach encoder
        if detach_encoder:
            image_features = image_features.detach()
            state_features = state_features.detach()
            if tactile_features is not None:
                tactile_features = tactile_features.detach()

        if tactile_features is not None:
            state_features = torch.cat([state_features, tactile_features], dim=-1)

        # Process actions (DSRL: should be noise, already flattened)
        if actions.dim() == 3:
            actions = actions[:, 0, :]  # [B, action_horizon, dim] -> [B, dim]

        # Compute Q values
        q_values = self.q_head(state_features, image_features, actions)

        return q_values

    # ===== NFT-specific methods =====

    def _init_nft_state(
        self,
        collect_nft_state: bool,
        x_t: torch.Tensor,
        num_steps: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor] | None:
        """Initialize NFT state buffers for rollout sampling."""
        if not collect_nft_state:
            return None
        return {
            "nft_step_index": torch.randint(
                0, num_steps, (x_t.shape[0],), device=device
            ),
            "nft_xcur": torch.zeros_like(x_t),
            "nft_v": torch.zeros_like(x_t),
            "nft_xnext": torch.zeros_like(x_t),
            "nft_noise_level": torch.zeros(
                x_t.shape[0], device=device, dtype=x_t.dtype
            ),
        }

    def _update_nft_state(
        self,
        nft_state: dict[str, torch.Tensor] | None,
        idx: int,
        x_t_prev: torch.Tensor,
        v_t: torch.Tensor,
        x_t: torch.Tensor,
        sample_method: str,
    ) -> None:
        """Update NFT state buffers for the selected denoising step."""
        if nft_state is None:
            return
        mask = nft_state["nft_step_index"] == idx
        if not mask.any():
            return
        mask_bc = mask[:, None, None]
        nft_state["nft_xcur"] = torch.where(
            mask_bc, x_t_prev.detach(), nft_state["nft_xcur"]
        )
        nft_state["nft_v"] = torch.where(mask_bc, v_t.detach(), nft_state["nft_v"])
        nft_state["nft_xnext"] = torch.where(
            mask_bc, x_t.detach(), nft_state["nft_xnext"]
        )
        noise_level = self._get_noise_level(
            device=x_t.device, dtype=x_t.dtype, sample_method=sample_method
        )
        nft_state["nft_noise_level"] = torch.where(
            mask,
            torch.full_like(nft_state["nft_noise_level"], float(noise_level.item())),
            nft_state["nft_noise_level"],
        )

    def _get_noise_level(
        self, device: torch.device, dtype: torch.dtype, sample_method: str | None = None
    ) -> torch.Tensor:
        method = sample_method or self.config.noise_method
        if method == "flow_ode":
            return torch.zeros((), device=device, dtype=dtype)
        if self.config.noise_anneal:
            noise_start, noise_end, anneal_steps = self.config.noise_params
            noise_level = (
                noise_start
                + (noise_end - noise_start)
                * min(self.global_step, anneal_steps)
                / anneal_steps
            )
        else:
            noise_level = self.config.noise_level
        return torch.full((), noise_level, device=device, dtype=dtype)

    def _preprocess_dsrl_images(self, images, train=False):
        """Preprocess ordered DSRL views independently at 64x64.

        Args:
            images: List of tensors.
                Can be [B, H, W, C] (NHWC) from environment or
                [B, C, H, W] (NCHW) from processed data.
                For Libero: images[0] is agentview, images[1] is wrist.
            train: Whether to use data augmentation (placeholder for now).

        Returns:
            Tensor of shape [B, N, C, 64, 64], resized, in [-1, 1].
        """
        if not isinstance(images, (list, tuple)):
            images = [images]
        resized_views = []
        for image in images:
            if image.shape[-1] == 3:
                image = image.permute(0, 3, 1, 2)
            elif image.shape[1] != 3:
                raise ValueError(
                    "OpenPI DSRL image expected RGB in NHWC or NCHW; "
                    f"got {tuple(image.shape)}."
                )
            if image.dtype == torch.uint8:
                image = image.float() / 255.0
            else:
                image = image.float()
                if image.min() < 0:
                    image = (image + 1.0) / 2.0
            image = image.clamp(0.0, 1.0)
            image = F.interpolate(
                image,
                size=(64, 64),
                mode="bilinear",
                align_corners=False,
            )
            resized_views.append(image * 2.0 - 1.0)
        return torch.stack(resized_views, dim=1)

    @staticmethod
    def _encode_dsrl_image_views(images, encoder):
        """Share one image encoder across views, then concatenate by view order."""
        if images.ndim != 5:
            raise ValueError(
                f"OpenPI DSRL encoded images expected [B,N,C,H,W]; got {tuple(images.shape)}."
            )
        batch_size, num_images, channels, height, width = images.shape
        view_batch = images.reshape(batch_size * num_images, 1, channels, height, width)
        per_view_features = encoder(view_batch)
        return per_view_features.reshape(batch_size, num_images, -1).reshape(
            batch_size, -1
        )

    def _preprocess_states(self, states):
        """
        Preprocess states: flatten to 2D and convert to bfloat16.

        Args:
            states: [B, ...] any shape

        Returns:
            states: [B, state_dim] flattened states as bfloat16
        """
        if states.dim() > 2:
            states = states.reshape(states.shape[0], -1)
        # Convert to bfloat16 to match encoder's dtype
        if states.dtype != torch.bfloat16:
            states = states.to(torch.bfloat16)
        return states

    def enable_torch_compile(
        self,
        mode: str = "max-autotune",
    ):
        if self.torch_compile_enabled:
            return

        self.paligemma_with_expert.paligemma.model.vision_tower.forward = torch.compile(
            self.paligemma_with_expert.paligemma.model.vision_tower.forward, mode=mode
        )

        # NOTE: paligemma.model.language_model and gemma_expert.model share the same LLM backbone.
        # Enabling cuda graph on both simultaneously causes mysterious crashes (likely due to
        # tensor aliasing in the shared computation graph). We disable cuda graph for
        # paligemma.model.language_model since it is not CPU-bound, while gemma_expert.model
        # benefits more from cuda graph.
        self.paligemma_with_expert.paligemma.model.language_model.forward = (
            torch.compile(
                self.paligemma_with_expert.paligemma.model.language_model.forward,
                mode="max-autotune-no-cudagraphs" if mode == "max-autotune" else mode,
            )
        )
        self.paligemma_with_expert.gemma_expert.model.forward = torch.compile(
            self.paligemma_with_expert.gemma_expert.model.forward,
            mode=mode,
            fullgraph=True,
        )
        self.get_logprob_norm = torch.compile(
            self.get_logprob_norm,
            mode="max-autotune-no-cudagraphs" if mode == "max-autotune" else mode,
        )

        self.torch_compile_enabled = True
