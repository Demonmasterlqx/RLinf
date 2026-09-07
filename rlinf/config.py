# Copyright 2025 The RLinf Authors.
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

import dataclasses
import importlib.util
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Callable, ClassVar, Optional, Union

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict
from omegaconf.dictconfig import DictConfig

from rlinf.envs import SupportedEnvType
from rlinf.scheduler.cluster import Cluster
from rlinf.utils.dsrl_observation import (
    DSRL_NUM_IMAGES,
    DSRL_OBSERVATION_SEMANTICS,
    REALWORLD_TACIMG_DSRL_NUM_IMAGES,
    REALWORLD_TACIMG_DSRL_OBSERVATION_SEMANTICS,
)
from rlinf.utils.dsrl_replay import (
    DSRL_REPLAY_BACKEND,
    DSRL_REPLAY_CAPACITY_TRANSITIONS,
    DSRL_REPLAY_CHECKPOINT_SHARD_TRANSITIONS,
    DSRL_REPLAY_MAX_RESIDENT_GIB,
    DSRL_REPLAY_SEMANTICS,
    REALWORLD_TACIMG_DSRL_REPLAY_CAPACITY_TRANSITIONS,
    REALWORLD_TACIMG_DSRL_REPLAY_CHECKPOINT_SHARD_TRANSITIONS,
    REALWORLD_TACIMG_DSRL_REPLAY_MAX_RESIDENT_GIB,
    REALWORLD_TACIMG_DSRL_REPLAY_SEMANTICS,
)
from rlinf.utils.dsrl_reward import DSRL_REWARD_SEMANTICS
from rlinf.utils.dsrl_rollout_sync import validate_dsrl_rollout_sync_config
from rlinf.utils.dsrl_transition import (
    DSRL_TRANSITION_BOUNDARY_SEMANTICS,
    REALWORLD_TACIMG_DSRL_CHUNK_BOUNDARY_MODE,
    REALWORLD_TACIMG_DSRL_TRANSITION_BOUNDARY_SEMANTICS,
    TABERO_DSRL_CHUNK_BOUNDARY_MODE,
)
from rlinf.utils.placement import (
    HybridComponentPlacement,
    ModelParallelComponentPlacement,
    PlacementMode,
)
from rlinf.utils.tabero_ppo_boundary import (
    TABERO_PI05_TACFIELD_CONFIG_NAMES,
    TABERO_PI05_TACIMG_CONFIG_NAME,
    TABERO_PPO_BOUNDARY_CONTRACTS,
    TABERO_PPO_CHECKPOINT_METADATA_KEY,
    TABERO_PPO_DEFAULT_RESET_TRANSITION_BOUNDARY_SEMANTICS,
    TABERO_XARM_GRIPPER_MAPPING_CONTRACT,
    validate_tabero_pi05_pirl_deployment_checkpoint,
)

if TYPE_CHECKING:
    from megatron.core.model_parallel_config import ModelParallelConfig
    from megatron.core.transformer.transformer_config import TransformerConfig

logging.getLogger().setLevel(logging.INFO)


@dataclasses.dataclass(frozen=True)
class SupportedModel:
    value: str

    models: ClassVar[dict[str, "SupportedModel"]] = {}

    @classmethod
    def register(cls, value: str, force: bool = False) -> "SupportedModel":
        if not value:
            raise ValueError("model_type must be a non-empty string.")
        if value in cls.models:
            if not force:
                raise ValueError(
                    f"Model type `{value}` is already registered. "
                    "Set force=True to override it."
                )
        else:
            cls.models[value] = cls.__private_create__(value)
        return cls.models[value]

    @classmethod
    def get(cls, value: str) -> "SupportedModel":
        if value not in cls.models:
            supported_models = sorted(cls.models)
            raise NotImplementedError(
                f"Model Type: {value} not supported. Supported models: {supported_models}"
            )
        return cls.models[value]

    def __new__(cls, value: str):
        return cls.get(value)

    @classmethod
    def __private_create__(cls, value: str) -> "SupportedModel":
        obj = object.__new__(cls)
        object.__setattr__(obj, "value", value)
        return obj


SupportedModel.QWEN2_5 = SupportedModel.register("qwen2.5", force=True)
SupportedModel.QWEN2_5_VL = SupportedModel.register("qwen2.5_vl", force=True)
SupportedModel.QWEN3 = SupportedModel.register("qwen3", force=True)
SupportedModel.QWEN3_VL = SupportedModel.register("qwen3_vl", force=True)
SupportedModel.QWEN3_MOE = SupportedModel.register("qwen3_moe", force=True)
SupportedModel.OPENVLA = SupportedModel.register("openvla", force=True)
SupportedModel.OPENVLA_OFT = SupportedModel.register("openvla_oft", force=True)
SupportedModel.OPENPI = SupportedModel.register("openpi", force=True)
SupportedModel.OPENPI_PYTORCH = SupportedModel.register("openpi_pytorch", force=True)
SupportedModel.STARVLA = SupportedModel.register("starvla", force=True)
SupportedModel.MLP_POLICY = SupportedModel.register("mlp_policy", force=True)
SupportedModel.RLT_MLP_POLICY = SupportedModel.register("rlt_mlp_policy", force=True)
SupportedModel.GR00T = SupportedModel.register("gr00t", force=True)
SupportedModel.DEXBOTIC_PI = SupportedModel.register("dexbotic_pi", force=True)
SupportedModel.DEXBOTIC_DM0 = SupportedModel.register("dexbotic_dm0", force=True)
SupportedModel.DREAMZERO = SupportedModel.register("dreamzero", force=True)
SupportedModel.CNN_POLICY = SupportedModel.register("cnn_policy", force=True)
SupportedModel.FLOW_POLICY = SupportedModel.register("flow_policy", force=True)
SupportedModel.CMA_POLICY = SupportedModel.register("cma", force=True)
SupportedModel.LINGBOTVLA = SupportedModel.register("lingbotvla", force=True)
SupportedModel.ABOT_M0 = SupportedModel.register("abot_m0", force=True)
SupportedModel.RESNET_REWARD = SupportedModel.register("resnet", force=True)
SupportedModel.CFG_MODEL = SupportedModel.register("cfg_model", force=True)
SupportedModel.RECAP_VALUE_MODEL = SupportedModel.register(
    "recap_value_model", force=True
)
SupportedModel.STEAM_VALUE_MODEL = SupportedModel.register(
    "steam_value_model", force=True
)

SupportedModel.QWEN2_5_VL_SFT = SupportedModel.register("qwen2.5_vl", force=True)
SupportedModel.QWEN3_VL_SFT = SupportedModel.register("qwen3_vl", force=True)
SupportedModel.QWEN3_VL_MOE_SFT = SupportedModel.register("qwen3_vl_moe", force=True)
SupportedModel.GR00T_N1D6 = SupportedModel.register("gr00t_n1d6", force=True)
SupportedModel.GR00T_N1D7 = SupportedModel.register("gr00t_n1d7", force=True)

EMBODIED_MODEL = set(
    {
        SupportedModel.OPENVLA,
        SupportedModel.OPENVLA_OFT,
        SupportedModel.OPENPI,
        SupportedModel.OPENPI_PYTORCH,
        SupportedModel.STARVLA,
        SupportedModel.MLP_POLICY,
        SupportedModel.RLT_MLP_POLICY,
        SupportedModel.GR00T,
        SupportedModel.DEXBOTIC_PI,
        SupportedModel.DEXBOTIC_DM0,
        SupportedModel.DREAMZERO,
        SupportedModel.CNN_POLICY,
        SupportedModel.FLOW_POLICY,
        SupportedModel.CMA_POLICY,
        SupportedModel.LINGBOTVLA,
        SupportedModel.ABOT_M0,
        SupportedModel.RESNET_REWARD,
        SupportedModel.GR00T_N1D6,
        SupportedModel.GR00T_N1D7,
        SupportedModel.CFG_MODEL,
        SupportedModel.RECAP_VALUE_MODEL,
        SupportedModel.STEAM_VALUE_MODEL,
    }
)


SUPPORTED_ROLLOUT_BACKENDS = ["sglang", "vllm"]
SUPPORTED_TASK_TYPE = [
    "embodied",
    "embodied_eval",
    "reasoning",
    "reasoning_eval",
    "coding_online_rl",
    "sft",
    "offline",
]
SUPPORTED_TRAINING_BACKENDS = ["megatron", "fsdp"]
__all__ = ["build_config"]


def torch_dtype_from_precision(
    precision: Union[int, str, None],
) -> Optional[torch.dtype]:
    if precision in ["bf16", "bf16-mixed"]:
        return torch.bfloat16
    elif precision in [16, "16", "fp16", "16-mixed"]:
        return torch.float16
    elif precision in [32, "32", "fp32", "32-true"]:
        return torch.float32
    elif precision in [None, "null"]:
        return None
    else:
        raise ValueError(
            f"Could not parse the precision of `{precision}` to a valid torch.dtype"
        )


@torch.jit.script
def gelu_impl(x):
    """
    OpenAI's gelu implementation.
    """
    return (
        0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))
    )


def openai_gelu(x):
    return gelu_impl(x)


try:
    jit_fuser = torch.compile
except Exception:
    jit_fuser = torch.jit.script


@jit_fuser
def squared_relu(x):
    return torch.pow(torch.nn.functional.relu(x), 2)


# This is actually Python equivalent of torch.nn.functional.gelu(), also with type hints for ONNX exporter
@torch.jit.script
def erf_gelu(x):
    return (
        x
        * 0.5
        * (
            torch.erf(x / 1.41421).to(dtype=x.dtype)
            + torch.ones_like(x).to(dtype=x.dtype)
        )
    )


def activation_to_func(
    activation: str, openai_gelu: bool = False, onnx_safe: bool = False
) -> Callable:
    """
    Converts an activation function represented as a string to a function.

    Args:
        activation (str): string representation of an activation function, typically gotten from the model config.
        openai_gelu (bool): whether to use the OpenAI GELU implementation. Used with HF compatibility.
        onnx_safe (bool): whether to use the ONNX-compatible implementation of GELU.

    Returns:
        Callable: the activation function.
    """

    supported_activations = [
        "gelu",
        "geglu",
        "reglu",
        "swiglu",
        "squared-relu",
        "fast-geglu",
        "fast-swiglu",
        "fast-reglu",
        "approx-gelu",
    ]

    if activation not in supported_activations:
        raise ValueError(
            f"Unsupported activation {activation}. Supported activations: {supported_activations} "
        )

    # Give openai_gelu precedence over other activations if set, for HF compatibility.
    # Normally this is off and shouldn't affect regular model training.
    if openai_gelu:
        activation_func = openai_gelu
    elif activation in ["gelu", "geglu", "fast-geglu"]:
        activation_func = F.gelu
    elif onnx_safe:
        activation_func = erf_gelu
    elif activation in ["reglu", "fast-reglu"]:
        activation_func = F.relu
    elif activation in ["swiglu", "fast-swiglu"]:
        # SiLU or sigmoid linear unit is the same as swish with beta = 1 (which is what https://arxiv.org/pdf/2002.05202.pdf uses.)
        activation_func = F.silu
    elif activation == "squared-relu":
        activation_func = squared_relu

    return activation_func


def validate_rollout_cfg(cfg, algorithm_cfg):
    SupportedModel(cfg.model.model_type)  # To validate model_type is supported

    def validate_sglang_cfg(cfg):
        assert cfg is not None, (
            "sglang config must be specified if rollout_backend is sglang."
        )
        cfg.attention_backend = cfg.get("attention_backend", "triton")
        cfg.decode_log_interval = cfg.get("decode_log_interval", 500000)
        cfg.use_torch_compile = cfg.get("use_torch_compile", False)
        cfg.torch_compile_max_bs = cfg.get("torch_compile_max_bs", 128)
        return cfg

    def validate_vllm_cfg(cfg):
        assert cfg is not None, (
            "vllm config must be specified if rollout_backend is vllm."
        )
        cfg.attention_backend = cfg.get("attention_backend", "FLASH_ATTN")
        cfg.enable_chunked_prefill = cfg.get("enable_chunked_prefill", True)
        cfg.enable_prefix_caching = cfg.get("enable_prefix_caching", True)
        cfg.enable_flash_infer_sampler = cfg.get("enable_flash_infer_sampler", True)
        cfg.max_num_batched_tokens = cfg.get("max_num_batched_tokens", None)
        cfg.torch_profiler_dir = cfg.get("torch_profiler_dir", None)
        return cfg

    with open_dict(cfg):
        cfg.gpu_memory_utilization = cfg.get("gpu_memory_utilization", 0.65)
        assert cfg.model.model_path is not None, (
            "rollout.model.model_path must be specified for rollout."
        )

        cfg.disable_log_stats = cfg.get("disable_log_stats", False)
        cfg.detokenize = cfg.get("detokenize", False)
        cfg.rollout_backend = cfg.get("rollout_backend", "sglang")
        assert cfg.rollout_backend in SUPPORTED_ROLLOUT_BACKENDS, (
            f"rollout_backend must be one of {SUPPORTED_ROLLOUT_BACKENDS}."
        )
        cfg.return_logprobs = cfg.return_logprobs or algorithm_cfg.get(
            "importance_sampling_fix", False
        )
        cfg.sglang = validate_sglang_cfg(cfg.sglang)
        cfg.vllm = validate_vllm_cfg(cfg.vllm)

    return cfg


def validate_model_cfg_by_hf_config(cfg, hf_model_path):
    # validate by hf config
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)

    if (
        "Qwen2ForCausalLM" in hf_config.architectures
        or "Qwen2_5ForCausalLM" in hf_config.architectures
        or "Qwen2_5_VLForConditionalGeneration" in hf_config.architectures
    ):
        qkv_bias = True
    else:
        qkv_bias = getattr(hf_config, "attention_bias", False)

    if (
        "Qwen3ForCausalLM" in hf_config.architectures
        or "Qwen3MoeForCausalLM" in hf_config.architectures
        or "Qwen3VLForConditionalGeneration" in hf_config.architectures
        or "Qwen3VLMoeForConditionalGeneration" in hf_config.architectures
    ):
        qk_layernorm = True
    else:
        qk_layernorm = getattr(cfg.model, "qk_layernorm", False)

    with open_dict(cfg):
        rs = getattr(hf_config, "rope_scaling", None)
        if isinstance(rs, dict):
            rtype = rs.get("type", "")
            if rtype in {"linear", "dynamic", "ntk", "yarn"}:
                f = rs.get("factor")
                if f is not None:
                    cfg.model.seq_len_interpolation_factor = float(f)
            else:
                # mrope
                cfg.model.seq_len_interpolation_factor = None
        model_type = getattr(cfg.model, "model_type", None)
        if model_type == "qwen3_vl" or model_type == "qwen3_vl_moe":
            # qwen3_vl and qwen3_vl_moe config.json set the model config in text_config
            hf_config = hf_config.text_config

        cfg.model.padded_vocab_size = hf_config.vocab_size
        cfg.model.max_position_embeddings = hf_config.max_position_embeddings
        cfg.model.rotary_base = hf_config.rope_theta
        cfg.model.share_embeddings_and_output_weights = getattr(
            hf_config, "tie_word_embeddings", False
        )
        cfg.model.num_layers = hf_config.num_hidden_layers
        cfg.model.hidden_size = hf_config.hidden_size
        cfg.model.num_attention_heads = hf_config.num_attention_heads
        cfg.model.num_query_groups = hf_config.num_key_value_heads
        cfg.model.ffn_hidden_size = hf_config.intermediate_size
        cfg.model.attention_dropout = hf_config.attention_dropout
        cfg.model.hidden_dropout = getattr(hf_config, "hidden_dropout", 0.0)
        cfg.model.add_qkv_bias = qkv_bias
        cfg.model.qk_layernorm = qk_layernorm
        cfg.model.layernorm_epsilon = hf_config.rms_norm_eps
        cfg.model.head_dim = getattr(
            hf_config,
            "head_dim",
            cfg.model.hidden_size // cfg.model.num_attention_heads,
        )
        if cfg.model.head_dim is not None:
            cfg.model.kv_channels = cfg.model.head_dim

        # MoE model
        cfg.model.num_moe_experts = getattr(hf_config, "num_experts", None)
        cfg.model.num_experts = getattr(hf_config, "num_experts", None)
        cfg.model.moe_ffn_hidden_size = getattr(
            hf_config, "moe_intermediate_size", None
        )
        cfg.model.moe_router_topk = getattr(hf_config, "num_experts_per_tok", 2)

    return cfg


def validate_fsdp_cfg(cfg: DictConfig) -> DictConfig:
    def validate_amp_cfg(config: DictConfig) -> DictConfig:
        """Validate AMP configuration and ensure mutual exclusivity with FSDP mixed_precision."""

        param_dtype = config.mixed_precision.param_dtype
        reduce_dtype = config.mixed_precision.reduce_dtype
        buffer_dtype = config.mixed_precision.buffer_dtype

        all_none = param_dtype is None and reduce_dtype is None and buffer_dtype is None

        all_fp32 = (
            param_dtype == "fp32" and reduce_dtype == "fp32" and buffer_dtype == "fp32"
        )

        use_fsdp_mixed_precision = not (all_none or all_fp32)

        amp_autocast = config.get("amp_autocast", {})
        config.amp_autocast = {
            "enabled": amp_autocast.get("enabled", False),
            "precision": amp_autocast.get("precision", "bf16"),
        }

        grad_scaler = config.get("grad_scaler", {})
        config.grad_scaler = {
            "enabled": grad_scaler.get("enabled", False),
            "init_scale": grad_scaler.get("init_scale", None),
            "growth_interval": grad_scaler.get("growth_interval", None),
        }

        if "amp" in config:
            logging.warning(
                "fsdp_config.amp is no longer supported, use fsdp_config.amp_autocast and fsdp_config.grad_scaler instead"
            )

        if config.amp_autocast.enabled and use_fsdp_mixed_precision:
            assert False, (
                "amp_autocast should not be enabled when fsdp mixed_precision is enabled"
            )
        assert config.amp_autocast.precision in ["fp16", "bf16", "fp32"], (
            "fsdp.amp_autocast.precision must be one of ['fp16', 'bf16', 'fp32']"
        )
        return config

    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.fsdp_config.strategy = cfg.fsdp_config.get("strategy", "fsdp")

        cfg.fsdp_config.sharding_strategy = cfg.fsdp_config.get(
            "sharding_strategy", "full_shard"
        )

        cfg.fsdp_config.forward_prefetch = cfg.fsdp_config.get(
            "forward_prefetch", False
        )
        cfg.fsdp_config.limit_all_gathers = cfg.fsdp_config.get(
            "limit_all_gathers", False
        )
        cfg.fsdp_config.backward_prefetch = cfg.fsdp_config.get(
            "backward_prefetch", None
        )
        cfg.fsdp_config.use_orig_params = cfg.fsdp_config.get("use_orig_params", False)
        cfg.fsdp_config.use_liger_kernel = cfg.fsdp_config.get(
            "use_liger_kernel", False
        )

        cfg.fsdp_config.cpu_offload = cfg.fsdp_config.get("cpu_offload", False)
        cfg.fsdp_config.offload_pin_memory = cfg.fsdp_config.get(
            "offload_pin_memory", False
        )
        cfg.fsdp_config.reshard_after_forward = cfg.fsdp_config.get(
            "reshard_after_forward", True
        )
        cfg.fsdp_config.enable_gradient_accumulation = cfg.fsdp_config.get(
            "enable_gradient_accumulation", False
        )

        assert cfg.fsdp_config.backward_prefetch in [
            None,
            "pre",
            "post",
        ], "fsdp_config.backward_prefetch must be one of [None, 'pre', 'post']"

        # validate mixed precision config
        assert hasattr(cfg.fsdp_config, "mixed_precision"), (
            "fsdp_config.mixed_precision is required in FSDP actor configuration."
        )
        mixed_precision_config = cfg.fsdp_config.mixed_precision
        mixed_precision_config.param_dtype = mixed_precision_config.get(
            "param_dtype", None
        )
        mixed_precision_config.reduce_dtype = mixed_precision_config.get(
            "reduce_dtype", None
        )
        mixed_precision_config.buffer_dtype = mixed_precision_config.get(
            "buffer_dtype", None
        )
        cfg.fsdp_config = validate_amp_cfg(cfg.fsdp_config)

    return cfg


def validate_megatron_cfg(cfg: DictConfig) -> DictConfig:
    OmegaConf.set_struct(cfg, True)

    with open_dict(cfg):
        cfg.mcore_gpt = cfg.get("mcore_gpt", True)
        spec_name = cfg.get("spec_name", "local_gpt")
        cfg.spec_name = spec_name

        # Pad the vocab size to be divisible by this value.
        cfg.model.make_vocab_size_divisible_by = cfg.model.get(
            "make_vocab_size_divisible_by", 8
        )
        cfg.use_torch_fsdp2 = False

        # training args for megatron
        cfg.megatron.load = cfg.model.get("megatron_checkpoint", None)
        use_hf_ckpt = cfg.megatron.get("use_hf_ckpt", False)
        if cfg.megatron.load is None:
            assert use_hf_ckpt, (
                "model.megatron_checkpoint is required if use_hf_ckpt is False"
            )
        else:
            assert not use_hf_ckpt, (
                "model.megatron_checkpoint should be None if use_hf_ckpt is True"
            )
        cfg.megatron.pretrained_checkpoint = cfg.get("pretrained_checkpoint", None)
        cfg.megatron.save = None
        cfg.megatron.micro_batch_size = cfg.get("micro_batch_size", 1)
        cfg.megatron.global_batch_size = cfg.get("global_batch_size", 1)
        cfg.megatron.tp_comm_overlap_cfg = cfg.megatron.get("tp_comm_overlap_cfg", None)
        cfg.megatron.decoder_tp_comm_overlap = cfg.megatron.get(
            "decoder_tp_comm_overlap", False
        )
        cfg.megatron.timing_log_level = cfg.megatron.get(
            "timing_log_level", 0
        )  # choices=range(0,3)
        cfg.megatron.timing_log_option = cfg.megatron.get(
            "timing_log_option", "minmax"
        )  # choices=['max', 'minmax', 'all']

        # Megatron >= 0.12.0
        cfg.megatron.init_model_with_meta_device = cfg.megatron.get(
            "init_model_with_meta_device", False
        )
        cfg.megatron.use_torch_fsdp2 = cfg.megatron.get("use_torch_fsdp2", False)
        cfg.megatron.use_custom_fsdp = cfg.megatron.get("use_custom_fsdp", False)
        cfg.megatron.check_for_large_grads = cfg.megatron.get(
            "check_for_large_grads", False
        )
        cfg.megatron.ddp_num_buckets = cfg.megatron.get("ddp_num_buckets", None)
        cfg.megatron.ddp_pad_buckets_for_high_nccl_busbw = cfg.megatron.get(
            "ddp_pad_buckets_for_high_nccl_busbw", False
        )
        cfg.megatron.enable_gloo_process_groups = cfg.megatron.get(
            "enable_gloo_process_groups", True
        )

        #  megatron >= 0.15.0
        cfg.megatron.skip_train = cfg.megatron.get("skip_train", False)

        # ddp config
        cfg.megatron.check_for_nan_in_loss_and_grad = cfg.megatron.get(
            "check_for_nan_in_loss_and_grad", False
        )
        cfg.megatron.ddp_bucket_size = cfg.megatron.get("ddp_bucket_size", None)
        cfg.megatron.ddp_average_in_collective = cfg.megatron.get(
            "ddp_average_in_collective", False
        )
        cfg.megatron.accumulate_allreduce_grads_in_fp32 = cfg.megatron.get(
            "accumulate_allreduce_grads_in_fp32", True
        )

        # profiler
        cfg.megatron.use_profiler = cfg.megatron.get("use_profiler", False)
        if cfg.megatron.use_profiler:
            cfg.megatron.profiler.schedule_warmup = cfg.megatron.profiler.get(
                "schedule_warmup", 3
            )
            cfg.megatron.profiler.schedule_active = cfg.megatron.profiler.get(
                "schedule_active", 1
            )

        # distributed
        # If set, distributed ranks initialize order is changed from tp-cp-ep-dp-pp to tp-cp-ep-pp-dp.
        cfg.megatron.use_tp_pp_dp_mapping = cfg.megatron.get(
            "use_tp_pp_dp_mapping", False
        )
        # Which backend to use for distributed training. Support 'nccl' and 'gloo'
        cfg.megatron.distributed_backend = cfg.megatron.get(
            "distributed_backend", "nccl"
        )
        cfg.megatron.distributed_timeout_minutes = cfg.megatron.get(
            "distributed_timeout_minutes", 10
        )
        cfg.megatron.num_distributed_optimizer_instances = cfg.megatron.get(
            "num_distributed_optimizer_instances", 1
        )
        cfg.megatron.nccl_communicator_config_path = cfg.megatron.get(
            "nccl_communicator_config_path", None
        )
        cfg.megatron.encoder_tensor_model_parallel_size = cfg.megatron.get(
            "encoder_tensor_model_parallel_size", 0
        )
        cfg.megatron.encoder_pipeline_model_parallel_size = cfg.megatron.get(
            "encoder_pipeline_model_parallel_size", 0
        )

        # checkpoint
        cfg.megatron.rerun_mode = cfg.megatron.get(
            "rerun_mode", "disabled"
        )  # choices=['disabled', 'validate_results', 'report_stats']
        cfg.megatron.error_injection_rate = cfg.megatron.get(
            "error_injection_rate", 0
        )  # Rate at which to inject unexpected results, e.g. 1000 means once every 1000 result validations
        cfg.megatron.error_injection_type = cfg.megatron.get(
            "error_injection_type", "transient_error"
        )  # choices=['correct_result', 'transient_error', 'persistent_error']

        cfg.megatron.moe_use_upcycling = cfg.megatron.get("moe_use_upcycling", False)
        cfg.megatron.async_save = cfg.megatron.get("async_save", False)
        cfg.megatron.use_dist_ckpt = cfg.megatron.get("use_dist_ckpt", False)
        cfg.megatron.no_load_optim = cfg.megatron.get("no_load_optim", False)
        cfg.megatron.no_load_rng = cfg.megatron.get("no_load_rng", False)
        cfg.megatron.no_save_optim = cfg.megatron.get("no_save_optim", False)
        cfg.megatron.no_save_rng = cfg.megatron.get("no_save_rng", False)
        cfg.megatron.ckpt_fully_parallel_save = cfg.megatron.get(
            "ckpt_fully_parallel_save", False
        )
        cfg.megatron.ckpt_format = cfg.megatron.get("ckpt_format", "torch")
        cfg.megatron.ckpt_convert_format = cfg.megatron.get(
            "ckpt_convert_format", None
        )  # choices=[None, 'torch', 'torch_dist', 'zarr']
        cfg.megatron.auto_detect_ckpt_format = cfg.megatron.get(
            "auto_detect_ckpt_format", False
        )
        cfg.megatron.non_persistent_save_interval = cfg.megatron.get(
            "non_persistent_save_interval", None
        )
        cfg.megatron.non_persistent_ckpt_type = cfg.megatron.get(
            "non_persistent_ckpt_type", None
        )
        cfg.megatron.non_persistent_local_ckpt_dir = cfg.megatron.get(
            "non_persistent_local_ckpt_dir", None
        )
        cfg.megatron.non_persistent_global_ckpt_dir = cfg.megatron.get(
            "non_persistent_global_ckpt_dir", None
        )
        cfg.megatron.non_persistent_local_ckpt_algo = cfg.megatron.get(
            "non_persistent_local_ckpt_algo", "fully_parallel"
        )  # choices=['fully_parallel', 'atomic']
        cfg.megatron.ckpt_convert_update_legacy_dist_opt_format = cfg.megatron.get(
            "ckpt_convert_update_legacy_dist_opt_format", False
        )
        cfg.megatron.finetune = cfg.megatron.get("finetune", False)
        cfg.megatron.ckpt_assume_constant_structure = cfg.megatron.get(
            "ckpt_assume_constant_structure", False
        )
        cfg.megatron.log_progress = cfg.megatron.get("log_progress", False)
        cfg.megatron.exit_on_missing_checkpoint = cfg.megatron.get(
            "exit_on_missing_checkpoint", True
        )
        cfg.megatron.retro_add_retriever = cfg.megatron.get(
            "retro_add_retriever", False
        )
        cfg.megatron.data_parallel_random_init = cfg.megatron.get(
            "data_parallel_random_init", False
        )
        cfg.megatron.use_tokenizer_model_from_checkpoint_args = cfg.megatron.get(
            "use_tokenizer_model_from_checkpoint_args", False
        )

        # cfg.model
        assert (
            cfg.model.get("precision", None) is not None
            and torch_dtype_from_precision(cfg.model.precision) is not None
        ), "model.precision is required"

        cfg.model.tensor_model_parallel_size = cfg.model.get(
            "tensor_model_parallel_size", 1
        )
        cfg.model.pipeline_model_parallel_size = cfg.model.get(
            "pipeline_model_parallel_size", 1
        )
        cfg.model.virtual_pipeline_model_parallel_size = cfg.model.get(
            "virtual_pipeline_model_parallel_size", None
        )
        cfg.model.pipeline_model_parallel_split_rank = cfg.model.get(
            "pipeline_model_parallel_split_rank", None
        )
        cfg.model.context_parallel_size = cfg.model.get("context_parallel_size", 1)

        cfg.model.expert_model_parallel_size = cfg.model.get(
            "expert_model_parallel_size", 1
        )

        cfg.model.expert_tensor_parallel_size = cfg.model.get(
            "expert_tensor_parallel_size", None
        )

        from rlinf.hybrid_engines.megatron.megatron_model_manager import HAVE_FUSCO

        if HAVE_FUSCO:
            assert (
                cfg.model.moe_token_dispatcher_type == "alltoall"
                and cfg.model.expert_model_parallel_size > 1
                and cfg.model.expert_tensor_parallel_size == 1
                and not cfg.model.variable_seq_lengths
            ), (
                f"FUSCO support detected. to enable FUSCO, moe_token_dispatcher_type must be 'alltoall', expert_model_parallel_size must be greater than 1, expert_tensor_parallel_size must be 1, and variable_seq_lengths must be False. get value ({cfg.model.moe_token_dispatcher_type}, {cfg.model.expert_model_parallel_size}, {cfg.model.expert_tensor_parallel_size}, {cfg.model.variable_seq_lengths})"
            )

        cfg.model.moe_grouped_gemm = cfg.model.get("moe_grouped_gemm", None)
        assert cfg.model.moe_grouped_gemm in [None, "te"], (
            f"grouped_gemm type only avail in [null, te]. get value ({cfg.model.moe_grouped_gemm})"
        )

        if (
            not getattr(cfg.megatron, "mbridge", False)
            and cfg.model.expert_tensor_parallel_size is not None
        ):
            assert (
                cfg.model.expert_tensor_parallel_size
                <= cfg.model.tensor_model_parallel_size
            ), (
                f"expert_tensor_parallel_size ({cfg.model.expert_tensor_parallel_size}) must be less than or equal to tensor_model_parallel_size ({cfg.model.tensor_model_parallel_size})"
            )

        cfg.model.position_embedding_type = cfg.model.get(
            "position_embedding_type", "learned_absolute"
        )
        cfg.model.rotary_percentage = cfg.model.get("rotary_percentage", 1.0)
        cfg.model.seq_len_interpolation_factor = cfg.model.get(
            "seq_len_interpolation_factor", None
        )
        cfg.model.rotary_base = cfg.model.get("rotary_base", 10000)
        cfg.model.share_embeddings_and_output_weights = cfg.model.get(
            "share_embeddings_and_output_weights", False
        )

        cfg.model.gradient_accumulation_fusion = cfg.model.get(
            "gradient_accumulation_fusion", True
        )
        cfg.model.masked_softmax_fusion = cfg.model.get("masked_softmax_fusion", True)
        cfg.model.persist_layer_norm = cfg.model.get("persist_layer_norm", True)

        cfg.model.padded_vocab_size = cfg.model.get("padded_vocab_size", None)
        cfg.model.use_cpu_initialization = cfg.model.get(
            "use_cpu_initialization", False
        )
        cfg.model.add_position_embedding = cfg.model.get("add_position_embedding", True)

        cfg.model.variable_seq_lengths = cfg.model.get("variable_seq_lengths", True)
        cfg.model.add_bias_linear = cfg.model.get("add_bias_linear", False)

        # optimizer config
        if cfg.optim.get("fp16", None) is None:
            cfg.optim.fp16 = (
                torch_dtype_from_precision(cfg.model.precision) == torch.float16
            )
        if cfg.optim.get("bf16", None) is None:
            cfg.optim.bf16 = (
                torch_dtype_from_precision(cfg.model.precision) == torch.bfloat16
            )
        cfg.optim.weight_decay = cfg.optim.get("weight_decay", 0.01)
        cfg.optim.overlap_param_gather_with_optimizer_step = cfg.optim.get(
            "overlap_param_gather_with_optimizer_step", False
        )
        cfg.optim.optimizer_cpu_offload = cfg.optim.get("optimizer_cpu_offload", False)
        cfg.optim.optimizer_offload_fraction = cfg.optim.get(
            "optimizer_offload_fraction", 0.0
        )
        cfg.optim.use_precision_aware_optimizer = cfg.optim.get(
            "use_precision_aware_optimizer", False
        )

        # learning rate
        cfg.lr_sched.lr = cfg.optim.get("lr", None)
        cfg.lr_sched.min_lr = cfg.lr_sched.get("min_lr", 0.0)
        # lr_decay_style choices=['constant', 'linear', 'cosine', 'inverse-square-root', 'WSD']
        cfg.lr_sched.lr_decay_style = cfg.lr_sched.get("lr_decay_style", "constant")
        # weight_decay_incr_style: Weight decay increment function. choices=['constant', 'linear', 'cosine']
        cfg.lr_sched.weight_decay_incr_style = cfg.lr_sched.get(
            "weight_decay_incr_style", "constant"
        )
        # lr_wsd_decay_style choices=['exponential', 'linear', 'cosine']
        cfg.lr_sched.lr_wsd_decay_style = cfg.lr_sched.get(
            "lr_wsd_decay_style", "exponential"
        )

        # TODO fix this
        cfg.megatron.train_iters = 100000
        # lr_decay_iters: number of iterations to decay learning rate over, defaults to train_iters
        cfg.lr_sched.lr_decay_iters = cfg.lr_sched.get("lr_decay_iters", None)
        cfg.lr_sched.lr_wsd_decay_iters = cfg.lr_sched.get("lr_wsd_decay_iters", None)
        cfg.lr_sched.lr_warmup_init = cfg.lr_sched.get("lr_warmup_init", 0.0)
        cfg.lr_sched.lr_warmup_iters = cfg.lr_sched.get("lr_warmup_iters", 0)
        cfg.lr_sched.lr_warmup_fraction = cfg.lr_sched.get("lr_warmup_fraction", None)
        cfg.lr_sched.use_checkpoint_opt_param_scheduler = cfg.lr_sched.get(
            "use_checkpoint_opt_param_scheduler", True
        )
        cfg.lr_sched.override_opt_param_scheduler = cfg.lr_sched.get(
            "override_opt_param_scheduler", False
        )

        if cfg.lr_sched.lr_decay_style == "constant":
            assert cfg.lr_sched.get("start_weight_decay") is None
            assert cfg.lr_sched.get("end_weight_decay") is None
            cfg.lr_sched.start_weight_decay = cfg.optim.weight_decay
            cfg.lr_sched.end_weight_decay = cfg.optim.weight_decay
        else:
            if not hasattr(cfg.lr_sched, "start_weight_decay"):
                raise ValueError(
                    "Error: 'start_weight_decay' is missing from 'cfg.lr_sched'"
                )
            if not hasattr(cfg.lr_sched, "end_weight_decay"):
                raise ValueError(
                    "Error: 'end_weight_decay' is missing from 'cfg.lr_sched'"
                )
            assert cfg.lr_sched.start_weight_decay is not None
            assert cfg.lr_sched.end_weight_decay is not None

        # TODO. Following args are needed for AUTO mode now, but will be removed in the future.
        cfg.megatron.transformer_impl = getattr(
            cfg.megatron, "transformer_impl", "transformer_engine"
        )
        cfg.megatron.swiglu = cfg.model.activation in ["swiglu", "fast-swiglu"]
        cfg.megatron.untie_embeddings_and_output_weights = (
            not cfg.model.share_embeddings_and_output_weights
        )
        # In RLinf, padded_vocab_size is set to hf_config.vocab_size, so make_vocab_size_divisible_by=1
        cfg.megatron.make_vocab_size_divisible_by = 1
        if cfg.model.normalization == "rmsnorm":
            cfg.megatron.normalization = "RMSNorm"

    return cfg


def _validate_tabero_realworld_action_filter_contract(
    cfg,
    checkpoint_metadata,
) -> str:
    """Validate the opt-in action filter and return its auditable metadata."""

    enabled_by_split = {}
    required_enabled_params = {
        "transition_steps": 3,
        "max_position_step_m": 0.008,
        "max_position_delta_change_m": 0.006,
        "max_orientation_step_deg": 2.0,
        "max_orientation_delta_change_deg": 1.5,
    }
    for split_name in ("train", "eval"):
        init_params = cfg.env[split_name].get("init_params", {})
        action_filter_cfg = init_params.get("action_filter")
        if action_filter_cfg is None:
            enabled = False
        else:
            if not hasattr(action_filter_cfg, "get"):
                raise ValueError(
                    "RealWorld Tabero PI0.5 PiRL requires "
                    f"env.{split_name}.init_params.action_filter to be a mapping."
                )
            enabled = action_filter_cfg.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError(
                "RealWorld Tabero PI0.5 PiRL requires "
                f"env.{split_name}.init_params.action_filter.enabled to be boolean; "
                f"got {enabled!r}."
            )
        enabled_by_split[split_name] = enabled
        if enabled:
            for key, expected in required_enabled_params.items():
                actual = action_filter_cfg.get(key)
                if actual != expected:
                    raise ValueError(
                        "RealWorld Tabero PI0.5 PiRL enabled action filter requires "
                        f"env.{split_name}.init_params.action_filter.{key}="
                        f"{expected!r}; got {actual!r}."
                    )

    if enabled_by_split["train"] != enabled_by_split["eval"]:
        raise ValueError(
            "RealWorld Tabero PI0.5 PiRL requires train and eval to use the "
            "same action_filter.enabled state; got "
            f"train={enabled_by_split['train']!r}, "
            f"eval={enabled_by_split['eval']!r}."
        )

    expected_metadata = (
        "xarm_sim_action_chunk_filter_v1" if enabled_by_split["train"] else "disabled"
    )
    actual_metadata = checkpoint_metadata.get("action_filter")
    if actual_metadata != expected_metadata:
        raise ValueError(
            "RealWorld Tabero PI0.5 PiRL requires auditable "
            "actor.fsdp_config.trainable_checkpoint_metadata.action_filter="
            f"{expected_metadata!r}; got {actual_metadata!r}."
        )
    return expected_metadata


def _validate_tabero_realworld_gripper_checkpoint_metadata(
    checkpoint_metadata,
) -> dict[str, object]:
    """Require audit metadata for RLinf's fixed XArm gripper boundary."""

    metadata_contract = {
        "gripper_mapping": TABERO_XARM_GRIPPER_MAPPING_CONTRACT["version"],
        "policy_gripper_coordinate": TABERO_XARM_GRIPPER_MAPPING_CONTRACT[
            "policy_coordinate"
        ],
        "sim_gripper_coordinate": TABERO_XARM_GRIPPER_MAPPING_CONTRACT[
            "sim_coordinate"
        ],
        "gripper_travel_m": TABERO_XARM_GRIPPER_MAPPING_CONTRACT["travel_m"],
    }
    for key, expected in metadata_contract.items():
        actual = checkpoint_metadata.get(key)
        if actual != expected:
            raise ValueError(
                "RealWorld Tabero training requires auditable "
                "actor.fsdp_config.trainable_checkpoint_metadata."
                f"{key}={expected!r}; got {actual!r}."
            )
    return metadata_contract


def _validate_tabero_realworld_pi05_pirl_contract(cfg, model_cfg) -> None:
    """Fail before Ray starts when the RealWorld PI0.5 PiRL contract drifts."""

    required_model_values = {
        "num_action_chunks": 10,
        "action_dim": 13,
        "is_lora": True,
        "lora_target": "action_expert",
        "freeze_non_lora": True,
        "lora_path": None,
        "use_proprio": True,
        "num_steps": 10,
        "add_value_head": True,
        "precision": None,
        "frozen_parameter_precision": "bf16",
        "trainable_parameter_precision": "fp32",
    }
    for key, expected in required_model_values.items():
        actual = model_cfg.get(key)
        if actual != expected:
            raise ValueError(
                "RealWorld Tabero PI0.5 PiRL requires "
                f"actor.model.{key}={expected!r}; got {actual!r}."
            )

    allowed_missing_prefixes = model_cfg.get("checkpoint_load_allowed_missing_prefixes")
    if allowed_missing_prefixes is None or list(allowed_missing_prefixes) != [
        "value_head."
    ]:
        raise ValueError(
            "RealWorld Tabero PI0.5 PiRL requires "
            "actor.model.checkpoint_load_allowed_missing_prefixes="
            "['value_head.'] so every SFT tensor except the new critic is strict."
        )

    openpi_cfg = model_cfg.get("openpi", {})
    config_name = openpi_cfg.get("config_name")
    common_openpi_values = {
        "pi05": True,
        "discrete_state_input": True,
        "train_expert_only": True,
        "action_chunk": 10,
        "action_env_dim": 13,
        "effective_action_dim": 13,
        "tactile_type": "expert_his_c_fut",
        "tactile_dim": 6,
        "tactile_dim_in": 0,
        "num_steps": 10,
        "add_value_head": True,
        "joint_logprob": False,
        "value_after_vlm": True,
        "detach_critic_input": True,
        "use_dsrl": False,
    }
    if config_name in TABERO_PI05_TACFIELD_CONFIG_NAMES:
        tactile_kind = "tacfield"
        action_horizon = openpi_cfg.get("action_horizon")
        if type(action_horizon) is not int or action_horizon not in (10, 50):
            raise ValueError("RealWorld TacField PiRL action horizon must be 10 or 50.")
        required_openpi_values = {
            **common_openpi_values,
            "config_name": config_name,
            "action_horizon": action_horizon,
            "num_images_in_input": 2,
            "tactile_prefix_dim_in": 9 * 440 * 2,
            "tactile_prefix_history": 8,
            "tactile_prefix_encoder_type": "tcn",
            "tactile_prefix_use_reference_frame": True,
            "tactile_prefix_diff_from_reference": False,
        }
        expected_tactile_streams = ["tactile_prefix"]
    elif config_name == TABERO_PI05_TACIMG_CONFIG_NAME:
        tactile_kind = "tacimg"
        required_openpi_values = {
            **common_openpi_values,
            "config_name": TABERO_PI05_TACIMG_CONFIG_NAME,
            "action_horizon": 50,
            "num_images_in_input": 3,
            "tactile_prefix_dim_in": None,
            "tactile_prefix_history": None,
            "tactile_prefix_encoder_type": None,
            "tactile_prefix_use_reference_frame": None,
            "tactile_prefix_diff_from_reference": None,
        }
        expected_tactile_streams = []
    else:
        raise ValueError(
            "RealWorld Tabero PI0.5 PiRL requires a supported TacField or TacImg "
            f"OpenPI config; got {config_name!r}."
        )
    for key, expected in required_openpi_values.items():
        actual = openpi_cfg.get(key)
        if actual != expected:
            raise ValueError(
                "RealWorld Tabero PI0.5 PiRL requires "
                f"actor.model.openpi.{key}={expected!r}; got {actual!r}."
            )
    tactile_streams = openpi_cfg.get("tactile_streams")
    if tactile_streams is None or list(tactile_streams) != expected_tactile_streams:
        raise ValueError(
            "RealWorld Tabero PI0.5 PiRL tactile stream contract mismatch: "
            f"expected {expected_tactile_streams!r}, got {tactile_streams!r}."
        )

    openpi_data_cfg = model_cfg.get("openpi_data")
    if openpi_data_cfg is not None and openpi_data_cfg.get("norm_stats_path"):
        raise ValueError(
            "RealWorld Tabero PI0.5 PiRL forbids actor.model.openpi_data."
            "norm_stats_path; normalization must load from the audited checkpoint."
        )

    actor_model_path = str(model_cfg.get("model_path", ""))
    rollout_model_path = str(cfg.rollout.model.get("model_path", ""))
    if (
        not actor_model_path
        or not rollout_model_path
        or Path(actor_model_path).expanduser().resolve()
        != Path(rollout_model_path).expanduser().resolve()
    ):
        raise ValueError(
            "RealWorld Tabero PI0.5 PiRL requires actor and rollout to load the "
            "same local model_path."
        )

    checkpoint_contract = model_cfg.get("tabero_pi05_checkpoint_contract")
    if checkpoint_contract is None:
        raise ValueError(
            "RealWorld Tabero PI0.5 PiRL requires "
            "actor.model.tabero_pi05_checkpoint_contract."
        )
    require_final = checkpoint_contract.get("require_final")
    allow_non_final_formal_training = checkpoint_contract.get(
        "allow_non_final_formal_training", False
    )
    if not isinstance(allow_non_final_formal_training, bool):
        raise ValueError(
            "RealWorld Tabero PI0.5 PiRL requires "
            "actor.model.tabero_pi05_checkpoint_contract."
            "allow_non_final_formal_training to be boolean."
        )
    if require_final is True and allow_non_final_formal_training:
        raise ValueError(
            "RealWorld Tabero PI0.5 PiRL cannot combine require_final=true with "
            "allow_non_final_formal_training=true."
        )
    expected_config_name = checkpoint_contract.get("expected_config_name")
    expected_norm_asset_id = checkpoint_contract.get("expected_norm_asset_id")
    expected_dataset = checkpoint_contract.get("expected_dataset")
    expected_gripper_coordinate = checkpoint_contract.get("expected_gripper_coordinate")
    checkpoint_info = validate_tabero_pi05_pirl_deployment_checkpoint(
        actor_model_path,
        expected_model_sha256=checkpoint_contract.get("expected_model_sha256"),
        expected_norm_stats_sha256=checkpoint_contract.get(
            "expected_norm_stats_sha256"
        ),
        expected_config_name=expected_config_name,
        expected_norm_asset_id=expected_norm_asset_id,
        expected_dataset=expected_dataset,
        expected_gripper_coordinate=expected_gripper_coordinate,
        require_final=require_final,
        expected_action_horizon=openpi_cfg.get("action_horizon"),
    )
    if allow_non_final_formal_training and checkpoint_info["is_final"] is not False:
        raise ValueError(
            "RealWorld Tabero PI0.5 PiRL "
            "allow_non_final_formal_training=true is only valid for an explicitly "
            "non-final initialization checkpoint."
        )

    checkpoint_metadata = (
        cfg.actor.get("fsdp_config", {}).get("trainable_checkpoint_metadata", {}) or {}
    )
    action_filter_metadata = _validate_tabero_realworld_action_filter_contract(
        cfg,
        checkpoint_metadata,
    )
    gripper_mapping_metadata = _validate_tabero_realworld_gripper_checkpoint_metadata(
        checkpoint_metadata,
    )
    required_checkpoint_metadata = {
        "method": "pirl",
        "task_domain": "realworld",
        "task_suite": "gentle_grasp",
        "task_id": 6,
        "target_object": "target_object_1",
        "task_description": "pick up the Vitasoy and put it into the basket",
        "control_mode": "hybrid_tactile",
        "reset_source": "task_config_default_reset",
        "gripper_coordinate": expected_gripper_coordinate,
        "action_filter": action_filter_metadata,
        **gripper_mapping_metadata,
        "camera_preprocess": "stretch_480x640_to_224x224_inter_area",
        "model_family": "pi05",
        "openpi_config_name": expected_config_name,
        "dataset": expected_dataset,
        "normalization_asset_id": expected_norm_asset_id,
        "base_model_sha256": checkpoint_contract.get("expected_model_sha256"),
        "base_norm_stats_sha256": checkpoint_contract.get("expected_norm_stats_sha256"),
        "base_checkpoint_require_final": require_final,
        "action_horizon": openpi_cfg.get("action_horizon"),
        "execution_horizon": 10,
        "effective_action_dim": 13,
        "state_dim": 7,
        "camera_count": 3 if tactile_kind == "tacimg" else 2,
        "target_global_step": cfg.runner.get("max_epochs"),
        "tabero_ppo_transition_boundary_semantics": (
            TABERO_PPO_DEFAULT_RESET_TRANSITION_BOUNDARY_SEMANTICS
        ),
    }
    if tactile_kind == "tacfield":
        required_checkpoint_metadata.update(
            {
                "tactile_input": "tactile_prefix",
                "tactile_prefix_dim_in": 9 * 440 * 2,
                "tactile_prefix_history": 8,
                "combined_marker_count": 440,
            }
        )
    else:
        required_checkpoint_metadata.update(
            {
                "tactile_input": "tactile_image",
                "tactile_image_history": 8,
                "tactile_mosaic_layout": "left_2x4_then_right_2x4",
                "excluded_tactile_inputs": [
                    "tactile_gripper_force",
                    "tactile_marker_motion",
                ],
            }
        )
    if allow_non_final_formal_training:
        required_checkpoint_metadata.update(
            {
                "base_checkpoint_allow_non_final_formal_training": True,
                "base_checkpoint_global_step": checkpoint_info["global_step"],
                "base_checkpoint_target_global_step": checkpoint_info[
                    "target_global_step"
                ],
                "base_checkpoint_is_final": checkpoint_info["is_final"],
            }
        )
    for key, expected in required_checkpoint_metadata.items():
        actual = checkpoint_metadata.get(key)
        if actual != expected:
            raise ValueError(
                "RealWorld Tabero PI0.5 PiRL requires auditable "
                "actor.fsdp_config.trainable_checkpoint_metadata."
                f"{key}={expected!r}; got {actual!r}."
            )
    training_config = checkpoint_metadata.get("training_config")
    if not isinstance(training_config, str) or not training_config.strip():
        raise ValueError(
            "RealWorld Tabero PI0.5 PiRL checkpoint metadata requires a "
            "non-empty training_config."
        )

    is_restricted_smoke = (
        cfg.runner.get("max_epochs") == 1
        and cfg.env.train.get("total_num_envs") == 1
        and cfg.actor.get("global_batch_size") == 1
    )
    if (
        require_final is False
        and not is_restricted_smoke
        and not allow_non_final_formal_training
    ):
        raise ValueError(
            "A non-final PI0.5 tactile checkpoint is restricted to a one-update, "
            "single-environment PiRL smoke run unless the training config explicitly "
            "sets allow_non_final_formal_training=true and records its provenance."
        )

    expected_env_values = {
        "id": "Isaac-RealWorld-GentleGrasp-XarmUmi-Hybrid-Tactile-v0",
        "target_object": "target_object_1",
        "task_description": "pick up the Vitasoy and put it into the basket",
        "reset_source": "task_config_default_reset",
        "task_suite": "gentle_grasp",
        "task_id": 6,
        "tactile_backend": "taxim_fots",
        "chunk_boundary_mode": "terminal_safe_v1",
        "marker_history_len": 8,
        "combined_marker_count": 440,
    }
    if tactile_kind == "tacimg":
        expected_env_values["tactile_image_history_len"] = 8
    for split_name in ("train", "eval"):
        init_params = cfg.env[split_name].get("init_params", {})
        for key, expected in expected_env_values.items():
            actual = init_params.get(key)
            if actual != expected:
                raise ValueError(
                    "RealWorld Tabero PI0.5 PiRL requires "
                    f"env.{split_name}.init_params.{key}={expected!r}; "
                    f"got {actual!r}."
                )

        success_cfg = init_params.get("success", {})
        if (
            success_cfg.get("required_consecutive_steps") != 8
            or success_cfg.get("terminal_reward") != 1.0
        ):
            raise ValueError(
                "RealWorld Tabero PI0.5 PiRL requires eight consecutive success "
                "steps and terminal_reward=1.0."
            )
        camera_cfg = init_params.get("camera_preprocess", {})
        required_camera_preprocess = {
            "source_height": 480,
            "source_width": 640,
            "target_height": 224,
            "target_width": 224,
            "mode": "stretch",
            "interpolation": "INTER_AREA",
        }
        for key, expected in required_camera_preprocess.items():
            actual = camera_cfg.get(key)
            if actual != expected:
                raise ValueError(
                    "RealWorld Tabero PI0.5 PiRL requires "
                    f"env.{split_name}.init_params.camera_preprocess.{key}="
                    f"{expected!r}; got {actual!r}."
                )
        for legacy_camera_override in ("agentview_cam", "eye_in_hand_cam"):
            if init_params.get(legacy_camera_override) is not None:
                raise ValueError(
                    "RealWorld Tabero PI0.5 PiRL must preserve the native 480x640 "
                    f"camera before stretch preprocessing; remove env.{split_name}."
                    f"init_params.{legacy_camera_override}."
                )

        if (
            cfg.env[split_name].get("max_episode_steps") != 300
            or cfg.env[split_name].get("max_steps_per_rollout_epoch") != 300
        ):
            raise ValueError(
                "RealWorld Tabero PI0.5 PiRL control-loop parity requires exactly "
                f"300 primitive steps in env.{split_name}."
            )

        extension_path = (
            Path(str(init_params.get("extension_path", ""))).expanduser().resolve()
        )
        if not extension_path.is_dir() or extension_path.parts[-3:] != (
            "Tabero_X",
            "source",
            "tac_manip",
        ):
            raise ValueError(
                "RealWorld Tabero PI0.5 PiRL extension_path must resolve to "
                "Tabero_X/source/tac_manip."
            )
        for directory_key in ("realworld_config_dir", "realworld_assets_dir"):
            directory = (
                Path(str(init_params.get(directory_key, ""))).expanduser().resolve()
            )
            if not directory.is_dir():
                raise ValueError(
                    "RealWorld Tabero PI0.5 PiRL requires existing "
                    f"env.{split_name}.init_params.{directory_key}; got {directory}."
                )


def _validate_tabero_realworld_pi05_dsrl_contract(cfg, model_cfg) -> None:
    """Fail before Ray starts when the RealWorld TacImg DSRL contract drifts."""

    algorithm_cfg = cfg.algorithm
    required_algorithm_values = {
        "adv_type": "embodied_sac",
        "loss_type": "embodied_sac",
        "reward_type": "chunk_level",
        "logprob_type": "chunk_level",
        "dsrl_reward_semantics": DSRL_REWARD_SEMANTICS,
        "dsrl_observation_semantics": (REALWORLD_TACIMG_DSRL_OBSERVATION_SEMANTICS),
        "dsrl_replay_semantics": REALWORLD_TACIMG_DSRL_REPLAY_SEMANTICS,
        "dsrl_transition_boundary_semantics": (
            REALWORLD_TACIMG_DSRL_TRANSITION_BOUNDARY_SEMANTICS
        ),
    }
    for key, expected in required_algorithm_values.items():
        actual = algorithm_cfg.get(key)
        if actual != expected:
            raise ValueError(
                "RealWorld TacImg DSRL requires "
                f"algorithm.{key}={expected!r}; got {actual!r}."
            )
    if cfg.rollout.get("collect_transitions") is not True:
        raise ValueError(
            "RealWorld TacImg DSRL requires rollout.collect_transitions=true."
        )

    required_model_values = {
        "num_action_chunks": 10,
        "action_dim": 13,
        "is_lora": False,
        "use_proprio": True,
        "num_steps": 10,
        "add_value_head": False,
        "add_q_head": True,
        "q_head_type": "default",
        "num_q_heads": 10,
        "precision": None,
    }
    for key, expected in required_model_values.items():
        actual = model_cfg.get(key)
        if actual != expected:
            raise ValueError(
                "RealWorld TacImg DSRL requires "
                f"actor.model.{key}={expected!r}; got {actual!r}."
            )

    expected_missing_prefixes = [
        "dsrl_action_noise_net.",
        "actor_image_encoder.",
        "actor_state_encoder.",
        "critic_image_encoder.",
        "critic_state_encoder.",
        "q_head.",
    ]
    allowed_missing_prefixes = model_cfg.get("checkpoint_load_allowed_missing_prefixes")
    if (
        allowed_missing_prefixes is None
        or list(allowed_missing_prefixes) != expected_missing_prefixes
    ):
        raise ValueError(
            "RealWorld TacImg DSRL requires actor.model."
            "checkpoint_load_allowed_missing_prefixes to list exactly the newly "
            f"initialized DSRL modules; expected {expected_missing_prefixes!r}."
        )

    openpi_cfg = model_cfg.get("openpi", {})
    required_openpi_values = {
        "config_name": TABERO_PI05_TACIMG_CONFIG_NAME,
        "pi05": True,
        "action_horizon": 50,
        "discrete_state_input": True,
        "num_images_in_input": 3,
        "action_chunk": 10,
        "num_steps": 10,
        "train_expert_only": True,
        "action_env_dim": 13,
        "effective_action_dim": 13,
        "add_value_head": False,
        "joint_logprob": False,
        "detach_critic_input": True,
        "tactile_type": "expert_his_c_fut",
        "tactile_dim": 6,
        "tactile_dim_in": 0,
        "tactile_prefix_dim_in": None,
        "tactile_prefix_history": None,
        "tactile_prefix_encoder_type": None,
        "tactile_prefix_use_reference_frame": None,
        "tactile_prefix_diff_from_reference": None,
        "use_dsrl": True,
        "dsrl_use_tactile": False,
        "dsrl_num_images": REALWORLD_TACIMG_DSRL_NUM_IMAGES,
        "dsrl_state_dim": 7,
        "dsrl_action_noise_dim": 32,
        "dsrl_num_q_heads": 10,
        "dsrl_agg_q": "mean",
        "dsrl_image_latent_dim": 64,
        "dsrl_state_latent_dim": 64,
        "dsrl_tactile_latent_dim": 64,
    }
    for key, expected in required_openpi_values.items():
        actual = openpi_cfg.get(key)
        if actual != expected:
            raise ValueError(
                "RealWorld TacImg DSRL requires "
                f"actor.model.openpi.{key}={expected!r}; got {actual!r}."
            )
    if list(openpi_cfg.get("tactile_streams", [])) != []:
        raise ValueError(
            "RealWorld TacImg DSRL requires actor.model.openpi.tactile_streams=[]; "
            f"got {openpi_cfg.get('tactile_streams')!r}."
        )
    if list(openpi_cfg.get("dsrl_hidden_dims", [])) != [128, 128, 128]:
        raise ValueError(
            "RealWorld TacImg DSRL requires "
            "actor.model.openpi.dsrl_hidden_dims=[128, 128, 128]."
        )
    openpi_data_cfg = model_cfg.get("openpi_data")
    if openpi_data_cfg is not None and openpi_data_cfg.get("norm_stats_path"):
        raise ValueError(
            "RealWorld TacImg DSRL forbids actor.model.openpi_data.norm_stats_path; "
            "normalization must load from the audited checkpoint."
        )

    actor_model_path = str(model_cfg.get("model_path", ""))
    rollout_model_path = str(cfg.rollout.model.get("model_path", ""))
    if (
        not actor_model_path
        or not rollout_model_path
        or Path(actor_model_path).expanduser().resolve()
        != Path(rollout_model_path).expanduser().resolve()
    ):
        raise ValueError(
            "RealWorld TacImg DSRL requires actor and rollout to load the same "
            "local model_path."
        )

    checkpoint_contract = model_cfg.get("tabero_pi05_checkpoint_contract")
    if checkpoint_contract is None:
        raise ValueError(
            "RealWorld TacImg DSRL requires "
            "actor.model.tabero_pi05_checkpoint_contract."
        )
    require_final = checkpoint_contract.get("require_final")
    allow_non_final_formal_training = checkpoint_contract.get(
        "allow_non_final_formal_training", False
    )
    if not isinstance(allow_non_final_formal_training, bool):
        raise ValueError(
            "RealWorld TacImg DSRL requires allow_non_final_formal_training to "
            "be boolean."
        )
    if require_final is True and allow_non_final_formal_training:
        raise ValueError(
            "RealWorld TacImg DSRL cannot combine require_final=true with "
            "allow_non_final_formal_training=true."
        )
    expected_config_name = checkpoint_contract.get("expected_config_name")
    expected_norm_asset_id = checkpoint_contract.get("expected_norm_asset_id")
    expected_dataset = checkpoint_contract.get("expected_dataset")
    expected_gripper_coordinate = checkpoint_contract.get("expected_gripper_coordinate")
    checkpoint_info = validate_tabero_pi05_pirl_deployment_checkpoint(
        actor_model_path,
        expected_model_sha256=checkpoint_contract.get("expected_model_sha256"),
        expected_norm_stats_sha256=checkpoint_contract.get(
            "expected_norm_stats_sha256"
        ),
        expected_config_name=expected_config_name,
        expected_norm_asset_id=expected_norm_asset_id,
        expected_dataset=expected_dataset,
        expected_gripper_coordinate=expected_gripper_coordinate,
        require_final=require_final,
    )
    if allow_non_final_formal_training and checkpoint_info["is_final"] is not False:
        raise ValueError(
            "RealWorld TacImg DSRL allow_non_final_formal_training=true is only "
            "valid for an explicitly non-final initialization checkpoint."
        )

    replay_cfg = algorithm_cfg.get("replay_buffer", {})
    required_replay_values = {
        "backend": DSRL_REPLAY_BACKEND,
        "capacity_transitions": (REALWORLD_TACIMG_DSRL_REPLAY_CAPACITY_TRANSITIONS),
        "checkpoint_shard_transitions": (
            REALWORLD_TACIMG_DSRL_REPLAY_CHECKPOINT_SHARD_TRANSITIONS
        ),
        "max_resident_gib": REALWORLD_TACIMG_DSRL_REPLAY_MAX_RESIDENT_GIB,
    }
    for key, expected in required_replay_values.items():
        actual = replay_cfg.get(key)
        if actual != expected:
            raise ValueError(
                "RealWorld TacImg DSRL compact replay requires "
                f"algorithm.replay_buffer.{key}={expected!r}; got {actual!r}."
            )
    legacy_replay_fields = {
        "enable_cache",
        "cache_size",
        "sample_window_size",
        "auto_save",
        "auto_save_path",
        "trajectory_format",
    }
    configured_legacy_fields = sorted(legacy_replay_fields.intersection(replay_cfg))
    if configured_legacy_fields:
        raise ValueError(
            "RealWorld TacImg DSRL compact replay forbids legacy trajectory-buffer "
            f"fields: {configured_legacy_fields}."
        )
    if algorithm_cfg.get("demo_buffer") is not None:
        raise ValueError(
            "RealWorld TacImg DSRL compact replay does not support demo_buffer."
        )

    fsdp_cfg = cfg.actor.get("fsdp_config", {})
    required_fsdp_values = {
        "sharding_strategy": "no_shard",
        "gradient_checkpointing": False,
        "use_orig_params": True,
        "checkpoint_format": "local_shard",
        "save_full_model_weights": False,
        "save_trainable_model_weights": True,
    }
    for key, expected in required_fsdp_values.items():
        actual = fsdp_cfg.get(key)
        if actual != expected:
            raise ValueError(
                "RealWorld TacImg DSRL requires "
                f"actor.fsdp_config.{key}={expected!r}; got {actual!r}."
            )

    checkpoint_metadata = fsdp_cfg.get("trainable_checkpoint_metadata", {}) or {}
    action_filter_metadata = _validate_tabero_realworld_action_filter_contract(
        cfg, checkpoint_metadata
    )
    gripper_mapping_metadata = _validate_tabero_realworld_gripper_checkpoint_metadata(
        checkpoint_metadata,
    )
    required_checkpoint_metadata = {
        "method": "dsrl",
        "task_domain": "realworld",
        "task_suite": "gentle_grasp",
        "task_id": 6,
        "target_object": "target_object_1",
        "task_description": "pick up the Vitasoy and put it into the basket",
        "control_mode": "hybrid_tactile",
        "reset_source": "task_config_default_reset",
        "gripper_coordinate": expected_gripper_coordinate,
        "action_filter": action_filter_metadata,
        **gripper_mapping_metadata,
        "camera_preprocess": "stretch_480x640_to_224x224_inter_area",
        "model_family": "pi05",
        "openpi_config_name": expected_config_name,
        "dataset": expected_dataset,
        "normalization_asset_id": expected_norm_asset_id,
        "base_model_sha256": checkpoint_contract.get("expected_model_sha256"),
        "base_norm_stats_sha256": checkpoint_contract.get("expected_norm_stats_sha256"),
        "base_checkpoint_require_final": require_final,
        "action_horizon": 50,
        "execution_horizon": 10,
        "effective_action_dim": 13,
        "state_dim": 7,
        "camera_count": 3,
        "tactile_input": "tactile_image",
        "tactile_image_history": 8,
        "tactile_mosaic_layout": "left_2x4_then_right_2x4",
        "excluded_tactile_inputs": [
            "tactile_gripper_force",
            "tactile_marker_motion",
        ],
        "reward_contract": "binary_success_plus_inverse_measured_force_v1",
        "force_reward_source": "policy_gripper_net_force",
        "dsrl_reward_semantics": DSRL_REWARD_SEMANTICS,
        "dsrl_observation_semantics": (REALWORLD_TACIMG_DSRL_OBSERVATION_SEMANTICS),
        "dsrl_replay_semantics": REALWORLD_TACIMG_DSRL_REPLAY_SEMANTICS,
        "dsrl_transition_boundary_semantics": (
            REALWORLD_TACIMG_DSRL_TRANSITION_BOUNDARY_SEMANTICS
        ),
        "target_global_step": cfg.runner.get("max_epochs"),
    }
    if allow_non_final_formal_training:
        required_checkpoint_metadata.update(
            {
                "base_checkpoint_allow_non_final_formal_training": True,
                "base_checkpoint_global_step": checkpoint_info["global_step"],
                "base_checkpoint_target_global_step": checkpoint_info[
                    "target_global_step"
                ],
                "base_checkpoint_is_final": checkpoint_info["is_final"],
            }
        )
    for key, expected in required_checkpoint_metadata.items():
        actual = checkpoint_metadata.get(key)
        if actual != expected:
            raise ValueError(
                "RealWorld TacImg DSRL requires auditable "
                "actor.fsdp_config.trainable_checkpoint_metadata."
                f"{key}={expected!r}; got {actual!r}."
            )
    training_config = checkpoint_metadata.get("training_config")
    if not isinstance(training_config, str) or not training_config.strip():
        raise ValueError(
            "RealWorld TacImg DSRL checkpoint metadata requires a non-empty "
            "training_config."
        )

    is_restricted_smoke = (
        cfg.runner.get("max_epochs") == 1
        and cfg.env.train.get("total_num_envs") == 1
        and cfg.actor.get("global_batch_size") == 1
    )
    if (
        require_final is False
        and not is_restricted_smoke
        and not allow_non_final_formal_training
    ):
        raise ValueError(
            "A non-final PI0.5 tactile checkpoint is restricted to a one-update, "
            "single-environment DSRL smoke unless the config explicitly allows "
            "and records non-final formal initialization."
        )

    expected_env_values = {
        "id": "Isaac-RealWorld-GentleGrasp-XarmUmi-Hybrid-Tactile-v0",
        "target_object": "target_object_1",
        "task_description": "pick up the Vitasoy and put it into the basket",
        "reset_source": "task_config_default_reset",
        "task_suite": "gentle_grasp",
        "task_id": 6,
        "tactile_backend": "taxim_fots",
        "chunk_boundary_mode": REALWORLD_TACIMG_DSRL_CHUNK_BOUNDARY_MODE,
        "marker_history_len": 8,
        "combined_marker_count": 440,
        "tactile_image_history_len": 8,
    }
    required_camera_preprocess = {
        "source_height": 480,
        "source_width": 640,
        "target_height": 224,
        "target_width": 224,
        "mode": "stretch",
        "interpolation": "INTER_AREA",
    }
    for split_name in ("train", "eval"):
        split_cfg = cfg.env.get(split_name)
        if split_cfg is None:
            raise ValueError(
                "RealWorld TacImg DSRL requires both env.train and env.eval."
            )
        if split_cfg.get("auto_reset") is not False:
            raise ValueError(
                f"RealWorld TacImg DSRL requires env.{split_name}.auto_reset=false."
            )
        if split_cfg.get("ignore_terminations") is not False:
            raise ValueError(
                "RealWorld TacImg DSRL requires "
                f"env.{split_name}.ignore_terminations=false."
            )
        init_params = split_cfg.get("init_params", {})
        for key, expected in expected_env_values.items():
            actual = init_params.get(key)
            if actual != expected:
                raise ValueError(
                    "RealWorld TacImg DSRL requires "
                    f"env.{split_name}.init_params.{key}={expected!r}; "
                    f"got {actual!r}."
                )
        success_cfg = init_params.get("success", {})
        if (
            success_cfg.get("required_consecutive_steps") != 8
            or success_cfg.get("terminal_reward") != 1.0
        ):
            raise ValueError(
                "RealWorld TacImg DSRL requires eight consecutive success steps "
                "and terminal_reward=1.0."
            )
        force_bonus_cfg = success_cfg.get("force_bonus", {})
        required_force_bonus = {
            "enabled": True,
            "coefficient": 20.0,
            "epsilon": 1.0,
            "max_bonus": 1.0,
            "min_valid_samples": 4,
            "contact_epsilon": 1.0,
        }
        for key, expected in required_force_bonus.items():
            actual = force_bonus_cfg.get(key)
            if actual != expected:
                raise ValueError(
                    "RealWorld TacImg DSRL force reward requires "
                    f"env.{split_name}.init_params.success.force_bonus.{key}="
                    f"{expected!r}; got {actual!r}."
                )
        camera_cfg = init_params.get("camera_preprocess", {})
        for key, expected in required_camera_preprocess.items():
            actual = camera_cfg.get(key)
            if actual != expected:
                raise ValueError(
                    "RealWorld TacImg DSRL requires "
                    f"env.{split_name}.init_params.camera_preprocess.{key}="
                    f"{expected!r}; got {actual!r}."
                )
        if (
            split_cfg.get("max_episode_steps") != 300
            or split_cfg.get("max_steps_per_rollout_epoch") != 300
        ):
            raise ValueError(
                "RealWorld TacImg DSRL control-loop parity requires exactly 300 "
                f"primitive steps in env.{split_name}."
            )
        extension_path = (
            Path(str(init_params.get("extension_path", ""))).expanduser().resolve()
        )
        if not extension_path.is_dir() or extension_path.parts[-3:] != (
            "Tabero_X",
            "source",
            "tac_manip",
        ):
            raise ValueError(
                "RealWorld TacImg DSRL extension_path must resolve to "
                "Tabero_X/source/tac_manip."
            )
        for directory_key in ("realworld_config_dir", "realworld_assets_dir"):
            directory = (
                Path(str(init_params.get(directory_key, ""))).expanduser().resolve()
            )
            if not directory.is_dir():
                raise ValueError(
                    "RealWorld TacImg DSRL requires existing "
                    f"env.{split_name}.init_params.{directory_key}; got {directory}."
                )

    expected_global_batch = int(cfg.env.train.total_num_envs) * int(
        cfg.env.train.rollout_epoch
    )
    if cfg.actor.get("global_batch_size") != expected_global_batch:
        raise ValueError(
            "RealWorld TacImg DSRL requires actor.global_batch_size to equal "
            "env.train.total_num_envs * env.train.rollout_epoch; expected "
            f"{expected_global_batch}, got {cfg.actor.get('global_batch_size')!r}."
        )


def validate_embodied_cfg(cfg):
    only_eval = (
        cfg.runner.get("only_eval", False)
        or cfg.runner.get("task_type") == "embodied_eval"
    )
    model_cfg = cfg.rollout.model if only_eval else cfg.actor.model
    algorithm_cfg = cfg.get("algorithm", {}) or {}
    model_type = SupportedModel(model_cfg.model_type)
    assert model_type in EMBODIED_MODEL, (
        f"Model type: '{model_cfg.model_type}' is not an embodied model. "
        f"Supported embodied models: {sorted([x.value for x in EMBODIED_MODEL])}."
    )
    use_dsrl = model_cfg.get("openpi", {}).get("use_dsrl", False)
    ppo_boundary_semantics = algorithm_cfg.get(
        "tabero_ppo_transition_boundary_semantics"
    )
    if ppo_boundary_semantics is not None:
        boundary_contract = TABERO_PPO_BOUNDARY_CONTRACTS.get(ppo_boundary_semantics)
        if boundary_contract is None:
            raise ValueError(
                "Unsupported algorithm.tabero_ppo_transition_boundary_semantics "
                f"{ppo_boundary_semantics!r}; expected one of "
                f"{sorted(TABERO_PPO_BOUNDARY_CONTRACTS)!r}."
            )
        if only_eval:
            raise ValueError(
                "Tabero PPO transition boundary semantics is a training contract and "
                "cannot be declared by an embodied-eval-only config."
            )
        openpi_cfg = model_cfg.get("openpi", {})
        expected_openpi_configs = (
            {
                *TABERO_PI05_TACFIELD_CONFIG_NAMES,
                TABERO_PI05_TACIMG_CONFIG_NAME,
            }
            if ppo_boundary_semantics
            == TABERO_PPO_DEFAULT_RESET_TRANSITION_BOUNDARY_SEMANTICS
            else {"pi0_lora_tacfield_tabero"}
        )
        if (
            model_type != SupportedModel.OPENPI
            or openpi_cfg.get("config_name") not in expected_openpi_configs
        ):
            raise ValueError(
                "Tabero PPO transition boundary semantics requires the Tabero tactile "
                "OpenPI model configuration in "
                f"{sorted(expected_openpi_configs)!r}."
            )
        required_ppo_values = {
            "adv_type": "gae",
            "loss_type": "actor_critic",
            "reward_type": "chunk_level",
            "logprob_type": "chunk_level",
        }
        for key, expected in required_ppo_values.items():
            actual = algorithm_cfg.get(key)
            if actual != expected:
                raise ValueError(
                    "Tabero PPO transition boundary semantics requires "
                    f"algorithm.{key}={expected!r}; got {actual!r}."
                )

        action_chunk = model_cfg.get("num_action_chunks")
        if (
            isinstance(action_chunk, bool)
            or not isinstance(action_chunk, int)
            or action_chunk <= 0
        ):
            raise ValueError(
                "Tabero PPO transition boundary semantics requires a positive integer "
                "actor.model.num_action_chunks."
            )
        if (
            ppo_boundary_semantics
            == TABERO_PPO_DEFAULT_RESET_TRANSITION_BOUNDARY_SEMANTICS
        ):
            if action_chunk != 10 or model_cfg.get("action_dim") != 13:
                raise ValueError(
                    "Tabero default-reset PPO boundary semantics requires "
                    "actor.model.num_action_chunks=10 and actor.model.action_dim=13; "
                    f"got {action_chunk!r} and {model_cfg.get('action_dim')!r}."
                )

        for split_name in ("train", "eval"):
            split_cfg = cfg.env.get(split_name)
            if split_cfg is None:
                raise ValueError(
                    "Tabero PPO transition boundary semantics requires both env.train "
                    "and env.eval configurations."
                )
            if split_cfg.get("auto_reset") is not False:
                raise ValueError(
                    f"Tabero PPO transition boundary semantics requires env.{split_name}."
                    "auto_reset=false."
                )
            if split_cfg.get("ignore_terminations") is not False:
                raise ValueError(
                    f"Tabero PPO transition boundary semantics requires env.{split_name}."
                    "ignore_terminations=false."
                )
            init_params = split_cfg.get("init_params", {})
            boundary_mode = init_params.get("chunk_boundary_mode")
            expected_boundary_mode = boundary_contract["chunk_boundary_mode"]
            if boundary_mode != expected_boundary_mode:
                raise ValueError(
                    f"Tabero PPO transition boundary semantics requires env.{split_name}."
                    "init_params.chunk_boundary_mode="
                    f"{expected_boundary_mode!r}; got {boundary_mode!r}."
                )
            hdf5_path = init_params.get("hdf5_initial_states_path")
            if boundary_contract["requires_hdf5"]:
                if not isinstance(hdf5_path, str) or not hdf5_path.strip():
                    raise ValueError(
                        f"Tabero PPO transition boundary semantics requires env.{split_name}."
                        "init_params.hdf5_initial_states_path."
                    )
                if init_params.get("hdf5_reset_assignment") != "cyclic":
                    raise ValueError(
                        f"Tabero PPO transition boundary semantics requires env.{split_name}."
                        "init_params.hdf5_reset_assignment='cyclic'."
                    )
            elif (
                hdf5_path is not None
                or init_params.get("hdf5_reset_assignment") is not None
            ):
                raise ValueError(
                    "Tabero default-reset PPO boundary semantics forbids "
                    f"env.{split_name} HDF5 reset configuration."
                )
            max_episode_steps = split_cfg.get("max_episode_steps")
            if (
                isinstance(max_episode_steps, bool)
                or not isinstance(max_episode_steps, int)
                or max_episode_steps <= 0
                or max_episode_steps % action_chunk != 0
            ):
                raise ValueError(
                    f"Tabero PPO transition boundary semantics requires env.{split_name}."
                    "max_episode_steps to be a positive multiple of "
                    f"actor.model.num_action_chunks={action_chunk}; got "
                    f"{max_episode_steps!r}."
                )

        if (
            ppo_boundary_semantics
            == TABERO_PPO_DEFAULT_RESET_TRANSITION_BOUNDARY_SEMANTICS
        ):
            _validate_tabero_realworld_pi05_pirl_contract(cfg, model_cfg)

        fsdp_cfg = cfg.actor.get("fsdp_config", {})
        if fsdp_cfg.get("save_trainable_model_weights") is not True:
            raise ValueError(
                "Tabero PPO transition boundary semantics requires "
                "actor.fsdp_config.save_trainable_model_weights=true for resume audit."
            )
        checkpoint_metadata = fsdp_cfg.get("trainable_checkpoint_metadata")
        actual_checkpoint_semantics = (
            checkpoint_metadata.get(TABERO_PPO_CHECKPOINT_METADATA_KEY)
            if checkpoint_metadata is not None
            else None
        )
        if actual_checkpoint_semantics != ppo_boundary_semantics:
            raise ValueError(
                "Tabero PPO transition boundary semantics requires matching "
                "actor.fsdp_config.trainable_checkpoint_metadata."
                f"{TABERO_PPO_CHECKPOINT_METADATA_KEY}; got "
                f"{actual_checkpoint_semantics!r}."
            )
    if use_dsrl and model_cfg.get("is_lora", False):
        raise ValueError("OpenPI DSRL requires actor.model.is_lora=false.")
    if use_dsrl and not only_eval:
        validate_dsrl_rollout_sync_config(cfg.actor)
        openpi_cfg = model_cfg.get("openpi", {})
        is_tabero_tactile_dsrl = openpi_cfg.get(
            "config_name"
        ) == "pi0_lora_tacfield_tabero" and openpi_cfg.get("dsrl_use_tactile", False)
        is_realworld_tacimg_dsrl = (
            openpi_cfg.get("config_name") == TABERO_PI05_TACIMG_CONFIG_NAME
            and openpi_cfg.get("dsrl_num_images") == REALWORLD_TACIMG_DSRL_NUM_IMAGES
            and openpi_cfg.get("dsrl_use_tactile") is False
        )
        if is_tabero_tactile_dsrl:
            required_algorithm_values = {
                "adv_type": "embodied_sac",
                "loss_type": "embodied_sac",
            }
            for key, expected in required_algorithm_values.items():
                actual = algorithm_cfg.get(key)
                if actual != expected:
                    raise ValueError(
                        "Tabero tactile OpenPI DSRL requires "
                        f"algorithm.{key}={expected!r}; got {actual!r}."
                    )
            if cfg.rollout.get("collect_transitions") is not True:
                raise ValueError(
                    "Tabero tactile OpenPI DSRL requires "
                    "rollout.collect_transitions=true."
                )

            required_model_values = {
                "num_action_chunks": 10,
                "action_dim": 13,
                "num_q_heads": 10,
            }
            for key, expected in required_model_values.items():
                actual = model_cfg.get(key)
                if actual != expected:
                    raise ValueError(
                        "Tabero tactile OpenPI DSRL requires "
                        f"actor.model.{key}={expected}; got {actual!r}."
                    )
            required_openpi_values = {
                "action_chunk": 10,
                "action_env_dim": 13,
                "dsrl_state_dim": 7,
                "dsrl_action_noise_dim": 32,
                "dsrl_num_q_heads": 10,
            }
            for key, expected in required_openpi_values.items():
                actual = openpi_cfg.get(key)
                if actual != expected:
                    raise ValueError(
                        "Tabero tactile OpenPI DSRL requires "
                        f"actor.model.openpi.{key}={expected}; got {actual!r}."
                    )

            base_config_path = (
                Path(str(model_cfg.get("model_path", ""))) / "config.json"
            )
            if not base_config_path.is_file():
                raise ValueError(
                    "Tabero tactile OpenPI DSRL requires a readable base "
                    f"config.json at {base_config_path}."
                )
            try:
                with base_config_path.open(encoding="utf-8") as file:
                    base_model_config = json.load(file)
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(
                    "Tabero tactile OpenPI DSRL could not read base config.json "
                    f"at {base_config_path}: {error}"
                ) from error
            required_base_values = {"action_horizon": 50, "action_dim": 32}
            for key, expected in required_base_values.items():
                actual = base_model_config.get(key)
                if actual != expected:
                    raise ValueError(
                        "Tabero tactile OpenPI DSRL requires Pi0 base "
                        f"config.json {key}={expected}; got {actual!r}."
                    )

            reward_semantics = algorithm_cfg.get("dsrl_reward_semantics")
            if reward_semantics != DSRL_REWARD_SEMANTICS:
                raise ValueError(
                    "Tabero tactile OpenPI DSRL requires "
                    "algorithm.dsrl_reward_semantics="
                    f"{DSRL_REWARD_SEMANTICS!r}; got {reward_semantics!r}."
                )
            observation_semantics = algorithm_cfg.get("dsrl_observation_semantics")
            if observation_semantics != DSRL_OBSERVATION_SEMANTICS:
                raise ValueError(
                    "Tabero tactile OpenPI DSRL requires "
                    "algorithm.dsrl_observation_semantics="
                    f"{DSRL_OBSERVATION_SEMANTICS!r}; got "
                    f"{observation_semantics!r}."
                )
            replay_semantics = algorithm_cfg.get("dsrl_replay_semantics")
            if replay_semantics != DSRL_REPLAY_SEMANTICS:
                raise ValueError(
                    "Tabero tactile OpenPI DSRL requires "
                    "algorithm.dsrl_replay_semantics="
                    f"{DSRL_REPLAY_SEMANTICS!r}; got {replay_semantics!r}."
                )
            transition_semantics = algorithm_cfg.get(
                "dsrl_transition_boundary_semantics"
            )
            if transition_semantics != DSRL_TRANSITION_BOUNDARY_SEMANTICS:
                raise ValueError(
                    "Tabero tactile OpenPI DSRL requires "
                    "algorithm.dsrl_transition_boundary_semantics="
                    f"{DSRL_TRANSITION_BOUNDARY_SEMANTICS!r}; got "
                    f"{transition_semantics!r}."
                )
            chunk_boundary_mode = cfg.env.train.init_params.get("chunk_boundary_mode")
            if chunk_boundary_mode != TABERO_DSRL_CHUNK_BOUNDARY_MODE:
                raise ValueError(
                    "Tabero tactile OpenPI DSRL requires "
                    "env.train.init_params.chunk_boundary_mode="
                    f"{TABERO_DSRL_CHUNK_BOUNDARY_MODE!r}; got "
                    f"{chunk_boundary_mode!r}."
                )

            def validate_firm_prompt_config(split_name: str, init_params) -> None:
                prompt_cfg = init_params.get("prompt_conditions")
                if prompt_cfg is None or prompt_cfg.get("enabled") is not True:
                    raise ValueError(
                        "Tabero tactile OpenPI DSRL requires Firm prompt "
                        f"conditions for env.{split_name}."
                    )
                condition_cycle = list(prompt_cfg.get("condition_cycle", []))
                if condition_cycle != ["firm"]:
                    raise ValueError(
                        "Tabero tactile OpenPI DSRL requires "
                        f"env.{split_name}.init_params.prompt_conditions."
                        "condition_cycle=['firm']; got "
                        f"{condition_cycle!r}."
                    )
                firm_adverbs = list(prompt_cfg.get("firm_adverbs", []))
                if firm_adverbs != ["firmly", "tightly"]:
                    raise ValueError(
                        "Tabero tactile OpenPI DSRL requires "
                        f"env.{split_name}.init_params.prompt_conditions."
                        "firm_adverbs=['firmly', 'tightly']; got "
                        f"{firm_adverbs!r}."
                    )

            validate_firm_prompt_config("train", cfg.env.train.init_params)
            eval_cfg = cfg.env.get("eval")
            if eval_cfg is not None:
                eval_boundary_mode = eval_cfg.init_params.get("chunk_boundary_mode")
                if eval_boundary_mode != TABERO_DSRL_CHUNK_BOUNDARY_MODE:
                    raise ValueError(
                        "Tabero tactile OpenPI DSRL requires "
                        "env.eval.init_params.chunk_boundary_mode="
                        f"{TABERO_DSRL_CHUNK_BOUNDARY_MODE!r}; got "
                        f"{eval_boundary_mode!r}."
                    )
                validate_firm_prompt_config("eval", eval_cfg.init_params)
            replay_cfg = algorithm_cfg.get("replay_buffer", {})
            required_replay_values = {
                "backend": DSRL_REPLAY_BACKEND,
                "capacity_transitions": DSRL_REPLAY_CAPACITY_TRANSITIONS,
                "checkpoint_shard_transitions": (
                    DSRL_REPLAY_CHECKPOINT_SHARD_TRANSITIONS
                ),
                "max_resident_gib": DSRL_REPLAY_MAX_RESIDENT_GIB,
            }
            for key, expected in required_replay_values.items():
                actual = replay_cfg.get(key)
                if actual != expected:
                    raise ValueError(
                        "Tabero tactile OpenPI DSRL compact replay requires "
                        f"algorithm.replay_buffer.{key}={expected!r}; got "
                        f"{actual!r}."
                    )
            legacy_replay_fields = {
                "enable_cache",
                "cache_size",
                "sample_window_size",
                "auto_save",
                "auto_save_path",
                "trajectory_format",
            }
            configured_legacy_fields = sorted(
                legacy_replay_fields.intersection(replay_cfg)
            )
            if configured_legacy_fields:
                raise ValueError(
                    "Tabero tactile OpenPI DSRL compact replay forbids legacy "
                    f"trajectory-buffer fields: {configured_legacy_fields}."
                )
            if algorithm_cfg.get("demo_buffer") is not None:
                raise ValueError(
                    "Tabero tactile OpenPI DSRL compact replay does not support "
                    "demo_buffer/intervention trajectories."
                )
            dsrl_num_images = openpi_cfg.get("dsrl_num_images")
            if dsrl_num_images != DSRL_NUM_IMAGES:
                raise ValueError(
                    "Tabero tactile OpenPI DSRL requires "
                    f"actor.model.openpi.dsrl_num_images={DSRL_NUM_IMAGES}; "
                    f"got {dsrl_num_images!r}."
                )
        elif is_realworld_tacimg_dsrl:
            _validate_tabero_realworld_pi05_dsrl_contract(cfg, model_cfg)
        elif openpi_cfg.get("config_name") == TABERO_PI05_TACIMG_CONFIG_NAME:
            raise ValueError(
                "RealWorld TacImg DSRL requires dsrl_num_images=3 and "
                "dsrl_use_tactile=false so the tactile RGB mosaic is the third "
                "image view and marker motion is excluded."
            )
    with open_dict(cfg):
        cfg.runner.val_check_interval = cfg.runner.get("val_check_interval", -1)
    enable_eval = cfg.runner.val_check_interval > 0 or only_eval

    with open_dict(cfg):
        if enable_eval:
            assert cfg.env.get("eval", None) is not None, (
                "env.eval config is required when runner.val_check_interval > 0, "
                "runner.only_eval=True, or runner.task_type=embodied_eval."
            )
            cfg.env.eval.group_size = cfg.env.eval.get("group_size", 1)
        if algorithm_cfg.get("rollout_epoch", None) is not None:
            logging.warning(
                "algorithm.rollout_epoch is deprecated; use env.train.rollout_epoch instead."
            )
            if cfg.env.get("train", None) is not None:
                cfg.env.train.rollout_epoch = cfg.env.train.get(
                    "rollout_epoch", algorithm_cfg.rollout_epoch
                )
        if algorithm_cfg.get("eval_rollout_epoch", None) is not None:
            logging.warning(
                "algorithm.eval_rollout_epoch is deprecated; use env.eval.rollout_epoch instead."
            )
            if cfg.env.get("eval", None) is not None:
                cfg.env.eval.rollout_epoch = cfg.env.eval.get(
                    "rollout_epoch", algorithm_cfg.eval_rollout_epoch
                )
        if cfg.env.get("train", None) is not None:
            cfg.env.train.rollout_epoch = cfg.env.train.get("rollout_epoch", 1)
        if cfg.env.get("eval", None) is not None:
            cfg.env.eval.rollout_epoch = cfg.env.eval.get("rollout_epoch", 1)
        if cfg.rollout.get("sampling_params", None) is None:
            if algorithm_cfg.get("sampling_params", None) is not None:
                logging.warning(
                    "algorithm.sampling_params is deprecated for embodied tasks; use "
                    "rollout.sampling_params instead."
                )
                cfg.rollout.sampling_params = OmegaConf.create(
                    OmegaConf.to_container(algorithm_cfg.sampling_params, resolve=False)
                )
        elif algorithm_cfg.get("sampling_params", None) is not None:
            logging.warning(
                "algorithm.sampling_params is deprecated for embodied tasks; use "
                "rollout.sampling_params instead."
            )
        sampling_params = cfg.rollout.get("sampling_params", None)
        if sampling_params is not None:
            sampling_params.do_sample = sampling_params.get("do_sample", True)
            sampling_params.temperature_train = sampling_params.get(
                "temperature_train", sampling_params.get("temperature", 1.0)
            )
            sampling_params.temperature_eval = sampling_params.get(
                "temperature_eval", sampling_params.get("temperature", 0.0)
            )
            sampling_params.top_k = sampling_params.get("top_k", 0)
            sampling_params.top_p = sampling_params.get("top_p", 1.0)
            sampling_params.repetition_penalty = sampling_params.get(
                "repetition_penalty", 1.0
            )
            if sampling_params.get("max_new_tokens", None) is None:
                sampling_params.max_new_tokens = cfg.rollout.get("max_new_tokens", None)
        if algorithm_cfg.get("length_params", None) is not None:
            logging.warning(
                "algorithm.length_params is deprecated for embodied tasks; use "
                "rollout.sampling_params.max_new_tokens instead."
            )
            if sampling_params is None:
                cfg.rollout.sampling_params = OmegaConf.create({})
                sampling_params = cfg.rollout.sampling_params
            if sampling_params.get("max_new_tokens", None) is None:
                sampling_params.max_new_tokens = algorithm_cfg.length_params.get(
                    "max_new_token", None
                )

    if not only_eval and cfg.runner.get("use_training_pipeline", False):
        assert cfg.algorithm.adv_type == "gae", (
            "algorithm.adv_type only supports 'gae' now"
            "when runner.use_training_pipeline is True."
        )

    # NOTE: Currently we only support actor_critic as PPO algorithm loss, and only support value_head as critic model.
    # This will be updated in the future to support more algorithms and critic models.
    # Check that actor_critic loss requires value_head (training only; eval does not need critic)
    if not only_eval and (
        cfg.algorithm.loss_type == "actor_critic"
        or cfg.algorithm.loss_type == "decoupled_actor_critic"
    ):
        add_value_head = cfg.actor.model.get("add_value_head", False)
        assert add_value_head, (
            f"When using PPO algorithm (algorithm.loss_type='actor_critic'), "
            f"actor.model.add_value_head must be True. "
            f"Current value: {add_value_head}"
        )

    # process num-envs
    component_placement = HybridComponentPlacement(cfg, Cluster())
    stage_num = cfg.rollout.pipeline_stage_num
    env_world_size = component_placement.get_world_size("env")

    use_reward_model = cfg.get("reward", {}).get("use_reward_model", False)
    standalone_realworld = cfg.get("reward", {}).get("standalone_realworld", False)
    if use_reward_model and not standalone_realworld:
        assert stage_num == 1, (
            "use_reward_model requires rollout.pipeline_stage_num to be 1"
        )
        reward_worker_type = str(cfg.reward.get("worker_type", "model")).lower()
        assert reward_worker_type in {"model", "api"}, (
            "reward.worker_type must be either 'model' or 'api'."
        )
        reward_model_cfg = cfg.reward.get("model", {})
        if reward_worker_type == "api":
            assert reward_model_cfg.get("model_type") == "history_vlm", (
                "reward.worker_type='api' currently requires "
                "reward.model.model_type='history_vlm'."
            )
            api_cfg = cfg.reward.get("api", {})
            api_base = str(api_cfg.get("api_base") or "").strip()
            # Empty api_base means the trainer will call
            # launch_sglang_router_and_server with top-level router_server_args.
            if not api_base:
                assert "router_server_args" in cfg, (
                    "reward.worker_type='api' requires either reward.api.api_base or "
                    "the standard top-level router_server_args block for "
                    "Ray-managed SGLang."
                )
                assert "reward_server" in cfg.cluster.get("component_placement", {}), (
                    "Ray-managed SGLang reward API requires "
                    "cluster.component_placement.reward_server."
                )

    if cfg.runner.get("enable_decoupled_mode", False):
        assert stage_num == 1, (
            "enable_decoupled_mode requires rollout.pipeline_stage_num to be 1"
        )

    if enable_eval:
        assert cfg.env.get("eval", None) is not None, (
            "env.eval config is required when runner.val_check_interval > 0, "
            "runner.only_eval=True, or runner.task_type=embodied_eval."
        )
        assert cfg.env.eval.total_num_envs > 0, (
            "Total number of parallel environments for evaluation must be greater than 0"
        )
        assert cfg.env.eval.total_num_envs % env_world_size == 0, (
            "Total number of parallel environments for evaluation must be divisible by the number of environment processes"
        )
        assert cfg.env.eval.total_num_envs % env_world_size % stage_num == 0, (
            "Total number of parallel environments for evaluation must be divisible by the number of environment processes and the number of pipeline stages"
        )
        assert cfg.env.eval.total_num_envs // env_world_size // stage_num > 0, (
            "env.eval.total_num_envs // env_world_size // rollout.pipeline_stage_num must be greater than 0"
        )
        assert (
            cfg.env.eval.total_num_envs
            // env_world_size
            // stage_num
            % cfg.env.eval.group_size
            == 0
        ), (
            "env.eval.total_num_envs // env_world_size // rollout.pipeline_stage_num must be divisible by the group size"
        )
        assert (
            cfg.env.eval.max_steps_per_rollout_epoch % model_cfg.num_action_chunks == 0
        ), (
            "env.eval.max_steps_per_rollout_epoch must be divisible by actor.model.num_action_chunks"
        )

    if not only_eval:
        assert cfg.env.train.total_num_envs > 0, (
            "Total number of parallel environments for training must be greater than 0"
        )
        assert cfg.env.train.total_num_envs % env_world_size == 0, (
            "Total number of parallel environments for training must be divisible by the number of environment processes"
        )
        assert cfg.env.train.total_num_envs % env_world_size % stage_num == 0, (
            "Total number of parallel environments for training must be divisible by the number of environment processes and the number of pipeline stages"
        )
        assert cfg.env.train.total_num_envs // env_world_size // stage_num > 0, (
            "env.train.total_num_envs // env_world_size // rollout.pipeline_stage_num must be greater than 0"
        )
        assert (
            cfg.env.train.total_num_envs
            // env_world_size
            // stage_num
            % cfg.env.train.group_size
            == 0
        ), (
            "env.train.total_num_envs // env_world_size // rollout.pipeline_stage_num must be divisible by the group size"
        )
        assert (
            cfg.env.train.max_steps_per_rollout_epoch % model_cfg.num_action_chunks == 0
        ), (
            "env.train.max_steps_per_rollout_epoch must be divisible by actor.model.num_action_chunks"
        )
    with open_dict(cfg):
        weight_sync_interval = cfg.runner.get("weight_sync_interval", 1)
        assert weight_sync_interval > 0, "weight_sync_interval must be greater than 0"
        cfg.runner.weight_sync_interval = weight_sync_interval
        # Overlap environment bootstrap (reset) with actor training to hide reset latency.
        # This is enabled only when offload is disabled to avoid resource contention.
        # Note: If EnvWorker and Actor share the same accelerator, this may increase GPU memory
        # pressure during the overlap period.
        cfg.runner.overlap_env_bootstrap = bool(
            cfg.runner.get("overlap_env_bootstrap", False)
        ) and not cfg.env.get("train", {}).get("enable_offload", False)
        train_env_type = (
            SupportedEnvType(cfg.env.train.env_type)
            if cfg.env.get("train", None) is not None
            else None
        )
        eval_env_type = (
            SupportedEnvType(cfg.env.eval.env_type)
            if cfg.env.get("eval", None) is not None
            else None
        )
        if (
            train_env_type == SupportedEnvType.MANISKILL
            or eval_env_type == SupportedEnvType.MANISKILL
        ):

            def get_robot_control_mode(robot: str):
                if robot == "panda-qpos":
                    return "pd_joint_delta_pos"
                elif robot == "panda-ee-dpos":
                    return "pd_ee_delta_pos"
                elif robot == "panda-ee-target-dpos":  # for GSEnv
                    return "pd_ee_target_delta_pose"
                elif "google_robot_static" in robot:
                    return "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
                elif "widowx" in robot:
                    return "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
                elif "panda" in robot:
                    return "pd_ee_body_target_delta_pose_real_root_frame"
                else:
                    raise NotImplementedError(f"Robot {robot} not supported")

            if cfg.env.get("train", None) is not None:
                cfg.env.train.init_params.control_mode = get_robot_control_mode(
                    model_cfg.policy_setup
                )
            if cfg.env.get("eval", None) is not None:
                cfg.env.eval.init_params.control_mode = get_robot_control_mode(
                    model_cfg.policy_setup
                )
        elif (
            train_env_type == SupportedEnvType.BEHAVIOR
            or eval_env_type == SupportedEnvType.BEHAVIOR
        ):
            if cfg.env.get("train", None) is not None:
                assert cfg.env.train.base_config_name == "r1pro_behavior", (
                    f"Only r1pro_behavior is supported for omnigibson, got {cfg.env.train.base_config_name}"
                )
    return cfg


def validate_offline_cfg(cfg: DictConfig) -> DictConfig:
    """Validation for offline tasks (e.g. IQL).

    Requires explicit offline IQL runtime fields and validates their ranges.
    """
    actor_global = cfg.actor.get("global_batch_size", None)
    actor_micro = cfg.actor.get("micro_batch_size", None)
    runner_local_update_steps = cfg.runner.get("local_update_steps", None)
    runner_max_steps = cfg.runner.get("max_steps", None)
    runner_only_eval = cfg.runner.get("only_eval", None)
    algorithm_gamma = cfg.algorithm.get("gamma", None)
    actor_seed = cfg.actor.get("seed", None)
    if actor_global is None:
        raise AssertionError("offline training requires actor.global_batch_size")
    if actor_micro is None:
        raise AssertionError("offline training requires actor.micro_batch_size")
    if runner_local_update_steps is None:
        raise AssertionError("offline training requires runner.local_update_steps")
    if runner_max_steps is None:
        raise AssertionError("offline training requires runner.max_steps")
    if runner_only_eval is None:
        raise AssertionError("offline training requires runner.only_eval")
    if algorithm_gamma is None:
        raise AssertionError("offline training requires algorithm.gamma")
    if actor_seed is None:
        raise AssertionError("offline training requires actor.seed")

    actor_global_int = int(actor_global)
    actor_micro_int = int(actor_micro)
    runner_local_update_steps_int = int(runner_local_update_steps)
    runner_max_steps_int = int(runner_max_steps)
    try:
        float(algorithm_gamma)
    except (TypeError, ValueError) as exc:
        raise AssertionError(
            f"algorithm.gamma must be numeric, got {algorithm_gamma!r}"
        ) from exc
    try:
        int(actor_seed)
    except (TypeError, ValueError) as exc:
        raise AssertionError(
            f"actor.seed must be int-castable, got {actor_seed!r}"
        ) from exc

    assert actor_global_int > 0, (
        f"actor.global_batch_size must be > 0, got {actor_global_int}"
    )
    assert actor_micro_int > 0, (
        f"actor.micro_batch_size must be > 0, got {actor_micro_int}"
    )
    assert runner_local_update_steps_int > 0, (
        "runner.local_update_steps must be > 0 for offline training"
    )
    assert runner_max_steps_int >= 0, (
        f"runner.max_steps must be >= 0, got {runner_max_steps_int}"
    )
    assert actor_global_int >= actor_micro_int, (
        "actor.global_batch_size must be >= actor.micro_batch_size for offline training"
    )

    with open_dict(cfg):
        cfg.runner.only_eval = bool(runner_only_eval)

    if cfg.runner.val_check_interval > 0 or cfg.runner.only_eval:
        component_placement = HybridComponentPlacement(cfg, Cluster())
        stage_num = cfg.rollout.pipeline_stage_num
        env_world_size = component_placement.get_world_size("env")
        assert cfg.env.eval.total_num_envs > 0, (
            "Total number of parallel environments for evaluation must be greater than 0"
        )
        assert cfg.env.eval.total_num_envs % env_world_size == 0, (
            "Total number of parallel environments for evaluation must be divisible by the number of environment processes"
        )
        assert cfg.env.eval.total_num_envs % env_world_size % stage_num == 0, (
            "Total number of parallel environments for evaluation must be divisible by the number of environment processes and the number of pipeline stages"
        )
        assert cfg.env.eval.total_num_envs // env_world_size // stage_num > 0, (
            "env.eval.total_num_envs // env_world_size // rollout.pipeline_stage_num must be greater than 0"
        )
        assert (
            cfg.env.eval.total_num_envs
            // env_world_size
            // stage_num
            % cfg.env.eval.group_size
            == 0
        ), (
            "env.eval.total_num_envs // env_world_size // rollout.pipeline_stage_num must be divisible by the group size"
        )
        assert (
            cfg.env.eval.max_steps_per_rollout_epoch % cfg.actor.model.num_action_chunks
            == 0
        ), (
            "env.eval.max_steps_per_rollout_epoch must be divisible by actor.model.num_action_chunks"
        )
    return cfg


def validate_sft_cfg(cfg: DictConfig) -> DictConfig:
    assert cfg.actor.get("global_batch_size", None) is not None, (
        "the actor.global_batch_size is not set"
    )
    assert cfg.actor.get("micro_batch_size", None) is not None, (
        "the actor.micro_batch_size is not set"
    )

    with open_dict(cfg):
        if cfg.data.get("train_data_paths", None) is None:
            # if train_data_paths is None, the code will just eval the model
            assert cfg.data.get("val_data_paths", None) is not None, (
                "the data.train_data_paths is None, so data.val_data_paths is required"
            )
        elif cfg.data.get("val_data_paths", None) is not None:
            # set the val_check_interval to max_epochs
            if cfg.runner.get("val_check_interval", None) is None:
                cfg.runner.val_check_interval = cfg.runner.max_epochs
        else:
            # set the val_check_interval to -1 if there is no eval data or is not set
            cfg.runner.val_check_interval = cfg.runner.get("val_check_interval", -1)

        model_type = cfg.actor.model.get("model_type", None)
        if (
            model_type is not None
            and SupportedModel(model_type) == SupportedModel.DREAMZERO
        ):
            from rlinf.models.embodiment.dreamzero.dreamzero_config import (
                validate_dreamzero_sft_model_cfg,
            )

            cfg.actor.model = validate_dreamzero_sft_model_cfg(cfg.actor.model)

        _validate_steam_ensemble_cfg(cfg.actor)

    return cfg


def _validate_steam_ensemble_cfg(actor_cfg: DictConfig) -> None:
    """Validate STEAM ensemble-specific settings."""
    model_cfg = actor_cfg.get("model", None)
    if model_cfg is None or model_cfg.get("model_type", None) != "steam_value_model":
        return

    # Import lazily to avoid a circular dependency:
    # rlinf.config -> rlinf.models.embodiment... -> rlinf.models -> rlinf.config
    from rlinf.models.embodiment.value_model.steam.configuration import (
        validate_steam_ensemble_settings,
    )

    try:
        ensemble_size = validate_steam_ensemble_settings(
            ensemble_size=model_cfg.get("ensemble_size", 1),
            micro_batch_size=actor_cfg.micro_batch_size,
            global_batch_size=actor_cfg.global_batch_size,
        )
    except ValueError as exc:
        raise AssertionError(str(exc)) from exc

    with open_dict(model_cfg):
        model_cfg.ensemble_size = ensemble_size


def validate_reasoning_cfg(cfg: DictConfig) -> DictConfig:
    assert cfg.algorithm.recompute_logprobs or cfg.rollout.return_logprobs, (
        "One of `algorithm.recompute_logprobs` or `rollout.return_logprobs` must be True to compute `prev_logprobs`."
    )

    with open_dict(cfg):
        cfg.algorithm.training_batch_size_per_gpu = cfg.algorithm.get(
            "training_batch_size_per_gpu", 1
        )
        cfg.algorithm.n_minibatches = cfg.algorithm.get("n_minibatches", 1)
        cfg.algorithm.max_num_gen_batches = cfg.algorithm.get("max_num_gen_batches", 1)
        cfg.actor.micro_batch_size = cfg.algorithm.training_batch_size_per_gpu
        cfg.actor.global_batch_size = (
            cfg.data.rollout_batch_size
            * cfg.algorithm.group_size
            // cfg.algorithm.n_minibatches
        )
        assert cfg.actor.micro_batch_size >= 1
        assert cfg.actor.global_batch_size >= 1
        if hasattr(cfg, "critic"):
            cfg.critic.micro_batch_size = cfg.algorithm.training_batch_size_per_gpu
            cfg.critic.global_batch_size = (
                cfg.data.rollout_batch_size
                * cfg.algorithm.group_size
                // cfg.algorithm.n_minibatches
            )
        assert cfg.runner.seq_length > cfg.data.max_prompt_length, (
            f"runner.seq_length ({cfg.runner.seq_length}) must be greater than data.max_prompt_length ({cfg.data.max_prompt_length})"
        )

        # add configs for importance sampling fix
        cfg.algorithm.recompute_logprobs = (
            cfg.algorithm.recompute_logprobs
            or cfg.algorithm.get("importance_sampling_fix", False)
        )

        cfg.rollout = validate_rollout_cfg(cfg.rollout, cfg.algorithm)
    return cfg


def validate_reasoning_eval_cfg(cfg: DictConfig) -> DictConfig:
    with open_dict(cfg):
        assert cfg.runner.seq_length > cfg.data.max_prompt_length, (
            f"runner.seq_length ({cfg.runner.seq_length}) must be greater than data.max_prompt_length ({cfg.data.max_prompt_length})"
        )
        cfg.rollout = validate_rollout_cfg(cfg.rollout, cfg.algorithm)
    return cfg


def validate_coding_online_rl_cfg(cfg: DictConfig) -> DictConfig:
    assert SupportedModel(cfg.rollout.model.model_type) == SupportedModel.QWEN2_5, (
        f"Model type {cfg.rollout.model.model_type} is not supported"
    )

    assert cfg.algorithm.recompute_logprobs or cfg.rollout.return_logprobs, (
        "One of `algorithm.recompute_logprobs` or `rollout.return_logprobs` must be True to compute `prev_logprobs`."
    )

    if cfg.algorithm.recompute_logprobs and cfg.rollout.return_logprobs:
        assert cfg.algorithm.get("importance_sampling_fix", False), (
            "Importance sampling fix must be enabled if both `algorithm.recompute_logprobs` and `rollout.return_logprobs` are True."
        )

    assert cfg.algorithm.recompute_logprobs, (
        "Online coding task must use recompute_logprobs"
    )

    assert cfg.actor.training_backend == "megatron", (
        "Online coding task must use megatron training backend"
    )

    cluster = Cluster()
    component_placement = ModelParallelComponentPlacement(cfg, cluster)
    assert component_placement.placement_mode == PlacementMode.DISAGGREGATED, (
        "Online coding task must use disaggregated placement mode"
    )

    with open_dict(cfg):
        cfg.algorithm.training_batch_size_per_gpu = cfg.algorithm.get(
            "training_batch_size_per_gpu", 1
        )
        cfg.algorithm.n_minibatches = cfg.algorithm.get("n_minibatches", 1)
        cfg.algorithm.max_num_gen_batches = cfg.algorithm.get("max_num_gen_batches", 1)
        cfg.actor.micro_batch_size = cfg.algorithm.training_batch_size_per_gpu
        cfg.actor.global_batch_size = (
            cfg.data.rollout_batch_size
            * cfg.algorithm.group_size
            // cfg.algorithm.n_minibatches
        )
        assert cfg.actor.micro_batch_size >= 1
        assert cfg.actor.global_batch_size >= 1
        assert cfg.runner.seq_length > cfg.data.max_prompt_length, (
            f"runner.seq_length ({cfg.runner.seq_length}) must be greater than data.max_prompt_length ({cfg.data.max_prompt_length})"
        )

        # add configs for importance sampling fix
        cfg.algorithm.recompute_logprobs = (
            cfg.algorithm.recompute_logprobs
            or cfg.algorithm.get("importance_sampling_fix", False)
        )

        cfg.rollout = validate_rollout_cfg(cfg.rollout, cfg.algorithm)
    return cfg


def validate_cfg(cfg: DictConfig) -> DictConfig:
    OmegaConf.set_struct(cfg, True)

    with open_dict(cfg):
        cfg.runner.per_worker_log = cfg.runner.get("per_worker_log", False)
        cfg.runner.per_worker_log_path = None
        if cfg.runner.per_worker_log:
            cfg.runner.per_worker_log_path = os.path.join(
                cfg.runner.logger.log_path, "worker_logs"
            )
        profiling_cfg = cfg.cluster.get("profiling", None)
        if profiling_cfg is not None and bool(profiling_cfg.get("enabled", True)):
            if not profiling_cfg.get("output_dir", None):
                cfg.cluster.profiling.output_dir = os.path.abspath(
                    os.path.join(
                        cfg.runner.logger.log_path,
                        cfg.runner.logger.experiment_name,
                        "profiling",
                    )
                )

    # Tracing defaults. The tracer is a cluster manager, so its config lives under
    # `cluster.tracer` and is launched by the Cluster below when enabled.
    with open_dict(cfg):
        if "tracer" not in cfg.cluster:
            cfg.cluster.tracer = {}
        cfg.cluster.tracer.enable = bool(cfg.cluster.tracer.get("enable", False))
        if cfg.cluster.tracer.enable and not cfg.cluster.tracer.get(
            "output_file", None
        ):
            cfg.cluster.tracer.output_file = os.path.join(
                cfg.runner.logger.log_path,
                cfg.runner.logger.experiment_name,
                "trace/trace_events.jsonl",
            )

    # Init cluster
    Cluster(
        cluster_cfg=cfg.cluster,
        distributed_log_dir=cfg.runner.per_worker_log_path,
    )

    assert cfg.runner.task_type in SUPPORTED_TASK_TYPE, (
        f"task_type must be one of {SUPPORTED_TASK_TYPE}"
    )
    if cfg.runner.task_type == "embodied":
        cfg = validate_embodied_cfg(cfg)
    elif cfg.runner.task_type == "embodied_eval":
        with open_dict(cfg):
            cfg.runner.only_eval = True
        cfg = validate_embodied_cfg(cfg)
        return cfg
    elif cfg.runner.task_type == "reasoning":
        cfg = validate_reasoning_cfg(cfg)
    elif cfg.runner.task_type == "coding_online_rl":
        cfg = validate_coding_online_rl_cfg(cfg)
    elif cfg.runner.task_type == "reasoning_eval":
        cfg = validate_reasoning_eval_cfg(cfg)
        return cfg
    elif cfg.runner.task_type == "sft":
        cfg = validate_sft_cfg(cfg)
    elif cfg.runner.task_type == "offline":
        cfg = validate_offline_cfg(cfg)

    if cfg.runner.task_type != "sft" and not cfg.runner.get("only_eval", False):
        if cfg.algorithm.adv_type in ("grpo", "grpo_dynamic", "reinpp_baseline"):
            assert cfg.algorithm.group_size > 1

    assert cfg.actor.training_backend in SUPPORTED_TRAINING_BACKENDS, (
        f"Unsupported training_backend {cfg.actor.training_backend}. Supported training backends are {SUPPORTED_TRAINING_BACKENDS}."
    )

    if cfg.actor.training_backend == "megatron":
        cfg.actor = validate_megatron_cfg(cfg.actor)
        if cfg.runner.task_type == "sft":
            cfg.actor = validate_model_cfg_by_hf_config(
                cfg.actor, cfg.actor.model.model_path
            )
        else:
            cfg.actor = validate_model_cfg_by_hf_config(
                cfg.actor, cfg.rollout.model.model_path
            )
        # TODO. Need actually pad padded_vocab_size.
        assert (
            cfg.actor.model.padded_vocab_size
            % cfg.actor.model.tensor_model_parallel_size
            == 0
        ), (
            f"padded_vocab_size ({cfg.actor.model.padded_vocab_size}) must be divisible by tensor_model_parallel_size ({cfg.actor.model.tensor_model_parallel_size})"
        )
    elif cfg.actor.training_backend == "fsdp":
        component_placement = HybridComponentPlacement(cfg, Cluster())
        actor_world_size = component_placement.get_world_size("actor")
        assert (
            cfg.actor.global_batch_size
            % (cfg.actor.micro_batch_size * actor_world_size)
            == 0
        ), (
            f"actor.global_batch_size ({cfg.actor.global_batch_size}) must be divisible by (actor.micro_batch_size ({cfg.actor.micro_batch_size}) * actor_world_size ({actor_world_size}))"
        )
        cfg.actor = validate_fsdp_cfg(cfg.actor)

    if cfg.get("critic", None) is not None:
        if cfg.critic.use_critic_model and cfg.critic.training_backend == "megatron":
            cfg.critic = validate_megatron_cfg(cfg.critic)
            cfg.critic = validate_model_cfg_by_hf_config(
                cfg.critic, cfg.rollout.model.model_path
            )
        elif cfg.critic.use_critic_model and cfg.critic.training_backend == "fsdp":
            cfg.critic = validate_fsdp_cfg(cfg.critic)

    return cfg


def build_config(cls, cfg):
    if not isinstance(cfg, (dict, DictConfig)):
        cfg = asdict(cfg)

    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in cfg:
            kwargs[f.name] = cfg.get(f.name)

    return cls(**kwargs)


def build_transformer_config(cfg) -> "TransformerConfig":
    """
    Builds the megatron core transformer config for the model.
    For attributes in the RLinf model config that are the same
    as the megatron core TransformerConfig, we will use the value from the RLinf model config.
    For attributes in TransformerConfig that are not in the RLinf model config, we add custom logic.
    """
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.utils import (
        init_method_normal,
        scaled_init_method_normal,
    )

    # get model parallel configs
    model_parallel_config = _build_model_parallel_config(cfg)

    # create a dictionary copy of the model config
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # create a dict to store the transformer config arguments
    transformer_config_dict = {}

    num_layers = cfg.get("num_layers", 1)
    if num_layers % cfg.get("pipeline_model_parallel_size", 1) != 0:
        raise ValueError(
            f"num_layers ({cfg.num_layers}) should be divisible by "
            f"pipeline_model_parallel_size ({cfg.get('pipeline_model_parallel_size', 1)})"
        )

    add_bias_linear = cfg.get("add_bias_linear", True)
    add_qkv_bias = cfg.get("add_qkv_bias", False)

    activation = cfg.get("activation", "gelu")
    gated_linear_unit = activation.endswith("glu")
    # TODO: need to check which activation functions are supported in mcore
    activation_func = activation_to_func(
        activation, openai_gelu=cfg.get("openai_gelu", False)
    )

    normalization = cfg.get("normalization", "layernorm").lower()
    layernorm_zero_centered_gamma = cfg.get(
        "normalization", "layernorm"
    ) == "layernorm1p" or cfg.get("layernorm_zero_centered_gamma", False)
    if normalization == "layernorm":
        normalization = "LayerNorm"
    elif normalization == "rmsnorm":
        normalization = "RMSNorm"
    elif normalization == "layernorm1p":
        normalization = "LayerNorm"
        layernorm_zero_centered_gamma = True
    else:
        logging.warning(
            f"The normalization type: {normalization} might not be supported in megatron core."
            f"Supported types are LayerNorm and RMSNorm."
        )

    tp_comm_overlap = cfg.get("tp_comm_overlap", False)

    if not cfg.get("fp8", False):
        fp8 = None
    elif cfg.get("fp8_e4m3", False):
        fp8 = "e4m3"
    elif cfg.get("fp8_hybrid", False):
        fp8 = "hybrid"
    else:
        raise ValueError(
            "fp8 enabled but fp8_format (fp8_e4m3 | fp8_hybrid) is not set."
        )

    init_method_std = cfg.get("init_method_std", 0.02)
    # default used in mcore
    init_method = init_method_normal(init_method_std)

    output_layer_init_method = init_method

    use_scaled_init_method = cfg.get("use_scaled_init_method", True)
    if use_scaled_init_method:
        output_layer_init_method = scaled_init_method_normal(
            init_method_std, num_layers=num_layers
        )

    attention_softmax_in_fp32 = cfg.get("attention_softmax_in_fp32", True)
    apply_query_key_layer_scaling = cfg.get("apply_query_key_layer_scaling", False)

    rotary_interleaved = cfg.get("rotary_interleaved", False)

    if apply_query_key_layer_scaling:
        if model_parallel_config.fp16:
            os.environ["NVTE_APPLY_QK_LAYER_SCALING"] = "1"
        else:
            logging.warning(
                "apply_query_key_layer_scaling is only enabled when using FP16, setting it to False "
                "and setting NVTE_APPLY_QK_LAYER_SCALING=0"
            )
            os.environ["NVTE_APPLY_QK_LAYER_SCALING"] = "0"
            apply_query_key_layer_scaling = False

    if apply_query_key_layer_scaling:
        attention_softmax_in_fp32 = True

    bias_activation_fusion = cfg.get("bias_activation_fusion", True)

    bias_dropout_fusion = cfg.get("bias_dropout_fusion", True)

    apply_rope_fusion = cfg.get("apply_rope_fusion", False)

    # TODO: need to check if recompute APIs are matching up properly
    recompute_granularity = cfg.get("recompute_granularity", None)
    recompute_method = cfg.get("recompute_method", None)
    recompute_num_layers = cfg.get("recompute_num_layers", None)

    tp_only_amax_red = cfg.get("tp_only_amax_red", False)

    if cfg.get("enable_cuda_graph", False):
        if importlib.util.find_spec("transformer_engine") is None:
            raise ImportError(
                "Can not import transformer_engine, which is required for cudagraphs."
            )
        assert cfg.get("use_te_rng_tracker", False), (
            "Transformer engine's RNG tracker is required for cudagraphs, this can be enabled with \
            'use_te_rng_tracker=True'."
        )

    # any configs that are not in the RLinf model config will be added here
    config_mapping = {
        "apply_query_key_layer_scaling": apply_query_key_layer_scaling,
        "apply_residual_connection_post_layernorm": False,  # we don't use this in NeMo
        "add_bias_linear": add_bias_linear,
        "add_qkv_bias": add_qkv_bias,
        "gated_linear_unit": gated_linear_unit,
        "activation_func": activation_func,
        "normalization": normalization,
        "layernorm_zero_centered_gamma": layernorm_zero_centered_gamma,
        "init_method": init_method,
        "output_layer_init_method": output_layer_init_method,
        "attention_softmax_in_fp32": attention_softmax_in_fp32,
        "bias_activation_fusion": bias_activation_fusion,
        "bias_dropout_fusion": bias_dropout_fusion,
        "apply_rope_fusion": apply_rope_fusion,
        "recompute_granularity": recompute_granularity,
        "recompute_method": recompute_method,
        "recompute_num_layers": recompute_num_layers,
        "distribute_saved_activations": False,  # not currently used in NeMo
        "fp8": fp8,
        "tp_comm_overlap": tp_comm_overlap,
        "rotary_interleaved": rotary_interleaved,
        "deallocate_pipeline_outputs": False,
        "tp_only_amax_red": tp_only_amax_red,
        "qk_layernorm": cfg.get("qk_layernorm", False),
        "kv_channels": cfg.get("head_dim", None),
        # MoE related
        "num_moe_experts": cfg.get("num_moe_experts", None),
        "moe_ffn_hidden_size": cfg.get("moe_ffn_hidden_size", None),
        # now the sequential mlp should ffn hidden size == moe_ffn_hidden_size
        "ffn_hidden_size": cfg.get("moe_ffn_hidden_size", None)
        or cfg.get("ffn_hidden_size", None),
        "moe_router_load_balancing_type": cfg.get(
            "moe_router_load_balancing_type", "aux_loss"
        ),
        "moe_router_topk": cfg.get("moe_router_topk", 2),
        "moe_grouped_gemm": cfg.get("moe_grouped_gemm", False),
        "moe_aux_loss_coeff": cfg.get(
            "moe_aux_loss_coeff", 0
        ),  # 1e-2 would be a good start value for load balance loss.
        "moe_z_loss_coeff": cfg.get(
            "moe_z_loss_coeff", None
        ),  # 1e-3 would be a good start value for z-loss
        "moe_input_jitter_eps": cfg.get("moe_input_jitter_eps", None),
        "moe_token_dropping": cfg.get("moe_token_dropping", False),
        "enable_cuda_graph": cfg.get("enable_cuda_graph", False),
    }

    # populate the transformer config dict
    for field in dataclasses.fields(TransformerConfig):
        # config mapping has second highest priority
        if field.name in config_mapping:
            transformer_config_dict[field.name] = config_mapping[field.name]
        # then config
        elif field.name in cfg:
            transformer_config_dict[field.name] = cfg[field.name]
        # then model parallel config
        elif field in dataclasses.fields(model_parallel_config):
            transformer_config_dict[field.name] = getattr(
                model_parallel_config, field.name
            )

    transformer_config = TransformerConfig(**transformer_config_dict)

    # pass mcore customization configs directly to mcore
    mcore_customization_config_dict = cfg.get("mcore_customization_config", {})
    for key, value in mcore_customization_config_dict.items():
        setattr(transformer_config, key, value)

    return transformer_config


def _build_model_parallel_config(cfg: DictConfig) -> "ModelParallelConfig":
    """
    For attributes in the RLinf model config that are the same as the
    megatron core ModelParallelConfig we will use the value from the RLinf config.
    For attributes in ModelParallelConfig that are not in the RLinf model config, we add custom logic.
    """
    from megatron.core.model_parallel_config import ModelParallelConfig
    from megatron.training.global_vars import get_timers
    # cfg = OmegaConf.to_container(cfg, resolve=True)

    # dtype used in p2p communication
    if cfg.get("precision", None) is None:
        raise f"precision not found in {cfg}."
    torch_dtype = torch_dtype_from_precision(cfg.precision)
    params_dtype = (
        torch_dtype if torch_dtype in [torch.bfloat16, torch.float16] else torch.float32
    )
    pipeline_dtype = cfg.get("pipeline_dtype", params_dtype)
    autocast_dtype = cfg.get("autocast_dtype", params_dtype)

    timers = get_timers()
    # maps NeMo model configs to ModelParallelConfig from megatron core
    config_mapping = {
        "perform_initialization": True,  # initailize weights when constructing the module
        "fp16": torch_dtype == torch.float16,
        "bf16": torch_dtype == torch.bfloat16,
        "params_dtype": params_dtype,
        "timers": timers,
        "async_tensor_model_parallel_allreduce": False,  # Deprecated in megatron
        "pipeline_dtype": pipeline_dtype,
        "grad_scale_func": None,
        "enable_autocast": False,  # torch_dtype in [torch.bfloat16, torch.float16],
        "autocast_dtype": autocast_dtype,
        "num_microbatches_with_partial_activation_checkpoints": cfg.get(
            "num_microbatches_with_partial_activation_checkpoints", None
        ),
        "batch_p2p_sync": True,  # call torch.cuda.synchronize() after batch isend/rcv
        "use_ring_exchange_p2p": False,
        "deallocate_pipeline_outputs": False,
        "no_sync_func": None,  # set dynamically during training
        "grad_sync_func": None,  # set dynamically during training
        "param_sync_func": None,  # set dynamically during training
        "tp_comm_overlap": cfg.get("tp_comm_overlap", False),
        "tp_comm_bootstrap_backend": cfg.get("tp_comm_bootstrap_backend", "nccl"),
    }

    # instantitate ModelParallelConfig from this dict
    mp_config_dict = {}

    for field in dataclasses.fields(ModelParallelConfig):
        # model config has priority
        if field.name in cfg:
            mp_config_dict[field.name] = cfg[field.name]
        # then config_mapping
        elif field.name in config_mapping:
            mp_config_dict[field.name] = config_mapping[field.name]

    model_parallel_config = ModelParallelConfig(**mp_config_dict)

    return model_parallel_config
