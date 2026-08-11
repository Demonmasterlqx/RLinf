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

from typing import Optional

import torch


def huber_loss(error: torch.Tensor, delta: float) -> torch.Tensor:
    return torch.where(
        error.abs() < delta, 0.5 * error**2, delta * (error.abs() - 0.5 * delta)
    )


def kl_penalty(
    logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty
) -> torch.FloatTensor:
    """
    Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def preprocess_embodied_advantages_inputs(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: Optional[torch.Tensor] = None,
    loss_mask: Optional[torch.Tensor] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    **kwargs,
) -> dict:
    """
    Preprocess inputs before computing advantages & returns.
    Unify names & formats, align with math interfaces.
    """
    if kwargs["reward_type"] == "chunk_level":
        # TODO: need check
        # rewards, dones, loss_mask, loss_mask_sum: [n_chunk_steps, bsz, num_action_chunks] -> [n_chunk_steps, bsz, 1]
        rewards = rewards.sum(dim=-1, keepdim=True)
        dones = dones.max(dim=-1, keepdim=True)[0]
        if loss_mask is not None:
            loss_mask = loss_mask.max(dim=-1, keepdim=True)[0]
        if loss_mask_sum is not None:
            loss_mask_sum = loss_mask_sum.max(dim=-1, keepdim=True)[0]

    num_chunk, bsz, chunk_size = rewards.shape
    n_steps = num_chunk * chunk_size
    kwargs.update(
        {
            "num_chunk": num_chunk,
            "batch_size": bsz,
            "chunk_size": chunk_size,
            "n_steps": n_steps,
        }
    )

    # Transpose(1, 2) -> [num-chunk, chunk-size, bsz]
    # Reshape -> [n_steps, bsz]
    # Rewards [n_steps, bsz]
    rewards = rewards.transpose(1, 2).reshape(n_steps, bsz)

    # Loss Mask (T steps) [bsz, n_steps]
    if loss_mask is not None:
        loss_mask = loss_mask.transpose(1, 2).reshape(n_steps, bsz)

    # Dones (T+1 steps) [num-chunk+1, bsz, chunk-size]
    flattened_dones_full = dones.transpose(1, 2).reshape(
        (num_chunk + 1) * chunk_size, bsz
    )
    dones = flattened_dones_full[-(n_steps + 1) :]

    if kwargs["adv_type"] == "gae":
        flattened_values_full = values.transpose(1, 2).reshape(
            (num_chunk + 1) * chunk_size, bsz
        )
        values = flattened_values_full[: n_steps + 1]

    kwargs.update(
        {
            "rewards": rewards,
            "dones": dones,
            "values": values,
            "loss_mask": loss_mask,
            "loss_mask_sum": loss_mask_sum,
        }
    )

    return kwargs


def calculate_scores(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    **kwargs,
) -> dict:
    scores = torch.zeros(kwargs["batch_size"])
    for step in reversed(range(kwargs["n_steps"])):
        scores = scores * ~dones[step + 1]
        scores += rewards[step]
    scores = scores.reshape(-1, kwargs["group_size"])

    kwargs.update(
        {
            "rewards": scores,
            "dones": dones,
        }
    )

    return kwargs


def postprocess_embodied_advantages_outputs(
    advantages: torch.Tensor,
    num_chunk: int,
    chunk_size: int,
    returns: Optional[torch.Tensor] = None,
    **kwargs,
) -> dict:
    """
    Post-process results for Embodiment tasks; unflatten tensors.
    """
    res = {}

    advantages = advantages.reshape(num_chunk, chunk_size, -1).transpose(1, 2)
    res.update({"advantages": advantages})

    if returns is not None:
        returns = returns.reshape(num_chunk, chunk_size, -1).transpose(1, 2)
        res.update({"returns": returns})

    return res


def preprocess_reasoning_advantages_inputs(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    values: Optional[torch.Tensor] = None,
    logprob: Optional[torch.Tensor] = None,
    ref_logprob: Optional[torch.Tensor] = None,
    **kwargs,
) -> dict:
    # NOTE: to align with embodied inputs, we transpose loss mask and rewards when needed.

    bsz, seq_len = loss_mask.shape
    loss_mask = loss_mask.transpose(0, 1)  # [seq_len, bsz]

    assert rewards.ndim == 1, f"Unsupported reward shape {rewards.shape}"

    if kwargs["adv_type"] == "gae":
        expanded_rewards = torch.zeros(
            (seq_len, bsz), dtype=rewards.dtype, device=rewards.device
        )
        expanded_rewards[-1] = rewards  # only last token has reward
        kwargs.update({"rewards": expanded_rewards})

    elif kwargs["adv_type"] == "grpo":
        grouped_rewards = rewards.reshape(-1, kwargs["group_size"]).contiguous()
        kwargs.update(
            {
                "rewards": grouped_rewards,
            }
        )

    elif kwargs["adv_type"] == "grpo_dynamic":
        grouped_rewards = (
            rewards.reshape(-1, kwargs["num_sequence"]).transpose(0, 1).contiguous()
        )
        kwargs.update(
            {
                "rewards": grouped_rewards,
            }
        )

    elif kwargs["adv_type"] == "reinpp":
        kwargs.update({"rewards": rewards.unsqueeze(0)})

    elif kwargs["adv_type"] == "raw":
        kwargs.update({"rewards": rewards})

    else:
        assert False, f"Unsupported adv_type {kwargs['adv_type']}"

    if values is not None:  # [bsz, seq_len]
        assert values.ndim == 2, f"Unsupported values shape {values.shape}"
        values = values.transpose(0, 1)  # [seq_len, bsz]
        # pad values with zeros at the end for bootstrapping
        values = torch.cat(
            [
                values,
                torch.zeros(
                    (1, values.shape[-1]), dtype=values.dtype, device=values.device
                ),
            ],
            dim=0,
        )  # [seq_len+1, bsz]

        kwargs.update({"values": values})

    if logprob is not None:
        logprob = logprob.transpose(0, 1)
        kwargs.update({"logprob": logprob})

    if ref_logprob is not None:
        ref_logprob = ref_logprob.transpose(0, 1)
        kwargs.update({"ref_logprob": ref_logprob})

    # Create done flags (episode ends at the last token)
    dones = torch.zeros(seq_len + 1, bsz, dtype=torch.bool, device=rewards.device)
    dones[-1] = True
    kwargs.update(
        {
            "dones": dones,
            "loss_mask": loss_mask,
        }
    )

    return kwargs


def postprocess_reasoning_advantages_outputs(
    advantages: torch.Tensor,
    returns: Optional[torch.Tensor] = None,
) -> dict:
    """
    Post-process results for Reasoning tasks; transpose tensors back.
    """

    # remember to call contiguous() to ensure correctness when being
    # transmitted through channels
    advantages = advantages.transpose(0, 1).contiguous()  # [bsz, seq_len]
    if returns is not None:
        returns = returns.transpose(0, 1).contiguous()  # [bsz, seq_len]

    return advantages, returns


def _reshape_and_mask_primitive_logprobs(
    values: torch.Tensor,
    *,
    batch_size: int,
    single_action_dim: int,
    primitive_loss_mask: torch.Tensor | None,
    name: str,
) -> torch.Tensor:
    reshaped = values.reshape(batch_size, -1, single_action_dim)
    if primitive_loss_mask is None:
        return reshaped

    mask = primitive_loss_mask.to(device=reshaped.device, dtype=torch.bool)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    expected_shape = reshaped.shape[:2]
    if tuple(mask.shape) != tuple(expected_shape):
        raise ValueError(
            f"{name} primitive_loss_mask expected shape {tuple(expected_shape)}, "
            f"got {tuple(mask.shape)}."
        )
    return torch.where(mask.unsqueeze(-1), reshaped, torch.zeros_like(reshaped))


def reduce_embodied_entropy_with_primitive_mask(
    entropy: torch.Tensor,
    *,
    primitive_loss_mask: torch.Tensor,
    entropy_type: str,
    single_action_dim: int,
) -> torch.Tensor:
    """Reduce entropy while excluding every post-terminal primitive action."""

    mask = primitive_loss_mask.to(device=entropy.device, dtype=torch.bool)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    if mask.ndim != 2 or entropy.shape[0] != mask.shape[0]:
        raise ValueError(
            "primitive entropy mask must have shape [B, action_chunk] aligned with "
            f"entropy batch; got entropy={tuple(entropy.shape)}, mask={tuple(mask.shape)}."
        )

    batch_size, action_chunk = mask.shape
    if entropy.ndim >= 2 and entropy.shape[1] == action_chunk:
        primitive_entropy = entropy
    else:
        per_batch_elements = entropy.numel() // batch_size
        if entropy.numel() % batch_size != 0 or per_batch_elements % action_chunk != 0:
            raise ValueError(
                "Cannot align entropy with primitive action mask: "
                f"entropy={tuple(entropy.shape)}, mask={tuple(mask.shape)}."
            )
        primitive_entropy = entropy.reshape(batch_size, action_chunk, -1)

    if entropy_type == "action_level":
        if primitive_entropy.ndim > 2:
            primitive_entropy = primitive_entropy.reshape(
                batch_size, action_chunk, -1
            ).sum(dim=-1)
        expanded_mask = mask
    elif entropy_type == "chunk_level":
        expanded_mask = mask
        while expanded_mask.ndim < primitive_entropy.ndim:
            expanded_mask = expanded_mask.unsqueeze(-1)
        masked_entropy = torch.where(
            expanded_mask, primitive_entropy, torch.zeros_like(primitive_entropy)
        )
        chunk_entropy = masked_entropy.reshape(batch_size, -1).sum(dim=-1)
        valid_chunks = mask.any(dim=-1)
        if not valid_chunks.any():
            return chunk_entropy.sum() * 0.0
        return chunk_entropy[valid_chunks].mean()
    elif entropy_type == "token_level":
        expanded_mask = mask
        while expanded_mask.ndim < primitive_entropy.ndim:
            expanded_mask = expanded_mask.unsqueeze(-1)
        expanded_mask = expanded_mask.expand_as(primitive_entropy)
    else:
        raise ValueError(f"Unsupported entropy_type {entropy_type!r}.")

    if not expanded_mask.any():
        return primitive_entropy.sum() * 0.0
    return primitive_entropy[expanded_mask].mean()


def preprocess_loss_inputs(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    logprob_type: Optional[str] = None,
    single_action_dim: Optional[int] = None,
    loss_mask: Optional[torch.Tensor] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    values: Optional[torch.Tensor] = None,
    prev_values: Optional[torch.Tensor] = None,
    returns: Optional[torch.Tensor] = None,
    reward_type: Optional[str] = None,
    versions: Optional[torch.Tensor] = None,
    primitive_loss_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> dict:
    if reward_type == "chunk_level":
        advantages = advantages.flatten()
        if loss_mask is not None:
            loss_mask = loss_mask.flatten()
        if loss_mask_sum is not None:
            loss_mask_sum = loss_mask_sum.flatten()
        if values is not None:
            values = values.flatten()
        if prev_values is not None:
            prev_values = prev_values.flatten()
        if returns is not None:
            returns = returns.flatten()

    bsz = logprobs.shape[0]
    proximal_logprobs = kwargs.get("proximal_logprobs", None)
    if logprob_type == "token_level":
        # logprobs, old_logprobs: [bsz, num_action_chunks, action_dim] -> [bsz, num_action_chunks, action_dim]
        logprobs = _reshape_and_mask_primitive_logprobs(
            logprobs,
            batch_size=bsz,
            single_action_dim=single_action_dim,
            primitive_loss_mask=primitive_loss_mask,
            name="logprobs",
        )
        old_logprobs = _reshape_and_mask_primitive_logprobs(
            old_logprobs,
            batch_size=bsz,
            single_action_dim=single_action_dim,
            primitive_loss_mask=primitive_loss_mask,
            name="old_logprobs",
        )
        if proximal_logprobs is not None:
            proximal_logprobs = _reshape_and_mask_primitive_logprobs(
                proximal_logprobs,
                batch_size=bsz,
                single_action_dim=single_action_dim,
                primitive_loss_mask=primitive_loss_mask,
                name="proximal_logprobs",
            )
        if versions is not None:
            versions = versions.reshape(bsz, -1, single_action_dim)
        if kwargs.get("loss_type") == "opd":
            assert advantages.shape == logprobs.shape, (
                f"OPD advantages shape {advantages.shape} must match "
                f"logprobs shape {logprobs.shape}."
            )
        else:
            advantages = advantages.unsqueeze(-1)
        if loss_mask is not None:
            loss_mask = loss_mask.unsqueeze(-1)
        if loss_mask_sum is not None:
            loss_mask_sum = loss_mask_sum.unsqueeze(-1)

    elif logprob_type == "action_level":
        # logprobs, old_logprobs: [bsz, num_action_chunks, action_dim] -> [bsz, num_action_chunks]
        logprobs = _reshape_and_mask_primitive_logprobs(
            logprobs,
            batch_size=bsz,
            single_action_dim=single_action_dim,
            primitive_loss_mask=primitive_loss_mask,
            name="logprobs",
        ).sum(dim=-1)
        old_logprobs = _reshape_and_mask_primitive_logprobs(
            old_logprobs,
            batch_size=bsz,
            single_action_dim=single_action_dim,
            primitive_loss_mask=primitive_loss_mask,
            name="old_logprobs",
        ).sum(dim=-1)
        if proximal_logprobs is not None:
            proximal_logprobs = _reshape_and_mask_primitive_logprobs(
                proximal_logprobs,
                batch_size=bsz,
                single_action_dim=single_action_dim,
                primitive_loss_mask=primitive_loss_mask,
                name="proximal_logprobs",
            ).sum(dim=-1)
        if versions is not None:
            versions = versions.reshape(bsz, -1, single_action_dim)[..., 0]

    elif logprob_type == "chunk_level":
        # logprobs, old_logprobs: [bsz, num_action_chunks, action_dim] -> [bsz]
        logprobs = _reshape_and_mask_primitive_logprobs(
            logprobs,
            batch_size=bsz,
            single_action_dim=single_action_dim,
            primitive_loss_mask=primitive_loss_mask,
            name="logprobs",
        ).sum(dim=[1, 2])
        old_logprobs = _reshape_and_mask_primitive_logprobs(
            old_logprobs,
            batch_size=bsz,
            single_action_dim=single_action_dim,
            primitive_loss_mask=primitive_loss_mask,
            name="old_logprobs",
        ).sum(dim=[1, 2])
        if proximal_logprobs is not None:
            proximal_logprobs = _reshape_and_mask_primitive_logprobs(
                proximal_logprobs,
                batch_size=bsz,
                single_action_dim=single_action_dim,
                primitive_loss_mask=primitive_loss_mask,
                name="proximal_logprobs",
            ).sum(dim=[1, 2])
        if versions is not None:
            versions = versions.reshape(bsz, -1, single_action_dim)[:, 0, 0]

    target_shape = logprobs.shape
    advantages = expand_to_target_dim(advantages, target_shape)
    loss_mask = expand_to_target_dim(loss_mask, target_shape)
    loss_mask_sum = expand_to_target_dim(loss_mask_sum, target_shape)
    values = expand_to_target_dim(values, target_shape)
    prev_values = expand_to_target_dim(prev_values, target_shape)
    returns = expand_to_target_dim(returns, target_shape)
    versions = expand_to_target_dim(versions, target_shape)

    kwargs.update(
        {
            "logprobs": logprobs,
            "old_logprobs": old_logprobs,
            "proximal_logprobs": proximal_logprobs,
            "versions": versions,
            "advantages": advantages,
            "loss_mask": loss_mask,
            "loss_mask_sum": loss_mask_sum,
            "values": values,
            "prev_values": prev_values,
            "returns": returns,
        }
    )

    return kwargs


def postprocess_loss_metric(metrics_data: dict) -> dict:
    for k, v in metrics_data.items():
        if isinstance(v, torch.Tensor):
            metrics_data[k] = v.detach().item()
        elif isinstance(v, (float, int)):
            metrics_data[k] = v
    return metrics_data


def expand_to_target_dim(tensor, target_shape):
    if tensor is None:
        return None
    if tensor.shape != target_shape:
        while len(tensor.shape) < len(target_shape):
            tensor = tensor.unsqueeze(-1)
    return tensor


def safe_normalize(array, loss_mask):
    valid_array = array[loss_mask]
    if len(valid_array) > 0:
        mean = valid_array.mean()
        std = valid_array.std()
        array = (array - mean) / (std + 1e-5)

    return array
