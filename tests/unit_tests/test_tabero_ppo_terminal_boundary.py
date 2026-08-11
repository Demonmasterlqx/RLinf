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

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.losses import compute_ppo_actor_loss
from rlinf.algorithms.utils import (
    preprocess_loss_inputs,
    reduce_embodied_entropy_with_primitive_mask,
)
from rlinf.models.embodiment.openpi.openpi_action_model import _reduce_openpi_entropy
from rlinf.utils.metric_utils import (
    compute_embodied_loss_masks,
    summarize_primitive_loss_mask,
)
from rlinf.utils.tabero_ppo_boundary import (
    TABERO_PPO_TRANSITION_BOUNDARY_SEMANTICS,
    validate_tabero_ppo_checkpoint_boundary_metadata,
)
from rlinf.utils.utils import preprocess_embodied_batch
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


def _primitive_dones() -> torch.Tensor:
    # One leading bootstrap row followed by two executed 4-action chunks.
    dones = torch.zeros((3, 3, 4), dtype=torch.bool)
    dones[1, 0, 0] = True
    dones[1, 1, 2] = True
    return dones


def test_prefix_masks_keep_terminal_primitive_and_mask_all_later_actions():
    masks = compute_embodied_loss_masks(
        _primitive_dones(),
        reward_type="chunk_level",
        use_primitive_prefix_logprobs=True,
    )

    expected = torch.tensor(
        [
            [[True, False, False, False], [True, True, True, False], [True] * 4],
            [[False] * 4, [False] * 4, [True] * 4],
        ],
        dtype=torch.bool,
    )
    torch.testing.assert_close(masks["primitive_loss_mask"], expected)
    torch.testing.assert_close(
        masks["chunk_loss_mask"], expected.any(dim=-1, keepdim=True)
    )
    torch.testing.assert_close(masks["loss_mask"], masks["chunk_loss_mask"])
    assert masks["loss_mask_sum"][:, :, 0].tolist() == [
        [1, 1, 2],
        [1, 1, 2],
    ]
    assert summarize_primitive_loss_mask(expected) == {
        "valid_primitive_actions": 12,
        "masked_post_done_actions": 12,
        "partial_chunk_count": 2,
    }


def _ppo_result(
    current_values: torch.Tensor,
    old_values: torch.Tensor,
    primitive_mask: torch.Tensor,
):
    current = current_values.clone().requires_grad_(True)
    prepared = preprocess_loss_inputs(
        logprobs=current,
        old_logprobs=old_values,
        advantages=torch.tensor([1.25], dtype=torch.float32),
        logprob_type="chunk_level",
        single_action_dim=current.shape[-1],
        loss_mask=torch.ones((1, 1), dtype=torch.bool),
        loss_mask_sum=torch.ones((1, 1), dtype=torch.long),
        reward_type="chunk_level",
        primitive_loss_mask=primitive_mask,
    )
    loss, metrics = compute_ppo_actor_loss(
        **prepared,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=None,
    )
    loss.backward()
    return loss.detach(), metrics, current.grad.detach()


def test_post_terminal_extreme_logprobs_do_not_change_ppo_or_gradients():
    primitive_mask = torch.tensor([[True, True, False, False]])
    current = torch.tensor(
        [[[0.1, -0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]],
        dtype=torch.float32,
    )
    old = current.detach().clone()
    extreme_current = current.detach().clone()
    extreme_old = old.clone()
    extreme_current[:, 2:] = 1.0e6
    extreme_old[:, 2:] = -1.0e6

    baseline_loss, baseline_metrics, baseline_grad = _ppo_result(
        current, old, primitive_mask
    )
    extreme_loss, extreme_metrics, extreme_grad = _ppo_result(
        extreme_current, extreme_old, primitive_mask
    )

    torch.testing.assert_close(extreme_loss, baseline_loss)
    for key in ("actor/ratio", "actor/approx_kl", "actor/policy_loss"):
        torch.testing.assert_close(extreme_metrics[key], baseline_metrics[key])
    torch.testing.assert_close(extreme_grad[:, :2], baseline_grad[:, :2])
    torch.testing.assert_close(
        extreme_grad[:, 2:], torch.zeros_like(extreme_grad[:, 2:])
    )


def test_all_valid_prefix_is_numerically_identical_to_legacy_chunk_sum():
    logprobs = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3) / 100.0
    old_logprobs = logprobs - 0.01
    common = {
        "logprobs": logprobs,
        "old_logprobs": old_logprobs,
        "advantages": torch.tensor([0.5, -0.25], dtype=torch.float32),
        "logprob_type": "chunk_level",
        "single_action_dim": 3,
        "loss_mask": torch.ones((2, 1), dtype=torch.bool),
        "loss_mask_sum": torch.ones((2, 1), dtype=torch.long),
        "reward_type": "chunk_level",
    }

    legacy = preprocess_loss_inputs(**common)
    boundary_safe = preprocess_loss_inputs(
        **common,
        primitive_loss_mask=torch.ones((2, 4), dtype=torch.bool),
    )

    torch.testing.assert_close(boundary_safe["logprobs"], legacy["logprobs"])
    torch.testing.assert_close(boundary_safe["old_logprobs"], legacy["old_logprobs"])


def test_partial_chunk_normalization_counts_one_macro_sample():
    primitive_mask = torch.tensor([[True, True, True, False], [False] * 4])
    prepared = preprocess_loss_inputs(
        logprobs=torch.zeros((2, 4, 1), dtype=torch.float32),
        old_logprobs=torch.zeros((2, 4, 1), dtype=torch.float32),
        advantages=torch.tensor([2.0, 999.0], dtype=torch.float32),
        logprob_type="chunk_level",
        single_action_dim=1,
        loss_mask=torch.tensor([[True], [False]]),
        loss_mask_sum=torch.ones((2, 1), dtype=torch.long),
        reward_type="chunk_level",
        primitive_loss_mask=primitive_mask,
    )

    loss, _ = compute_ppo_actor_loss(
        **prepared,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=None,
        max_episode_steps=2,
    )

    assert loss.item() == pytest.approx(-2.0)


def test_entropy_ignores_post_terminal_suffix_and_has_zero_suffix_gradient():
    entropy = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [1.0e6, 1.0e6], [2.0e6, 2.0e6]]],
        requires_grad=True,
    )
    primitive_mask = torch.tensor([[True, True, False, False]])

    reduced = reduce_embodied_entropy_with_primitive_mask(
        entropy,
        primitive_loss_mask=primitive_mask,
        entropy_type="token_level",
        single_action_dim=2,
    )
    reduced.backward()

    assert reduced.item() == pytest.approx(2.5)
    torch.testing.assert_close(
        entropy.grad[:, 2:], torch.zeros_like(entropy.grad[:, 2:])
    )


def test_openpi_masks_entropy_before_model_reduction():
    entropy = torch.arange(24, dtype=torch.float32).reshape(1, 2, 4, 3)
    entropy[:, :, 2:] = 1.0e6
    entropy.requires_grad_(True)
    primitive_mask = torch.tensor([[True, True, False, False]])

    reduced = _reduce_openpi_entropy(entropy, primitive_mask)
    expected = entropy.detach()[:, :, :2].mean().reshape(1, 1)
    torch.testing.assert_close(reduced, expected)
    reduced.sum().backward()
    torch.testing.assert_close(
        entropy.grad[:, :, 2:], torch.zeros_like(entropy.grad[:, :, 2:])
    )

    all_valid = _reduce_openpi_entropy(
        entropy.detach(), torch.ones((1, 4), dtype=torch.bool)
    )
    torch.testing.assert_close(all_valid, entropy.detach().mean((1, 2, 3))[:, None])


def test_pipeline_and_non_pipeline_build_identical_boundary_masks():
    source = {"dones": _primitive_dones()}
    actor = object.__new__(EmbodiedFSDPActor)
    actor.cfg = OmegaConf.create(
        {
            "env": {
                "train": {
                    "rollout_epoch": 1,
                    "auto_reset": False,
                    "ignore_terminations": False,
                }
            },
            "algorithm": {
                "reward_type": "chunk_level",
                "filter_rewards": False,
                "group_size": 1,
            },
        }
    )
    actor._tabero_ppo_transition_boundary_semantics = (
        TABERO_PPO_TRANSITION_BOUNDARY_SEMANTICS
    )

    non_pipeline = EmbodiedFSDPActor._process_received_rollout_batch(actor, source)
    pipeline = preprocess_embodied_batch(
        source,
        rollout_epoch=1,
        auto_reset=False,
        ignore_terminations=False,
        reward_type="chunk_level",
        filter_rewards=False,
        group_size=1,
        transition_boundary_semantics=TABERO_PPO_TRANSITION_BOUNDARY_SEMANTICS,
    )

    for key in (
        "primitive_loss_mask",
        "chunk_loss_mask",
        "loss_mask",
        "loss_mask_sum",
    ):
        torch.testing.assert_close(non_pipeline[key], pipeline[key])


def _write_checkpoint_sidecar(path, semantics=None):
    sidecar = path / "model_state_dict" / "trainable_weights.pt"
    sidecar.parent.mkdir(parents=True)
    metadata = {"global_step": 1}
    if semantics is not None:
        metadata["tabero_ppo_transition_boundary_semantics"] = semantics
    torch.save({"model": {}, "metadata": metadata}, sidecar)
    return sidecar


def test_checkpoint_boundary_metadata_accepts_only_matching_sidecars(tmp_path):
    valid = tmp_path / "valid"
    _write_checkpoint_sidecar(valid, TABERO_PPO_TRANSITION_BOUNDARY_SEMANTICS)
    metadata = validate_tabero_ppo_checkpoint_boundary_metadata(
        valid,
        expected_semantics=TABERO_PPO_TRANSITION_BOUNDARY_SEMANTICS,
    )
    assert metadata["global_step"] == 1

    legacy = tmp_path / "legacy"
    _write_checkpoint_sidecar(legacy)
    with pytest.raises(ValueError, match="must restart from the base model"):
        validate_tabero_ppo_checkpoint_boundary_metadata(
            legacy,
            expected_semantics=TABERO_PPO_TRANSITION_BOUNDARY_SEMANTICS,
        )

    with pytest.raises(ValueError, match="requires checkpoint sidecar"):
        validate_tabero_ppo_checkpoint_boundary_metadata(
            tmp_path / "missing",
            expected_semantics=TABERO_PPO_TRANSITION_BOUNDARY_SEMANTICS,
        )
