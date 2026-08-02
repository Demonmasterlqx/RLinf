import pytest
import torch

from rlinf.algorithms.losses import compute_ppo_critic_loss
from rlinf.utils.metric_utils import (
    CRITIC_EXPLAINED_VARIANCE_KEY,
    CRITIC_EXPLAINED_VARIANCE_STAT_KEYS,
    compute_critic_explained_variance_from_stats,
    compute_critic_explained_variance_stats,
    pop_critic_explained_variance_stats,
)


def test_critic_explained_variance_is_finite_for_one_masked_sample():
    _, metrics = compute_ppo_critic_loss(
        values=torch.tensor([0.25, 0.75]),
        returns=torch.tensor([0.5, 1.0]),
        prev_values=torch.tensor([0.2, 0.7]),
        value_clip=0.2,
        huber_delta=1.0,
        loss_mask=torch.tensor([True, False]),
    )

    explained_variance = metrics["critic/explained_variance"]
    assert torch.isfinite(explained_variance)
    assert explained_variance.item() == 0.0


@pytest.mark.parametrize(
    ("values", "returns", "loss_mask"),
    [
        ([float("nan"), 0.75], [0.5, 1.0], [True, False]),
        ([float("inf"), 0.75], [0.5, 1.0], [True, False]),
        ([0.25, 0.75], [float("nan"), 1.0], [True, False]),
        ([0.25, float("nan")], [0.5, 0.5], [True, True]),
    ],
)
def test_critic_explained_variance_preserves_nonfinite_inputs(
    values, returns, loss_mask
):
    _, metrics = compute_ppo_critic_loss(
        values=torch.tensor(values),
        returns=torch.tensor(returns),
        prev_values=torch.tensor([0.2, 0.7]),
        value_clip=0.2,
        huber_delta=1.0,
        loss_mask=torch.tensor(loss_mask),
    )

    assert not torch.isfinite(metrics["critic/explained_variance"])


def test_critic_explained_variance_is_zero_for_empty_mask():
    _, metrics = compute_ppo_critic_loss(
        values=torch.tensor([0.25, 0.75]),
        returns=torch.tensor([0.5, 1.0]),
        prev_values=torch.tensor([0.2, 0.7]),
        value_clip=0.2,
        huber_delta=1.0,
        loss_mask=torch.tensor([False, False]),
    )

    explained_variance = metrics["critic/explained_variance"]
    assert torch.isfinite(explained_variance)
    assert explained_variance.item() == 0.0


def test_critic_explained_variance_is_zero_for_constant_returns():
    _, metrics = compute_ppo_critic_loss(
        values=torch.tensor([0.25, 0.75]),
        returns=torch.tensor([0.5, 0.5]),
        prev_values=torch.tensor([0.2, 0.7]),
        value_clip=0.2,
        huber_delta=1.0,
    )

    explained_variance = metrics[CRITIC_EXPLAINED_VARIANCE_KEY]
    assert torch.isfinite(explained_variance)
    assert explained_variance.item() == 0.0


def test_critic_explained_variance_combines_microbatch_statistics():
    returns_parts = (torch.tensor([0.0, 1.0]), torch.tensor([2.0, 3.0]))
    values_parts = (torch.tensor([0.0, 1.0]), torch.tensor([1.0, 4.0]))
    stats_parts = [
        compute_critic_explained_variance_stats(returns, values)
        for returns, values in zip(returns_parts, values_parts, strict=True)
    ]
    metrics = {
        key: [part[key] for part in stats_parts]
        for key in CRITIC_EXPLAINED_VARIANCE_STAT_KEYS
    }
    metrics[CRITIC_EXPLAINED_VARIANCE_KEY] = [
        torch.tensor(0.0),
        torch.tensor(0.0),
    ]

    combined_stats = pop_critic_explained_variance_stats(metrics)
    explained_variance = compute_critic_explained_variance_from_stats(combined_stats)

    assert CRITIC_EXPLAINED_VARIANCE_KEY not in metrics
    assert not any(key in metrics for key in CRITIC_EXPLAINED_VARIANCE_STAT_KEYS)
    assert torch.isclose(explained_variance, torch.tensor(0.6))
