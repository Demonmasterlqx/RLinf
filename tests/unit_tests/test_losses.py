import torch
import pytest

from rlinf.algorithms.losses import compute_ppo_critic_loss


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
