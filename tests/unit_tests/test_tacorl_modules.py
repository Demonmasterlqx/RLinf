"""Synthetic TacCoRL module and conversion checks."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from torch import nn

from rlinf.models.embodiment.openpi import openpi_action_model
from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0ForRLActionPrediction,
)
from rlinf.models.embodiment.openpi.tactile_encoder import (
    MarkerTokenEncoder,
    TactilePrefixCrossAttention,
)


def _load_reference():
    t2vla_root = os.environ.get("T2VLA_ROOT")
    if t2vla_root is None:
        t2vla_root = Path(__file__).resolve().parents[2].parent / "T2-VLA"
    path = Path(t2vla_root) / "src/openpi/models_pytorch/tactile_encoder_pytorch.py"
    if not path.is_file():
        pytest.skip("T2-VLA reference checkout is unavailable")
    spec = importlib.util.spec_from_file_location("t2vla_tactile_encoder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _taccorl_config(**overrides):
    values = {
        "tactile_fusion": "taccorl",
        "tactile_type": "EXPERT_HIS_C_FUT",
        "tactile_streams": ("tactile_prefix", "tactile_suffix"),
        "tactile_suffix_placement": "suffix",
        "tactile_prefix_dim_in": 4,
        "tactile_prefix_history": 2,
        "tactile_prefix_use_reference_frame": False,
        "tactile_prefix_diff_from_reference": False,
        "tactile_prefix_num_tokens": 3,
        "tactile_suffix_num_tokens": 2,
        "tactile_prefix_cross_attn": True,
        "tactile_cross_attn_heads": 2,
        "tactile_prefix_encoder_type": "tcn",
        "paligemma_variant": "tiny",
        "action_expert_variant": "tiny",
        "pi05": True,
        "action_horizon": 4,
        "action_dim": 6,
        "action_chunk": 4,
        "action_env_dim": 6,
        "use_rlt": False,
        "use_dsrl": False,
        "is_nft": False,
        "image_noise_std": 0.0,
        "tactile_prefix_encoder": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _bare_taccorl_model(**config_overrides):
    model = OpenPi0ForRLActionPrediction.__new__(OpenPi0ForRLActionPrediction)
    nn.Module.__init__(model)
    model.config = _taccorl_config(**config_overrides)
    model.marker_token_encoder = MarkerTokenEncoder(
        in_dim=4,
        history_len=2,
        has_reference_frame=False,
        diff_from_reference=False,
        prefix_width=8,
        suffix_width=6,
        prefix_num_tokens=3,
        suffix_num_tokens=2,
    )
    model.tactile_cross_attn = TactilePrefixCrossAttention(8, 2)
    model.tactile_prefix_encoder = None
    model.gradient_checkpointing_enabled = False
    model._rlinf_gradient_checkpointing_enabled = False
    model.register_parameter("_test_parameter", nn.Parameter(torch.zeros(())))
    # sft_forward would otherwise try to access a real Gemma backbone while
    # disabling checkpointing.  The synthetic tests do not need that seam.
    model.gradient_checkpointing_disable = lambda: None
    return model


def _install_embedding_stubs(monkeypatch):
    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        del img_masks, lang_tokens, lang_masks
        batch_size = images[0].shape[0]
        base = torch.arange(batch_size * 4 * 8, dtype=torch.float32).reshape(
            batch_size, 4, 8
        )
        return (
            base,
            torch.ones(batch_size, 4, dtype=torch.bool),
            torch.zeros(batch_size, 4, dtype=torch.bool),
        )

    def embed_suffix(self, state, noisy_actions, timestep):
        del state, timestep
        batch_size, action_horizon = noisy_actions.shape[:2]
        base = torch.arange(
            batch_size * action_horizon * 6,
            dtype=noisy_actions.dtype,
            device=noisy_actions.device,
        ).reshape(batch_size, action_horizon, 6)
        return (
            base,
            torch.ones(batch_size, action_horizon, dtype=torch.bool),
            torch.zeros(batch_size, action_horizon, dtype=torch.bool),
            None,
        )

    monkeypatch.setattr(PI0Pytorch, "embed_prefix", embed_prefix)
    monkeypatch.setattr(PI0Pytorch, "embed_suffix", embed_suffix)


def _prefix_inputs(batch_size):
    return (
        [torch.zeros(batch_size, 1, 1, 1)],
        [torch.ones(batch_size, dtype=torch.bool)],
        None,
        None,
    )


def _encoder(cls):
    return cls(
        in_dim=6,
        history_len=2,
        has_reference_frame=True,
        diff_from_reference=False,
        prefix_width=8,
        suffix_width=4,
        prefix_num_tokens=3,
        suffix_num_tokens=2,
    )


def test_marker_encoder_matches_reference_with_shared_weights():
    reference = _load_reference()
    torch.manual_seed(7)
    ours = _encoder(MarkerTokenEncoder)
    torch.manual_seed(7)
    expected = _encoder(reference.MarkerTokenEncoder)
    expected.load_state_dict(ours.state_dict())
    tactile = torch.randn(2, 3, 6)
    actual = ours(tactile)
    wanted = expected(tactile)
    assert set(ours.state_dict()) == set(expected.state_dict())
    for got, want in zip(actual, wanted, strict=True):
        torch.testing.assert_close(got, want)


def test_cross_attention_matches_reference_and_closed_gate_is_identity():
    reference = _load_reference()
    torch.manual_seed(9)
    ours = TactilePrefixCrossAttention(8, 2)
    torch.manual_seed(9)
    expected = reference.TactilePrefixCrossAttention(8, 2)
    expected.load_state_dict(ours.state_dict())
    with torch.no_grad():
        ours.alpha.fill_(0.7)
        expected.alpha.copy_(ours.alpha)
    base = torch.randn(2, 5, 8)
    tactile = torch.randn(2, 3, 8)
    mask = torch.tensor([[True, True, False], [False, False, False]])
    torch.testing.assert_close(ours(base, tactile, mask), expected(base, tactile, mask))
    torch.testing.assert_close(ours(base, tactile, torch.zeros_like(mask)), base)


def test_taccorl_converter_transposes_linear_kernels():
    pytest.importorskip("openpi")
    from rlinf.utils.ckpt_convertor.convert_tabero_openpi_jax_to_safetensors import (
        convert_taccorl_params,
    )

    def linear(in_dim, out_dim, offset):
        values = torch.arange(in_dim * out_dim, dtype=torch.float32).reshape(in_dim, out_dim) + offset
        return {"kernel": values.numpy(), "bias": torch.arange(out_dim).numpy() + offset}

    params = {
        "marker_token_encoder": {
            "blocks": {
                "block_0": {
                    "kernels": {"kernel_0": linear(2, 4, 1)},
                    "residual_proj": linear(2, 4, 2),
                },
                "block_1": {"kernels": {"kernel_0": linear(4, 4, 3)}},
            },
            "prefix_proj": linear(4, 6, 4),
            "suffix_proj": linear(4, 8, 5),
            "prefix_token_embedding": torch.ones(3, 6).numpy(),
            "suffix_token_embedding": torch.ones(2, 8).numpy() * 2,
        },
        "tactile_cross_attn": {
            "norm": {"scale": torch.ones(8).numpy(), "bias": torch.zeros(8).numpy()},
            "query": linear(8, 8, 6),
            "key": linear(8, 8, 7),
            "value": linear(8, 8, 8),
            "out": linear(8, 8, 9),
            "alpha": torch.zeros(()).numpy(),
        },
    }
    converted = convert_taccorl_params(params)
    assert converted["marker_token_encoder.blocks.0.kernels.0.weight"].shape == (4, 2)
    assert converted["tactile_cross_attn.norm.weight"].shape == (8,)
    assert set(converted) == {
        "marker_token_encoder.blocks.0.kernels.0.weight",
        "marker_token_encoder.blocks.0.kernels.0.bias",
        "marker_token_encoder.blocks.0.residual_proj.weight",
        "marker_token_encoder.blocks.0.residual_proj.bias",
        "marker_token_encoder.blocks.1.kernels.0.weight",
        "marker_token_encoder.blocks.1.kernels.0.bias",
        "marker_token_encoder.prefix_proj.weight",
        "marker_token_encoder.prefix_proj.bias",
        "marker_token_encoder.suffix_proj.weight",
        "marker_token_encoder.suffix_proj.bias",
        "marker_token_encoder.prefix_token_embedding",
        "marker_token_encoder.suffix_token_embedding",
        "tactile_cross_attn.norm.weight",
        "tactile_cross_attn.norm.bias",
        "tactile_cross_attn.query.weight",
        "tactile_cross_attn.query.bias",
        "tactile_cross_attn.key.weight",
        "tactile_cross_attn.key.bias",
        "tactile_cross_attn.value.weight",
        "tactile_cross_attn.value.bias",
        "tactile_cross_attn.out.weight",
        "tactile_cross_attn.out.bias",
        "tactile_cross_attn.alpha",
    }


def test_embed_prefix_closed_contact_keeps_base_tokens_and_masks_tactile_tokens(
    monkeypatch,
):
    _install_embedding_stubs(monkeypatch)
    model = _bare_taccorl_model()
    with torch.no_grad():
        model.tactile_cross_attn.alpha.fill_(1.0)
    images, image_masks, lang_tokens, lang_masks = _prefix_inputs(2)
    tactile = torch.randn(2, 2, 4)
    marker_tokens = model._encode_marker_tokens(tactile)

    base, _, _ = PI0Pytorch.embed_prefix(
        model, images, image_masks, lang_tokens, lang_masks
    )
    result, pad_masks, att_masks = model.embed_prefix(
        images,
        image_masks,
        lang_tokens,
        lang_masks,
        marker_tokens=marker_tokens,
        tactile_contact=torch.tensor([False, True]),
    )

    # Closed gate (sample 0) leaves base tokens untouched; open gate (sample 1, alpha=1) must change them.
    torch.testing.assert_close(result[0, :4], base[0])
    assert not torch.allclose(result[1, :4], base[1])
    assert pad_masks.shape == (2, 7)
    assert pad_masks[0, 4:].tolist() == [False, False, False]
    assert pad_masks[1, 4:].tolist() == [True, True, True]
    assert att_masks[:, 4:].eq(False).all()

    open_result, open_pad_masks, _ = model.embed_prefix(
        images,
        image_masks,
        lang_tokens,
        lang_masks,
        marker_tokens=marker_tokens,
        tactile_contact=None,
    )
    assert open_pad_masks[:, 4:].all()
    assert not torch.equal(open_result[1, :4], base[1])
    explicit_open, _, _ = model.embed_prefix(
        images,
        image_masks,
        lang_tokens,
        lang_masks,
        marker_tokens=marker_tokens,
        tactile_contact=torch.ones(2, dtype=torch.bool),
    )
    torch.testing.assert_close(open_result, explicit_open)


def test_embed_suffix_places_marker_tokens_before_actions_with_contact_padding(
    monkeypatch,
):
    _install_embedding_stubs(monkeypatch)
    model = _bare_taccorl_model()
    noisy_actions = torch.zeros(2, 4, 6)
    tactile = torch.randn(2, 2, 4)
    marker_tokens = model._encode_marker_tokens(tactile)

    suffix, pad_masks, att_masks, _ = model.embed_suffix(
        torch.zeros(2, 7),
        noisy_actions,
        torch.zeros(2),
        marker_tokens=marker_tokens,
        tactile_contact=torch.tensor([False, True]),
    )

    assert suffix.shape == (2, 6, 6)
    assert pad_masks[:, :2].tolist() == [[False, False], [True, True]]
    assert att_masks[:, :6].tolist() == [
        [True, False, False, False, False, False],
        [True, False, False, False, False, False],
    ]
    assert pad_masks[:, 2:].all()


def test_prefix_cache_attention_masks_closed_tactile_keys(monkeypatch):
    _install_embedding_stubs(monkeypatch)
    model = _bare_taccorl_model()
    captured = {}

    class PrefixBackbone:
        def __init__(self):
            self.paligemma = SimpleNamespace(
                language_model=SimpleNamespace(
                    config=SimpleNamespace(), gradient_checkpointing=False
                )
            )

        def forward(self, **kwargs):
            captured["attention_mask"] = kwargs["attention_mask"]
            return (kwargs["inputs_embeds"][0], None), "cache"

    model.paligemma_with_expert = PrefixBackbone()
    images, image_masks, lang_tokens, lang_masks = _prefix_inputs(2)
    tactile = torch.randn(2, 2, 4)
    prefix_output, pad_masks, cache = model._build_prefix_cache(
        images,
        image_masks,
        lang_tokens,
        lang_masks,
        tactile,
        marker_tokens=model._encode_marker_tokens(tactile),
        tactile_contact=torch.tensor([False, True]),
    )

    assert cache == "cache"
    assert prefix_output.shape == (2, 7, 8)
    assert pad_masks[0, -3:].eq(False).all()
    attention_mask = captured["attention_mask"]
    assert attention_mask.shape == (2, 1, 7, 7)
    assert (attention_mask[0, 0, :, -3:] < -1e30).all()
    assert attention_mask[1, 0, :, -3:].eq(0).all()


@pytest.mark.parametrize(
    "invoke",
    [
        lambda model: model._build_rlt_prefix_cache(None, train=False),
        lambda model: model._select_rlt_prefix_embeddings(None, None, None),
        lambda model: model._extract_rlt_prefix_embeddings(None, train=False),
        lambda model: model.extract_rlt_obs(None),
        lambda model: model.nft_forward({}),
        lambda model: model.sample_actions(None),
        lambda model: model.predict_action_batch({}),
        lambda model: model.sac_forward(),
        lambda model: model.sac_q_forward(),
        lambda model: model.freeze_non_dsrl_parameters(),
    ],
    ids=[
        "rlt_cache",
        "rlt_select",
        "rlt_extract",
        "rlt_obs",
        "nft_forward",
        "sample_actions",
        "dsrl_predict",
        "sac_forward",
        "sac_q_forward",
        "dsrl_freeze",
    ],
)
def test_taccorl_rejects_unsupported_rl_paths(invoke):
    model = _bare_taccorl_model(use_rlt=True, use_dsrl=True, is_nft=True)

    with pytest.raises(NotImplementedError, match="TacCoRL"):
        invoke(model)


def test_sft_forward_returns_per_sample_loss_with_matching_mean(monkeypatch):
    model = _bare_taccorl_model(
        tactile_type="EXPERT_HIS_C_FUT",
        use_rlt=False,
        action_dim=4,
        effective_action_dim=4,
        tactile_dim=1,
        tactile_loss_weight=1.0,
        padding_loss_weight=1.0,
        expert_his_c_fut_loss_mode="weighted_full",
        action_env_dim=4,
        action_chunk=3,
    )
    monkeypatch.setattr(openpi_action_model, "register_pytree_dataclasses", lambda _: None)
    monkeypatch.setattr(openpi_action_model, "tree_map", lambda fn, value: value)
    per_element = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    model._sft_forward_taccorl = lambda observation, actions: per_element
    observation = SimpleNamespace(
        tactile_prefix=torch.zeros(2, 2, 4),
        tactile_contact=torch.tensor([False, True]),
    )

    result = model.sft_forward(
        {"observation": observation, "actions": torch.zeros(2, 3, 4)}
    )

    assert result["per_sample_loss"].shape == (2,)
    torch.testing.assert_close(result["loss"], result["per_sample_loss"].mean())
    torch.testing.assert_close(
        result["per_sample_loss"], per_element.mean(dim=(1, 2))
    )


def test_image_noise_is_train_only_bounded_and_zero_is_a_noop(monkeypatch):
    image = torch.full((2, 2, 3), 0.8)
    state = torch.zeros(2, 1)

    def preprocess(self, observation, *, train=True):
        del observation, train
        return ([image.clone()], [torch.ones(2, dtype=torch.bool)], None, None, state, None)

    monkeypatch.setattr(PI0Pytorch, "_preprocess_observation", preprocess)
    monkeypatch.setattr(torch, "randn_like", lambda value: torch.ones_like(value))
    model = _bare_taccorl_model(image_noise_std=0.2)

    train_result = model._preprocess_observation(object(), train=True)
    eval_result = model._preprocess_observation(object(), train=False)
    torch.testing.assert_close(eval_result[0][0], image)
    assert train_result[0][0].min() >= -1
    assert train_result[0][0].max() <= 1
    torch.testing.assert_close(train_result[0][0], torch.ones_like(image))

    model.config.image_noise_std = 0.0
    zero_result = model._preprocess_observation(object(), train=True)
    torch.testing.assert_close(zero_result[0][0], image)


def test_legacy_tactile_encoder_contract_is_unchanged(monkeypatch):
    monkeypatch.setattr(
        openpi_action_model._gemma,
        "get_config",
        lambda _: SimpleNamespace(width=8),
    )
    legacy = _bare_taccorl_model(
        tactile_fusion="none",
        tactile_streams=("tactile_prefix",),
        tactile_prefix_encoder_type="tcn",
        tactile_prefix_cross_attn=False,
    )
    del legacy.marker_token_encoder
    del legacy.tactile_cross_attn
    legacy._init_tactile_prefix_encoder()

    assert legacy.tactile_prefix_encoder is not None
    assert not hasattr(legacy, "marker_token_encoder")
    legacy_keys = [key for key in legacy.state_dict() if key != "_test_parameter"]
    assert all(
        key.startswith("tactile_prefix_encoder.")
        for key in legacy_keys
    )

    taccorl = _bare_taccorl_model()
    taccorl._init_tactile_prefix_encoder()
    assert taccorl.tactile_prefix_encoder is None
    assert any(key.startswith("marker_token_encoder.") for key in taccorl.state_dict())
    assert any(key.startswith("tactile_cross_attn.") for key in taccorl.state_dict())
