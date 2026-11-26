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

"""Synthetic co-training datasets, rank streams and domain loss diagnostics."""

import copy
import itertools
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rlinf.data.openpi_cotrain import (
    CoTrainConfig,
    CoTrainDatasetSpec,
    FrameIndexDataset,
    MixtureDataset,
    StratifiedBatchSampler,
    create_cotrain_data_loader,
    domain_loss_metrics,
    episodes_for_source,
    frame_indices_for_episodes,
    validate_dataset_compatibility,
)


class SyntheticDataset:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {"value": np.int64(index)}


def test_interleaved_episode_subset_preserves_full_dataset_indices():
    frames = frame_indices_for_episodes([0, 1, 0, 2, 1, 3, 2, 1, 3], [1, 3])
    np.testing.assert_array_equal(frames, [1, 4, 5, 7, 8])
    subset = FrameIndexDataset(SyntheticDataset(9), frames)
    assert [subset[i]["value"] for i in range(len(subset))] == [1, 4, 5, 7, 8]
    assert subset[-1]["value"] == 8


@pytest.mark.parametrize("episodes", [[], [7]])
def test_invalid_episode_selection_raises(episodes):
    with pytest.raises(ValueError):
        frame_indices_for_episodes([0, 1, 0], episodes)


def test_episode_source_annotations(tmp_path):
    (tmp_path / "meta").mkdir()
    rows = [
        {"episode_index": 0, "source": "realworld"},
        {"episode_index": 1, "source": "replay"},
        {"episode_index": 2, "source": "realworld"},
    ]
    (tmp_path / "meta/episode_sources.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows)
    )
    assert episodes_for_source(tmp_path, "realworld") == (0, 2)
    with pytest.raises(ValueError, match="No episodes"):
        episodes_for_source(tmp_path, "unknown")


@pytest.mark.parametrize("sim_weight,n_sim", [(0.0, 0), (0.3, 3), (0.5, 5), (1.0, 10)])
def test_each_rank_batch_has_exact_domain_counts(sim_weight, n_sim):
    for rank in range(3):
        batches = list(
            StratifiedBatchSampler(
                31,
                19,
                10,
                sim_weight,
                seed=42,
                rank=rank,
                world_size=3,
                num_batches=17,
            )
        )
        assert all(
            len(batch) == 10 and sum(i >= 31 for i in batch) == n_sim
            for batch in batches
        )


def test_stratified_sampler_rank_determinism_and_disjointness():
    streams = []
    for rank in range(3):
        kwargs = {"seed": 12, "rank": rank, "world_size": 3, "num_batches": 20}
        sampler = StratifiedBatchSampler(31, 19, 8, **kwargs)
        sampler.set_epoch(4)
        batches = list(sampler)
        same = StratifiedBatchSampler(31, 19, 8, **kwargs)
        same.set_epoch(4)
        assert list(same) == batches
        same.set_epoch(5)
        assert list(same) != batches
        streams.append(set(itertools.chain.from_iterable(batches)))
    assert all(a.isdisjoint(b) for a, b in itertools.combinations(streams, 2))


def test_tiny_domain_allows_unavoidable_overlap_without_empty_rank():
    for rank in range(4):
        sampler = StratifiedBatchSampler(1, 2, 8, rank=rank, world_size=4)
        batch = next(iter(sampler))
        assert len(batch) == 8
        assert sum(i == 0 for i in batch) == 4


@pytest.mark.parametrize("weight", [0.5, 0.3])
def test_non_integer_micro_batch_split_raises(weight):
    with pytest.raises(ValueError, match="must be an integer"):
        StratifiedBatchSampler(4, 4, 7, weight)


@pytest.mark.parametrize("weight", [-0.1, 1.1, float("nan"), float("inf")])
def test_out_of_range_weight_raises(weight):
    with pytest.raises(ValueError, match="sim_weight"):
        StratifiedBatchSampler(4, 4, 8, weight)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_epoch_offset_resume_is_independent_of_prefetch(num_workers):
    def loader():
        sampler = StratifiedBatchSampler(
            50, 40, 8, seed=9, rank=1, world_size=2, num_batches=12
        )
        sampler.set_epoch(3)
        return torch.utils.data.DataLoader(
            MixtureDataset(SyntheticDataset(50), SyntheticDataset(40)),
            batch_sampler=sampler,
            num_workers=num_workers,
            generator=torch.Generator().manual_seed(23),
        )

    original = iter(loader())
    for _ in range(4):
        next(original)
    expected = list(original)
    restored = iter(loader())
    for _ in range(4):
        next(restored)
    actual = list(restored)
    assert len(expected) == len(actual) == 8
    for a, b in zip(expected, actual, strict=True):
        torch.testing.assert_close(a["value"], b["value"])
        torch.testing.assert_close(a["domain"], b["domain"])


def test_mixture_domain_and_indices():
    mixed = MixtureDataset(SyntheticDataset(2), SyntheticDataset(3))
    assert len(mixed) == 5
    assert [mixed[i]["domain"] for i in range(5)] == [0, 0, 1, 1, 1]
    assert mixed[-1]["value"] == 2
    with pytest.raises(IndexError):
        mixed[5]


@pytest.mark.parametrize("field", ["fps", "keys", "tactile_marker_motion", "actions"])
def test_compatibility_checks_all_shapes_and_fps(field):
    real = SimpleNamespace(
        fps=20,
        features={
            "state": {"shape": [13]},
            "actions": {"shape": [13]},
            "tactile_marker_motion": {"shape": [9, 440, 2]},
        },
    )
    sim = copy.deepcopy(real)
    validate_dataset_compatibility(real, sim)
    if field == "fps":
        sim.fps = 10
    elif field == "keys":
        sim.features.pop("state")
    else:
        sim.features[field]["shape"][-1] += 1
    with pytest.raises(ValueError):
        validate_dataset_compatibility(real, sim)


def test_specs_require_two_domains_and_mutually_exclusive_selections():
    with pytest.raises(ValueError, match="requires real and sim"):
        CoTrainConfig.from_mapping({"real": {"dataset_path": "synthetic"}})
    with pytest.raises(ValueError, match="mutually exclusive"):
        CoTrainDatasetSpec("synthetic", episodes=(1,), episode_source="realworld")


def test_native_frames_builder_uses_each_domains_timestamps(monkeypatch, tmp_path):
    import openpi.training.data_loader as upstream

    roots = {domain: tmp_path / domain for domain in ("real", "sim")}
    for root in roots.values():
        (root / "meta").mkdir(parents=True)
        (root / "meta/info.json").write_text("{}")
    fps = {str(roots["real"]): 5, str(roots["sim"]): 20}
    monkeypatch.setattr(
        upstream.lerobot_dataset,
        "LeRobotDatasetMetadata",
        lambda repo, **kw: SimpleNamespace(
            root=kw["root"], fps=fps[repo], features={"actions": {"shape": [13]}}
        ),
    )
    timestamps = {}

    def dataset(repo, **kwargs):
        timestamps[repo] = kwargs["delta_timestamps"]["actions"]
        return SyntheticDataset(8)

    monkeypatch.setattr(upstream.lerobot_dataset, "LeRobotDataset", dataset)
    monkeypatch.setattr(upstream, "transform_dataset", lambda dataset, cfg: dataset)
    data = SimpleNamespace(
        norm_stats={"shared": object()},
        rlds_data_dir=None,
        action_sequence_keys=("actions",),
        prompt_from_task=False,
    )
    config = SimpleNamespace(
        data=SimpleNamespace(create=lambda *args: data),
        assets_dirs="unused",
        model=SimpleNamespace(action_horizon=3),
        num_workers=0,
    )
    specs = {domain: {"dataset_path": str(root)} for domain, root in roots.items()}
    with pytest.raises(ValueError, match="fps mismatch"):
        create_cotrain_data_loader(
            config,
            CoTrainConfig.from_mapping(specs),
            micro_batch_size=8,
            rank=0,
            world_size=1,
            seed=42,
        )
    cotrain = CoTrainConfig.from_mapping({**specs, "temporal_mode": "native_frames"})
    loader, _ = create_cotrain_data_loader(
        config,
        cotrain,
        micro_batch_size=8,
        rank=0,
        world_size=1,
        seed=42,
    )
    assert timestamps[str(roots["real"])] == [0.0, 0.2, 0.4]
    assert timestamps[str(roots["sim"])] == [0.0, 0.05, 0.1]
    assert next(iter(loader))["domain"].tolist().count(1) == 4
    with pytest.raises(ValueError, match="temporal_mode"):
        CoTrainConfig.from_mapping({**specs, "temporal_mode": "resample"})


@pytest.mark.parametrize("domain", [[0, 1, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1]])
def test_domain_loss_means_and_absent_domains(domain):
    losses = torch.tensor([1.0, 3.0, 5.0, 7.0], requires_grad=True)
    metrics = domain_loss_metrics({"per_sample_loss": losses}, domain)
    expected = {}
    for value, name in ((0, "loss_real"), (1, "loss_sim")):
        selected = [
            loss
            for loss, tag in zip([1.0, 3.0, 5.0, 7.0], domain, strict=True)
            if tag == value
        ]
        if selected:
            expected[name] = sum(selected) / len(selected)
    assert metrics == expected
    assert losses.grad is None


@pytest.mark.parametrize("output", [torch.tensor(2.0), {"loss": torch.tensor(2.0)}])
def test_models_without_per_sample_loss_keep_working(output):
    assert domain_loss_metrics(output, [0, 1]) == {}
    assert domain_loss_metrics(output, None) == {}


def test_builder_uses_full_indices_shared_transforms_and_domain_after_transform(
    monkeypatch, tmp_path
):
    import openpi.training.data_loader as upstream

    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text("{}")
    (tmp_path / "meta/episode_sources.jsonl").write_text(
        "\n".join(
            json.dumps({"episode_index": i, "source": source})
            for i, source in enumerate(["realworld", "replay", "realworld", "replay"])
        )
    )
    calls, configs = [], []
    meta = SimpleNamespace(root=tmp_path, fps=20, features={"actions": {"shape": [13]}})
    monkeypatch.setattr(
        upstream.lerobot_dataset, "LeRobotDatasetMetadata", lambda *a, **kw: meta
    )

    class FullDataset(SyntheticDataset):
        hf_dataset = {"episode_index": [0, 1, 0, 2, 1, 3, 2, 1, 3]}

    def full_dataset(*args, **kwargs):
        assert "episodes" not in kwargs
        assert kwargs["delta_timestamps"]["actions"] == [0.0, 0.05, 0.1]
        calls.append(args)
        return FullDataset(9)

    def transform(dataset, config):
        configs.append(config)
        assert "domain" not in dataset[0]
        return dataset

    monkeypatch.setattr(upstream.lerobot_dataset, "LeRobotDataset", full_dataset)
    monkeypatch.setattr(upstream, "transform_dataset", transform)
    data = SimpleNamespace(
        norm_stats={"shared": object()},
        rlds_data_dir=None,
        action_sequence_keys=("actions",),
        prompt_from_task=False,
    )
    config = SimpleNamespace(
        data=SimpleNamespace(create=lambda *args: data),
        assets_dirs="unused",
        model=SimpleNamespace(action_horizon=3),
        num_workers=0,
    )
    cotrain = CoTrainConfig(
        CoTrainDatasetSpec(str(tmp_path), episode_source="realworld"),
        CoTrainDatasetSpec(str(tmp_path), episode_source="replay"),
    )
    loader, got = create_cotrain_data_loader(
        config, cotrain, micro_batch_size=4, rank=0, world_size=1, seed=5
    )
    assert got is data and configs == [data, data]
    assert len(calls) == 1
    np.testing.assert_array_equal(loader.dataset.real.frame_indices, [0, 2, 3, 6])
    np.testing.assert_array_equal(loader.dataset.sim.frame_indices, [1, 4, 5, 7, 8])
    assert next(iter(loader))["domain"].tolist().count(1) == 2
