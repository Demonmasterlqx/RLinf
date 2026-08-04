#!/usr/bin/env python3
"""Audit raw and model-side shapes for the Tabero Firm OpenPI SFT pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import openpi.training.data_loader as openpi_data_loader
import pyarrow.parquet as pq

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config


def _shape(value) -> list[int]:
    return list(value.shape)


def audit_batch(dataset_path: Path, norm_stats_path: Path) -> dict:
    info = json.loads((dataset_path / "meta" / "info.json").read_text())
    first_parquet = sorted((dataset_path / "data").rglob("*.parquet"))[0]
    raw = pq.read_table(
        first_parquet,
        columns=["state", "actions", "tactile_marker_motion"],
    ).slice(0, 1)
    raw_shapes = {
        key: list(info["features"][key]["shape"])
        for key in ("state", "actions", "tactile_marker_motion")
    }
    for key, expected in raw_shapes.items():
        actual = raw.column(key)[0].as_py()
        if key == "tactile_marker_motion":
            actual_shape = [len(actual), len(actual[0]), len(actual[0][0])]
        else:
            actual_shape = [len(actual)]
        if actual_shape != expected:
            raise ValueError(
                f"Raw {key} shape mismatch: expected={expected}, actual={actual_shape}"
            )

    config = get_openpi_config(
        "pi0_lora_tacfield_tabero",
        model_path="/data/home/sim6g/code/tabero/models/pi0_base",
        batch_size=1,
        repo_id=str(dataset_path),
        data_kwargs={"norm_stats_path": str(norm_stats_path)},
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = openpi_data_loader.create_torch_dataset(
        data_config, config.model.action_horizon, config.model
    )
    training_input_dataset = openpi_data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
        ],
    )
    training_input = training_input_dataset[0]
    training_input_shapes = {
        "state": _shape(training_input["state"]),
        "actions": _shape(training_input["actions"]),
        "tactile_prefix": _shape(training_input["tactile_prefix"]),
    }
    expected_training_input_shapes = {
        "state": [7],
        "actions": [50, 13],
        "tactile_prefix": [9, 396],
    }
    if training_input_shapes != expected_training_input_shapes:
        raise ValueError(
            "Training-input shape mismatch: "
            f"expected={expected_training_input_shapes}, "
            f"actual={training_input_shapes}"
        )

    loader = openpi_data_loader.create_data_loader(
        config, framework="pytorch", shuffle=False
    )
    batch = next(iter(loader._data_loader))
    model_shapes = {
        "state": _shape(batch["state"])[1:],
        "actions": _shape(batch["actions"])[1:],
        "tactile_prefix": _shape(batch["tactile_prefix"])[1:],
    }
    expected_model_shapes = {
        "state": [32],
        "actions": [50, 32],
        "tactile_prefix": [9, 396],
    }
    if model_shapes != expected_model_shapes:
        raise ValueError(
            "Model-side batch shape mismatch: "
            f"expected={expected_model_shapes}, actual={model_shapes}"
        )

    action_padding = batch["actions"][..., 13:]
    if action_padding.count_nonzero().item() != 0:
        raise ValueError("Model action padding dimensions 13:32 must be zero.")
    return {
        "dataset": str(dataset_path),
        "raw_frame": raw_shapes,
        "training_input": training_input_shapes,
        "model": model_shapes,
        "action_padding_13_32_all_zero": True,
        "action_horizon": config.model.action_horizon,
        "num_images_in_input": 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--norm-stats-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_batch(args.dataset_path.resolve(), args.norm_stats_path.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
