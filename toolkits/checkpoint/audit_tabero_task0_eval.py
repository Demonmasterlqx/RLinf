#!/usr/bin/env python3
"""Cross-check a completed Task 0 plain-prompt Tabero evaluation."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


def audit_evaluation(json_path: Path, txt_path: Path, client_log_path: Path) -> dict:
    """Validate raw JSON, TXT, episode coverage, protocol, and SR arithmetic."""

    payload = json.loads(json_path.read_text())
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Evaluation JSON has no metadata mapping.")
    expected_metadata = {
        "prompt_mode": "plain",
        "prompt_adverbs": [],
        "num_total_experiments": 50,
        "num_success_steps": 8,
        "replan_steps": 10,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Evaluation metadata {key!r} differs: "
                f"expected={expected!r}, actual={metadata.get(key)!r}"
            )
    max_policy = metadata.get("max_inference_steps_policy")
    if not isinstance(max_policy, dict) or max_policy.get("libero_object") != 30:
        raise ValueError("Evaluation metadata does not specify 30 Task 0 chunks.")

    results = payload.get("results")
    if not isinstance(results, dict) or set(results) != {"libero_object_task0"}:
        raise ValueError("Evaluation JSON must contain exactly libero_object_task0.")
    result = results["libero_object_task0"]
    if result.get("status") != "completed":
        raise ValueError(f"Task 0 is not completed: {result.get('status')}")
    if result.get("total_experiments") != 50:
        raise ValueError("Task 0 does not contain 50 experiments.")
    episodes = result.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 50:
        raise ValueError("Task 0 does not contain 50 episode records.")
    if [episode.get("experiment_index") for episode in episodes] != list(range(50)):
        raise ValueError("Task 0 experiment indexes are incomplete or unordered.")
    if [episode.get("hdf5_episode_index") for episode in episodes] != list(range(50)):
        raise ValueError("Task 0 HDF5 reset indexes are incomplete or unordered.")
    if result.get("metrics_status") != "complete" or result.get("metrics_warnings"):
        raise ValueError("Task 0 episode metrics are incomplete or contain warnings.")

    successes = sum(episode.get("success") is True for episode in episodes)
    expected_rate = successes / 50 * 100.0
    if result.get("successful_experiments") != successes:
        raise ValueError("Task 0 success count differs from episode records.")
    if not math.isclose(result.get("success_rate"), expected_rate, abs_tol=1.0e-9):
        raise ValueError("Task 0 success-rate arithmetic differs.")

    client_log = client_log_path.read_text(errors="replace")
    starts = re.findall(r"\[(\d+)/50\] Starting experiment", client_log)
    if starts != [str(index) for index in range(1, 51)]:
        raise ValueError("Client log episode starts are incomplete or unordered.")
    for marker in (
        "TASK COMPLETED: libero_object - Task 0",
        "Progress: 1/1 tasks completed",
        f"Success Rate: {expected_rate:.2f}%",
    ):
        if client_log.count(marker) != 1:
            raise ValueError(f"Client log must contain exactly one {marker!r}.")

    txt = txt_path.read_text(errors="replace")
    if "Task 0 (" not in txt or f"({successes}/50)" not in txt:
        raise ValueError("TXT summary does not match Task 0 success count.")

    return {
        "status": "completed_and_cross_checked",
        "json": str(json_path),
        "txt": str(txt_path),
        "client_log": str(client_log_path),
        "successes": successes,
        "episodes": 50,
        "success_rate": expected_rate,
        "success_force_metrics": {
            "squeeze_avg_pred": result.get("avg_squeeze_pred"),
            "squeeze_avg_meas": result.get("avg_squeeze_meas"),
            "squeeze_max_pred": result.get("task_squeeze_max_mean"),
            "squeeze_max_meas": result.get("task_squeeze_max_meas_mean"),
            "ap_avg_pred": result.get("task_app_mean_mean"),
            "ap_avg_meas": result.get("task_ap_mean_meas_mean"),
            "ap_max_pred": result.get("task_app_max_mean"),
            "ap_max_meas": result.get("task_ap_max_meas_mean"),
        },
        "damage_threshold_statistics": result.get("damage_threshold_statistics"),
        "step_statistics": result.get("step_statistics"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--txt", type=Path, required=True)
    parser.add_argument("--client-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_evaluation(
        args.json.resolve(), args.txt.resolve(), args.client_log.resolve()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
