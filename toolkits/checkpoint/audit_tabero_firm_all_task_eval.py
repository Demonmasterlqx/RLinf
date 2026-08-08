#!/usr/bin/env python3
"""Cross-check a completed 9-task Tabero Firm evaluation."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

_TASK_IDS = (0, 1, 2, 3, 5, 6, 7, 8, 9)


def audit_evaluation(json_path: Path, txt_path: Path, client_log_path: Path) -> dict:
    """Validate raw JSON, TXT, client progress, and SR arithmetic."""
    payload = json.loads(json_path.read_text())
    results = payload.get("results")
    if not isinstance(results, dict):
        raise ValueError("Evaluation JSON has no results mapping.")
    expected_keys = {f"libero_object_task{task_id}" for task_id in _TASK_IDS}
    if set(results) != expected_keys:
        raise ValueError(
            f"Evaluation task keys differ: expected={sorted(expected_keys)}, "
            f"actual={sorted(results)}"
        )

    task_results = []
    total_successes = 0
    total_episodes = 0
    for task_id in _TASK_IDS:
        result = results[f"libero_object_task{task_id}"]
        if result.get("status") != "completed":
            raise ValueError(f"Task {task_id} is not completed: {result.get('status')}")
        if result.get("total_experiments") != 50:
            raise ValueError(f"Task {task_id} does not contain 50 experiments.")
        episodes = result.get("episodes")
        if not isinstance(episodes, list) or len(episodes) != 50:
            raise ValueError(f"Task {task_id} does not contain 50 episode records.")
        indexes = [episode.get("experiment_index") for episode in episodes]
        if indexes != list(range(50)):
            raise ValueError(f"Task {task_id} experiment indexes are incomplete.")
        successes = sum(episode.get("success") is True for episode in episodes)
        if result.get("successful_experiments") != successes:
            raise ValueError(f"Task {task_id} success count differs from episodes.")
        expected_rate = successes / 50 * 100
        if not math.isclose(result.get("success_rate"), expected_rate, abs_tol=1e-9):
            raise ValueError(f"Task {task_id} success-rate arithmetic differs.")
        if result.get("metrics_status") != "complete":
            raise ValueError(f"Task {task_id} episode metrics are incomplete.")
        total_successes += successes
        total_episodes += 50
        task_results.append(
            {
                "task_id": task_id,
                "successes": successes,
                "episodes": 50,
                "success_rate": expected_rate,
                "execution_time": result.get("execution_time"),
                "step_statistics": result.get("step_statistics"),
            }
        )

    client_log = client_log_path.read_text(errors="replace")
    starts = re.findall(r"\[(\d+)/50\] Starting experiment", client_log)
    expected_starts = [str(index) for index in range(1, 51)] * len(_TASK_IDS)
    if starts != expected_starts:
        raise ValueError(
            f"Client log episode starts are incomplete or unordered: {len(starts)}"
        )
    for task_id in _TASK_IDS:
        marker = f"TASK COMPLETED: libero_object - Task {task_id}"
        if client_log.count(marker) != 1:
            raise ValueError(
                f"Client log completion marker differs for Task {task_id}."
            )
    if client_log.count("Progress: 9/9 tasks completed") != 1:
        raise ValueError(
            "Client log does not contain exactly one 9/9 completion marker."
        )

    txt = txt_path.read_text(errors="replace")
    for result in task_results:
        marker = f"Task {result['task_id']} ("
        fraction = f"({result['successes']}/50)"
        if marker not in txt or fraction not in txt:
            raise ValueError(
                f"TXT summary is missing Task {result['task_id']} or {fraction}."
            )

    return {
        "status": "completed_and_cross_checked",
        "json": str(json_path),
        "txt": str(txt_path),
        "client_log": str(client_log_path),
        "tasks": task_results,
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "overall_success_rate": total_successes / total_episodes * 100,
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
