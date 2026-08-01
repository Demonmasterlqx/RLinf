# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Host-memory stress audit for the Tabero compact DSRL replay ring.

This is intentionally a standalone stress tool rather than a pytest test. The
formal defaults materialize 75,600 transitions and an approximately 8 GiB
rank-local replay checkpoint.
"""

import argparse
import gc
import json
import os
import resource
import threading
import time
from pathlib import Path

import torch

from rlinf.data.dsrl_replay_buffer import CompactDSRLReplayBuffer
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.utils.dsrl_replay import (
    DSRL_REPLAY_CAPACITY_TRANSITIONS,
    DSRL_REPLAY_CHECKPOINT_SHARD_TRANSITIONS,
    DSRL_REPLAY_MAX_RESIDENT_GIB,
    dsrl_replay_bytes_per_transition,
)


def _rss_bytes() -> int:
    with Path("/proc/self/statm").open(encoding="utf-8") as file:
        resident_pages = int(file.read().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _thread_count() -> int:
    return len(list(Path("/proc/self/task").iterdir()))


class _ProcessMonitor:
    def __init__(self) -> None:
        self.peak_rss_bytes = _rss_bytes()
        self.peak_threads = _thread_count()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(0.02):
            self.peak_rss_bytes = max(self.peak_rss_bytes, _rss_bytes())
            self.peak_threads = max(self.peak_threads, _thread_count())

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_rss_bytes = max(self.peak_rss_bytes, _rss_bytes())
        self.peak_threads = max(self.peak_threads, _thread_count())


def _synthetic_trajectory(
    *, trajectory_length: int, batch_size: int, value: float
) -> Trajectory:
    obs_shape = (trajectory_length, batch_size)

    def observation(offset: float) -> dict[str, torch.Tensor]:
        return {
            "dsrl_images": torch.full(
                (*obs_shape, 2, 3, 64, 64),
                value + offset,
                dtype=torch.bfloat16,
            ),
            "states": torch.full((*obs_shape, 7), value + offset, dtype=torch.bfloat16),
            "tactile_marker_motion": torch.full(
                (*obs_shape, 9, 198, 2),
                value + offset,
                dtype=torch.bfloat16,
            ),
        }

    return Trajectory(
        curr_obs=observation(0.0),
        next_obs=observation(0.5),
        actions=torch.full((*obs_shape, 32), value, dtype=torch.bfloat16),
        rewards=torch.zeros(*obs_shape, 10, dtype=torch.float32),
        terminations=torch.zeros(*obs_shape, 10, dtype=torch.bool),
        truncations=torch.zeros(*obs_shape, 10, dtype=torch.bool),
    )


def run_stress(args: argparse.Namespace) -> dict[str, object]:
    torch.set_num_threads(1)
    checkpoint_dir = args.checkpoint_dir.resolve()
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise ValueError(f"checkpoint directory must be empty: {checkpoint_dir}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    buffer = CompactDSRLReplayBuffer(
        seed=args.seed,
        capacity_transitions=args.capacity_transitions,
        checkpoint_shard_transitions=args.checkpoint_shard_transitions,
        max_resident_gib=args.max_resident_gib,
    )
    baseline_rss = _rss_bytes()
    baseline_threads = _thread_count()
    ingest_started = time.monotonic()
    with _ProcessMonitor() as ingest_monitor:
        for step in range(args.steps):
            trajectory = _synthetic_trajectory(
                trajectory_length=args.trajectory_length,
                batch_size=args.batch_size,
                value=float(step % 8),
            )
            buffer.add_trajectories([trajectory])
            del trajectory
            gc.collect()
    ingest_seconds = time.monotonic() - ingest_started

    transitions_per_trajectory = args.trajectory_length * args.batch_size
    expected_samples = min(
        args.steps * transitions_per_trajectory,
        args.capacity_transitions,
    )
    expected_resident_bytes = expected_samples * dsrl_replay_bytes_per_transition()
    stats = buffer.get_stats()
    if buffer.total_samples != expected_samples:
        raise AssertionError(
            f"resident transition mismatch: {buffer.total_samples} != {expected_samples}"
        )
    if int(stats["resident_tensor_bytes"]) != expected_resident_bytes:
        raise AssertionError("compact replay resident byte accounting mismatch")
    if expected_resident_bytes > int(args.max_resident_gib * (2**30)):
        raise AssertionError("compact replay exceeded configured resident budget")

    checkpoint_baseline_rss = _rss_bytes()
    checkpoint_baseline_threads = _thread_count()
    checkpoint_started = time.monotonic()
    with _ProcessMonitor() as checkpoint_monitor:
        buffer.save_checkpoint(str(checkpoint_dir))
    checkpoint_seconds = time.monotonic() - checkpoint_started
    checkpoint_metadata = CompactDSRLReplayBuffer.validate_checkpoint_metadata(
        str(checkpoint_dir),
        expected_capacity=args.capacity_transitions,
    )

    checkpoint_bytes = sum(
        path.stat().st_size for path in checkpoint_dir.iterdir() if path.is_file()
    )
    shard_tensor_bytes = (
        args.checkpoint_shard_transitions * dsrl_replay_bytes_per_transition()
    )
    checkpoint_rss_delta = max(
        0, checkpoint_monitor.peak_rss_bytes - checkpoint_baseline_rss
    )
    checkpoint_thread_delta = max(
        0, checkpoint_monitor.peak_threads - checkpoint_baseline_threads
    )
    checkpoint_rss_limit = checkpoint_baseline_rss + 2 * shard_tensor_bytes + 2**30
    if checkpoint_monitor.peak_rss_bytes > checkpoint_rss_limit:
        raise AssertionError(
            "checkpoint temporary RSS exceeded two shards plus 1 GiB slack"
        )
    if checkpoint_thread_delta > 2:
        raise AssertionError(
            "checkpoint created unexpected worker threads: "
            f"delta={checkpoint_thread_delta}"
        )
    if checkpoint_bytes > int(expected_resident_bytes * 1.03) + 2**20:
        raise AssertionError("checkpoint serialization overhead exceeded 3 percent")

    result = {
        "status": "passed",
        "steps": args.steps,
        "trajectory_shape": [args.trajectory_length, args.batch_size],
        "transitions_per_trajectory": transitions_per_trajectory,
        "resident_transitions": buffer.total_samples,
        "bytes_per_transition": dsrl_replay_bytes_per_transition(),
        "resident_tensor_bytes": expected_resident_bytes,
        "resident_tensor_gib": expected_resident_bytes / (2**30),
        "projected_four_rank_resident_gib": 4 * expected_resident_bytes / (2**30),
        "capacity_transitions": args.capacity_transitions,
        "capacity_bytes": int(stats["capacity_bytes"]),
        "max_resident_gib": args.max_resident_gib,
        "process_baseline_rss_bytes": baseline_rss,
        "process_peak_ingest_rss_bytes": ingest_monitor.peak_rss_bytes,
        "process_peak_ingest_rss_gib": ingest_monitor.peak_rss_bytes / (2**30),
        "process_peak_rss_delta_bytes": ingest_monitor.peak_rss_bytes - baseline_rss,
        "ingest_seconds": ingest_seconds,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_gib": checkpoint_bytes / (2**30),
        "projected_four_rank_checkpoint_gib": 4 * checkpoint_bytes / (2**30),
        "checkpoint_seconds": checkpoint_seconds,
        "checkpoint_num_shards": checkpoint_metadata["num_shards"],
        "checkpoint_shard_tensor_bytes": shard_tensor_bytes,
        "checkpoint_peak_rss_bytes": checkpoint_monitor.peak_rss_bytes,
        "checkpoint_peak_rss_gib": checkpoint_monitor.peak_rss_bytes / (2**30),
        "checkpoint_peak_rss_delta_bytes": checkpoint_rss_delta,
        "checkpoint_peak_rss_delta_gib": checkpoint_rss_delta / (2**30),
        "checkpoint_peak_rss_limit_bytes": checkpoint_rss_limit,
        "baseline_threads": baseline_threads,
        "peak_ingest_threads": ingest_monitor.peak_threads,
        "checkpoint_baseline_threads": checkpoint_baseline_threads,
        "checkpoint_peak_threads": checkpoint_monitor.peak_threads,
        "checkpoint_thread_delta": checkpoint_thread_delta,
        "process_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
    }
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--trajectory-length", type=int, default=72)
    parser.add_argument("--batch-size", type=int, default=21)
    parser.add_argument(
        "--capacity-transitions", type=int, default=DSRL_REPLAY_CAPACITY_TRANSITIONS
    )
    parser.add_argument(
        "--checkpoint-shard-transitions",
        type=int,
        default=DSRL_REPLAY_CHECKPOINT_SHARD_TRANSITIONS,
    )
    parser.add_argument(
        "--max-resident-gib", type=float, default=DSRL_REPLAY_MAX_RESIDENT_GIB
    )
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run_stress(_parse_args()), indent=2, sort_keys=True))
