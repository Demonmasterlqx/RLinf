# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_HELPER = Path(__file__).with_name("_isaaclab_venv_startup_scenario.py")


def _run_startup_scenario(scenario):
    started = time.monotonic()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    process = subprocess.Popen(
        [sys.executable, str(SCENARIO_HELPER), scenario],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate(timeout=2)
        pytest.fail(
            f"IsaacLab startup scenario {scenario!r} timed out; "
            f"stdout={stdout!r}, stderr={stderr!r}"
        )

    assert time.monotonic() - started < 5
    assert process.returncode == 0, (stdout, stderr)
    report = json.loads(stdout.splitlines()[-1])
    assert report["elapsed"] < 5
    assert report["exception_type"] == "RuntimeError"
    assert report["child_alive"] is False
    assert report["resources_closed_by_init"] == {
        "pipes_closed": True,
        "queues_closed": True,
    }
    assert report["repeat_cleanup_errors"] == []
    assert report["active_child_pids"] == []
    return report


def test_subproc_isaaclab_env_reports_spawned_env_initialization_error():
    report = _run_startup_scenario("value_error")

    assert "ValueError" in report["exception_message"]
    assert (
        "episode horizon mismatch: expected 300 steps, got 160"
        in report["exception_message"]
    )


def test_subproc_isaaclab_env_reports_child_exit_without_handshake():
    report = _run_startup_scenario("child_exit")

    assert "before startup handshake" in report["exception_message"]
    assert "exit code 23" in report["exception_message"]
