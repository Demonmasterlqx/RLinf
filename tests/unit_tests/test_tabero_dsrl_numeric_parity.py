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
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
T2_REPO = REPO_ROOT.parent / "T2-VLA"
RUNNER = REPO_ROOT / "examples" / "embodiment" / "run_tabero_dsrl_parity.py"
ACTOR_STAGES = {
    "image_input": [1, 2, 3, 64, 64],
    "main_image_input": [1, 3, 64, 64],
    "wrist_image_input": [1, 3, 64, 64],
    "state_input": [1, 7],
    "tactile_input": [1, 9, 396],
    "state_features": [1, 64],
    "image_features": [1, 128],
    "main_image_features": [1, 64],
    "wrist_image_features": [1, 64],
    "tactile_features": [1, 64],
    "gaussian_mean": [1, 32],
    "deterministic_noise": [1, 32],
    "broadcast_noise": [1, 50, 32],
}


def test_actor_parity_runner_emits_complete_passing_contract(tmp_path):
    output = tmp_path / "parity.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(T2_REPO / "src"), str(REPO_ROOT), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--device",
            "cpu",
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["schema_version"] == 1
    assert report["passed"] is True
    assert report["fixture"]["weight_key_count"] == 48
    assert report["fixture"]["weight_dtype"] == "bfloat16"
    assert report["fixture"]["raw_image_shape"] == [256, 256, 3]
    assert report["fixture"]["raw_wrist_image_shape"] == [256, 256, 3]
    assert set(report["revisions"]) == {"base", "t2", "rlinf"}
    assert all(len(item["git_sha"]) == 40 for item in report["revisions"].values())
    for name, shape in ACTOR_STAGES.items():
        stage = report["stages"][name]
        assert stage["passed"] is True
        assert stage["max_abs"] <= report["tolerance"]
        assert stage["rlinf"] == {"shape": shape, "dtype": "bfloat16"}
        assert stage["t2"] == {"shape": shape, "dtype": "bfloat16"}
    assert report["stages"]["final_pi0_action"]["status"] == "not_requested"


def test_failed_stage_makes_runner_exit_nonzero(tmp_path):
    output = tmp_path / "parity.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(T2_REPO / "src"), str(REPO_ROOT), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--device",
            "cpu",
            "--tolerance",
            "-1",
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode != 0
    report = json.loads(output.read_text())
    assert report["passed"] is False
    assert any(stage.get("passed") is False for stage in report["stages"].values())


def test_requested_final_pi0_reports_missing_checkpoint_without_loading_model(
    tmp_path,
):
    output = tmp_path / "parity.json"
    missing_checkpoint = tmp_path / "missing-base-model"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(T2_REPO / "src"), str(REPO_ROOT), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--device",
            "cpu",
            "--with-final-pi0",
            "--base-checkpoint",
            str(missing_checkpoint),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode != 0
    report = json.loads(output.read_text())
    final = report["stages"]["final_pi0_action"]
    assert report["passed"] is False
    assert final["status"] == "error"
    assert final["passed"] is False
    assert final["error_type"] == "FileNotFoundError"
