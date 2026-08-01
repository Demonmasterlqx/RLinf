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

import os
import re
import shlex
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "examples" / "embodiment" / "run_tabero_task0_firm_dsrl_smoke.sh"
CONFIG_NAME = "isaaclab_pi0_dsrl_tacfield_tabero_task0_firm_8gpu_smoke"


def _run(
    tmp_path,
    *args,
    run_id=None,
    wandb_id="wandb-dsrl-test-id",
    visible_gpus="0,1,2,3,4,5,6,7",
    installed_gpus="0,1,2,3,4,5,6,7",
):
    env = os.environ.copy()
    fake_bin = tmp_path.parent / f"{tmp_path.name}_fake_bin"
    fake_bin.mkdir(exist_ok=True)
    fake_nvidia_smi = fake_bin / "nvidia-smi"
    gpu_rows = " ".join(installed_gpus.split(","))
    fake_nvidia_smi.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$*" == "--query-gpu=index --format=csv,noheader" ]]; then\n'
        f"  printf '%s\\n' {gpu_rows}\n"
        "  exit 0\n"
        "fi\n"
        "exit 0\n"
    )
    fake_nvidia_smi.chmod(0o755)
    env.update(
        {
            "TABERO_RESULTS_ROOT": str(tmp_path),
            "CUDA_VISIBLE_DEVICES": visible_gpus,
            "PATH": f"{fake_bin}:{env['PATH']}",
        }
    )
    if wandb_id is None:
        env.pop("WANDB_RUN_ID", None)
    else:
        env["WANDB_RUN_ID"] = wandb_id
    if run_id is None:
        env.pop("TABERO_TASK0_DSRL_RUN_ID", None)
    else:
        env["TABERO_TASK0_DSRL_RUN_ID"] = run_id
    return subprocess.run(
        ["bash", str(LAUNCHER), *args, "--dry-run"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )


def _read_env(path):
    result = subprocess.run(
        ["bash", "-c", f"set -a; source {shlex.quote(str(path))}; env"],
        check=True,
        text=True,
        capture_output=True,
    )
    return dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )


def _run_fake_resume(tmp_path, checkpoint, *, exit_status):
    fake_repo = tmp_path / "fake_repo"
    launcher = (
        fake_repo / "examples" / "embodiment" / "run_tabero_task0_firm_dsrl_smoke.sh"
    )
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_text(LAUNCHER.read_text())
    launcher.chmod(0o755)
    (launcher.parent / "train_embodied_agent.py").write_text("")
    config_dir = launcher.parent / "config"
    config_dir.mkdir(exist_ok=True)
    (config_dir / f"{CONFIG_NAME}.yaml").write_text("{}\n")

    setup = fake_repo / "isaac_sim" / "setup_conda_env.sh"
    setup.parent.mkdir(parents=True, exist_ok=True)
    setup.write_text("#!/usr/bin/env bash\n")

    capture = tmp_path / "child.env"
    fake_python = fake_repo / ".venv" / "bin" / "python"
    fake_python.parent.mkdir(parents=True, exist_ok=True)
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'WANDB_RUN_ID=%s\\nWANDB_RESUME=%s\\n' "
        '"${WANDB_RUN_ID}" "${WANDB_RESUME}" >"${TABERO_CAPTURE_FILE}"\n'
        'exit "${TABERO_FAKE_EXIT_STATUS}"\n'
    )
    fake_python.chmod(0o755)

    fake_bin = tmp_path / "fake_runtime_bin"
    fake_bin.mkdir(exist_ok=True)
    fake_nvidia_smi = fake_bin / "nvidia-smi"
    fake_nvidia_smi.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$*" == "--query-gpu=index --format=csv,noheader" ]]; then\n'
        "  printf '%s\\n' 0 1 2 3 4 5 6 7\n"
        "fi\n"
        "exit 0\n"
    )
    fake_nvidia_smi.chmod(0o755)
    fake_date = fake_bin / "date"
    fake_date.write_text(
        "#!/usr/bin/env bash\n"
        'case "$*" in\n'
        "  '-u +%Y-%m-%dT%H:%M:%SZ') printf '%s\\n' '2026-07-26T12:00:00Z' ;;\n"
        "  '+%Y-%m-%dT%H:%M:%S%z') printf '%s\\n' '2026-07-26T20:00:00+0800' ;;\n"
        "  '+%Y%m%d_%H%M%S') printf '%s\\n' '20260726_200000' ;;\n"
        "  '+%s') printf '%s\\n' '1000' ;;\n"
        "  *) exit 2 ;;\n"
        "esac\n"
    )
    fake_date.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "PATH": f"{fake_bin}:{env['PATH']}",
            "TABERO_CAPTURE_FILE": str(capture),
            "TABERO_FAKE_EXIT_STATUS": str(exit_status),
        }
    )
    result = subprocess.run(
        ["bash", str(launcher), "--resume-dir", str(checkpoint)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    return result, capture


def test_dsrl_smoke_dry_run_writes_metadata_and_exact_command(tmp_path):
    run_id = "20260726_123456_smoke"
    result = _run(tmp_path, run_id=run_id)

    assert result.returncode == 0, result.stderr
    output = tmp_path / f"tabero_task0_firm_tactile_dsrl_8gpu_smoke_{run_id}"
    metadata = _read_env(output / "run.env")
    assert metadata["TABERO_TASK0_DSRL_RUN_ID"] == run_id
    assert metadata["WANDB_RUN_ID"] == "wandb-dsrl-test-id"
    assert metadata["WANDB_RESUME"] == "allow"
    assert metadata["TABERO_OUTPUT_DIR"] == str(output)
    assert metadata["TABERO_CONFIG_NAME"] == CONFIG_NAME
    assert metadata["TABERO_START_TIME_UTC"]
    assert metadata["TABERO_START_TIME_LOCAL"]
    assert f"--config-name {CONFIG_NAME}" in result.stdout
    assert f"runner.logger.log_path={output}" in result.stdout
    assert "runner.max_epochs=" not in result.stdout
    assert (output / "command.txt").read_text().startswith("Command:")
    assert not (output / "train.log").exists()


def test_dsrl_smoke_rejects_output_collision(tmp_path):
    run_id = "20260726_123457_smoke"
    first = _run(tmp_path, run_id=run_id)
    second = _run(tmp_path, run_id=run_id)

    assert first.returncode == 0, first.stderr
    assert second.returncode != 0
    assert "already exists" in second.stderr


def test_dsrl_resume_reuses_wandb_and_requires_associated_run_env(tmp_path):
    run_id = "20260726_123458_smoke"
    fresh = _run(tmp_path, run_id=run_id)
    assert fresh.returncode == 0, fresh.stderr
    output = tmp_path / f"tabero_task0_firm_tactile_dsrl_8gpu_smoke_{run_id}"
    original_run_env = (output / "run.env").read_text()
    checkpoint = output / output.name / "checkpoints" / "global_step_1"
    (checkpoint / "actor").mkdir(parents=True)

    resumed = _run(
        tmp_path,
        "--resume-dir",
        str(checkpoint),
        wandb_id="must-not-replace-existing-id",
    )

    assert resumed.returncode == 0, resumed.stderr
    metadata = _read_env(output / "run.env")
    assert metadata["WANDB_RUN_ID"] == "wandb-dsrl-test-id"
    assert metadata["WANDB_RESUME"] == "allow"
    assert (output / "run.env").read_text() == original_run_env
    assert f"runner.resume_dir={checkpoint}" in resumed.stdout
    assert list(output.glob("resume_command_*.txt"))

    missing = tmp_path / "unassociated" / "global_step_1"
    (missing / "actor").mkdir(parents=True)
    rejected = _run(tmp_path, "--resume-dir", str(missing))
    assert rejected.returncode != 0
    assert "run.env" in rejected.stderr


def test_dsrl_resume_rejects_run_env_shell_code(tmp_path):
    output = tmp_path / "malicious"
    checkpoint = output / output.name / "checkpoints" / "global_step_1"
    (checkpoint / "actor").mkdir(parents=True)
    marker = tmp_path / "injected"
    (output / "run.env").write_text(f"touch {marker}\n")

    result = _run(tmp_path, "--resume-dir", str(checkpoint))

    assert result.returncode != 0
    assert "invalid run.env" in result.stderr
    assert not marker.exists()


def test_dsrl_resume_exports_must_and_writes_unique_runtime_artifacts(tmp_path):
    run_id = "20260726_123459_smoke"
    fresh = _run(tmp_path, run_id=run_id)
    assert fresh.returncode == 0, fresh.stderr
    output = tmp_path / f"tabero_task0_firm_tactile_dsrl_8gpu_smoke_{run_id}"
    checkpoint = output / output.name / "checkpoints" / "global_step_1"
    (checkpoint / "actor").mkdir(parents=True)

    failed, capture = _run_fake_resume(tmp_path, checkpoint, exit_status=7)
    assert failed.returncode == 7, failed.stderr
    child_env = _read_env(capture)
    assert child_env["WANDB_RUN_ID"] == "wandb-dsrl-test-id"
    assert child_env["WANDB_RESUME"] == "must"
    [failed_status] = output.glob("resume_status_*.env")
    assert _read_env(failed_status)["TABERO_EXIT_STATUS"] == "7"

    succeeded, _ = _run_fake_resume(tmp_path, checkpoint, exit_status=0)
    assert succeeded.returncode == 0, succeeded.stderr
    assert len(list(output.glob("resume_command_*.txt"))) == 2
    assert len(list(output.glob("resume_*.log"))) == 2
    assert len(list(output.glob("resume_status_*.env"))) == 2


def test_dsrl_launcher_requires_eight_unique_physical_gpus(tmp_path):
    duplicate = _run(
        tmp_path,
        run_id="20260726_123500_smoke",
        visible_gpus="0,1,2,3,4,5,6,6",
    )
    assert duplicate.returncode != 0
    assert "unique" in duplicate.stderr

    too_few = _run(
        tmp_path,
        run_id="20260726_123501_smoke",
        visible_gpus="0,1,2,3,4,5,6",
    )
    assert too_few.returncode != 0
    assert "exactly 8" in too_few.stderr

    unavailable = _run(
        tmp_path,
        run_id="20260726_123502_smoke",
        visible_gpus="100,101,102,103,104,105,106,107",
    )
    assert unavailable.returncode != 0
    assert "not installed" in unavailable.stderr


def test_dsrl_launcher_generates_ids_without_persisting_secrets(tmp_path):
    result = _run(tmp_path, run_id=None, wandb_id=None)

    assert result.returncode == 0, result.stderr
    [output] = list(tmp_path.iterdir())
    metadata_text = (output / "run.env").read_text()
    metadata = _read_env(output / "run.env")
    assert re.fullmatch(r"\d{8}_\d{6}_smoke", metadata["TABERO_TASK0_DSRL_RUN_ID"])
    assert metadata["WANDB_RUN_ID"]
    assert "WANDB_API_KEY" not in metadata_text
    assert "API_KEY" not in metadata_text


def test_dsrl_launcher_sources_isaac_before_selecting_rlinf_python():
    launcher = LAUNCHER.read_text()

    source_position = launcher.index('source "${ISAAC_SETUP}"')
    python_position = launcher.index('PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"')
    assert source_position < python_position


def test_dsrl_launcher_samples_gpu_and_process_metrics_every_five_seconds():
    launcher = LAUNCHER.read_text()

    assert 'GPU_SAMPLES_FILE="${OUTPUT_DIR}/gpu_samples.csv"' in launcher
    assert (
        'GPU_PROCESS_SAMPLES_FILE="${OUTPUT_DIR}/gpu_process_samples.csv"' in launcher
    )
    assert (
        'HOST_MEMORY_SAMPLES_FILE="${OUTPUT_DIR}/host_memory_samples.csv"' in launcher
    )
    assert "memory.used,memory.total,utilization.gpu,power.draw,pstate" in launcher
    assert "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory" in launcher
    assert "mem_available_kib" in launcher
    assert "sac_actor_rss_kib" in launcher
    assert "sleep 5" in launcher


def test_dsrl_launcher_requires_compute_idle_gpus():
    launcher = LAUNCHER.read_text()

    assert "existing_compute_processes" in launcher
    assert "all 8 GPUs must be compute-idle before smoke" in launcher
    assert "start_gpu_sampler" in launcher
    assert "stop_gpu_sampler" in launcher
