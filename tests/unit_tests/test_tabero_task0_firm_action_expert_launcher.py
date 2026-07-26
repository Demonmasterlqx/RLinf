from pathlib import Path
import os
import re
import shlex
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    REPO_ROOT
    / "examples"
    / "embodiment"
    / "run_tabero_task0_firm_action_expert.sh"
)
CONFIG_NAME = "isaaclab_pi0_peft_lora_tacfield_tabero_task0_firm_8gpu_50step"


def _run(
    tmp_path,
    mode,
    *args,
    run_id=None,
    wandb_id="wandb-test-id",
    visible_gpus="0,1,2,3,4,5,6,7",
):
    env = os.environ.copy()
    env.update(
        {
            "TABERO_RESULTS_ROOT": str(tmp_path),
            "CUDA_VISIBLE_DEVICES": visible_gpus,
        }
    )
    if wandb_id is None:
        env.pop("WANDB_RUN_ID", None)
    else:
        env["WANDB_RUN_ID"] = wandb_id
    if run_id is None:
        env.pop("TABERO_TASK0_RUN_ID", None)
    else:
        env["TABERO_TASK0_RUN_ID"] = run_id
    return subprocess.run(
        ["bash", str(LAUNCHER), mode, *args, "--dry-run"],
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
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def test_task0_smoke_dry_run_writes_metadata_and_exact_capacity_command(tmp_path):
    run_id = "20260726_123456_smoke"
    result = _run(tmp_path, "smoke", run_id=run_id)

    assert result.returncode == 0, result.stderr
    output = tmp_path / f"tabero_task0_firm_action_expert_lora_8gpu_capacity_smoke_{run_id}"
    metadata = _read_env(output / "run.env")
    assert metadata["TABERO_TASK0_RUN_ID"] == run_id
    assert metadata["WANDB_RUN_ID"] == "wandb-test-id"
    assert metadata["WANDB_RESUME"] == "allow"
    assert metadata["TABERO_RUN_MODE"] == "smoke"
    assert metadata["TABERO_OUTPUT_DIR"] == str(output)
    assert metadata["TABERO_CONFIG_NAME"] == CONFIG_NAME
    assert metadata["TABERO_START_TIME_UTC"]
    assert metadata["TABERO_START_TIME_LOCAL"]
    assert f"--config-name {CONFIG_NAME}" in result.stdout
    assert "runner.max_epochs=1" in result.stdout
    assert "runner.save_interval=1" in result.stdout
    assert "env.train.total_num_envs" not in result.stdout
    assert f"runner.logger.log_path={output}" in result.stdout


def test_task0_formal_dry_run_keeps_yaml_defaults_and_rejects_collision(tmp_path):
    run_id = "20260726_123457_formal"
    first = _run(tmp_path, "formal", run_id=run_id)

    assert first.returncode == 0, first.stderr
    output = tmp_path / f"tabero_task0_firm_action_expert_lora_8gpu_50step_{run_id}"
    assert output.is_dir()
    assert "runner.max_epochs=" not in first.stdout
    assert "runner.save_interval=" not in first.stdout
    assert f"runner.logger.experiment_name={output.name}" in first.stdout
    second = _run(tmp_path, "formal", run_id=run_id)
    assert second.returncode != 0
    assert "already exists" in second.stderr


def test_task0_resume_reuses_wandb_id_and_requires_existing_metadata(tmp_path):
    run_id = "20260726_123458_smoke"
    fresh = _run(tmp_path, "smoke", run_id=run_id)
    assert fresh.returncode == 0, fresh.stderr
    output = tmp_path / f"tabero_task0_firm_action_expert_lora_8gpu_capacity_smoke_{run_id}"
    checkpoint = output / output.name / "checkpoints" / "global_step_1"
    (checkpoint / "actor").mkdir(parents=True)

    resumed = _run(
        tmp_path,
        "smoke",
        "--resume-dir",
        str(checkpoint),
        wandb_id="must-not-replace-existing-id",
    )
    assert resumed.returncode == 0, resumed.stderr
    metadata = _read_env(output / "run.env")
    assert metadata["TABERO_TASK0_RUN_ID"] == run_id
    assert metadata["WANDB_RUN_ID"] == "wandb-test-id"
    assert metadata["WANDB_RESUME"] == "must"
    assert f"runner.resume_dir={checkpoint}" in resumed.stdout

    missing = tmp_path / "unassociated" / "global_step_1"
    (missing / "actor").mkdir(parents=True)
    rejected = _run(tmp_path, "smoke", "--resume-dir", str(missing))
    assert rejected.returncode != 0
    assert "run.env" in rejected.stderr


def test_task0_resume_rejects_run_env_shell_code_without_executing_it(tmp_path):
    output = tmp_path / "malicious"
    checkpoint = output / output.name / "checkpoints" / "global_step_1"
    (checkpoint / "actor").mkdir(parents=True)
    marker = tmp_path / "injected"
    (output / "run.env").write_text(f"touch {marker}\n")

    result = _run(tmp_path, "smoke", "--resume-dir", str(checkpoint))

    assert result.returncode != 0
    assert "invalid run.env" in result.stderr
    assert not marker.exists()


def test_task0_launcher_requires_eight_unique_physical_gpus(tmp_path):
    duplicate = _run(
        tmp_path,
        "smoke",
        run_id="20260726_123500_smoke",
        visible_gpus="0,1,2,3,4,5,6,6",
    )
    assert duplicate.returncode != 0
    assert "unique" in duplicate.stderr

    too_few = _run(
        tmp_path,
        "smoke",
        run_id="20260726_123501_smoke",
        visible_gpus="0,1,2,3,4,5,6",
    )
    assert too_few.returncode != 0
    assert "exactly 8" in too_few.stderr


def test_task0_dry_run_generates_ids_without_persisting_secrets(tmp_path):
    result = _run(tmp_path, "formal", run_id=None, wandb_id=None)

    assert result.returncode == 0, result.stderr
    [output] = list(tmp_path.iterdir())
    metadata_text = (output / "run.env").read_text()
    metadata = _read_env(output / "run.env")
    assert re.fullmatch(r"\d{8}_\d{6}_formal", metadata["TABERO_TASK0_RUN_ID"])
    assert metadata["WANDB_RUN_ID"]
    assert "WANDB_API_KEY" not in metadata_text
    assert "API_KEY" not in metadata_text


def test_task0_launcher_sources_isaac_environment_before_selecting_python():
    launcher = LAUNCHER.read_text()

    source_position = launcher.index('source "${ISAAC_SETUP}"')
    python_position = launcher.index('PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"')
    assert source_position < python_position


def test_task0_launcher_samples_gpu_metrics_and_processes_every_five_seconds():
    launcher = LAUNCHER.read_text()

    assert 'GPU_SAMPLES_FILE="${OUTPUT_DIR}/gpu_samples.csv"' in launcher
    assert 'GPU_PROCESS_SAMPLES_FILE="${OUTPUT_DIR}/gpu_process_samples.csv"' in launcher
    assert "memory.used,memory.total,utilization.gpu,power.draw,pstate" in launcher
    assert "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory" in launcher
    assert "sleep 5" in launcher
    assert "start_gpu_sampler" in launcher
    assert "stop_gpu_sampler" in launcher
    assert 'if [[ ! -s "${GPU_SAMPLES_FILE}" ]]; then' in launcher
    assert 'if [[ ! -s "${GPU_PROCESS_SAMPLES_FILE}" ]]; then' in launcher
