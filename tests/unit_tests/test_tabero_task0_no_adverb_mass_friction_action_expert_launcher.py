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

import os
import re
import shlex
import subprocess
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    REPO_ROOT
    / "examples"
    / "embodiment"
    / "run_tabero_task0_no_adverb_mass_friction_action_expert.sh"
)
CONFIG_NAME = (
    "isaaclab_pi0_peft_lora_tacfield_tabero_task0_no_adverb_mass_friction_2gpu_100step"
)
BOUNDARY_SEMANTICS = "terminal_observation_first_done_prefix_logprob_hdf5_reset_v1"


def _run(
    tmp_path,
    mode,
    *args,
    run_id=None,
    wandb_id="wandb-test-id",
    visible_gpus="0,1",
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
        env.pop("TABERO_TASK0_NO_ADVERB_RUN_ID", None)
    else:
        env["TABERO_TASK0_NO_ADVERB_RUN_ID"] = run_id
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
    return dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )


def _write_checkpoint_sidecar(checkpoint, semantics=BOUNDARY_SEMANTICS):
    sidecar = checkpoint / "actor" / "model_state_dict" / "trainable_weights.pt"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"global_step": 1}
    if semantics is not None:
        metadata["tabero_ppo_transition_boundary_semantics"] = semantics
    torch.save({"model": {}, "metadata": metadata}, sidecar)
    return sidecar


def test_smoke_dry_run_writes_metadata_and_exact_capacity_command(tmp_path):
    run_id = "20260810_220001_smoke"
    result = _run(tmp_path, "smoke", run_id=run_id)

    assert result.returncode == 0, result.stderr
    output = (
        tmp_path
        / f"tabero_task0_no_adverb_mass_friction_action_expert_lora_2gpu_capacity_smoke_{run_id}"
    )
    metadata = _read_env(output / "run.env")
    assert metadata["TABERO_TASK0_NO_ADVERB_RUN_ID"] == run_id
    assert metadata["WANDB_RUN_ID"] == "wandb-test-id"
    assert metadata["WANDB_RESUME"] == "allow"
    assert metadata["TABERO_RUN_MODE"] == "smoke"
    assert metadata["TABERO_OUTPUT_DIR"] == str(output)
    assert metadata["TABERO_CONFIG_NAME"] == CONFIG_NAME
    assert metadata["TABERO_VISIBLE_GPUS"] == "0,1"
    assert metadata["TABERO_PPO_TRANSITION_BOUNDARY_SEMANTICS"] == BOUNDARY_SEMANTICS
    assert f"--config-name {CONFIG_NAME}" in result.stdout
    assert "runner.max_epochs=1" in result.stdout
    assert "runner.save_interval=1" in result.stdout
    assert (
        "actor.fsdp_config.trainable_checkpoint_metadata.target_global_step=1"
        in result.stdout
    )
    assert "env.train.total_num_envs" not in result.stdout
    assert "CUDA_VISIBLE_DEVICES: 0,1" in result.stdout


def test_formal_dry_run_keeps_yaml_defaults_and_rejects_collision(tmp_path):
    run_id = "20260810_220002_formal"
    first = _run(tmp_path, "formal", run_id=run_id)

    assert first.returncode == 0, first.stderr
    output = (
        tmp_path
        / f"tabero_task0_no_adverb_mass_friction_action_expert_lora_2gpu_100step_{run_id}"
    )
    assert output.is_dir()
    assert "runner.max_epochs=" not in first.stdout
    assert "runner.save_interval=" not in first.stdout
    assert f"runner.logger.experiment_name={output.name}" in first.stdout
    second = _run(tmp_path, "formal", run_id=run_id)
    assert second.returncode != 0
    assert "already exists" in second.stderr


def test_resume_reuses_wandb_id_and_requires_same_gpu_mapping(tmp_path):
    run_id = "20260810_220003_smoke"
    fresh = _run(tmp_path, "smoke", run_id=run_id)
    assert fresh.returncode == 0, fresh.stderr
    output = (
        tmp_path
        / f"tabero_task0_no_adverb_mass_friction_action_expert_lora_2gpu_capacity_smoke_{run_id}"
    )
    checkpoint = output / output.name / "checkpoints" / "global_step_1"
    _write_checkpoint_sidecar(checkpoint)

    resumed = _run(
        tmp_path,
        "smoke",
        "--resume-dir",
        str(checkpoint),
        wandb_id="must-not-replace-existing-id",
    )
    assert resumed.returncode == 0, resumed.stderr
    metadata = _read_env(output / "run.env")
    assert metadata["WANDB_RUN_ID"] == "wandb-test-id"
    assert metadata["WANDB_RESUME"] == "must"
    assert f"runner.resume_dir={checkpoint}" in resumed.stdout

    rejected = _run(
        tmp_path,
        "smoke",
        "--resume-dir",
        str(checkpoint),
        visible_gpus="2,3",
    )
    assert rejected.returncode != 0
    assert "GPU mapping" in rejected.stderr


def test_resume_rejects_missing_legacy_or_mismatched_boundary_sidecar(tmp_path):
    run_id = "20260810_220006_smoke"
    fresh = _run(tmp_path, "smoke", run_id=run_id)
    assert fresh.returncode == 0, fresh.stderr
    output = (
        tmp_path
        / f"tabero_task0_no_adverb_mass_friction_action_expert_lora_2gpu_capacity_smoke_{run_id}"
    )
    checkpoint = output / output.name / "checkpoints" / "global_step_1"
    (checkpoint / "actor").mkdir(parents=True)

    missing = _run(tmp_path, "smoke", "--resume-dir", str(checkpoint))
    assert missing.returncode != 0
    assert "requires checkpoint sidecar" in missing.stderr

    _write_checkpoint_sidecar(checkpoint, semantics=None)
    legacy = _run(tmp_path, "smoke", "--resume-dir", str(checkpoint))
    assert legacy.returncode != 0
    assert "legacy or mismatched checkpoint" in legacy.stderr

    _write_checkpoint_sidecar(checkpoint, semantics="other_boundary")
    mismatch = _run(tmp_path, "smoke", "--resume-dir", str(checkpoint))
    assert mismatch.returncode != 0
    assert "legacy or mismatched checkpoint" in mismatch.stderr


def test_resume_rejects_run_env_boundary_semantics_mismatch(tmp_path):
    run_id = "20260810_220007_smoke"
    fresh = _run(tmp_path, "smoke", run_id=run_id)
    assert fresh.returncode == 0, fresh.stderr
    output = (
        tmp_path
        / f"tabero_task0_no_adverb_mass_friction_action_expert_lora_2gpu_capacity_smoke_{run_id}"
    )
    checkpoint = output / output.name / "checkpoints" / "global_step_1"
    _write_checkpoint_sidecar(checkpoint)
    run_env = output / "run.env"
    run_env.write_text(
        run_env.read_text().replace(BOUNDARY_SEMANTICS, "legacy_boundary")
    )

    result = _run(tmp_path, "smoke", "--resume-dir", str(checkpoint))

    assert result.returncode != 0
    assert "PPO boundary semantics" in result.stderr


def test_resume_rejects_run_env_shell_code_without_executing_it(tmp_path):
    output = tmp_path / "malicious"
    checkpoint = output / output.name / "checkpoints" / "global_step_1"
    (checkpoint / "actor").mkdir(parents=True)
    marker = tmp_path / "injected"
    (output / "run.env").write_text(f"touch {marker}\n")

    result = _run(tmp_path, "smoke", "--resume-dir", str(checkpoint))

    assert result.returncode != 0
    assert "invalid run.env" in result.stderr
    assert not marker.exists()


def test_launcher_requires_two_unique_physical_gpus(tmp_path):
    duplicate = _run(
        tmp_path,
        "smoke",
        run_id="20260810_220004_smoke",
        visible_gpus="0,0",
    )
    assert duplicate.returncode != 0
    assert "unique" in duplicate.stderr

    too_many = _run(
        tmp_path,
        "smoke",
        run_id="20260810_220005_smoke",
        visible_gpus="0,1,2",
    )
    assert too_many.returncode != 0
    assert "exactly 2" in too_many.stderr


def test_dry_run_generates_ids_without_persisting_secrets(tmp_path):
    result = _run(tmp_path, "formal", run_id=None, wandb_id=None)

    assert result.returncode == 0, result.stderr
    [output] = list(tmp_path.iterdir())
    metadata_text = (output / "run.env").read_text()
    metadata = _read_env(output / "run.env")
    assert re.fullmatch(
        r"\d{8}_\d{6}_formal", metadata["TABERO_TASK0_NO_ADVERB_RUN_ID"]
    )
    assert metadata["WANDB_RUN_ID"]
    assert "WANDB_API_KEY" not in metadata_text
    assert "API_KEY" not in metadata_text


def test_launcher_sources_isaac_environment_before_selecting_python():
    launcher = LAUNCHER.read_text()

    source_position = launcher.index('source "${ISAAC_SETUP}"')
    python_position = launcher.index('PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"')
    assert source_position < python_position


def test_launcher_samples_gpu_metrics_and_processes_every_five_seconds():
    launcher = LAUNCHER.read_text()

    assert 'GPU_SAMPLES_FILE="${OUTPUT_DIR}/gpu_samples.csv"' in launcher
    assert 'GPU_PROCESS_SAMPLES_FILE="${OUTPUT_DIR}/gpu_process_samples.csv"' in (
        launcher
    )
    assert "memory.used,memory.total,utilization.gpu,power.draw,pstate" in launcher
    assert "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory" in launcher
    assert "sleep 5" in launcher
    assert "start_gpu_sampler" in launcher
    assert "stop_gpu_sampler" in launcher
