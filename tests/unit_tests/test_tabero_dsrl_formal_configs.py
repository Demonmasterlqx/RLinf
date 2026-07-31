# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import hashlib
import os
import subprocess
import time
from pathlib import Path

import pytest
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "examples" / "embodiment" / "config"
LAUNCHER = REPO_ROOT / "examples" / "embodiment" / "run_tabero_firm_matrix_train.sh"
MODEL_PATH = "/data/home/sim6g/code/tabero/models/pi0_lora_tacfield_tabero_safetensors"
PREFIXES = [
    "dsrl_action_noise_net.",
    "actor_image_encoder.",
    "actor_state_encoder.",
    "actor_tactile_encoder.",
]
TASKS = {
    0: {
        "instruction": "pick up the alphabet soup and place it in the basket",
        "hdf5": "libero_object_task0_pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5",
    },
    5: {
        "instruction": "pick up the tomato sauce and place it in the basket",
        "hdf5": "libero_object_task5_pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo.hdf5",
    },
}


def _config_name(task_id: int) -> str:
    return f"isaaclab_pi0_dsrl_tacfield_tabero_task{task_id}_firm_8gpu_50step"


@pytest.mark.parametrize("task_id", [0, 5])
def test_formal_dsrl_configs_preserve_training_contract(monkeypatch, task_id):
    monkeypatch.setenv("TABERO_MATRIX_RUN_ID", "20260728_120000_formal")
    cfg = OmegaConf.load(CONFIG_DIR / f"{_config_name(task_id)}.yaml")

    assert cfg.runner.max_epochs == 50
    assert cfg.runner.max_steps == -1
    assert cfg.runner.save_interval == 10
    assert cfg.runner.logger.logger_backends == ["tensorboard", "wandb"]
    assert cfg.runner.logger.project_name == "tabero-rlinf"
    assert cfg.env.train.total_num_envs == 84
    assert cfg.env.train.rollout_epoch == 2
    assert cfg.actor.global_batch_size == 168
    assert cfg.actor.micro_batch_size == 2
    assert cfg.algorithm.update_epoch == 200
    assert cfg.algorithm.replay_buffer.min_buffer_size == 10
    assert cfg.algorithm.train_actor_steps == 10
    assert cfg.algorithm.gamma == 0.999
    assert cfg.algorithm.tau == 0.005
    assert cfg.algorithm.entropy_tuning.target_entropy == -16
    assert cfg.actor.optim.lr == 1.0e-4
    assert cfg.actor.critic_optim.lr == 3.0e-4
    assert cfg.actor.rollout_sync_prefixes == PREFIXES
    assert cfg.actor.fsdp_config.sharding_strategy == "no_shard"
    assert cfg.actor.fsdp_config.use_orig_params is True
    assert cfg.actor.fsdp_config.checkpoint_format == "local_shard"
    assert cfg.actor.fsdp_config.trainable_checkpoint_metadata.method == "dsrl"
    assert cfg.actor.fsdp_config.trainable_checkpoint_metadata.task_id == task_id
    assert cfg.actor.fsdp_config.trainable_checkpoint_metadata.target_global_step == 50
    assert cfg.actor.model.model_path == MODEL_PATH
    assert cfg.rollout.model.model_path == MODEL_PATH
    assert cfg.actor.model.openpi.use_dsrl is True
    assert cfg.actor.model.openpi.dsrl_use_tactile is True
    assert cfg.actor.model.openpi.dsrl_state_dim == 7
    assert cfg.actor.model.openpi.dsrl_action_noise_dim == 32
    assert cfg.actor.model.openpi.dsrl_num_q_heads == 10
    assert cfg.actor.model.openpi.dsrl_image_latent_dim == 64
    assert cfg.actor.model.openpi.dsrl_state_latent_dim == 64
    assert cfg.actor.model.openpi.dsrl_tactile_latent_dim == 64
    assert cfg.actor.model.openpi.dsrl_hidden_dims == [128, 128, 128]


@pytest.mark.parametrize("task_id", [0, 5])
def test_formal_dsrl_configs_select_exact_task_and_firm_prompt(monkeypatch, task_id):
    monkeypatch.setenv("TABERO_MATRIX_RUN_ID", "20260728_120000_formal")
    cfg = OmegaConf.load(CONFIG_DIR / f"{_config_name(task_id)}.yaml")
    expected = TASKS[task_id]

    for split in (cfg.env.train.init_params, cfg.env.eval.init_params):
        assert split.task_suite == "libero_object"
        assert split.task_id == task_id
        assert OmegaConf.to_container(split.tasks) == [
            {"task_suite": "libero_object", "task_id": task_id}
        ]
        assert split.task_description == expected["instruction"]
        assert split.hdf5_initial_states_path.endswith(expected["hdf5"])
        assert split.hdf5_reset_assignment == "cyclic"
        assert split.success.required_consecutive_steps == 8
        assert split.marker_history_len == 8
        assert split.combined_marker_count == 198
        assert split.agentview_cam.height == 256
        assert split.agentview_cam.width == 256
    prompt = cfg.env.train.init_params.prompt_conditions
    assert prompt.enabled is True
    assert prompt.assignment == "cyclic"
    assert prompt.condition_cycle == ["firm"]
    assert prompt.firm_adverbs == ["firmly", "tightly"]
    assert prompt.prompt_seed == 0


def _fake_nvidia_smi(tmp_path: Path) -> Path:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    script = fake_bin / "nvidia-smi"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$*\" == '--query-gpu=index --format=csv,noheader' ]]; then\n"
        '  for ((gpu_id = 0; gpu_id < "${FAKE_GPU_COUNT:-8}"; gpu_id++)); do\n'
        "    printf '%s\\n' \"${gpu_id}\"\n"
        "  done\n"
        'elif [[ "$*" == --id=*" --query-compute-apps=pid '
        '--format=csv,noheader,nounits" ]]; then\n'
        '  gpu_id="${1#--id=}"\n'
        '  if [[ "${FAKE_DELAY_GPU_ID:-}" == "${gpu_id}" ]]; then\n'
        '    sleep "${FAKE_COMPUTE_QUERY_DELAY:-2}"\n'
        "  fi\n"
        '  if [[ "${FAKE_BUSY_GPU_ID:-}" == "${gpu_id}" ]]; then\n'
        "    printf '%s\\n' 4242\n"
        "  fi\n"
        "fi\n"
        "exit 0\n"
    )
    script.chmod(0o755)
    date_script = fake_bin / "date"
    date_script.write_text(
        "#!/usr/bin/env bash\n"
        'case "$*" in\n'
        "  '-u +%Y-%m-%dT%H:%M:%SZ') printf '%s\\n' "
        "'2026-07-28T04:00:00Z' ;;\n"
        "  '+%Y-%m-%dT%H:%M:%S%z') printf '%s\\n' "
        "'2026-07-28T12:00:00+0800' ;;\n"
        "  '+%s') printf '%s\\n' 1785211200 ;;\n"
        "  '+%Y%m%d_%H%M%S') printf '%s\\n' 20260728_120000 ;;\n"
        '  *) /usr/bin/date "$@" ;;\n'
        "esac\n"
    )
    date_script.chmod(0o755)
    df_script = fake_bin / "df"
    df_script.write_text(
        "#!/usr/bin/env bash\n"
        'target="${@: -1}"\n'
        'if [[ "${FAKE_DF_DELAY_PATH:-}" == "${target}" ]]; then\n'
        '  sleep "${FAKE_DF_DELAY:-2}"\n'
        "fi\n"
        "available=2147483648\n"
        'if [[ -n "${FAKE_DF_LOW_PATH:-}" && "${target}" == '
        '"${FAKE_DF_LOW_PATH}" ]]; then\n'
        "  available=0\n"
        "fi\n"
        "printf '%s\\n' 'Filesystem 1024-blocks Used Available Capacity Mounted on'\n"
        'printf "fake 4294967296 0 %s 0%% %s\\n" "${available}" "${target}"\n'
    )
    df_script.chmod(0o755)
    return fake_bin


def _dry_run(
    tmp_path: Path,
    task_id: int,
    *extra: str,
    env_overrides: dict[str, str] | None = None,
):
    env = os.environ.copy()
    fake_bin = _fake_nvidia_smi(tmp_path)
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "TABERO_RESULTS_ROOT": str(tmp_path / "results"),
            "TABERO_MIN_FREE_KIB": "1",
            "TABERO_TEST_MODEL_SHA256": "0" * 64,
            "TABERO_GPU_LOCK_DIR": str(tmp_path / "gpu-locks"),
            "TABERO_MATRIX_RUN_ID": "20260728_120000_formal",
            "WANDB_RUN_ID": "matrix-dsrl-test",
        }
    )
    env.update(env_overrides or {})
    return subprocess.run(
        ["bash", str(LAUNCHER), "dsrl", str(task_id), "formal", *extra, "--dry-run"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )


def _fake_resume_checkpoint(
    tmp_path: Path,
    task_id: int = 0,
    step: int = 10,
    *,
    current_run_env: bool = False,
) -> Path:
    run_id = "20260728_120000_formal"
    experiment_name = f"tabero_firm_matrix_dsrl_task{task_id}_formal_{run_id}"
    output_dir = tmp_path / "results" / experiment_name
    checkpoint = output_dir / experiment_name / "checkpoints" / f"global_step_{step}"
    (checkpoint / "actor").mkdir(parents=True)
    (output_dir / "tensorboard").mkdir()
    (output_dir / "tensorboard" / "config.yaml").write_text("legacy: true\n")
    fields = [
        f"TABERO_MATRIX_RUN_ID={run_id}",
        "WANDB_RUN_ID=hnwwyz7o",
        "WANDB_RESUME=allow",
        "TABERO_METHOD=dsrl",
        f"TABERO_TASK_ID={task_id}",
        "TABERO_RUN_MODE=formal",
        f"TABERO_OUTPUT_DIR={output_dir}",
        f"TABERO_CONFIG_NAME={_config_name(task_id)}",
        "TABERO_CONFIG_DIR=examples/embodiment/config",
        f"TABERO_EXPERIMENT_NAME={experiment_name}",
        "TABERO_START_TIME_UTC=2026-07-28T12:00:00Z",
        "TABERO_START_TIME_LOCAL=2026-07-28T20:00:00+0800",
    ]
    if current_run_env:
        fields.append("TABERO_LAUNCH_KIND=new")
    (output_dir / "run.env").write_text("\n".join(fields) + "\n")
    return checkpoint


@pytest.mark.parametrize("task_id", [0, 5])
def test_matrix_launcher_builds_exact_dsrl_formal_command(tmp_path, task_id):
    result = _dry_run(tmp_path, task_id)
    assert result.returncode == 0, result.stderr
    assert f"--config-name {_config_name(task_id)}" in result.stdout
    assert (
        f"tabero_firm_matrix_dsrl_task{task_id}_formal_20260728_120000_formal"
        in result.stdout
    )
    assert "runner.max_epochs=" not in result.stdout
    assert "runner.save_interval=" not in result.stdout
    output_dir = (
        tmp_path
        / "results"
        / f"tabero_firm_matrix_dsrl_task{task_id}_formal_20260728_120000_formal"
    )
    provenance = (output_dir / "provenance.env").read_text()
    assert "TABERO_PROVENANCE_MODE=fresh" in provenance
    assert "TABERO_CONFIG_SHA256=" in provenance
    assert "TABERO_GIT_COMMIT=" in provenance
    assert f"TABERO_BASE_MODEL_SHA256={'0' * 64}" in provenance
    assert (output_dir / "config_snapshot.yaml").is_file()


def test_matrix_launcher_builds_task0_dsrl_60step_command(tmp_path):
    result = _dry_run(tmp_path, 0, "--target-steps", "60")

    assert result.returncode == 0, result.stderr
    config_name = "isaaclab_pi0_dsrl_tacfield_tabero_task0_firm_8gpu_60step"
    assert f"--config-name {config_name}" in result.stdout
    output_dir = (
        tmp_path
        / "results"
        / "tabero_firm_matrix_dsrl_task0_formal_20260728_120000_formal"
    )
    run_env = (output_dir / "run.env").read_text()
    assert f"TABERO_CONFIG_NAME={config_name}" in run_env
    snapshot = OmegaConf.load(output_dir / "config_snapshot.yaml")
    assert snapshot.runner.max_epochs == 60
    assert snapshot.runner.save_interval == 10
    metadata = snapshot.actor.fsdp_config.trainable_checkpoint_metadata
    assert metadata.training_config == config_name
    assert metadata.target_global_step == 60


def test_matrix_launcher_can_trace_only_the_training_driver(tmp_path):
    result = _dry_run(
        tmp_path,
        0,
        "--target-steps",
        "60",
        env_overrides={"TABERO_TRACE_DRIVER_SIGNALS": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert "/usr/bin/strace -qq -e trace=none" in result.stdout
    assert "-e signal=SIGTERM\\,SIGINT\\,SIGHUP" in result.stdout
    assert "driver_signals.log" in result.stdout
    assert " -f " not in result.stdout


def test_matrix_launcher_rejects_invalid_driver_signal_trace_setting(tmp_path):
    result = _dry_run(
        tmp_path,
        0,
        env_overrides={"TABERO_TRACE_DRIVER_SIGNALS": "yes"},
    )

    assert result.returncode != 0
    assert "TABERO_TRACE_DRIVER_SIGNALS must be 0 or 1" in result.stderr


def test_matrix_launcher_rejects_task5_dsrl_60step_profile(tmp_path):
    result = _dry_run(tmp_path, 5, "--target-steps", "60")

    assert result.returncode != 0
    assert "60-step formal training supports only DSRL Task 0" in result.stderr


def test_matrix_launcher_restore_only_requires_resume(tmp_path):
    result = _dry_run(tmp_path, 0, "--restore-only")
    assert result.returncode != 0
    assert "--restore-only requires --resume-dir" in result.stderr


def test_matrix_launcher_resumes_exact_checkpoint_and_wandb_run(tmp_path):
    checkpoint = _fake_resume_checkpoint(tmp_path)
    result = _dry_run(tmp_path, 0, "--resume-dir", str(checkpoint))

    assert result.returncode == 0, result.stderr
    assert f"runner.resume_dir={checkpoint}" in result.stdout
    assert "runner.max_epochs=" not in result.stdout
    output_dir = checkpoint.parents[2]
    resume_commands = list(output_dir.glob("resume_command_*.txt"))
    assert len(resume_commands) == 1
    command_text = resume_commands[0].read_text()
    assert "WANDB_RUN_ID=hnwwyz7o" in command_text
    assert "WANDB_RESUME=must" in command_text
    assert "runner.resume_dir=" in command_text
    provenance = (output_dir / "provenance.env").read_text()
    assert "TABERO_PROVENANCE_MODE=legacy_migration" in provenance
    source_config = output_dir / "tensorboard" / "config.yaml"
    expected_source_hash = hashlib.sha256(source_config.read_bytes()).hexdigest()
    assert f"TABERO_SOURCE_CONFIG_SHA256={expected_source_hash}" in provenance


def test_matrix_launcher_rejects_current_run_without_provenance(tmp_path):
    checkpoint = _fake_resume_checkpoint(tmp_path, current_run_env=True)

    result = _dry_run(tmp_path, 0, "--resume-dir", str(checkpoint))

    assert result.returncode != 0
    assert "current run.env requires provenance.env" in result.stderr


def test_matrix_launcher_rejects_changed_resume_provenance(tmp_path):
    checkpoint = _fake_resume_checkpoint(tmp_path)
    first = _dry_run(tmp_path, 0, "--resume-dir", str(checkpoint))
    assert first.returncode == 0, first.stderr
    provenance_path = checkpoint.parents[2] / "provenance.env"
    provenance_lines = provenance_path.read_text().splitlines()
    provenance_path.write_text(
        "\n".join(
            f"TABERO_CONFIG_SHA256={'f' * 64}"
            if line.startswith("TABERO_CONFIG_SHA256=")
            else line
            for line in provenance_lines
        )
        + "\n"
    )

    second = _dry_run(tmp_path, 0, "--resume-dir", str(checkpoint))

    assert second.returncode != 0
    assert "config SHA-256 does not match provenance" in second.stderr


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("TABERO_PROVENANCE_VERSION", "2", "unsupported provenance version"),
        ("TABERO_PROVENANCE_MODE", "unknown", "unsupported provenance mode"),
    ],
)
def test_matrix_launcher_rejects_unknown_provenance_contract(
    tmp_path,
    key,
    value,
    message,
):
    checkpoint = _fake_resume_checkpoint(tmp_path)
    first = _dry_run(tmp_path, 0, "--resume-dir", str(checkpoint))
    assert first.returncode == 0, first.stderr
    provenance_path = checkpoint.parents[2] / "provenance.env"
    lines = provenance_path.read_text().splitlines()
    provenance_path.write_text(
        "\n".join(
            f"{key}={value}" if line.startswith(f"{key}=") else line for line in lines
        )
        + "\n"
    )

    second = _dry_run(tmp_path, 0, "--resume-dir", str(checkpoint))

    assert second.returncode != 0
    assert message in second.stderr


def test_matrix_launcher_rejects_changed_legacy_source_config(tmp_path):
    checkpoint = _fake_resume_checkpoint(tmp_path)
    first = _dry_run(tmp_path, 0, "--resume-dir", str(checkpoint))
    assert first.returncode == 0, first.stderr
    source_config = checkpoint.parents[2] / "tensorboard" / "config.yaml"
    source_config.write_text("legacy: tampered\n")

    second = _dry_run(tmp_path, 0, "--resume-dir", str(checkpoint))

    assert second.returncode != 0
    assert "legacy source config SHA-256 does not match provenance" in second.stderr


def test_matrix_launcher_never_overwrites_resume_evidence(tmp_path):
    checkpoint = _fake_resume_checkpoint(tmp_path)

    first = _dry_run(tmp_path, 0, "--resume-dir", str(checkpoint))
    second = _dry_run(tmp_path, 0, "--resume-dir", str(checkpoint))

    assert first.returncode == second.returncode == 0
    output_dir = checkpoint.parents[2]
    assert len(list(output_dir.glob("resume_command_*.txt"))) == 2


def test_matrix_launcher_restore_only_loads_step_without_training(tmp_path):
    checkpoint = _fake_resume_checkpoint(tmp_path)
    result = _dry_run(
        tmp_path,
        0,
        "--resume-dir",
        str(checkpoint),
        "--restore-only",
    )

    assert result.returncode == 0, result.stderr
    assert f"runner.resume_dir={checkpoint}" in result.stdout
    assert "runner.max_epochs=10" in result.stdout
    assert "runner.save_interval=-1" in result.stdout
    output_dir = checkpoint.parents[2]
    assert len(list(output_dir.glob("restore_only_command_*.txt"))) == 1


def test_matrix_launcher_rejects_unknown_method_task_and_mode(tmp_path):
    env = os.environ.copy()
    env["PATH"] = f"{_fake_nvidia_smi(tmp_path)}:{env['PATH']}"
    for args in (
        ("bad", "0", "formal"),
        ("dsrl", "4", "formal"),
        ("dsrl", "0", "quick"),
    ):
        result = subprocess.run(
            ["bash", str(LAUNCHER), *args, "--dry-run"],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        assert result.returncode != 0


def test_matrix_launcher_rejects_busy_visible_gpu(tmp_path):
    result = _dry_run(
        tmp_path,
        0,
        env_overrides={"FAKE_BUSY_GPU_ID": "3"},
    )

    assert result.returncode != 0
    assert "GPU 3 is busy with compute PID 4242" in result.stderr


def test_matrix_launcher_checks_resume_output_filesystem(tmp_path):
    checkpoint = _fake_resume_checkpoint(tmp_path)
    output_dir = checkpoint.parents[2]

    result = _dry_run(
        tmp_path,
        0,
        "--resume-dir",
        str(checkpoint),
        env_overrides={
            "TABERO_RESULTS_ROOT": str(tmp_path / "different-results-root"),
            "FAKE_DF_LOW_PATH": str(output_dir),
        },
    )

    assert result.returncode != 0
    assert f"insufficient disk space on {output_dir}" in result.stderr


def test_matrix_launcher_serializes_exclusive_gpu_lease(tmp_path):
    fake_bin = _fake_nvidia_smi(tmp_path)
    base_env = os.environ.copy()
    base_env.update(
        {
            "PATH": f"{fake_bin}:{base_env['PATH']}",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "TABERO_RESULTS_ROOT": str(tmp_path / "results"),
            "TABERO_MIN_FREE_KIB": "1",
            "TABERO_TEST_MODEL_SHA256": "0" * 64,
            "TABERO_GPU_LOCK_DIR": str(tmp_path / "gpu-locks"),
            "FAKE_DELAY_GPU_ID": "0",
            "FAKE_COMPUTE_QUERY_DELAY": "2",
            "WANDB_RUN_ID": "gpu-lock-test",
        }
    )
    first_env = base_env | {"TABERO_MATRIX_RUN_ID": "20260728_120000_formal"}
    second_env = base_env | {"TABERO_MATRIX_RUN_ID": "20260728_120001_formal"}
    first = subprocess.Popen(
        ["bash", str(LAUNCHER), "dsrl", "0", "formal", "--dry-run"],
        cwd=REPO_ROOT,
        env=first_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        time.sleep(0.2)
        second = subprocess.run(
            ["bash", str(LAUNCHER), "dsrl", "5", "formal", "--dry-run"],
            cwd=REPO_ROOT,
            env=second_env,
            text=True,
            capture_output=True,
            timeout=10,
        )
    finally:
        first.communicate(timeout=10)

    assert second.returncode != 0
    assert "GPU 0 lease is already held" in second.stderr


def test_matrix_launcher_serializes_resume_output_lease(tmp_path):
    checkpoint = _fake_resume_checkpoint(tmp_path)
    output_dir = checkpoint.parents[2]
    fake_bin = _fake_nvidia_smi(tmp_path)
    base_env = os.environ.copy()
    base_env.update(
        {
            "PATH": f"{fake_bin}:{base_env['PATH']}",
            "TABERO_RESULTS_ROOT": str(tmp_path / "unused-results"),
            "TABERO_MIN_FREE_KIB": "1",
            "TABERO_TEST_MODEL_SHA256": "0" * 64,
            "TABERO_GPU_LOCK_DIR": str(tmp_path / "gpu-locks"),
            "FAKE_GPU_COUNT": "16",
            "FAKE_DF_DELAY_PATH": str(output_dir),
            "FAKE_DF_DELAY": "2",
        }
    )
    command = [
        "bash",
        str(LAUNCHER),
        "dsrl",
        "0",
        "formal",
        "--resume-dir",
        str(checkpoint),
        "--dry-run",
    ]
    first = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=base_env | {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        time.sleep(0.2)
        second = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=base_env | {"CUDA_VISIBLE_DEVICES": "8,9,10,11,12,13,14,15"},
            text=True,
            capture_output=True,
            timeout=10,
        )
    finally:
        first.communicate(timeout=10)

    assert second.returncode != 0
    assert "formal output lease is already held" in second.stderr


def test_matrix_launcher_rejects_disk_override_outside_dry_run(tmp_path):
    resume_dir = tmp_path / "global_step_10"
    resume_dir.mkdir()
    env = os.environ.copy()
    fake_bin = _fake_nvidia_smi(tmp_path)
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "TABERO_RESULTS_ROOT": str(tmp_path / "results"),
            "TABERO_MIN_FREE_KIB": "1",
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "dsrl",
            "0",
            "formal",
            "--resume-dir",
            str(resume_dir),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "TABERO_MIN_FREE_KIB is only allowed with --dry-run" in result.stderr
