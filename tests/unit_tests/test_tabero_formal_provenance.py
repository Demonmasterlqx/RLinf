# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Synthetic provenance tests: no local models, datasets, or training configs."""

import argparse
import hashlib
import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def provenance(monkeypatch):
    source = (
        Path(__file__).resolve().parents[2]
        / "examples/embodiment/tabero_formal_provenance.py"
    )
    spec = importlib.util.spec_from_file_location("tabero_formal_provenance", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_git_state", lambda _: ("commit-a", True))

    def forbid_hash(*args, **kwargs):
        pytest.fail("Training provenance must never calculate SHA-256")

    monkeypatch.setattr(hashlib, "sha256", forbid_hash)
    return module


@pytest.fixture
def args(tmp_path):
    config = tmp_path / "input.txt"
    config.write_text("synthetic input\n")
    output = tmp_path / "output"
    output.mkdir()
    model = tmp_path / "empty-model-directory"
    model.mkdir()
    return argparse.Namespace(
        config_path=config,
        model_path=model,
        repo_root=tmp_path,
        output_dir=output,
    )


def test_capture_and_resume_without_hashes_or_clean_worktree(
    provenance, args, monkeypatch
):
    provenance.capture(args)
    record = (args.output_dir / provenance.PROVENANCE_FILENAME).read_text()
    assert "TABERO_PROVENANCE_VERSION=2" in record
    assert "TABERO_GIT_DIRTY=true" in record
    assert "SHA256" not in record
    monkeypatch.setattr(provenance, "_git_state", lambda _: ("commit-b", False))
    provenance.verify(args)


def test_resume_keeps_config_and_model_path_lineage(provenance, args):
    provenance.capture(args)
    args.config_path.write_text("changed input\n")
    with pytest.raises(ValueError, match="differs from the recorded snapshot"):
        provenance.verify(args)
    args.config_path.write_text("synthetic input\n")
    args.model_path = args.repo_root / "another-model-directory"
    args.model_path.mkdir()
    with pytest.raises(ValueError, match="base model path"):
        provenance.verify(args)


def test_capture_never_overwrites_existing_evidence(provenance, args):
    provenance.capture(args)
    before = (args.output_dir / provenance.PROVENANCE_FILENAME).read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        provenance.capture(args)
    assert (args.output_dir / provenance.PROVENANCE_FILENAME).read_bytes() == before


def test_old_provenance_is_rejected_without_hashing(provenance, args):
    provenance.capture(args)
    path = args.output_dir / provenance.PROVENANCE_FILENAME
    path.write_text(
        path.read_text().replace(
            "TABERO_PROVENANCE_VERSION=2", "TABERO_PROVENANCE_VERSION=1"
        )
    )
    with pytest.raises(ValueError, match="unsupported provenance version"):
        provenance.verify(args)


def test_migration_mode_and_argument_are_rejected(provenance, args):
    provenance.capture(args)
    path = args.output_dir / provenance.PROVENANCE_FILENAME
    path.write_text(
        path.read_text().replace(
            "TABERO_PROVENANCE_MODE=fresh", "TABERO_PROVENANCE_MODE=legacy_migration"
        )
    )
    with pytest.raises(ValueError, match="unsupported provenance mode"):
        provenance.verify(args)
    with pytest.raises(SystemExit):
        provenance._parser().parse_args(
            [
                "capture",
                "--output-dir",
                str(args.output_dir),
                "--config-path",
                str(args.config_path),
                "--model-path",
                str(args.model_path),
                "--repo-root",
                str(args.repo_root),
                "--legacy-source-config",
                "old.txt",
            ]
        )


@pytest.fixture
def launcher_fixture(tmp_path):
    """Build a synthetic launcher workspace, without any local assets/configs."""
    source = Path(__file__).resolve().parents[2] / "examples/embodiment"
    repo = tmp_path / "RLinf"
    scripts = repo / "examples/embodiment"
    config_dir = scripts / "config"
    config_dir.mkdir(parents=True)
    for name in ("run_tabero_firm_matrix_train.sh", "tabero_formal_provenance.py"):
        shutil.copyfile(source / name, scripts / name)
    (scripts / "train_embodied_agent.py").write_text(
        'raise AssertionError("dry-run must never start training")\n'
    )
    config = (
        config_dir / "isaaclab_pi0_dsrl_tacfield_tabero_task0_firm_8gpu_50step.yaml"
    )
    config.write_text(
        "cluster:\n  num_nodes: 1\n  component_placement:\n"
        '    actor: "4"\n    rollout: "2"\n    env: "2"\n'
    )
    shutil.copyfile(
        config,
        config_dir
        / "isaaclab_pi0_peft_lora_tacfield_tabero_task0_firm_8gpu_50step.yaml",
    )
    rlt_config_dir = repo / "examples/tabero"
    rlt_config_dir.mkdir()
    shutil.copyfile(config, rlt_config_dir / "tabero_rlt_stage2_ac_task0_firm.yaml")
    (repo / ".venv/bin").mkdir(parents=True)
    python = repo / ".venv/bin/python"
    python.write_text(f'#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} "$@"\n')
    python.chmod(0o755)
    setup = tmp_path / "setup.sh"
    setup.write_text("# synthetic setup\n")
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"not a real model")
    commands = tmp_path / "bin"
    commands.mkdir()
    for name, body in {
        "git": 'if [[ "$1" == rev-parse ]]; then echo fixture-commit; else echo " M fixture"; fi',
        "nvidia-smi": 'if [[ "$*" == "--query-gpu=index --format=csv,noheader" ]]; then printf "2\\n4\\n"; fi',
        "sha256sum": 'echo "SHA command must not run" >&2; exit 99',
    }.items():
        command = commands / name
        command.write_text("#!/usr/bin/env bash\n" + body + "\n")
        command.chmod(0o755)
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(
        PATH=f"{commands}:{env['PATH']}",
        ROOT=str(tmp_path),
        TABERO_MODEL_PATH=str(model),
        TABERO_ISAAC_SETUP=str(setup),
        TABERO_RESULTS_ROOT=str(tmp_path / "results"),
        TABERO_GPU_LOCK_DIR=str(tmp_path / "locks"),
        TABERO_MIN_FREE_KIB="1",
        TABERO_MATRIX_RUN_ID="20000101_000000_formal",
        WANDB_RUN_ID="synthetic-wandb",
    )
    return scripts / "run_tabero_firm_matrix_train.sh", env


@pytest.mark.parametrize("method", ["dsrl", "pirl", "rlt"])
def test_launcher_uses_configured_gpus_and_does_not_hash(launcher_fixture, method):
    launcher, env = launcher_fixture
    result = subprocess.run(
        ["bash", str(launcher), method, "0", "formal", "--dry-run"],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    locks = {path.name for path in Path(env["TABERO_GPU_LOCK_DIR"]).glob("gpu_*.lock")}
    assert locks == {"gpu_2.lock", "gpu_4.lock"}
    records = list(Path(env["TABERO_RESULTS_ROOT"]).glob("*/provenance.env"))
    assert len(records) == 1
    assert "SHA256" not in records[0].read_text()
    assert "TABERO_GIT_DIRTY=true" in records[0].read_text()
    if method == "rlt":
        assert "actor.model.model_path=" not in result.stdout
        assert "rollout.rlt_feature_model.openpi_data.norm_stats_path=" in result.stdout
    else:
        assert f"actor.model.model_path={env['TABERO_MODEL_PATH']}" in result.stdout


def test_launcher_rejects_cuda_mask_allocation(launcher_fixture):
    launcher, env = launcher_fixture
    env["CUDA_VISIBLE_DEVICES"] = "2,4"
    result = subprocess.run(
        ["bash", str(launcher), "dsrl", "0", "formal", "--dry-run"],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "cluster.component_placement" in result.stderr


def test_launcher_rejects_old_run_env_without_migration(launcher_fixture):
    launcher, env = launcher_fixture
    command = ["bash", str(launcher), "dsrl", "0", "formal", "--dry-run"]
    fresh = subprocess.run(command, env=env, text=True, capture_output=True, timeout=30)
    assert fresh.returncode == 0, fresh.stderr
    record = next(Path(env["TABERO_RESULTS_ROOT"]).glob("*/run.env"))
    values = dict(line.split("=", 1) for line in record.read_text().splitlines())
    checkpoint = (
        record.parent / values["TABERO_EXPERIMENT_NAME"] / "checkpoints/global_step_1"
    )
    (checkpoint / "actor").mkdir(parents=True)
    record.write_text(
        "\n".join(
            line
            for line in record.read_text().splitlines()
            if not line.startswith("TABERO_LAUNCH_KIND=")
        )
        + "\n"
    )
    resumed = subprocess.run(
        command + ["--resume-dir", str(checkpoint)],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert resumed.returncode != 0
    assert "unsupported run.env" in resumed.stderr
