# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Record training provenance and guard resume lineage without hashing weights."""

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

PROVENANCE_FILENAME = "provenance.env"
CONFIG_SNAPSHOT_FILENAME = "config_snapshot.yaml"
PROVENANCE_KEYS = (
    "TABERO_PROVENANCE_VERSION",
    "TABERO_PROVENANCE_MODE",
    "TABERO_CONFIG_PATH",
    "TABERO_GIT_COMMIT",
    "TABERO_GIT_DIRTY",
    "TABERO_BASE_MODEL_PATH",
    "TABERO_SOURCE_CONFIG_PATH",
)


def _git_state(repo_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, bool(status.strip())


def _read_provenance(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or not value:
            raise ValueError(f"invalid provenance line: {line!r}")
        if key in values:
            raise ValueError(f"duplicate provenance key: {key}")
        values[key] = value
    version = values.get("TABERO_PROVENANCE_VERSION")
    if version != "2":
        raise ValueError("unsupported provenance version")
    required = set(PROVENANCE_KEYS)
    missing = required - values.keys()
    if missing:
        raise ValueError(f"missing provenance keys: {sorted(missing)}")
    if values["TABERO_PROVENANCE_MODE"] != "fresh":
        raise ValueError("unsupported provenance mode")
    return values


def _current_values(args: argparse.Namespace) -> dict[str, str]:
    config_path = args.config_path.resolve()
    model_path = args.model_path.resolve()
    repo_root = args.repo_root.resolve()
    if not config_path.is_file():
        raise ValueError(f"config does not exist: {config_path}")
    if not model_path.is_dir():
        raise ValueError(f"base model does not exist: {model_path}")
    git_commit, git_dirty = _git_state(repo_root)
    return {
        "TABERO_PROVENANCE_VERSION": "2",
        "TABERO_PROVENANCE_MODE": "fresh",
        "TABERO_CONFIG_PATH": str(config_path),
        "TABERO_GIT_COMMIT": git_commit,
        "TABERO_GIT_DIRTY": str(git_dirty).lower(),
        "TABERO_BASE_MODEL_PATH": str(model_path),
        "TABERO_SOURCE_CONFIG_PATH": "none",
    }


def _write_atomic(path: Path, lines: list[str]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as file:
            file.write("\n".join(lines) + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.link(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def capture(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        raise ValueError(f"output directory does not exist: {output_dir}")
    provenance_path = output_dir / PROVENANCE_FILENAME
    snapshot_path = output_dir / CONFIG_SNAPSHOT_FILENAME
    if provenance_path.exists() or snapshot_path.exists():
        raise ValueError("formal provenance already exists")
    values = _current_values(args)
    temporary_snapshot = output_dir / f".{CONFIG_SNAPSHOT_FILENAME}.{os.getpid()}"
    shutil.copyfile(args.config_path, temporary_snapshot)
    created_snapshot = False
    try:
        os.link(temporary_snapshot, snapshot_path)
        created_snapshot = True
        _write_atomic(
            provenance_path,
            [f"{key}={values[key]}" for key in PROVENANCE_KEYS],
        )
    except Exception:
        if created_snapshot:
            snapshot_path.unlink(missing_ok=True)
        raise
    finally:
        temporary_snapshot.unlink(missing_ok=True)


def verify(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    provenance_path = output_dir / PROVENANCE_FILENAME
    snapshot_path = output_dir / CONFIG_SNAPSHOT_FILENAME
    if not provenance_path.is_file() or not snapshot_path.is_file():
        raise ValueError(
            "current run.env requires provenance.env and config_snapshot.yaml"
        )
    recorded = _read_provenance(provenance_path)
    current = _current_values(args)
    # Configuration continuity still matters; no digest or model-file scan is needed.
    if snapshot_path.read_bytes() != args.config_path.read_bytes():
        raise ValueError("training config differs from the recorded snapshot")
    if current["TABERO_BASE_MODEL_PATH"] != recorded["TABERO_BASE_MODEL_PATH"]:
        raise ValueError("base model path does not match provenance")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("capture", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--output-dir", type=Path, required=True)
        subparser.add_argument("--config-path", type=Path, required=True)
        subparser.add_argument("--model-path", type=Path, required=True)
        subparser.add_argument("--repo-root", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.command == "capture":
            capture(args)
        else:
            verify(args)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
