# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Capture and verify immutable provenance for Tabero formal training."""

import argparse
import hashlib
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
    "TABERO_CONFIG_SHA256",
    "TABERO_CONFIG_SNAPSHOT_SHA256",
    "TABERO_GIT_COMMIT",
    "TABERO_GIT_DIRTY",
    "TABERO_BASE_MODEL_PATH",
    "TABERO_BASE_MODEL_SHA256",
    "TABERO_SOURCE_CONFIG_SHA256",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _validate_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _read_provenance(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or not value:
            raise ValueError(f"invalid provenance line: {line!r}")
        if key not in PROVENANCE_KEYS:
            raise ValueError(f"unknown provenance key: {key}")
        if key in values:
            raise ValueError(f"duplicate provenance key: {key}")
        values[key] = value
    if set(values) != set(PROVENANCE_KEYS):
        missing = sorted(set(PROVENANCE_KEYS) - set(values))
        extra = sorted(set(values) - set(PROVENANCE_KEYS))
        raise ValueError(
            f"invalid provenance keyspace: missing={missing}, extra={extra}"
        )
    if values["TABERO_PROVENANCE_VERSION"] != "1":
        raise ValueError("unsupported provenance version")
    mode = values["TABERO_PROVENANCE_MODE"]
    if mode not in {"fresh", "legacy_migration"}:
        raise ValueError("unsupported provenance mode")
    _validate_sha256(values["TABERO_CONFIG_SHA256"], "config SHA-256")
    _validate_sha256(
        values["TABERO_CONFIG_SNAPSHOT_SHA256"],
        "config snapshot SHA-256",
    )
    _validate_sha256(values["TABERO_BASE_MODEL_SHA256"], "base model SHA-256")
    source_hash = values["TABERO_SOURCE_CONFIG_SHA256"]
    if mode == "fresh" and source_hash != "none":
        raise ValueError("fresh provenance must not contain a source config SHA-256")
    if mode == "legacy_migration":
        _validate_sha256(source_hash, "legacy source config SHA-256")
    return values


def _current_values(args: argparse.Namespace, mode: str) -> dict[str, str]:
    config_path = args.config_path.resolve()
    model_path = args.model_path.resolve()
    repo_root = args.repo_root.resolve()
    if not config_path.is_file():
        raise ValueError(f"config does not exist: {config_path}")
    if not model_path.is_dir():
        raise ValueError(f"base model does not exist: {model_path}")
    _validate_sha256(args.base_model_sha256, "base model SHA-256")
    git_commit, git_dirty = _git_state(repo_root)
    if git_dirty and not args.allow_dirty:
        raise ValueError("RLinf worktree must be clean for formal training")
    source_hash = "none"
    if args.source_config is not None:
        source_config = args.source_config.resolve()
        if not source_config.is_file():
            raise ValueError(f"legacy source config does not exist: {source_config}")
        source_hash = _sha256(source_config)
    return {
        "TABERO_PROVENANCE_VERSION": "1",
        "TABERO_PROVENANCE_MODE": mode,
        "TABERO_CONFIG_SHA256": _sha256(config_path),
        "TABERO_CONFIG_SNAPSHOT_SHA256": _sha256(config_path),
        "TABERO_GIT_COMMIT": git_commit,
        "TABERO_GIT_DIRTY": str(git_dirty).lower(),
        "TABERO_BASE_MODEL_PATH": str(model_path),
        "TABERO_BASE_MODEL_SHA256": args.base_model_sha256,
        "TABERO_SOURCE_CONFIG_SHA256": source_hash,
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
    mode = "legacy_migration" if args.legacy_source_config is not None else "fresh"
    args.source_config = args.legacy_source_config
    values = _current_values(args, mode)
    temporary_snapshot = output_dir / f".{CONFIG_SNAPSHOT_FILENAME}.{os.getpid()}"
    shutil.copyfile(args.config_path, temporary_snapshot)
    try:
        os.link(temporary_snapshot, snapshot_path)
        _write_atomic(
            provenance_path,
            [f"{key}={values[key]}" for key in PROVENANCE_KEYS],
        )
    except Exception:
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
    args.source_config = None
    current = _current_values(args, recorded["TABERO_PROVENANCE_MODE"])
    snapshot_hash = _sha256(snapshot_path)
    if snapshot_hash != recorded["TABERO_CONFIG_SNAPSHOT_SHA256"]:
        raise ValueError("config snapshot SHA-256 does not match provenance")
    if recorded["TABERO_PROVENANCE_MODE"] == "legacy_migration":
        legacy_source_config = output_dir / "tensorboard" / "config.yaml"
        if not legacy_source_config.is_file():
            raise ValueError(
                f"legacy source config does not exist: {legacy_source_config}"
            )
        if _sha256(legacy_source_config) != recorded["TABERO_SOURCE_CONFIG_SHA256"]:
            raise ValueError("legacy source config SHA-256 does not match provenance")
    comparisons = (
        ("TABERO_CONFIG_SHA256", "config SHA-256 does not match provenance"),
        ("TABERO_GIT_COMMIT", "Git commit does not match provenance"),
        ("TABERO_GIT_DIRTY", "Git dirty state does not match provenance"),
        ("TABERO_BASE_MODEL_PATH", "base model path does not match provenance"),
        ("TABERO_BASE_MODEL_SHA256", "base model SHA-256 does not match provenance"),
    )
    for key, message in comparisons:
        if current[key] != recorded[key]:
            raise ValueError(message)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("capture", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--output-dir", type=Path, required=True)
        subparser.add_argument("--config-path", type=Path, required=True)
        subparser.add_argument("--model-path", type=Path, required=True)
        subparser.add_argument("--base-model-sha256", required=True)
        subparser.add_argument("--repo-root", type=Path, required=True)
        subparser.add_argument("--allow-dirty", action="store_true")
        if command == "capture":
            subparser.add_argument("--legacy-source-config", type=Path)
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
