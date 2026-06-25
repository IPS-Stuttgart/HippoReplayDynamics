"""Small provenance helpers for script-generated analysis artifacts."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Mapping, Sequence


def build_script_provenance(
    *,
    input_paths: Mapping[str, str | Path | None] | None = None,
    argv: Sequence[str] | None = None,
    cwd: str | Path | None = None,
) -> dict[str, object]:
    """Return reproducibility metadata for a script invocation.

    The helper is intentionally dependency-free and tolerant of non-git contexts.
    File hashes are only computed for regular files; directories and missing paths
    are recorded with a null hash rather than making provenance collection fatal.
    """

    working_directory = Path(cwd or os.getcwd()).resolve()
    path_map = {key: str(value) for key, value in (input_paths or {}).items() if value not in (None, "")}
    hash_paths = {key: _resolve_input_path(value, working_directory) for key, value in path_map.items()}
    provenance = {
        **git_metadata(working_directory),
        "command_line": command_line(argv or sys.argv),
        "working_directory": str(working_directory),
        "input_file_paths": path_map,
        "input_file_sha256": {key: file_sha256(path) for key, path in hash_paths.items()},
    }
    return provenance


def git_metadata(cwd: str | Path | None = None) -> dict[str, object]:
    """Return commit, branch, dirty flag, and error details if git is unavailable."""

    working_directory = Path(cwd or os.getcwd()).resolve()
    env_commit = os.environ.get("GITHUB_SHA") or os.environ.get("CI_COMMIT_SHA")
    env_branch = os.environ.get("GITHUB_REF_NAME") or os.environ.get("CI_COMMIT_REF_NAME")
    metadata: dict[str, object] = {
        "code_commit": env_commit or "unavailable",
        "git_branch": env_branch or "unavailable",
        "git_dirty": None,
        "git_error": "",
    }
    try:
        if not env_commit:
            metadata["code_commit"] = _git_stdout(["rev-parse", "HEAD"], cwd=working_directory)
        if not env_branch:
            branch = _git_stdout(["branch", "--show-current"], cwd=working_directory)
            metadata["git_branch"] = branch or _git_stdout(["rev-parse", "--abbrev-ref", "HEAD"], cwd=working_directory)
        status = _git_stdout(["status", "--porcelain"], cwd=working_directory)
        metadata["git_dirty"] = bool(status)
    except Exception as exc:  # pragma: no cover - exercised only outside git checkouts.
        metadata["git_error"] = str(exc)
    return metadata


def command_line(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in argv)


def _resolve_input_path(path: str | Path, working_directory: Path) -> Path:
    file_path = Path(path)
    return file_path if file_path.is_absolute() else working_directory / file_path


def file_sha256(path: str | Path) -> str | None:
    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_stdout(args: Sequence[str], *, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()
