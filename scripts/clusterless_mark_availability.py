#!/usr/bin/env python3
"""Audit whether a dataset copy can support clusterless mark-based evidence.

The clusterless consistency screen is only scientifically meaningful when
spike-mark or waveform-feature data are present. This audit deliberately uses
a broad, file-format-tolerant scan so it can run before the benchmark code tries
to instantiate clusterless emissions. It is a feasibility gate, not a decoder.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_SESSIONS = (
    "Rat1/Open1",
    "Rat1/Open2",
    "Rat2/Open1",
    "Rat2/Open2",
    "Rat3/Open1",
    "Rat3/Open2",
    "Rat4/Open1",
    "Rat4/Open2",
)

SUPPORTED_SUFFIXES = {
    ".csv",
    ".tsv",
    ".txt",
    ".json",
    ".jsonl",
    ".mat",
    ".npy",
    ".npz",
    ".h5",
    ".hdf5",
}

MARK_TOKENS = (
    "mark",
    "marks",
    "clusterless",
    "waveform",
    "waveforms",
    "feature",
    "features",
    "amplitude",
    "amplitudes",
    "peak",
    "peaks",
    "energy",
    "width",
    "channel_feature",
    "channel_features",
)

SPIKE_TOKENS = (
    "spike",
    "spikes",
    "spike_time",
    "spike_times",
    "spiketimes",
    "unit",
    "units",
)

MARK_PATH_RE = re.compile(
    r"(^|[/_.\-\s])"
    r"(marks?|clusterless|waveforms?|features?|amplitudes?|peaks?|energy|width|channel[_-]?features?)"
    r"($|[/_.\-\s])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SessionMarkAvailability:
    session: str
    status: str
    session_path_found: bool
    candidate_roots: str
    candidate_files_scanned: int
    spike_like_files: int
    mark_path_hits: int
    mark_key_hits: int
    mark_like_files: int
    has_mark_like_paths: bool
    has_mark_like_keys: bool
    has_clusterless_marks: bool
    example_mark_paths: str
    example_mark_keys: str


def parse_sessions(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return list(DEFAULT_SESSIONS)
    if isinstance(value, str):
        tokens = [token.strip() for token in re.split(r"[\s,]+", value) if token.strip()]
    else:
        tokens = [str(token).strip() for token in value if str(token).strip()]
    return tokens or list(DEFAULT_SESSIONS)


def discover_session_roots(dataset_root: Path, session: str) -> list[Path]:
    """Return plausible roots for a session without assuming one dataset layout."""

    rat, _, open_name = session.partition("/")
    candidates: list[Path] = []
    direct = dataset_root / rat / open_name if open_name else dataset_root / session
    flat_underscore = dataset_root / session.replace("/", "_")
    flat_dash = dataset_root / session.replace("/", "-")
    flat_plain = dataset_root / session.replace("/", "")
    for path in (direct, flat_underscore, flat_dash, flat_plain):
        if path.exists() and path not in candidates:
            candidates.append(path)

    rat_lower = rat.lower()
    open_lower = open_name.lower()
    if dataset_root.exists():
        for path in dataset_root.rglob("*"):
            if not path.is_dir():
                continue
            text = str(path.relative_to(dataset_root)).replace("\\", "/").lower()
            if rat_lower in text and (not open_lower or open_lower in text):
                if path not in candidates:
                    candidates.append(path)
            if len(candidates) >= 20:
                break
    return candidates


def iter_session_files(
    dataset_root: Path,
    session_roots: list[Path],
    session: str,
    *,
    max_files: int,
) -> list[Path]:
    """Return bounded candidate files for a session."""

    files: list[Path] = []
    seen: set[Path] = set()
    for root in session_roots:
        if root.is_file():
            candidates = [root]
        else:
            candidates = [path for path in root.rglob("*") if path.is_file()]
        for path in candidates:
            if path in seen:
                continue
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            seen.add(path)
            files.append(path)
            if len(files) >= max_files:
                return files

    if files or not dataset_root.exists():
        return files

    rat, _, open_name = session.partition("/")
    rat_lower = rat.lower()
    open_lower = open_name.lower()
    for path in dataset_root.rglob("*"):
        if not path.is_file() or path in seen:
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = str(path.relative_to(dataset_root)).replace("\\", "/").lower()
        if rat_lower in text and (not open_lower or open_lower in text):
            seen.add(path)
            files.append(path)
            if len(files) >= max_files:
                break
    return files


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def inspect_text_like_file(path: Path, *, max_bytes: int = 64_000) -> list[str]:
    try:
        text = path.read_bytes()[:max_bytes].decode("utf-8", errors="ignore").lower()
    except OSError:
        return []
    return sorted({token for token in MARK_TOKENS if token in text})


def inspect_npz_file(path: Path) -> list[str]:
    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception:
        return []
    try:
        with np.load(path, allow_pickle=False) as data:
            keys = [str(key).lower() for key in data.files]
    except Exception:
        return []
    return sorted({key for key in keys if any(token in key for token in MARK_TOKENS)})


def inspect_mat_file(path: Path) -> list[str]:
    try:
        from scipy.io import whosmat  # type: ignore[import-not-found]
    except Exception:
        return []
    try:
        keys = [str(item[0]).lower() for item in whosmat(path)]
    except Exception:
        return []
    return sorted({key for key in keys if any(token in key for token in MARK_TOKENS)})


def inspect_hdf5_file(path: Path, *, max_keys: int = 2000) -> list[str]:
    try:
        import h5py  # type: ignore[import-not-found]
    except Exception:
        return []
    keys: list[str] = []
    try:
        with h5py.File(path, "r") as handle:

            def visitor(name: str) -> None:
                if len(keys) >= max_keys:
                    return
                keys.append(name.lower())

            handle.visit(visitor)
    except Exception:
        return []
    return sorted({key for key in keys if any(token in key for token in MARK_TOKENS)})


def inspect_mark_keys(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt", ".json", ".jsonl"}:
        return inspect_text_like_file(path)
    if suffix == ".npz":
        return inspect_npz_file(path)
    if suffix == ".mat":
        return inspect_mat_file(path)
    if suffix in {".h5", ".hdf5"}:
        return inspect_hdf5_file(path)
    return []


def audit_session(
    dataset_root: Path,
    session: str,
    *,
    max_files_per_session: int,
) -> SessionMarkAvailability:
    roots = discover_session_roots(dataset_root, session)
    files = iter_session_files(
        dataset_root,
        roots,
        session,
        max_files=max_files_per_session,
    )
    mark_path_examples: list[str] = []
    mark_key_examples: list[str] = []
    mark_path_hits = 0
    mark_key_hits = 0
    mark_like_files = 0
    spike_like_files = 0
    for path in files:
        rel = _safe_relative(path, dataset_root)
        lower_rel = rel.lower()
        has_spike = any(token in lower_rel for token in SPIKE_TOKENS)
        has_mark_path = MARK_PATH_RE.search(lower_rel) is not None
        key_hits = inspect_mark_keys(path)
        has_mark_keys = bool(key_hits)
        if has_spike:
            spike_like_files += 1
        if has_mark_path:
            mark_path_hits += 1
            if len(mark_path_examples) < 5:
                mark_path_examples.append(rel)
        if has_mark_keys:
            mark_key_hits += 1
            if len(mark_key_examples) < 10:
                mark_key_examples.extend(f"{rel}:{key}" for key in key_hits[:3])
        if has_mark_path or has_mark_keys:
            mark_like_files += 1

    session_found = bool(roots or files)
    has_marks = mark_like_files > 0
    if has_marks:
        status = "marks_detected"
    elif session_found:
        status = "no_marks_detected"
    else:
        status = "session_missing"
    return SessionMarkAvailability(
        session=session,
        status=status,
        session_path_found=session_found,
        candidate_roots=";".join(_safe_relative(path, dataset_root) for path in roots[:10]),
        candidate_files_scanned=len(files),
        spike_like_files=spike_like_files,
        mark_path_hits=mark_path_hits,
        mark_key_hits=mark_key_hits,
        mark_like_files=mark_like_files,
        has_mark_like_paths=mark_path_hits > 0,
        has_mark_like_keys=mark_key_hits > 0,
        has_clusterless_marks=has_marks,
        example_mark_paths=";".join(mark_path_examples),
        example_mark_keys=";".join(mark_key_examples[:10]),
    )


def audit_clusterless_mark_availability(
    dataset_root: str | Path,
    *,
    sessions: Iterable[str] = DEFAULT_SESSIONS,
    max_files_per_session: int = 3000,
) -> list[SessionMarkAvailability]:
    root = Path(dataset_root)
    return [
        audit_session(root, session, max_files_per_session=max_files_per_session)
        for session in sessions
    ]


def gate_summary(
    rows: list[SessionMarkAvailability],
    *,
    dataset_root: Path,
) -> list[dict[str, object]]:
    requested = len(rows)
    present = sum(row.session_path_found for row in rows)
    with_marks = sum(row.has_clusterless_marks for row in rows)
    rows_with_status = {row.status for row in rows}
    any_marks = with_marks > 0
    all_marks = requested > 0 and with_marks == requested
    partial_marks = any_marks and not all_marks
    gates: list[dict[str, object]] = []

    def add(gate: str, passed: bool, observed: object, criterion: str) -> None:
        gates.append(
            {
                "gate": gate,
                "passed": bool(passed),
                "observed": observed,
                "criterion": criterion,
            }
        )

    add("dataset_root_exists", dataset_root.exists(), str(dataset_root), "dataset root exists")
    add(
        "sessions_found",
        present == requested and requested > 0,
        f"{present}/{requested}",
        "all requested sessions are discoverable",
    )
    add(
        "any_session_has_clusterless_marks",
        any_marks,
        f"{with_marks}/{requested}",
        "at least one requested session has spike marks",
    )
    add(
        "all_sessions_have_clusterless_marks",
        all_marks,
        f"{with_marks}/{requested}",
        "all requested sessions have spike marks",
    )
    if all_marks:
        readiness = "ready_for_true_clusterless_consistency"
    elif partial_marks:
        readiness = "partial_mark_coverage"
    elif "session_missing" in rows_with_status:
        readiness = "dataset_or_session_layout_unresolved"
    else:
        readiness = "blocked_no_marks_detected"
    add(
        "overall_clusterless_goal3_ready",
        all_marks,
        readiness,
        "true clusterless consistency requires marks for all audited sessions",
    )
    return gates


def write_outputs(
    rows: list[SessionMarkAvailability],
    output: str | Path,
    *,
    dataset_root: str | Path,
    max_files_per_session: int,
) -> None:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    availability_path = out / "clusterless_mark_availability.csv"
    gate_path = out / "clusterless_mark_gate_summary.csv"
    manifest_path = out / "clusterless_mark_availability_manifest.json"

    row_dicts = [asdict(row) for row in rows]
    with availability_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row_dicts[0].keys()) if row_dicts else [])
        writer.writeheader()
        writer.writerows(row_dicts)

    gates = gate_summary(rows, dataset_root=Path(dataset_root))
    with gate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["gate", "passed", "observed", "criterion"])
        writer.writeheader()
        writer.writerows(gates)

    manifest = {
        "schema_version": 1,
        "dataset_root": str(dataset_root),
        "sessions_requested": [row.session for row in rows],
        "max_files_per_session": int(max_files_per_session),
        "sessions_with_marks": [row.session for row in rows if row.has_clusterless_marks],
        "sessions_without_marks": [
            row.session for row in rows if row.status == "no_marks_detected"
        ],
        "missing_sessions": [row.session for row in rows if row.status == "session_missing"],
        "gate_summary": gates,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--sessions", default=" ".join(DEFAULT_SESSIONS))
    parser.add_argument("--max-files-per-session", type=int, default=3000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sessions = parse_sessions(args.sessions)
    rows = audit_clusterless_mark_availability(
        args.dataset_root,
        sessions=sessions,
        max_files_per_session=args.max_files_per_session,
    )
    write_outputs(
        rows,
        args.output,
        dataset_root=args.dataset_root,
        max_files_per_session=args.max_files_per_session,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
