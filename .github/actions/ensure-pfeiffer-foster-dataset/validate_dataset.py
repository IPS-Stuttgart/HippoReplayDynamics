from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_SESSION_FILES = (
    "Position_Data.mat",
    "Ripple_Events.mat",
    "Spike_Data.mat",
    "Epochs.mat",
)
OPTIONAL_SESSION_FILES = ("Well_Sequence.mat", "Experiment_Information.mat")
MANIFEST_NAMES = {"MANIFEST.txt", "dataset_manifest.json"}


def session_is_valid(session_dir: Path) -> bool:
    return session_dir.is_dir() and all((session_dir / name).is_file() for name in REQUIRED_SESSION_FILES)


def valid_sessions(dataset_root: Path) -> list[Path]:
    if not dataset_root.is_dir():
        return []
    return sorted(
        session
        for session in dataset_root.glob("Rat*/Open*")
        if session_is_valid(session)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tracked_files(dataset_root: Path) -> list[Path]:
    return sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.name not in MANIFEST_NAMES
    )


def _file_record(path: Path, dataset_root: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": path.relative_to(dataset_root).as_posix(),
        "size_bytes": stat.st_size,
        "sha256": _sha256(path),
    }


def _session_record(session_dir: Path, dataset_root: Path) -> dict[str, object]:
    required_files = []
    missing_required = []
    for name in REQUIRED_SESSION_FILES:
        path = session_dir / name
        if path.is_file():
            required_files.append(_file_record(path, dataset_root))
        else:
            missing_required.append(name)
    optional_files = []
    for name in OPTIONAL_SESSION_FILES:
        path = session_dir / name
        if path.is_file():
            optional_files.append(_file_record(path, dataset_root))
    return {
        "session": session_dir.relative_to(dataset_root).as_posix(),
        "missing_required_files": missing_required,
        "required_files": required_files,
        "optional_files": optional_files,
    }


def _write_manifest(dataset_root: Path, files: list[Path], target: Path | None = None) -> Path:
    sessions = valid_sessions(dataset_root)
    manifest = {
        "dataset_root_name": dataset_root.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "session_count": len(sessions),
        "sessions": [_session_record(session, dataset_root) for session in sessions],
        "files": [_file_record(path, dataset_root) for path in files],
        "total_bytes": sum(path.stat().st_size for path in files),
    }
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest["manifest_sha256_without_this_field"] = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()
    target = target or dataset_root / "dataset_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a Pfeiffer/Foster dataset tree.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--session", default="", help="Optional session such as Rat1/Open1")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument(
        "--manifest-output",
        type=Path,
        help="Optional manifest path to use instead of writing into the dataset root.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser()
    if args.session:
        session_dir = dataset_root / args.session
        if not session_is_valid(session_dir):
            missing = [
                name for name in REQUIRED_SESSION_FILES if not (session_dir / name).is_file()
            ]
            raise SystemExit(
                f"Invalid Pfeiffer/Foster dataset at {dataset_root}: "
                f"session {args.session!r} is missing {missing or 'the session directory'}"
            )
        sessions = [session_dir]
    else:
        sessions = valid_sessions(dataset_root)
        if not sessions:
            raise SystemExit(
                f"Invalid Pfeiffer/Foster dataset at {dataset_root}: "
                "no Rat*/Open* session with the required .mat files was found."
            )

    files = _tracked_files(dataset_root)
    total_bytes = sum(path.stat().st_size for path in files)
    if args.write_manifest:
        manifest_path = _write_manifest(dataset_root, files, args.manifest_output)
        print(f"Wrote dataset manifest: {manifest_path}")

    print(f"Validated Pfeiffer/Foster dataset at {dataset_root}")
    print(f"Checked session(s): {', '.join(str(s.relative_to(dataset_root)) for s in sessions)}")
    print(f"Files: {len(files)}")
    print(f"Bytes: {total_bytes}")


if __name__ == "__main__":
    main()
