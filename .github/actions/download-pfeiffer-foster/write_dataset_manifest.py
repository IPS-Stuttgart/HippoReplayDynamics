from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_SESSION_FILES = (
    "Position_Data.mat",
    "Ripple_Events.mat",
    "Spike_Data.mat",
    "Epochs.mat",
)
OPTIONAL_SESSION_FILES = ("Well_Sequence.mat", "Experiment_Information.mat")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, root: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": stat.st_size,
        "sha256": _sha256(path),
    }


def _session_record(session_dir: Path, root: Path) -> dict[str, object]:
    files = []
    missing_required = []
    for name in REQUIRED_SESSION_FILES:
        path = session_dir / name
        if path.is_file():
            files.append(_file_record(path, root))
        else:
            missing_required.append(name)
    optional = []
    for name in OPTIONAL_SESSION_FILES:
        path = session_dir / name
        if path.is_file():
            optional.append(_file_record(path, root))
    return {
        "session": session_dir.relative_to(root).as_posix(),
        "missing_required_files": missing_required,
        "required_files": files,
        "optional_files": optional,
    }


def main() -> None:
    root = Path(os.environ["DATASET_ROOT"]).resolve()
    if not root.is_dir():
        raise SystemExit(f"DATASET_ROOT is not a directory: {root}")
    sessions = [
        _session_record(session, root)
        for session in sorted(root.glob("Rat*/Open*"))
        if session.is_dir()
    ]
    manifest = {
        "dataset_root_name": root.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "session_count": len(sessions),
        "valid_session_count": sum(
            1 for session in sessions if not session["missing_required_files"]
        ),
        "sessions": sessions,
    }
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    manifest["manifest_sha256_without_this_field"] = digest
    (root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "Dataset manifest written to "
        f"{root / 'dataset_manifest.json'} "
        f"({manifest['valid_session_count']} valid sessions)."
    )


if __name__ == "__main__":
    main()
