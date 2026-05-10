from __future__ import annotations

import argparse
from pathlib import Path

REQUIRED_SESSION_FILES = (
    "Position_Data.mat",
    "Ripple_Events.mat",
    "Spike_Data.mat",
    "Epochs.mat",
)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a Pfeiffer/Foster dataset tree.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--session", default="", help="Optional session such as Rat1/Open1")
    parser.add_argument("--write-manifest", action="store_true")
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

    files = sorted(path for path in dataset_root.rglob("*") if path.is_file())
    total_bytes = sum(path.stat().st_size for path in files)
    if args.write_manifest:
        manifest = dataset_root / "MANIFEST.txt"
        manifest.write_text(
            "Pfeiffer/Foster dataset cache manifest\n"
            f"Root: {dataset_root}\n"
            f"Valid sessions: {len(valid_sessions(dataset_root))}\n"
            f"Files: {len(files)}\n"
            f"Bytes: {total_bytes}\n\n"
            + "\n".join(
                f"{path.stat().st_size} {path.relative_to(dataset_root)}" for path in files
            )
            + "\n",
            encoding="utf-8",
        )

    print(f"Validated Pfeiffer/Foster dataset at {dataset_root}")
    print(f"Checked session(s): {', '.join(str(s.relative_to(dataset_root)) for s in sessions)}")
    print(f"Files: {len(files)}")
    print(f"Bytes: {total_bytes}")


if __name__ == "__main__":
    main()
