from __future__ import annotations

import os
import shutil
from pathlib import Path


REQUIRED_SESSION_FILES = (
    "Position_Data.mat",
    "Ripple_Events.mat",
    "Spike_Data.mat",
    "Epochs.mat",
)


def looks_like_dataset(path: Path) -> bool:
    if not path.is_dir():
        return False
    for session in path.glob("Rat*/Open*"):
        if all((session / name).is_file() for name in REQUIRED_SESSION_FILES):
            return True
    return False


def find_dataset_roots(staging_dir: Path) -> list[Path]:
    """Return valid dataset roots, including roots below archive/WebDAV wrappers."""

    candidates = [staging_dir, staging_dir / "DataSetFromPfeifferFoster"]
    candidates.extend(
        session.parent.parent
        for session in staging_dir.rglob("Rat*/Open*")
        if session.is_dir()
    )

    matches: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if looks_like_dataset(candidate):
            matches.append(candidate)
    return matches


def main() -> None:
    dataset_root = Path(os.environ["DATASET_ROOT"])
    staging_dir = Path(os.environ["RUNNER_TEMP"]) / "pfeiffer-foster-webdav"

    matches = find_dataset_roots(staging_dir)
    if not matches:
        visible = sorted(
            str(path.relative_to(staging_dir))
            for path in staging_dir.rglob("*")
            if path.is_dir()
        )[:80]
        raise SystemExit(
            "Could not find a Pfeiffer-Foster dataset root after WebDAV download. "
            f"Visible directories: {visible}"
        )
    if len(matches) > 1:
        relative_matches = sorted(str(path.relative_to(staging_dir)) for path in matches)
        raise SystemExit(
            "Found multiple Pfeiffer-Foster dataset roots after WebDAV download; "
            "refusing to choose one based on filesystem traversal order. "
            f"Matches: {relative_matches}"
        )

    source = matches[0]
    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    shutil.move(str(source), str(dataset_root))
    print(f"Dataset root prepared at: {dataset_root}")


if __name__ == "__main__":
    main()
