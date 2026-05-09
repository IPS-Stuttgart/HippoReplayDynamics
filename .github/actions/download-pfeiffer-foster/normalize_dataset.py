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


def main() -> None:
    dataset_root = Path(os.environ["DATASET_ROOT"])
    staging_dir = Path(os.environ["RUNNER_TEMP"]) / "pfeiffer-foster-webdav"

    candidates = [staging_dir, staging_dir / "DataSetFromPfeifferFoster"]
    candidates.extend(child for child in staging_dir.iterdir() if child.is_dir())
    matches = [candidate for candidate in candidates if looks_like_dataset(candidate)]
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

    source = matches[0]
    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    shutil.move(str(source), str(dataset_root))
    print(f"Dataset root prepared at: {dataset_root}")


if __name__ == "__main__":
    main()
