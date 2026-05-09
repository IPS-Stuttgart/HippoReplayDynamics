from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path


def main() -> None:
    temp = Path(os.environ["RUNNER_TEMP"])
    zip_path = temp / "rclone-download" / "rclone.zip"
    extract_dir = temp / "rclone-download" / "extract"
    bin_dir = temp / "rclone-bin"

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)

    matches = list(extract_dir.glob("rclone-*/rclone"))
    if not matches:
        raise SystemExit("Downloaded rclone archive did not contain rclone binary")

    target = bin_dir / "rclone"
    shutil.copy2(matches[0], target)
    target.chmod(0o755)


if __name__ == "__main__":
    main()
