#!/usr/bin/env python3
"""Resume a frozen DANDI000978 release and verify published SHA-256 digests."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath


def stamp():
    return datetime.now(UTC).isoformat()


def write_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def destination(root, item):
    rel = PurePosixPath(item["path"])
    if rel.is_absolute() or ".." in rel.parts or "\\" in str(rel):
        raise ValueError("Unsafe asset path")
    target = (root / "raw" / str(rel)).resolve()
    if not target.is_relative_to((root / "raw").resolve()):
        raise ValueError("Asset escapes raw directory")
    digest = item["digest"]["dandi:sha2-256"]
    if not re.fullmatch("[0-9a-f]{64}", digest) or item["size"] <= 0:
        raise ValueError("Published SHA-256 and positive size required")
    if not item["url"].startswith("https://dandiarchive.s3.amazonaws.com/"):
        raise ValueError("Unexpected public download host")
    return target


def verify(path, item):
    if path.stat().st_size != item["size"]:
        raise ValueError(f"Size mismatch: {path}")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != item["digest"]["dandi:sha2-256"]:
        raise ValueError(f"SHA-256 mismatch: {path}; retaining file for inspection")
    return actual


def run(root):
    manifest_path = root / "metadata" / "asset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest["dataset"], manifest["version"]) != ("000978", "0.240511.0307"):
        raise ValueError("Unexpected dataset/version")
    assets = manifest["assets"]
    if len(assets) != 9 or len({x["path"] for x in assets}) != len(assets):
        raise ValueError("Expected nine unique published asset paths")
    targets = [destination(root, item) for item in assets]
    state = {
        "dataset": manifest["dataset"],
        "version": manifest["version"],
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "started_at_utc": stamp(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "status": "running",
        "assets_expected": len(assets),
        "assets_verified": 0,
        "bytes_expected": sum(x["size"] for x in assets),
        "verified": [],
    }
    status_path = root / "download_status.json"
    write_json(status_path, state)
    try:
        for item, target in zip(assets, targets, strict=True):
            partial = target.with_name(target.name + ".part")
            state.update(current_asset=item["path"], updated_at_utc=stamp(), phase="download")
            write_json(status_path, state)
            print(json.dumps({"started": item["path"], "time": stamp()}), flush=True)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                existing = partial.stat().st_size if partial.exists() else 0
                if existing > item["size"]:
                    raise ValueError("Partial asset exceeds published size")
                if shutil.disk_usage(root).free < item["size"] - existing + 10 * 1024**3:
                    raise RuntimeError("Insufficient free space with 10-GiB safety reserve")
                if existing < item["size"]:
                    subprocess.run(
                        [
                            "curl",
                            "--fail",
                            "--location",
                            "--silent",
                            "--show-error",
                            "--continue-at",
                            "-",
                            "--retry",
                            "8",
                            "--retry-delay",
                            "15",
                            "--retry-connrefused",
                            "--connect-timeout",
                            "30",
                            "--speed-time",
                            "120",
                            "--speed-limit",
                            "1024",
                            "--output",
                            str(partial),
                            item["url"],
                        ],
                        check=True,
                    )
                state.update(phase="checksum", updated_at_utc=stamp())
                write_json(status_path, state)
                digest = verify(partial, item)
                # Do not overwrite a concurrently created final asset.
                os.link(partial, target)
                partial.unlink()
            else:
                digest = verify(target, item)
            state["verified"].append(
                {
                    "path": item["path"],
                    "size": item["size"],
                    "sha256": digest,
                    "verified_at_utc": stamp(),
                }
            )
            state["assets_verified"] = len(state["verified"])
            state["updated_at_utc"] = stamp()
            write_json(status_path, state)
            print(json.dumps({"verified": item["path"], "time": stamp()}), flush=True)
        state.update(status="complete_verified", phase="done", completed_at_utc=stamp())
        write_json(status_path, state)
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}", updated_at_utc=stamp())
        write_json(status_path, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    with (root / "download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        run(root)
        print(f"Completed in {time.monotonic() - started:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()
