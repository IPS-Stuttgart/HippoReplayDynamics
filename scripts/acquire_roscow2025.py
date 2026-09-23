#!/usr/bin/env python3
"""Acquire a pinned public Roscow release subset, without analysis-result objects."""

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
import urllib.request
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

COMMIT = "d8564850791358b7909c787990b6a22b0d1e01b9"
REPO = "EmmaRoscow/QlearningReplay"


def stamp():
    return datetime.now(UTC).isoformat()


def write_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def wanted(path):
    return (
        path in {"README.md", "LICENSE"}
        or path.startswith(("code/", "data/behavioural_data/"))
        or (path.startswith("data/ephys_data/") and not path.endswith("/binnedfr.mat"))
    )


def destination(root, item):
    rel = PurePosixPath(item["path"])
    if rel.is_absolute() or ".." in rel.parts or "\\" in str(rel):
        raise ValueError("Unsafe repository path")
    path = (root / "raw" / str(rel)).resolve()
    if not path.is_relative_to((root / "raw").resolve()):
        raise ValueError("Asset escapes raw directory")
    if not re.fullmatch("[0-9a-f]{40}", item["sha"]) or item["size"] < 0:
        raise ValueError("Missing Git blob digest or size")
    return path


def verify(path, item):
    size = path.stat().st_size
    if size != item["size"]:
        raise ValueError(f"Size mismatch: {path}")
    git_hash = hashlib.sha1(f"blob {size}\0".encode())
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            git_hash.update(chunk)
            sha256.update(chunk)
    if git_hash.hexdigest() != item["sha"]:
        raise ValueError(f"Published Git blob mismatch: {path}; file retained")
    return sha256.hexdigest()


def run(root):
    manifest_path = root / "asset_manifest.json"
    if not manifest_path.exists():
        url = f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1"
        with urllib.request.urlopen(url, timeout=60) as response:
            tree = json.load(response)
        if tree.get("truncated") or tree.get("sha") != COMMIT:
            raise ValueError("Incomplete or unpinned repository tree")
        blobs = [x for x in tree["tree"] if x["type"] == "blob"]
        write_json(manifest_path, {
            "repository": REPO, "source_commit": COMMIT, "tree_url": url,
            "created_at_utc": stamp(), "assets": [x for x in blobs if wanted(x["path"])],
            "omitted": [x["path"] for x in blobs if not wanted(x["path"])],
            "scope": "spikes, epochs, ripple times, behavioral timestamps, speed, behavior, source",
        })
    manifest = json.loads(manifest_path.read_text())
    if (manifest["repository"], manifest["source_commit"]) != (REPO, COMMIT):
        raise ValueError("Unexpected source identity")
    assets = manifest["assets"]
    if not assets or len({x["path"] for x in assets}) != len(assets):
        raise ValueError("Empty or duplicate assets")
    state = {
        "status": "running", "source_commit": COMMIT, "host": socket.gethostname(),
        "pid": os.getpid(), "started_at_utc": stamp(), "verified": [],
        "assets_expected": len(assets), "assets_verified": 0,
        "bytes_expected": sum(x["size"] for x in assets),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    status_path = root / "download_status.json"
    write_json(status_path, state)
    try:
        for item in assets:
            path = destination(root, item)
            partial = path.with_name(path.name + ".part")
            state.update(current_asset=item["path"], updated_at_utc=stamp())
            write_json(status_path, state)
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                existing = partial.stat().st_size if partial.exists() else 0
                if existing > item["size"]:
                    raise ValueError("Partial download exceeds expected size")
                if shutil.disk_usage(root).free < item["size"] - existing + 10 * 1024**3:
                    raise RuntimeError("Insufficient space with 10-GiB reserve")
                if existing < item["size"] or not partial.exists():
                    subprocess.run([
                        "curl", "--fail", "--location", "--silent", "--show-error",
                        "--continue-at", "-", "--retry", "6", "--retry-delay", "10",
                        "--connect-timeout", "30", "--speed-time", "120", "--speed-limit", "1024",
                        "--output", str(partial),
                        f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{item['path']}",
                    ], check=True)
                digest = verify(partial, item)
                os.link(partial, path)
                partial.unlink()
            else:
                digest = verify(path, item)
            state["verified"].append({"path": item["path"], "size": item["size"], "sha256": digest})
            state.update(assets_verified=len(state["verified"]), updated_at_utc=stamp())
            write_json(status_path, state)
            print(json.dumps({"verified": item["path"], "count": len(state["verified"])}), flush=True)
        state.update(status="complete_verified", completed_at_utc=stamp())
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}", updated_at_utc=stamp())
        raise
    finally:
        write_json(status_path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(root)


if __name__ == "__main__":
    main()

