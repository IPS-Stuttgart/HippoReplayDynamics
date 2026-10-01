#!/usr/bin/env python3
"""Pinned odor-place feasibility audit. No replay-behavior association analysis."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._provenance import build_script_provenance
from scripts.odor_place_source_io import atomic_json, digest, download_verified, get_json

PREFIX = "odor_place_post_error_"


def acquire(args, protocol):
    metadata_dir = args.dataset_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    source_url = f"https://api.figshare.com/v2/articles/{protocol['figshare_article']}/versions/{protocol['figshare_version']}"
    source = get_json(source_url)
    if source["version"] != protocol["figshare_version"]:
        raise ValueError("Figshare did not return the pinned version")
    atomic_json(metadata_dir / "figshare_version.json", source)
    files = [entry for entry in source["files"] if entry["name"] == args.source_archive.name]
    if len(files) != 1:
        raise ValueError("--source-archive must name an actual file in the pinned Figshare version; newer releases are not substituted")
    entry = files[0]
    if "source_published_md5" in protocol and (
        entry["name"] != protocol["source_filename"]
        or entry["computed_md5"] != protocol["source_published_md5"]
        or entry["size"] != protocol["source_size_bytes"]
    ):
        raise ValueError("Published source metadata differs from the frozen source amendment")
    verified = download_verified(entry["download_url"], args.source_archive, entry["size"], entry["computed_md5"], "md5", protocol["download_reserve_bytes"])
    atomic_json(metadata_dir / "source_verified.json", verified)
    base = f"https://api.dandiarchive.org/api/dandisets/{protocol['dandi_id']}/versions/{protocol['dandi_version']}/"
    version = get_json(base)
    atomic_json(metadata_dir / "dandi_version.json", version)
    listing = get_json(base + "assets/?page_size=100")
    if listing.get("next"):
        raise ValueError("Asset inventory exceeds one page; no silent truncation allowed")
    atomic_json(metadata_dir / "dandi_assets.json", listing)
    for index, asset in enumerate(listing["results"]):
        target = metadata_dir / (asset["asset_id"] + ".json")
        if not target.exists():
            atomic_json(target, get_json(f"https://api.dandiarchive.org/api/assets/{asset['asset_id']}/"))
        print(f"asset_metadata {index + 1}/{len(listing['results'])}: {asset['path']}", flush=True)
    return {"source": verified, "n_assets": len(listing["results"]), "full_nwb_files_downloaded": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["acquire", "inventory", "run-qc", "verify", "report"])
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if args.seed != protocol["seed"] or protocol["scope"] != "feasibility_only_no_replay_behavior_association":
        parser.error("Seed and scope must match the frozen protocol")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = build_script_provenance(input_paths={"protocol": args.protocol})
    if args.stage == "acquire":
        result = acquire(args, protocol)
    else:
        from scripts.odor_place_feasibility_core import dispatch
        result = dispatch(args, protocol)
    environment = {}
    for package in ["numpy", "scipy", "h5py", "matplotlib", "pytest"]:
        try:
            environment[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            environment[package] = "not_installed"
    atomic_json(args.output_dir / (PREFIX + args.stage.replace("-", "_") + "_manifest.json"), {
        **provenance, "created_at_utc": datetime.now(UTC).isoformat(), "stage": args.stage,
        "protocol_sha256": digest(args.protocol), "python": platform.python_version(), "environment_versions": environment,
        "dataset_root": str(args.dataset_root.resolve()), "result": result,
        "association_fitted": False, "manuscript_claims_changed": False,
    })
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
