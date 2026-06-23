#!/usr/bin/env python3
"""Prepare a manifest for the Olafsdottir et al. 2016 Axona dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from hipporeplayimm.olafsdottir2016 import (
    EXPECTED_MD5,
    ZENODO_URL,
    prepare_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/Olafsdottir2016"))
    parser.add_argument("--zenodo-url", default=ZENODO_URL)
    parser.add_argument("--expected-md5", default=EXPECTED_MD5)
    parser.add_argument("--archive-path", type=Path, default=None)
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the Zenodo zip if it is absent. Off by default so CI never fetches 9.9 GB.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload the archive even if --archive-path already exists.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract the zip into --dataset-root before building the manifest.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path, records = prepare_dataset(
        dataset_root=args.dataset_root,
        zenodo_url=args.zenodo_url,
        expected_md5=args.expected_md5,
        archive_path=args.archive_path,
        manifest_output=args.manifest_output,
        download=args.download,
        extract=args.extract,
        force_download=args.force_download,
    )
    print(f"Wrote {manifest_path}")
    print(f"Sessions: {len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
