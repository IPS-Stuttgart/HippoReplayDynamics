#!/usr/bin/env python3
"""Independently trace every overview value back to its source-table cell."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import build_script_provenance, file_sha256


def verify_values(values, sources):
    checked = 0
    for row in values.to_dict("records"):
        frame = sources[row["source_table"]]
        filters = json.loads(row["source_filter"])
        selected = frame.set_index(list(filters)).loc[[tuple(filters.values())]]
        if len(selected) != 1:
            raise ValueError("source extraction is not unique")
        source = selected.iloc[0]
        expected = float(source[row["source_field"]]) * row["scale"]
        if not np.isclose(expected, row["value"], atol=1e-12, rtol=0):
            raise ValueError("overview point estimate differs from source")
        ci = json.loads(row["source_ci_fields"])
        if ci is not None:
            expected = [float(source[c]) * row["scale"] for c in ci]
            if not np.allclose(expected, [float(row["ci_low"]), float(row["ci_high"])], atol=1e-12, rtol=0):
                raise ValueError("overview interval differs from source")
        checked += 1
    return checked


def run(args):
    root = args.report_dir.resolve()
    path = root / "replay_measurement_paper_manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "complete" or manifest.get("non_rescoring") is not True or manifest.get("git_dirty"):
        raise ValueError("complete clean non-rescoring report required")
    for name, expected in manifest["output_sha256"].items():
        if file_sha256(root / name) != expected:
            raise ValueError("changed overview output")
    for key, expected in manifest["input_file_sha256"].items():
        if file_sha256(manifest["input_file_paths"][key]) != expected:
            raise ValueError("changed overview input")
    sources = {key.removeprefix("table_"): pd.read_csv(p) for key, p in manifest["input_file_paths"].items() if key.startswith("table_")}
    # Preserve the JSON literal null in source_ci_fields rather than parsing it as NaN.
    values = pd.read_csv(root / "replay_measurement_paper_figure_values.csv", keep_default_na=False)
    if values.duplicated(["panel", "dataset", "condition"]).any() or values.groupby("panel").size().to_dict() != {"A": 6, "B": 10, "C": 4, "D": 8}:
        raise ValueError("overview panel coverage mismatch")
    n = verify_values(values, sources)
    pixels = np.asarray(Image.open(root / "replay_measurement_paper_overview.png").convert("RGB"))
    nonwhite = float(np.mean(np.any(pixels < 245, axis=2)))
    if nonwhite <= 0.015:
        raise ValueError("overview figure is blank")
    result = build_script_provenance(input_paths={"report_manifest": path, "auditor": Path(__file__)}, cwd=ROOT)
    result.update(
        status="pass",
        source_rows_checked=n,
        figure_shape=list(pixels.shape),
        figure_nonwhite_fraction=nonwhite,
        verification_scope="all used input/output hashes; independent source-cell extraction of every overview value/CI; panel coverage and nonblank figure; no underlying scientific rescore",
    )
    with args.output_json.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "source_rows_checked": n, "figure_nonwhite_fraction": nonwhite}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    run(parser.parse_args())
