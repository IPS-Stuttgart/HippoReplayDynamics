"""Replay saved event seeds without modifying the shared bank."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd

from hipporeplayimm.regional_blind_bank import Geometry, sample_conditional_event
from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_edge_support_content import load_npz
from scripts.build_regional_blind_bank import read_csv, write_json
from scripts.measure_edge_support_content import occupied_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    bank, out = Path(args.bank), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for session, strata in read_csv(bank / "strata.csv").groupby("session"):
        parent = bank / session.replace("/", "_")
        config = json.loads((parent / "config.json").read_text())
        pool = config["pool"]
        f, t = load_npz(parent / "source_inputs.npz"), load_npz(parent / "templates.npz")
        geometry = Geometry.from_grid(f["grid"], occupied_graph(f["grid"]))
        for row in strata.itertuples():
            p = load_npz(Path(row.folder) / "panel_000.npz")
            a, b = t["offsets"][:2]
            template = {"start": t["starts"][0], "end": t["endpoints"][0], "times": t["times"][a:b]}
            result = sample_conditional_event(geometry, row.generator, pool["moving_speed"], pool["lengths"],
                pool["intervals"], row.scale, np.asarray(config["center"]), row.delta_ms/1000.,
                p["desired_label"][0], template, f["rates"], np.random.default_rng(int(p["event_seeds"][0])),
                max_attempts=config["max_attempts"])
            c, d = p["path_offsets"][:2]
            np.testing.assert_array_equal(result["identities"], p["identities"][a:b])
            np.testing.assert_array_equal(result["path"].ages, p["path_ages"][c:d])
            np.testing.assert_array_equal(result["path"].nodes, p["path_nodes"][c:d])
            for key in result["contents"][0]:
                np.testing.assert_array_equal([r[key] for r in result["contents"]], p[key][0])
            assert result["attempts"] == p["attempts"][0]
            rows.append({"session": session, "generator": row.generator, "scale": row.scale,
                         "delta_ms": row.delta_ms, "event_id": int(t["event_ids"][0]),
                         "seed": str(int(p["event_seeds"][0])), "status": "exact_match"})
    pd.DataFrame(rows).to_csv(out / "seed_reproduction.csv", index=False)
    manifest = build_script_provenance(input_paths={"bank": bank / "manifest.json", "verifier": __file__})
    manifest.update(status="pass", events_checked=len(rows), outputs={p.name: file_sha256(p) for p in out.iterdir()})
    write_json(out / "manifest.json", manifest)
    print("exact seed reproduction", len(rows), flush=True)


if __name__ == "__main__":
    main()
