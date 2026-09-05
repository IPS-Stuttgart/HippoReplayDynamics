#!/usr/bin/env python3
"""Non-rescoring paired contrasts and denominator audit for known-path recovery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts._provenance import build_script_provenance, file_sha256


def paired_effect(table, condition, first, second, value, rng, bootstraps=5000):
    """First-minus-second within session, then equal sessions within animal."""
    pairs = table.pivot(index=["animal", "session"], columns=condition, values=value)
    if first not in pairs or second not in pairs:
        return {"paired_sessions": 0, "animals": 0, "effect": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
    difference = (pairs[first] - pairs[second]).dropna()
    animals = difference.groupby("animal").mean().to_numpy()
    lo, hi = np.nan, np.nan
    if len(animals) > 1:
        lo, hi = np.quantile(rng.choice(animals, (bootstraps, len(animals)), replace=True).mean(axis=1), [.025, .975])
    return {"paired_sessions": len(difference), "animals": len(animals),
            "effect": float(animals.mean()) if len(animals) else np.nan, "ci95_low": lo, "ci95_high": hi}


def make_tables(sessions, gradients, seed):
    rng = np.random.default_rng(seed)
    s, g = sessions, gradients
    q = s[s.truth_kind.eq("continuous") & s.gradient.eq(0) & s.cell_fraction.isin([1., .5])]
    rows = []
    groups = ["dataset", "regime", "likelihood", "bin_filter", "estimator"]
    for key, local in q.groupby(groups, sort=True):
        rows.append({**dict(zip(groups, key, strict=True)), "contrast": "half_minus_full_truth_eligible_continuity",
                     **paired_effect(local, "cell_fraction", .5, 1., "continuous_eligible_recovery_fraction", rng)})
    continuity = pd.DataFrame(rows)
    q = g[g.cell_fraction.eq(1.) & g.gradient.ne(0)]
    groups = ["dataset", "regime", "likelihood", "bin_filter", "estimator", "selection", "coordinate", "readout"]
    rows = []
    for key, local in q.groupby(groups, sort=True):
        rows.append({**dict(zip(groups, key, strict=True)), "contrast": "positive_minus_negative_injected_gradient",
                     "injected_gradient_difference": 1.,
                     **paired_effect(local, "gradient", .5, -.5, "normalized_slope", rng)})
    response = pd.DataFrame(rows)
    availability = g.groupby(groups + ["cell_fraction", "gradient"], as_index=False).agg(
        sessions=("session", "size"), available_sessions=("status", lambda v: v.eq("available").sum()),
        contributing_events=("events", "sum"))
    return continuity, response, availability


def run(args):
    root = args.result_dir.resolve()
    manifest_path = root / "coverage_recovery_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    filenames = ["coverage_recovery_session_summary.csv", "coverage_recovery_session_gradient_recovery.csv", "coverage_recovery_gate_summary.csv"]
    for filename in filenames:
        if file_sha256(root / filename) != manifest["output_sha256"][filename]:
            raise ValueError("source output hash mismatch")
    gates = pd.read_csv(root / filenames[-1])
    if gates.loc[gates.gate.eq("overall_technical"), "passed"].tolist() != [True]:
        raise ValueError("technical recovery gates must pass")
    tables = make_tables(pd.read_csv(root / filenames[0]), pd.read_csv(root / filenames[1]), args.seed)
    for name, table in zip(["paired_continuity", "paired_gradient_response", "gradient_availability"], tables, strict=True):
        table.to_csv(root / f"coverage_recovery_{name}.csv", index=False)
    provenance = build_script_provenance(input_paths={"manifest": manifest_path, "reporter": Path(__file__),
                                                     **{p: root / p for p in filenames}}, cwd=ROOT)
    provenance.update({"seed": args.seed, "bootstraps": 5000,
                       "claim_boundary": "paired development simulation contrasts; no biological equivalence and no independent evaluation",
                       "uncertainty_scope": "animal_bootstrap_after_session_pairing_conditional_on_fixed_simulation_draws"})
    (root / "coverage_recovery_paired_report_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260908)
    run(parser.parse_args())
