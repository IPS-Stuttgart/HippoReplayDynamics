"""Verify the frozen joint-order calibration tables and selected integrations."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from calibrate_hc11_joint_order_null import relabelling_check, replicate_joint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["git_dirty"] is False
    assert manifest["known_trajectory"] is False and manifest["real_data_scored"] is False
    assert manifest["real_data_analysis_authorized"] is False
    for name, digest in manifest["output_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    rows = pd.read_csv(root / "joint_order_phase_scores.csv")
    assert len(rows) == 1280
    assert not rows.duplicated(["replicate_seed", "scenario", "emission_mode", "phase"]).any()
    assert np.isfinite(rows.select_dtypes(include="number").to_numpy()).all()
    assert rows.n_events.eq(40).all()
    summary = pd.read_csv(root / "joint_order_recovery_summary.csv")
    for result in summary.itertuples(index=False):
        paired = rows[rows.emission_mode.eq(result.emission_mode)]
        paired = paired.groupby(["scenario", "replicate_seed"]).mean(numeric_only=True).reset_index()
        calibration = paired[paired.scenario.eq("unchanged") & paired.replicate_seed.mod(2).eq(0)]
        q = np.percentile(calibration.mean_event_gain_per_spike, 95)
        test = paired[paired.scenario.eq(result.scenario) & paired.replicate_seed.mod(2).eq(1)]
        assert len(test) == 32
        values = test.mean_event_gain_per_spike
        assert np.isclose(q, result.null_calibration_p95, atol=1e-14, rtol=0)
        assert np.isclose(values.median(), result.median_gain_per_spike, atol=1e-14, rtol=0)
        assert np.isclose((values > q).mean(), result.above_null_p95_fraction, atol=1e-14, rtol=0)
    gates = pd.read_csv(root / "joint_order_gate_summary.csv").set_index("gate").passed
    assert bool(gates.all_fits_converged) == bool(rows.null_converged.all() and rows.alternative_converged.all())
    assert bool(gates.shared_order_constraint) == bool((rows.null_routing_invariance_error < 1e-10).all())
    for mode in ("oracle_phase_map", "pooled_map"):
        s = summary[summary.emission_mode.eq(mode)]
        assert bool(gates[mode + "_negative_control_screen"]) == bool((s[s.interpretation.eq("negative_control")].above_null_p95_fraction <= .15).all())
        assert bool(gates[mode + "_positive_control_screen"]) == bool((s[s.interpretation.eq("positive_control")].above_null_p95_fraction >= .8).all())
    assert manifest["known_map_screen_passed"] == bool(gates.all())
    identity = json.loads((root / "state_identity_counterexample.json").read_text())
    assert identity == relabelling_check()
    checked = 0
    for scenario in rows.scenario.unique():
        repeat = pd.DataFrame(replicate_joint((20260924, scenario, 80, 40)))
        original = rows[rows.scenario.eq(scenario) & rows.replicate_seed.eq(20260924)]
        joined = original.merge(repeat, on=["replicate_seed", "scenario", "emission_mode", "phase"], validate="one_to_one", suffixes=("_saved", "_repeat"))
        assert len(joined) == 4
        np.testing.assert_allclose(joined.mean_event_gain_per_spike_saved, joined.mean_event_gain_per_spike_repeat, atol=1e-12, rtol=0)
        checked += len(joined)
    result = {"status": "passed", "rows_checked": len(rows), "summary_rows_recomputed": len(summary),
              "phase_rows_deterministically_reproduced": checked, "output_hashes_checked": len(manifest["output_sha256"]),
              "relabelling_counterexample_reproduced": True, "producer_commit": manifest["code_commit"],
              "scientific_validation_passed": False, "assay_screen_passed": manifest["known_map_screen_passed"]}
    (root / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
