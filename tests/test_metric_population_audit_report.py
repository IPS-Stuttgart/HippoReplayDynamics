import json
from itertools import product

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.metric_population_recovery import CONDITIONS, SCENARIOS, population_fit, summarize
from scripts._provenance import file_sha256
from scripts.report_2d_metric_population_recovery import report
from scripts.verify_2d_metric_finite_recovery import assert_table
from scripts.verify_2d_metric_population_recovery import certify, reference_summary, verify_fit


def fake_populations():
    return pd.DataFrame(
        [
            {"dataset": d, "condition": c, "repeat": r, "scenario": q, "events_per_recording": n, "phi_hat": q, "phi_low": q - 0.03, "phi_high": q + 0.03}
            for d, c, r, q, n in product(("pfeiffer_foster", "tanni2022"), CONDITIONS, range(50), SCENARIOS, (32, 128))
        ]
    )


def test_independent_kkt_certificate_and_profile_boundaries():
    labels = np.repeat(np.arange(4), [70, 30, 30, 30])
    ll = np.where(labels[:, None] == np.arange(4)[None, :], 0.0, -800.0)
    row = pd.Series(population_fit(ll))
    verify_fit(ll, row)
    with pytest.raises(AssertionError):
        certify(ll, [0.25, 0.25, 0.25, 0.25])
    bad = row.copy()
    bad["phi_low"] += 0.05
    with pytest.raises(AssertionError):
        verify_fit(ll, bad)


def test_independent_summary_matches_all_factors_and_rejects_missing():
    fits, summary, gates = summarize(fake_populations())
    expected_rows, expected_summary, expected_gates = reference_summary(fits)
    assert_table(fits, expected_rows, ["dataset", "condition", "repeat", "scenario", "events_per_recording"])
    assert_table(summary, expected_summary, ["dataset", "condition", "scenario", "events_per_recording"])
    assert_table(gates, expected_gates, ["dataset", "condition", "events_per_recording"])
    with pytest.raises(AssertionError):
        reference_summary(fits.iloc[:-1])


def test_report_requires_verified_run_and_keeps_scientific_boundary(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    fits, summary, gates = summarize(fake_populations())
    decision = pd.DataFrame(
        [
            {
                "exact_population_recovery_pass": True,
                "robust_population_recovery_pass": True,
                "real_events_rescored": False,
                "biological_mechanism_established": False,
                "new_real_scoring_authorized": False,
            }
        ]
    )
    for name, frame in (("fits", fits), ("summary", summary), ("gates", gates), ("decision", decision)):
        frame.to_csv(run / f"metric_population_{name}.csv", index=False)
    manifest = {"status": "complete", "code_commit": "synthetic_test_only", "output_sha256": {f.name: file_sha256(f) for f in run.iterdir()}}
    mp = run / "metric_population_manifest.json"
    mp.write_text(json.dumps(manifest))
    audit = tmp_path / "audit.json"
    data = {
        "status": "pass",
        "input_file_sha256": {"run_manifest": file_sha256(mp)},
        "likelihoods_checked": 0,
        "fresh_paths_regenerated": 0,
        "count_arrays_regenerated": 0,
        "mixture_fits_checked": 1800,
    }
    audit.write_text(json.dumps(data))
    out = tmp_path / "report"
    report(run, audit, out)
    text = (out / "metric_population_report.md").read_text()
    assert "No real replay population" in text and "not genuine held-out prediction" in text
    assert "common generating composition" in text and "misspecification" in text
    assert (out / "metric_population_recovery.png").stat().st_size > 10000
    assert (out / "metric_population_coverage.png").stat().st_size > 10000
    data["status"] = "failed"
    audit.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        report(run, audit, tmp_path / "bad")
