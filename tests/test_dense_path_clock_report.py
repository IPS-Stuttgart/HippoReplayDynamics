import json
from itertools import product

import pandas as pd
import pytest

from hipporeplayimm.dense_path_clocks import summarize
from scripts._provenance import file_sha256
from scripts.report_dense_path_clock_recovery import NAMES, classify, report


def fixture_run(tmp_path):
    fits = pd.DataFrame(
        [
            {
                "dataset": d,
                "source_teacher": t,
                "scenario": q,
                "repeat": r,
                "candidate_bank": b,
                "support": h,
                "phi_hat": q,
                "phi_low": q - 0.02,
                "phi_high": q + 0.02,
                "coherent_weight": 0.6,
            }
            for d, t, q, r, b, h in product(("pfeiffer_foster", "tanni2022"), ("original_bank_0", "original_bank_1"), (0.25, 0.5, 0.75), range(50), (0, 1), (1024, 4096, 8192))
        ]
    )
    run, audit_dir = tmp_path / "run", tmp_path / "audit"
    run.mkdir()
    audit_dir.mkdir()
    for name, frame in zip(NAMES, summarize(fits), strict=True):
        frame.to_csv(run / f"dense_path_clock_{name}.csv", index=False)
    manifest = {
        "status": "complete",
        "code_commit": "synthetic",
        "oracle_knows_path": False,
        "real_events_rescored": False,
        "all_scorer_libraries_independent": True,
        "supports": [1024, 4096, 8192],
        "candidate_banks": [0, 1],
        "observations": 1267200,
        "score_rows": 7603200,
        "n_fits": 3600,
        "output_sha256": {p.name: file_sha256(p) for p in run.iterdir()},
    }
    mp = run / "dense_path_clock_manifest.json"
    mp.write_text(json.dumps(manifest))
    audit = {
        "status": "pass",
        "population_fits_certified": 3600,
        "paths_regenerated": 540672,
        "observations_audited": 19800,
        "likelihoods_recomputed": 594000,
        "input_file_sha256": {"run_manifest": file_sha256(mp)},
    }
    (audit_dir / "dense_path_clock_audit.json").write_text(json.dumps(audit))
    return run, audit_dir


def test_nonrescoring_report_and_claim_boundary(tmp_path):
    run, audit = fixture_run(tmp_path)
    output = tmp_path / "report"
    assert report(run, audit, output) == "simulation_recovery_pass_restricted_prior_only"
    assert (output / "dense_path_clock_recovery.png").stat().st_size > 10000
    content = (output / "dense_path_clock_report.md").read_text()
    assert "NOT a biological speed" in content
    assert "not every production likelihood" in content
    assert "not a matched scorer condition" in content
    provenance = json.loads((output / "dense_path_clock_report_manifest.json").read_text())
    assert not provenance["real_events_rescored"]
    for name, digest in provenance["output_sha256"].items():
        assert file_sha256(output / name) == digest


def test_recovery_and_integration_are_separate_nonvacuous_gates(tmp_path):
    run, _ = fixture_run(tmp_path)
    gates = pd.read_csv(run / "dense_path_clock_gates.csv")
    convergence = pd.read_csv(run / "dense_path_clock_convergence.csv")
    convergence.loc[0, "integration_stable"] = False
    assert classify(gates, convergence) == ("path_integration_not_stable", True, False)
    gates.loc[gates.support.eq(8192), "practical_pass"] = False
    assert classify(gates, convergence) == ("dense_path_recovery_failed", False, False)
    with pytest.raises(ValueError):
        classify(gates.iloc[:0], convergence)


@pytest.mark.parametrize("corruption", ["table", "audit", "incomplete_audit"])
def test_missing_or_tampered_inputs_rejected(tmp_path, corruption):
    run, audit = fixture_run(tmp_path)
    if corruption == "table":
        path = run / "dense_path_clock_gates.csv"
        path.write_text(path.read_text() + "\n")
    else:
        path = audit / "dense_path_clock_audit.json"
        data = json.loads(path.read_text())
        if corruption == "audit":
            data["input_file_sha256"]["run_manifest"] = "wrong"
        else:
            data["population_fits_certified"] -= 1
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        report(run, audit, tmp_path / "report")
    assert not (tmp_path / "report").exists()
