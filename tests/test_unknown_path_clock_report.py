import json
from itertools import product

import pandas as pd
import pytest

from hipporeplayimm.unknown_path_clocks import MODELS, SCENARIOS, SUPPORTS, TEACHERS, summarize
from scripts._provenance import file_sha256
from scripts.report_unknown_path_clock_populations import report


def test_report_is_audited_nonrescorer_and_rejects_mismatched_audit(tmp_path):
    run, audit = tmp_path / "run", tmp_path / "audit"
    run.mkdir()
    audit.mkdir()
    rows = []
    for d, q, r, t, h in product(("pfeiffer_foster", "tanni2022"), SCENARIOS, range(50), TEACHERS, SUPPORTS):
        rows.append(
            {
                "dataset": d,
                "scenario": q,
                "repeat": r,
                "teacher": t,
                "support": h,
                "phi_hat": q,
                "phi_low": 0,
                "phi_high": 1,
                "coherent_weight": 0.6,
                **{"weight_" + m: 0.2 for m in MODELS},
            }
        )
    for name, frame in zip(("fits", "summary", "gates"), summarize(pd.DataFrame(rows)), strict=True):
        frame.to_csv(run / f"unknown_path_clock_{name}.csv", index=False)
    manifest = {
        "status": "complete",
        "oracle_knows_path": False,
        "real_events_rescored": False,
        "code_commit": "synthetic_fixture",
        "n_fits": 1200,
        "n_observations": 1267200,
        "n_rows": 2534400,
        "output_sha256": {p.name: file_sha256(p) for p in run.iterdir()},
    }
    mp = run / "unknown_path_clock_manifest.json"
    mp.write_text(json.dumps(manifest))
    a = {
        "status": "pass",
        "population_fits_certified": 1200,
        "observations_regenerated": 1267200,
        "likelihoods_checked": 12672000,
        "input_file_sha256": {"run_manifest": file_sha256(mp)},
    }
    ap = audit / "unknown_path_clock_audit.json"
    ap.write_text(json.dumps(a))
    output = tmp_path / "report"
    verdict = report(run, audit, output)
    assert verdict == "population_clock_recovery_not_ready_even_with_matched_path_library"
    assert (output / "unknown_path_clock_recovery.png").stat().st_size > 10000
    assert "No real replay events were rescored" in (output / "unknown_path_clock_population_report.md").read_text()
    a["input_file_sha256"]["run_manifest"] = "wrong"
    ap.write_text(json.dumps(a))
    with pytest.raises(ValueError, match="matching"):
        report(run, audit, tmp_path / "invalid")
