import json

import pandas as pd
import pytest

from scripts._provenance import file_sha256
from scripts.report_replay_clock_timing_recovery import report


def test_nonrescoring_report_and_tamper_guard(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    values = {"coarse_identity": 0.6, "fine_identity": 0.65, "fine_timing": 0.55, "fine_joint": 0.7}
    rows = []
    pairs = []
    for dataset in ["pfeiffer_foster", "tanni2022"]:
        for animal in ["a", "b"]:
            for condition in ["exact", "gain_drift"]:
                for mode, value in values.items():
                    rows.append({"dataset": dataset, "animal": animal, "condition": condition, "observation": mode, "oracle_correct": value})
                pairs.append({"dataset": dataset, "animal": animal, "condition": condition, **values, "identity_timing_gain": 0.05, "joint_timing_gain": 0.1})
    animals = pd.DataFrame(rows)
    summary = animals.groupby(["dataset", "condition", "observation"], as_index=False).oracle_correct.mean()
    for name, df in [("summary", summary), ("animals", animals), ("paired_animals", pd.DataFrame(pairs))]:
        df.to_csv(run / f"clock_timing_{name}.csv", index=False)
    mp = run / "clock_timing_manifest.json"
    mp.write_text(json.dumps({"status": "complete", "code_commit": "synthetic_fixture", "completed": [{}] * 4, "output_sha256": {p.name: file_sha256(p) for p in run.iterdir()}}))
    ap = tmp_path / "audit.json"
    ap.write_text(
        json.dumps(
            {
                "status": "pass",
                "input_file_sha256": {"run_manifest": file_sha256(mp)},
                "scores_checked": 32,
                "numerical_readiness_pass": True,
                "max_probability_error": 1e-7,
                "max_parent_marginal_error": 1e-6,
            }
        )
    )
    out = tmp_path / "report"
    report(run, ap, out)
    assert (out / "clock_timing_recovery.png").stat().st_size > 10000
    assert "not a real-replay classification" in (out / "clock_timing_report.md").read_text()
    assert not json.loads((out / "clock_timing_report_manifest.json").read_text())["biological_mechanism_established"]
    (run / "clock_timing_summary.csv").write_text("tampered")
    with pytest.raises(ValueError, match="changed"):
        report(run, ap, tmp_path / "invalid")
