import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import logit

from scripts.report_tirole_composition_calibration import analyze
from scripts.verify_tirole_composition_calibration import run, sha


def fixture(tmp_path):
    rows = []
    for eid in range(2):
        for split in range(5):
            for truth in [1, 2]:
                q = 0.1 + 0.1 * eid if truth == 1 else 0.8 + 0.1 * eid
                sign = 1 if truth == 1 else -1
                for generator in ["ordered", "whole_bin_shuffled"]:
                    for fraction in [1.0, 0.5]:
                        rows.append(
                            {
                                "session": "fixture",
                                "animal": "rat",
                                "cohort_stratum": "strict_RUN_pass",
                                "epoch": "POST",
                                "anchor_ripple_positive": True,
                                "anchor_event": eid,
                                "split": split,
                                "truth_track": truth,
                                "generator": generator,
                                "fraction": fraction,
                                "sequence_accepted": (fraction == 1 or truth == 2),
                                "evaluation_true_signed_log_odds": -sign * logit(q),
                                "evaluation_track2_probability": q,
                                "conditional_true_signed_log_odds": -sign * logit(q),
                            }
                        )
    x = pd.DataFrame(rows)
    source = tmp_path / "source"
    source.mkdir()
    x.to_csv(source / "known_track_calibration.csv", index=False)
    sm = {"output_sha256": {"known_track_calibration.csv": sha(source / "known_track_calibration.csv")}}
    (source / "manifest.json").write_text(json.dumps(sm))
    report = tmp_path / "report"
    report.mkdir()
    for name, frame in analyze(x).items():
        frame.to_csv(report / (name + ".csv"), index=False)
    m = {
        "git_dirty": False,
        "non_rescoring": True,
        "changes_real_data_selection": False,
        "source_manifests": {str(source / "manifest.json"): sha(source / "manifest.json")},
        "output_sha256": {p.name: sha(p) for p in report.iterdir()},
    }
    (report / "manifest.json").write_text(json.dumps(m))
    return report


def test_independent_arithmetic_and_primary_bootstrap(tmp_path):
    report = fixture(tmp_path)
    out = tmp_path / "verification.json"
    run(report, out)
    m = json.loads(out.read_text())
    assert m["status"] == "pass"
    assert m["primary_bootstrap_intervals_reconstructed"] == 10
    assert m["max_absolute_error"] < 1e-11


def test_valid_checksum_does_not_hide_incorrect_statistic(tmp_path):
    report = fixture(tmp_path)
    p = report / "known_selection_interventions.csv"
    x = pd.read_csv(p)
    x.loc[0, "observed_shift"] = 0.3
    x.to_csv(p, index=False)
    m = json.loads((report / "manifest.json").read_text())
    m["output_sha256"][p.name] = sha(p)
    (report / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="reconstruction mismatch"):
        run(report, tmp_path / "verification.json")


def test_perfect_zero_intervention_is_analytical_not_new_draws(tmp_path):
    report = fixture(tmp_path)
    x = pd.read_csv(report / "known_selection_interventions.csv")
    null = x[x.true_shift == 0]
    np.testing.assert_allclose(null.observed_shift, 0, atol=1e-15)
