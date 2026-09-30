import numpy as np
import pandas as pd
import pytest

from scripts.report_regional_content_mua_selection_null import tables, technical_gates


def fixture_tables():
    fits, bounds = [], []
    for replicate in (0, 1):
        for cohort, tv in (("selected", .2), ("independent_same_window", .1)):
            key = {"animal": "Rat1", "session": "Rat1/Open1", "replicate": replicate,
                       "peak_gain": 6, "cohort": cohort, "readout": "frozen_poisson", "calibration": "generator_unselected"}
            fits.append(dict(**key, fit_tv=tv, true_prevalence=.3, prevalence=.32,
                prevalence_bias=.02, auc=.9, mean_spikes=5., silent_fraction=0., status="fit"))
            bounds.append(dict(**key, slack=0., lower=.2, upper=.4, status="feasible", contains_truth=True))
    return pd.DataFrame(fits), pd.DataFrame(bounds)


def test_paired_same_truth_and_condition_summary():
    joined, sessions, animals, summary, paired = tables(*fixture_tables())
    assert len(joined) == 4 and len(sessions) == len(animals) == len(summary) == 2
    np.testing.assert_allclose(paired.fit_tv_selection_excess, .1)
    assert summary.truth_contained_runs.eq(2).all()
    assert summary.incompatible_runs.eq(0).all()
    np.testing.assert_allclose(summary.abs_prevalence_bias, .02)


def test_paired_different_truth_rejected():
    f, b = fixture_tables()
    f.loc[f.cohort.eq("selected"), "true_prevalence"] = .4
    with pytest.raises(AssertionError):
        tables(f, b)


def test_missing_bounds_rejected():
    f, b = fixture_tables()
    with pytest.raises(ValueError, match="missing zero-slack"):
        tables(f, b.iloc[:-1])


def test_empty_run_cannot_pass():
    manifest = {"status": "complete", "parameters": {"replicates": 4}}
    sessions = pd.DataFrame(columns=["session", "status"])
    detection = pd.DataFrame(columns=["peak_gain", "selected_endpoints", "session", "replicate"])
    fits = pd.DataFrame(columns=["status", "prevalence", "fit_tv", "auc"])
    gates = technical_gates(manifest, sessions, detection, fits, {"status": "pass"}).set_index("gate")
    assert gates.loc["overall", "status"] == "fail"
    assert gates.loc["nonempty_primary_selected", "status"] == "fail"
    assert gates.loc["calibration_rows_complete", "status"] == "fail"


def test_missing_audit_prevents_pass():
    manifest = {"status": "complete", "parameters": {"replicates": 1}}
    sessions = pd.DataFrame([{"session": "S", "status": "complete"}])
    detection = pd.DataFrame([{"session": "S", "replicate": 0, "peak_gain": g, "selected_endpoints": 2} for g in (1, 3, 6)])
    f, _ = fixture_tables()
    gates = technical_gates(manifest, sessions, detection, f, {}).set_index("gate")
    assert gates.loc["independent_audit_pass", "status"] == "fail"
