"""Report decisions must retain nonreplication and missing denominators."""

import pandas as pd

from scripts.report_coverage_dose_forecasts import decisions


def table():
    rows = []
    for dataset, n in [("pfeiffer_foster", 4), ("tanni2022", 5)]:
        base = {"dataset": dataset, "fraction": 0.5, "animals": n, "positive_animals": n, "mean": 0.05, "ci_low": 0.02, "ci_high": 0.08}
        rows.append(base | {"arm": "fixed_codebook", "group": "all", "metric": "pass_change", "mean": -0.1, "ci_low": -0.2, "ci_high": -0.05})
        for arm in ["fixed_codebook", "restricted_calibration"]:
            for group in ["lost", "lost_supported"]:
                for metric in ["dynamic_minus_matched_own", "dynamic_minus_global", "dynamic_minus_no_history"]:
                    rows.append(base | {"arm": arm, "group": group, "metric": metric})
    return pd.DataFrame(rows)


def test_requires_both_axes_and_keeps_failed_replication():
    t = table()
    assert decisions(t).joint_statement_supported.all()
    t.loc[t.dataset.eq("tanni2022") & t.metric.eq("dynamic_minus_matched_own"), "ci_low"] = -0.01
    d = decisions(t)
    assert d[d.dataset.eq("pfeiffer_foster")].joint_statement_supported.all()
    assert not d[d.dataset.eq("tanni2022")].joint_statement_supported.any()
    t = table()
    t.loc[t.metric.eq("pass_change"), "ci_high"] = 0.01
    assert not decisions(t).joint_statement_supported.any()


def test_missing_animals_or_adequacy_control_cannot_pass():
    t = table()
    t.loc[t.dataset.eq("pfeiffer_foster") & t.metric.eq("dynamic_minus_matched_own"), "animals"] = 3
    assert not decisions(t).query("dataset == 'pfeiffer_foster'").joint_statement_supported.any()
    assert not decisions(table().query("metric != 'dynamic_minus_global'")).joint_statement_supported.any()
