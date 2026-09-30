import numpy as np
import pandas as pd
import pytest

from scripts.audit_regional_terminal_mixtures import exhaustive_any_bf, separate_optimizer
from scripts.report_regional_terminal_mixtures import summarize_envelopes


def records(errors, identified=True):
    return pd.DataFrame([{"population": "full", "family": "full", "readout": "independent_bin_any_home",
                          "representation": "continuous_zero_mass", "delta_ms": 20, "session": "a", "animal": "r",
                          "prevalence": .3, "replica": i, "worst_absolute_error": e,
                          "within_5pp": bool(e <= .05 and identified), "identified": identified} for i, e in enumerate(errors)])


def test_report_budget_needs_error_and_fraction():
    assert summarize_envelopes(records([.01, .02, .03, .04])).iloc[0].development_budget_pass
    assert not summarize_envelopes(records([.01, .01, .01, .10])).iloc[0].development_budget_pass


def test_report_unidentified_and_empty_cannot_pass():
    assert not summarize_envelopes(records([.01]*4, identified=False)).iloc[0].development_budget_pass
    with pytest.raises(ValueError, match="nonempty"):
        summarize_envelopes(pd.DataFrame())
    with pytest.raises(ValueError, match="missing"):
        summarize_envelopes(records([np.nan]))


def test_report_replicas_not_duplicated():
    f = records([.02])
    with pytest.raises(ValueError, match="duplicate"):
        summarize_envelopes(pd.concat([f, f]))


def test_independent_union_enumeration_matches_stable_readout():
    from hipporeplayimm.regional_terminal_mixture import independent_any_bf
    for scores, totals in [([3., -4., 1.], [1, 2, 3]), ([99., -3.], [0, 1]), ([-10000., -10000.], [1, 1])]:
        check = exhaustive_any_bf(np.array(scores), np.array(totals), .035)
        actual = independent_any_bf([scores], [totals], .035)[0]
        assert np.isfinite(actual)
        assert check == pytest.approx(actual, abs=1e-8)


def test_second_optimizer_recovers_known_mixture():
    f = np.array([[.9, .1]]*7 + [[.1, .9]]*3)
    assert separate_optimizer(f) == pytest.approx(.25, abs=1e-6)
