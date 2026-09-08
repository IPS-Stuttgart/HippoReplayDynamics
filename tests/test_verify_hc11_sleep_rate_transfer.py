import numpy as np
import pandas as pd
import pytest
from test_hc11_sleep_rate_transfer import score_fixture

from scripts.audit_hc11_sleep_rate_transfer import calibrate, paired_contrasts, summarize
from scripts.verify_hc11_sleep_rate_transfer import assert_equal, global_score, reconstruct_gain, verify_aggregates


@pytest.mark.parametrize("regime,alpha", [("run_original", None), ("sleep_alpha100", 100), ("sleep_alpha1000", 1000)])
def test_independent_gains(regime, alpha):
    rates = np.array([[1.0, 5.0, 8.0], [2.0, 2.0, 2.0], [4.0, 1.0, 0.01]])
    counts = np.array([30, 1, 7])
    independent = reconstruct_gain(rates, counts, regime)
    production = calibrate(rates, counts, alpha)
    for a, b in zip(independent, production, strict=True):
        assert_equal(a, b, "gain")


def test_proper_multinomial():
    probability = np.array([0.25, 0.75])
    assert global_score(np.array([[1, 1]]), probability) == pytest.approx(np.log(2 * 0.25 * 0.75))
    assert global_score(np.array([[0, 0]]), probability) == 0


def test_corrupted_values_rejected():
    with pytest.raises(ValueError):
        assert_equal([1.0, 2.0], [1.0, 3.0], "corrupt")
    with pytest.raises(ValueError):
        assert_equal([1.0], [1.0, 1.0], "missing")


@pytest.mark.parametrize("corrupt", [False, True])
def test_aggregate_and_interval_audit(tmp_path, corrupt):
    table = pd.concat([score_fixture().assign(rat=rat, session=rat) for rat in ("a", "b", "c", "d")], ignore_index=True)
    split, events = paired_contrasts(table)
    summary, animals, _ = summarize(events)
    if corrupt:
        summary.loc[0, "ci_low"] = -1.0
    for name, frame in (("split_contrasts", split), ("event_contrasts", events), ("summary", summary), ("by_animal", animals)):
        frame.to_csv(tmp_path / f"sleep_rate_transfer_{name}.csv", index=False)
    if corrupt:
        with pytest.raises(ValueError, match="aggregate"):
            verify_aggregates(table, tmp_path)
    else:
        result = verify_aggregates(table, tmp_path)
        assert result["max_aggregate_error"] == 0
