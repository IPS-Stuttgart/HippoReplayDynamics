import numpy as np
import pandas as pd
import pytest

from scripts.build_regional_blind_bank import fit_anchors, jobs, quota_labels


def transitions():
    rows = []
    for event, origin in ((1, 0.), (2, 100.)):
        for i, length in enumerate([1., 30., 40., 1., 1., 25., 1.]):
            rows.append({"session": "a", "event_index": event, "transition_index": i,
                         "transition_time_s": origin+i*.004, "posterior_mean_step_speed_cm_s": length/.004})
    return pd.DataFrame(rows)


def test_empirical_episode_pool_does_not_cross_events():
    rows, pools = fit_anchors(transitions())
    p = pools["a"]
    np.testing.assert_allclose(p["lengths"], [40., 25., 40., 25.])
    np.testing.assert_allclose(p["intervals"], [.016, .016])
    assert p["moving_speed"] == 250.
    assert rows.iloc[0].events == 2


def test_missing_within_event_intervals_fails():
    data = transitions()
    data.loc[data.transition_index >= 4, "posterior_mean_step_speed_cm_s"] = 1.
    with pytest.raises(ValueError, match="missing within-session"):
        fit_anchors(data)


def test_missing_transition_is_not_bridged():
    with pytest.raises(ValueError, match="contiguous"):
        fit_anchors(transitions().drop(index=2))


@pytest.mark.parametrize("p", [.05, .15, .30, .50])
def test_free_prevalence_quota(p):
    a = quota_labels(140, p, np.random.default_rng(1))
    b = quota_labels(140, p, np.random.default_rng(1))
    assert a.sum() == round(140*p)
    np.testing.assert_array_equal(a, b)


def test_calibration_and_validation_jobs_are_distinct():
    j = jobs(4)
    assert len(j) == len(set(j)) == 36
    assert len([x for x in j if x[0] == "calibration"]) == 4
    assert {x[1] for x in j if x[0] == "validation"} == {.05, .15, .3, .5}
