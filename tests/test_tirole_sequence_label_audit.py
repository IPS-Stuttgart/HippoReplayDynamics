import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from hipporeplayimm.two_track_content import weighted_correlation
from scripts.audit_tirole_sequence_labels import confusion, contributions, enrich, preference, summarize, track_masses


def fixture():
    rows = []
    for eid in range(4):
        for split in range(5):
            for repeat in range(-1, 5):
                accepted = repeat == -1 or eid >= 2
                label = 1 if eid < 2 else 2
                rows.append(
                    {
                        "event_id": eid,
                        "split": split,
                        "repeat": repeat,
                        "start_s": eid * 61.0,
                        "truth_track": 1 if eid < 2 else 2,
                        "fold": 0,
                        "time_block": eid,
                        "sequence_accepted": accepted,
                        "sequence_eligible": True,
                        "inferred_track": label,
                        "n_inference_spikes": 5,
                        "A_poisson_track2_mass": 0.2,
                        "A_conditional_track2_mass": 0.8,
                        "B_track2_mass": 0.8,
                        "B_track2_z": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def test_correlation_does_not_require_track_identity_mass():
    posterior = np.zeros((10, 2, 10))
    posterior[:, 0, :] = np.eye(10) * 1e-8
    posterior[:, 1, 5] = 1 - 1e-8
    corr = weighted_correlation(posterior, np.arange(10))
    assert corr[0] == pytest.approx(1)
    assert posterior[:, 0].sum(axis=1).mean() == pytest.approx(1e-8)
    assert np.all(preference(posterior[:, 1].sum(axis=1)) == 2)


def test_track_mass_matches_scalar_likelihood_and_zero_spikes_stay_missing():
    rng = np.random.default_rng(4)
    c = rng.poisson(0.5, (8, 6))
    rates = rng.uniform(0.01, 6, (2, 6, 9))
    valid = rng.random((2, 9)) > 0.2
    c[0] = 0
    expected = []
    for conditional in [False, True]:
        lam = np.maximum(rates, 1e-4)
        if conditional:
            lam /= lam.sum(axis=1, keepdims=True)
        masses = []
        for count in c[c.sum(axis=1) > 0]:
            ll = np.stack([(np.log(lam[k]) * count[:, None]).sum(axis=0) - (0 if conditional else 0.1 * lam[k].sum(axis=0)) - np.log(valid[k].sum()) for k in range(2)])
            ll[~valid] = -np.inf
            p = np.exp(ll - logsumexp(ll))
            masses.append(p[1].sum())
        expected.append(np.mean(masses))
    np.testing.assert_allclose(track_masses(c, rates, valid, 0.1), expected)
    assert np.isnan(track_masses(c * 0, rates, valid, 0.1)).all()


def test_ties_never_become_confident_disagreement():
    p = preference([0.5, np.nan, 0.5 + 1e-14, 0.2, 0.8])
    assert np.isnan(p[:3]).all() and np.array_equal(p[3:], [1, 2])
    with pytest.raises(ValueError):
        preference([1.01])


def test_paired_label_groups_and_denominators():
    rows = fixture()
    table = contributions(rows)
    windows = rows[["event_id", "start_s", "truth_track", "fold", "time_block"]].drop_duplicates()
    summary, _ = summarize(table, windows, "POST", "fixture")

    def point(group, metric):
        return summary[(summary.group == group) & (summary.metric == metric)].estimate.iloc[0]

    assert point("full", "A_poisson_agreement") == 0.5
    assert point("half", "A_poisson_agreement") == 0
    assert point("lost", "B_agreement") == 0
    assert point("half", "B_agreement") == 1
    assert point("half", "label_behavior_agreement") == 1
    assert np.isnan(point("gained", "B_agreement"))
    cf = confusion(rows)
    for _, g in cf[cf.total_selection_weight.gt(0)].groupby(["group", "reference"]):
        assert g.joint_fraction.sum() == pytest.approx(1)


def test_independent_B_must_remain_fixed():
    rows = fixture()
    rows.loc[rows.repeat.eq(0), "B_track2_mass"] = 0.6
    with pytest.raises(ValueError, match="B changes"):
        contributions(rows)


def test_missing_frozen_copy_and_changed_counts_fail():
    rows = fixture()
    windows = rows[["event_id", "start_s", "truth_track", "fold", "time_block"]].drop_duplicates()
    b = rows[rows.repeat.eq(-1)][["event_id", "split", "B_track2_mass", "B_track2_z"]]
    with pytest.raises(ValueError, match="incomplete frozen"):
        enrich(windows, rows.iloc[1:], b, None)

    def observation(*_):
        return np.zeros((10, 5)), np.ones((2, 5, 10)), np.ones((2, 10), bool), 0.1

    with pytest.raises(ValueError, match="changed inference"):
        enrich(windows, rows, b, observation)


def test_one_informative_block_has_no_interval():
    rows = fixture()
    rows["sequence_accepted"] = rows.event_id.eq(0)
    table = contributions(rows)
    windows = rows[["event_id", "start_s", "truth_track", "fold", "time_block"]].drop_duplicates()
    summary, _ = summarize(table, windows, "POST", "fixture")
    row = summary[(summary.group == "full") & (summary.metric == "B_agreement")].iloc[0]
    assert row.informative_blocks == 1 and not row.interval_supported and np.isnan(row.ci025)
