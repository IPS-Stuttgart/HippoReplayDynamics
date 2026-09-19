from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from scripts.audit_tirole_evaluation_neuron_influence import METRICS, delete_permutation_index, event_vectors, lost_membership


def test_cycle_deletion_preserves_uniform_null_and_rate_groups():
    p = np.array([list(a) + list(b) for a in permutations(range(3)) for b in permutations(range(3, 6))])
    for drop in range(6):
        q = delete_permutation_index(p, drop)
        np.testing.assert_array_equal(np.sort(q, axis=1), np.tile(np.arange(5), (len(p), 1)))
        keep = np.delete(np.arange(6), drop)
        assert np.all((keep[q] // 3) == (keep // 3))
        _, counts = np.unique(q, axis=0, return_counts=True)
        assert np.all(counts == 3)


@pytest.mark.parametrize("matrix,drop", [(np.array([[0, 0, 2]]), 0), (np.array([[0.0, 1.0]]), 0), (np.array([[0, 1]]), 2)])
def test_invalid_permutations_fail(matrix, drop):
    with pytest.raises(ValueError):
        delete_permutation_index(matrix, drop)


def test_frozen_membership_never_uses_evaluation_content():
    d = pd.DataFrame(
        [
            {
                "session": "rat",
                "event_id": e,
                "split": 0,
                "repeat": 0,
                "fraction": f,
                "arm": "real",
                "epoch": "POST",
                "candidate_stratum": "ripple",
                "sequence_accepted": (f == 1 or e == 2),
                "sequence_eligible": True,
                "inferred_track": e,
                "evaluation_z_log_odds": -999.0,
            }
            for e in [1, 2]
            for f in [1.0, 0.5]
        ]
    )
    a = lost_membership(d)
    assert a.event_id.tolist() == [1]
    d.evaluation_z_log_odds = 999.0
    pd.testing.assert_frame_equal(a, lost_membership(d))
    with pytest.raises(ValueError):
        lost_membership(d.iloc[1:])


def test_omission_only_replaces_affected_splits_and_collapses_repeats():
    member = pd.DataFrame([{"event_id": 1, "split": s, "repeat": r, "full_track": 1} for s in [0, 1] for r in range(5)])
    readout = pd.DataFrame(
        [{"event_id": 1, "split": s, "omitted_index": -1, **{k: 2.0 for k in METRICS}} for s in [0, 1]]
        + [{"event_id": 1, "split": 0, "omitted_index": 8, **{k: -2.0 for k in METRICS}}]
    )
    baseline = event_vectors(member, readout, -1)
    modified = event_vectors(member, readout, 8)
    assert len(baseline) == len(modified) == 1
    assert baseline.iloc[0, 0] == 2 and modified.iloc[0, 0] == 0
    member.full_track = 2
    assert event_vectors(member, readout, -1).iloc[0, 0] == -2


def test_unsupported_deletion_stays_missing_not_zero():
    member = pd.DataFrame([{"event_id": 1, "split": 0, "repeat": 0, "full_track": 1}])
    readout = pd.DataFrame(
        [{"event_id": 1, "split": 0, "omitted_index": -1, **{k: 1.0 for k in METRICS}}, {"event_id": 1, "split": 0, "omitted_index": 5, **{k: np.nan for k in METRICS}}]
    )
    assert event_vectors(member, readout, 5).isna().all().all()
