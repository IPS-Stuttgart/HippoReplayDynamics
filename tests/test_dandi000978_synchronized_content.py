import numpy as np
import pandas as pd
import pytest

from scripts.calibrate_dandi000978_synchronized_content import (
    CONDITIONS,
    make_fold,
    nearest_other_trial,
    regional_counts,
    summarize,
    thin_rows,
)


def fixture_data():
    bins = pd.DataFrame(
        [{"trial_id": 2 * route + trial, "route": route, "x_cm": 40.0 * route + phase * 5, "y_cm": float(trial)} for route in range(4) for trial in range(2) for phase in range(2)]
    )
    counts = np.zeros((16, 8), dtype=int)
    for ix, row in bins.iterrows():
        counts[ix, int(row.route)] = 8
        counts[ix, 4 + int(row.route)] = 6
    ca_ix = np.arange(0, 16, 4)
    pf_ix = ca_ix + 3
    frames = []
    audits = []
    for region, indices, start, target in (("CA1", ca_ix, 0, 4), ("PFC", pf_ix, 4, 3)):
        frame = pd.DataFrame(
            {
                "target_index": [0] * 4,
                "repeat": [0] * 4,
                "scope": ["window_250ms"] * 4,
                "route": range(4),
                "trial_id": bins.trial_id.to_numpy()[indices],
                "window_index": indices,
                "target_count": [target] * 4,
                "matched": [True] * 4,
            }
        )
        native = counts[indices, start : start + 4]
        sparse, _ = thin_rows(native, np.full(4, target), (region,))
        rates = np.eye(4) * 3 + 0.1
        audits.append(
            {
                "source_unit_ids": np.arange(start, start + 4),
                "native": native,
                "sparse": sparse,
                "rates": rates,
                "target_counts": np.array([target]),
                "target_event_ids": np.array(["synthetic"]),
                "training_trial_ids": np.array([100, 101]),
                "test_trial_ids": np.arange(8),
            }
        )
        frames.append(frame)
    return bins, counts, np.arange(8), *frames, *audits


def test_nearest_donor_other_trial_same_route():
    bins = fixture_data()[0]
    selected, distance = nearest_other_trial(bins, [0, 1, 4])
    np.testing.assert_array_equal(selected, [2, 3, 6])
    np.testing.assert_allclose(distance, 1)
    assert (bins.trial_id.to_numpy()[selected] != bins.trial_id.to_numpy()[[0, 1, 4]]).all()


def test_no_other_trial_fails_explicitly():
    bins = fixture_data()[0]
    bins.trial_id = bins.route
    with pytest.raises(ValueError, match="No same-route"):
        nearest_other_trial(bins, [0])


def test_nonfinite_position_rejected():
    bins = fixture_data()[0]
    bins.loc[1, "x_cm"] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        nearest_other_trial(bins, [0])


def test_exact_thinning_deterministic_and_unsupported():
    counts = np.array([[3, 5], [0, 0], [1, 0]])
    targets = np.array([5, 0, 3])
    a, ok = thin_rows(counts, targets, ("a",))
    b, other = thin_rows(counts, targets, ("a",))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(ok, [True, True, False])
    np.testing.assert_array_equal(ok, other)
    assert (a <= counts).all() and a[0].sum() == 5


def test_regional_ids_explicit_not_positional():
    counts = np.arange(12).reshape(3, 4)
    np.testing.assert_array_equal(regional_counts(counts, [8, 5, 1, 7], [7, 5]), counts[:, [3, 1]])
    with pytest.raises(ValueError, match="Duplicate"):
        regional_counts(counts, [8, 5, 1, 7], [7, 7])


def test_synced_source_and_four_route_scores():
    args = fixture_data()
    frame, blocks, bank, audit = make_fold(*args, ("test", 3))
    assert blocks.common_matched.all()
    np.testing.assert_array_equal(audit["native_synchronous_pfc_counts"], args[1][frame.window_index, 4:])
    for condition in CONDITIONS:
        for ref in ("decoded", "oracle"):
            matrix = bank[f"sleep_matched_{condition}_{ref}"]
            assert matrix.shape == (1, 4, 4)
            assert np.diagonal(matrix, axis1=1, axis2=2).mean() > matrix.mean()


def test_raw_source_mismatch_fails():
    args = list(fixture_data())
    args[1][0, 0] += 1
    with pytest.raises(ValueError, match="archived donor"):
        make_fold(*args, ("test", 3))


def test_train_trial_leakage_fails():
    args = list(fixture_data())
    args[-2]["training_trial_ids"] = np.array([0])
    with pytest.raises(ValueError, match="leakage"):
        make_fold(*args, ("test", 3))


def test_common_support_does_not_clip_or_replace():
    args = list(fixture_data())
    # A synchronous PFC anchor has too few spikes; old independent donors do not.
    args[1][0, 4] = 1
    frame, blocks, _, _ = make_fold(*args, ("test", 3))
    assert not frame.synchronous_matched.iloc[0]
    assert not blocks.common_matched.iloc[0]
    assert blocks.independent_matched.iloc[0]


def test_empty_common_support_stays_unsupported(tmp_path):
    args = list(fixture_data())
    args[1][0, 4] = 1
    _, blocks, bank, _ = make_fold(*args, ("test", 3))
    blocks["animal"], blocks["file"], blocks["heldout_epoch"] = "JS14", "x.nwb", 3
    primary = pd.DataFrame({"animal": ["JS14"], "event_id": ["synthetic"]})
    (tmp_path / "simulations").mkdir()
    summarize(blocks, bank, primary, tmp_path)
    rows = pd.read_csv(tmp_path / "conditional_sensitivity.csv")
    assert len(rows) == 24
    assert (rows.status == "unsupported").all()
    assert "positive_control_rejection" not in rows or rows.positive_control_rejection.isna().all()
