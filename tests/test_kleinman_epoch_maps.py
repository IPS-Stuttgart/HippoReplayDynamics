import numpy as np
import pandas as pd
import pytest

from scripts.audit_kleinman_epoch_maps import common_maps, decode_metrics, epoch_groups, group_split, run_mask, summarize_windows


def runs_fixture():
    return [{"epoch": e, "start_s": e * 100 + i * 2, "end_s": e * 100 + i * 2 + 1, "traversal": e * 100 + i} for e in [1, 2, 3] for i in range(17)]


def test_grouping_does_not_straddle_epochs_and_splits_are_disjoint_matched():
    runs = epoch_groups(runs_fixture())
    assert len(runs) == 48
    for split in range(5):
        a, b, test = group_split(runs, "a", "s", 1, 2, split)
        assert len(a) == len(b) == 4
        assert not set(b) & set(test)
        assert set(b) | set(test) == set(range(8))
        repeat = group_split(runs, "a", "s", 1, 2, split)
        for x, y in zip((a, b, test), repeat, strict=True):
            np.testing.assert_array_equal(x, y)
    t = np.arange(100, 340, 0.5)
    assert not (run_mask(t, runs, 2, b) & run_mask(t, runs, 2, test)).any()
    assert not (run_mask(t, runs, 1, a) & run_mask(t, runs, 2, test)).any()


def test_insufficient_groups_fail_instead_of_reusing_test_laps():
    runs = epoch_groups(runs_fixture())
    runs = [r for r in runs if r["epoch"] != 1 or r["epoch_group"] < 3]
    with pytest.raises(ValueError, match="insufficient_epoch"):
        group_split(runs, "a", "s", 1, 2, 0)


def test_common_unit_alignment_uses_composite_training_order():
    all_rates = np.arange(1, 25).reshape(6, 4)
    ua = np.array([True, False, True, True])
    ub = np.array([False, True, True, True])
    sa = np.array([True, True, True, False, False, False])
    sb = np.array([False, True, True, True, False, False])
    ra, rb, support, units = common_maps((all_rates[:, ua], sa, ua, None), (2 * all_rates[:, ub], sb, ub, None))
    assert units.tolist() == [False, False, True, True]
    np.testing.assert_array_equal(ra, all_rates[:, 2:])
    np.testing.assert_array_equal(rb, 2 * all_rates[:, 2:])
    np.testing.assert_array_equal(support, sa & sb)


def decode_fixture():
    centers = np.arange(1, 60, 2)
    rates = np.full((60, 24), 0.02)
    for d in [0, 1]:
        rates[d * 30 : (d + 1) * 30, d * 12 : (d + 1) * 12] += 45 * np.exp(-0.5 * ((centers[:, None] - np.linspace(3, 57, 12)) / 5) ** 2)
    counts = np.random.default_rng(7).poisson(0.25 * rates)
    return counts, rates, np.ones(60, bool), centers, np.tile(centers, 2), np.repeat([0, 1], 30), np.arange(0, 62, 2)


def test_spatially_shifted_map_degrades_transfer_for_both_arms():
    args = decode_fixture()
    counts, rates, support, centers, truth, direction, edges = args
    shifted = np.roll(rates.reshape(2, 30, 24), 12, axis=1).reshape(60, 24)
    for arm in ["poisson", "composition"]:
        good = decode_metrics(*args, arm)
        bad = decode_metrics(counts, shifted, support, centers, truth, direction, edges, arm)
        assert good["mean_error_cm"].mean() < 5
        assert bad["mean_error_cm"].mean() > good["mean_error_cm"].mean() + 15


def test_composition_is_invariant_to_global_gain_but_keeps_support_loss_visible():
    counts, rates, support, centers, truth, direction, edges = decode_fixture()
    a = decode_metrics(counts, rates, support, centers, truth, direction, edges, "composition")
    b = decode_metrics(counts, rates * 7, support, centers, truth, direction, edges, "composition")
    for key in a:
        np.testing.assert_allclose(a[key], b[key], atol=1e-10)
    support[0] = False
    c = decode_metrics(counts, rates, support, centers, truth, direction, edges, "composition")
    assert not c["truth_supported"][0]
    assert np.isnan(c["true_log_score_above_uniform"][0])
    assert np.isfinite(c["mean_error_cm"][0])


def test_summary_pairs_same_windows_and_retains_failed_qc():
    records = []
    for split in range(2):
        for _ in range(25):
            row = {"animal": "a", "session": "s", "source_epoch": 1, "target_epoch": 2, "arm": "poisson", "split": split, "n_common_units": 8, "mean_map_hellinger_squared": 0.2}
            for arm in ["within", "cross"]:
                row.update(
                    {
                        arm + "_mean_error_cm": 10 if arm == "within" else 40,
                        arm + "_map_error_cm": 12,
                        arm + "_direction_correct": 0.8,
                        arm + "_entropy": 0.5,
                        arm + "_true_log_score_above_uniform": 1,
                    }
                )
            row.update(error_increase_cm=30, within_truth_supported=True)
            records.append(row)
    summary = summarize_windows(pd.DataFrame(records))
    assert len(summary) == 2 and summary.n_windows.eq(25).all()
    assert summary.within_qc.all() and not summary.cross_qc.any()
    assert summary.error_increase_cm.eq(30).all()
