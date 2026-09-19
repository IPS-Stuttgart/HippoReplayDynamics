import numpy as np
import pandas as pd
import pytest

from scripts.validate_hippocampal_cortical_run_readouts import (
    COHORTS,
    PARAMETERS,
    clock_segments,
    decode,
    evaluate,
    fit_composition,
    fold_masks,
    prepare_run,
    unit_mask,
    write_summaries,
)


def synthetic(seed=3):
    rng = np.random.default_rng(seed)
    n = 3000
    position = np.abs((np.arange(n) / 80) % 2 - 1)
    centers = np.linspace(0, 1, 25)
    rates = 0.3 + 40 * np.exp(-0.5 * ((position[:, None] - centers) / 0.075)**2)
    return position, rng.poisson(rates * PARAMETERS["bin_s"])


def test_author_rsc_rule_is_not_the_export_pyramidal_label():
    areas = np.array(["CA1", "CA3", "RSC", "RSC", "RSC", "RSC"])
    types = np.array(["Pyramidal Cell", "Pyramidal Cell", "Pyramidal Cell", "Wide Interneuron", "Unknown", "Narrow Interneuron"])
    assert np.flatnonzero(unit_mask(areas, types, "RSC_author_non_narrow")).tolist() == [2, 3, 4]
    assert np.flatnonzero(unit_mask(areas, types, "RSC_export_pyramidal")).tolist() == [2]
    assert np.flatnonzero(unit_mask(areas, types, "CA1_pyramidal")).tolist() == [0]
    assert np.flatnonzero(unit_mask(areas, types, "CA3_pyramidal")).tolist() == [1]
    with pytest.raises(ValueError):
        unit_mask(areas, types, "RSC_unknown_choice")


def test_blocked_folds_are_disjoint_and_guarded():
    n = 1000
    covered = np.zeros(n, int)
    for fold in range(5):
        train, test = fold_masks(n, np.ones(n, bool), fold)
        assert not (train & test).any()
        assert test.sum() == 200
        a, b = np.flatnonzero(test)[[0, -1]]
        assert not train[max(0, a - 4):min(n, b + 5)].any()
        covered += test
    np.testing.assert_array_equal(covered, 1)


def test_heldout_spikes_and_labels_cannot_change_encoding():
    position, counts = synthetic()
    train, _test = fold_masks(len(position), np.ones(len(position), bool), 2)
    model, use, occupancy = fit_composition(counts, position, train)
    changed = counts.copy()
    changed[~train] = 1000
    different_positions = position.copy()
    different_positions[~train] = 0.99
    other, other_use, other_occ = fit_composition(changed, different_positions, train)
    np.testing.assert_array_equal(model, other)
    np.testing.assert_array_equal(use, other_use)
    np.testing.assert_array_equal(occupancy, other_occ)


def test_known_fields_recover_position_and_beats_shift_control():
    position, counts = synthetic()
    train, test = fold_masks(len(position), np.ones(len(position), bool), 2)
    metrics, audit = evaluate(counts, position, train, test, ("synthetic",))
    assert metrics["median_absolute_error_fraction"] < 0.08
    assert metrics["gain_over_null_mae_fraction"] > 0.1
    assert metrics["empirical_shift_p"] <= 0.05
    repeat, repeated = evaluate(counts, position, train, test, ("synthetic",))
    assert repeat == metrics
    np.testing.assert_array_equal(repeated["shifts"], audit["shifts"])


def test_decoder_is_time_independent_and_zero_count_is_flat():
    position, counts = synthetic()
    train, test = fold_masks(len(position), np.ones(len(position), bool), 0)
    composition, use, _ = fit_composition(counts, position, train)
    selected = counts[test][:, use]
    posterior = decode(selected, composition)
    np.testing.assert_allclose(decode(selected[::-1], composition), posterior[::-1])
    np.testing.assert_allclose(decode(np.zeros((2, use.sum())), composition), 1 / 25)
    row = 33
    changed = selected.copy()
    changed[row] += 10
    mask = np.arange(len(selected)) != row
    np.testing.assert_array_equal(decode(changed, composition)[mask], posterior[mask])


def test_total_activity_alone_does_not_decode_uniform_composition():
    composition = np.tile(np.array([0.2, 0.3, 0.5]), (25, 1))
    posterior = decode(np.array([[2, 3, 5], [200, 300, 500], [0, 0, 0]]), composition)
    np.testing.assert_allclose(posterior, 1 / 25)


def test_gaps_are_segments_not_maze_labels():
    assert clock_segments([0., 0.04, 5., 5.04]) == [(0, 2), (2, 4)]
    for clock in ([1., 0.], [0., np.nan], [1.]):
        with pytest.raises(ValueError):
            clock_segments(clock)


def test_run_binning_uses_timestamps_and_half_open_edges():
    clock = 10 + np.arange(2501) / 25
    xy = np.column_stack((np.abs(np.sin(clock / 10)), np.zeros(len(clock))))
    spikes = [np.array([10., 10.249, 10.25, 10.5, 109.999, 110.])]
    _, _, counts, eligible, geometry = prepare_run(clock, xy, np.ones(len(clock)) * 5, spikes)
    assert counts[:, 0].sum() == 5  # Final boundary is excluded.
    np.testing.assert_array_equal(counts[:3, 0], [2, 1, 1])
    assert eligible.all()
    assert geometry["coordinate"].endswith("not_cm")
    assert geometry["axis_variance_fraction"] == 1


def test_missing_training_units_or_test_bins_fails():
    position, counts = synthetic()
    train, test = fold_masks(len(position), np.ones(len(position), bool), 0)
    with pytest.raises(ValueError, match="training-active"):
        evaluate(np.zeros_like(counts), position, train, test, ("missing",))
    with pytest.raises(ValueError, match="test bins"):
        evaluate(counts, position, train, np.zeros_like(test), ("missing",))


def test_empty_and_all_failed_summaries_are_not_vacuous_success(tmp_path):
    for rows in ([], [{"status": "failed", "animal": "a", "asset_id": "s", "segment": 0, "cohort": COHORTS[0]}]):
        table = write_summaries(rows, tmp_path)
        assert (table.status == "scored").sum() == 0
        for name in ("segment_summary.csv", "file_summary.csv", "by_animal_summary.csv"):
            assert pd.read_csv(tmp_path / name).empty
