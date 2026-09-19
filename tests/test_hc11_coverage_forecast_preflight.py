import numpy as np
import pytest

from scripts.preflight_hc11_coverage_forecasts import freeze_selection, intersect_nrem


def candidate(i):
    return {
        "event_id": i,
        "start_s": float(i),
        "end_s": i + 0.1,
        "peak_s": i + 0.05,
        "duration_s": 0.1,
        "detector_spikes": 4,
        "detector_active_cells": 2,
        "detector_peak_z": 4.0,
        "mean_speed_cm_s": 0.0,
    }


def test_interval_intersection_guard_and_no_overlap():
    np.testing.assert_allclose(intersect_nrem([[0, 4], [5, 10]], [[3, 8]]), [[3.05, 3.95], [5.05, 7.95]])
    assert intersect_nrem([[0, 1]], [[2, 3]]).shape == (0, 2)
    assert intersect_nrem([[0, 0.05]], [[0, 1]]).shape == (0, 2)


@pytest.mark.parametrize("intervals", [[[1, 0]], [[0, np.nan]], [[0, 2], [1, 3]], []])
def test_invalid_intervals_fail(intervals):
    with pytest.raises(ValueError):
        intersect_nrem(intervals, [[0, 10]])


def test_selection_deterministic_and_independent_of_strength():
    events = [candidate(i) for i in range(300)]
    original = freeze_selection(events, "s", [], cap=200)
    changed = freeze_selection([dict(e, detector_spikes=1000 - i, detector_peak_z=i) for i, e in enumerate(events)], "s", [], cap=200)
    np.testing.assert_array_equal(original.selected, changed.selected)
    assert original.selected.sum() == 200
    assert not original.overlaps_previous_cohort.any()
    assert "mean_speed_cm_s" not in original
    assert original.detection_domain.eq("native_POST_NREM").all()


def test_previous_overlap_is_annotation_not_exclusion():
    events = [candidate(i) for i in range(7)]
    table = freeze_selection(events, "s", [[0.05, 0.3], [1.1, 2.0]])
    assert table.selected.all()
    assert table.overlaps_previous_cohort.tolist() == [True, False, False, False, False, False, False]


def test_empty_selection_is_explicit():
    table = freeze_selection([], "s", [])
    assert table.empty
    assert "selected" in table and "event_id" in table
    assert "event_id" in table[table.selected]


def test_detector_population_disjoint_and_heldout_fixed():
    from hipporeplayimm.independent_rejected_forecast import partitions

    detector, population, splits = partitions(30, ("hc11", "a", "s"))
    assert not set(detector) & set(population)
    assert len(splits) == 5
    for full, half, held in splits:
        assert set(half) <= set(full)
        assert not set(full) & set(held)
        assert len(half) >= 2 and len(held) >= 2
        assert not set(population[held]) & set(detector)
