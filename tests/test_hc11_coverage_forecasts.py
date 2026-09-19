import numpy as np
import pytest

from hipporeplayimm.training_continuity import geometry
from scripts.audit_hc11_coverage_forecasts import classify_native


def mocked_path(monkeypatch, path):
    import scripts.audit_hc11_coverage_forecasts as module

    monkeypatch.setattr(module, "poisson_map", lambda counts, rates: np.asarray(path, int))
    monkeypatch.setattr(module, "overlapping_counts", lambda base, durations: np.ones((len(path), 2), int))


def classify(path, centers, topology, length):
    return classify_native(np.ones((10, 2), int), np.full(10, 0.005), np.ones((2, len(centers))), np.asarray(centers, float), np.arange(2), topology, length)[0]


def test_linear_identical_to_original_screen(monkeypatch):
    centers = np.arange(0, 160, 4)
    path = np.arange(0, 24, 2)
    mocked_path(monkeypatch, path)
    actual = classify(path, centers, "linear", 160)
    expected = geometry(path, np.column_stack((centers, np.zeros(len(centers)))), np.ones((len(path), 2), int))
    assert actual == expected
    assert actual["geometric_pass"]


def test_circular_seam_does_not_break_continuity(monkeypatch):
    centers = np.arange(0, 160, 4)
    path = (np.arange(12) * 2 + 32) % len(centers)
    mocked_path(monkeypatch, path)
    result = classify(path, centers, "circular", 160)
    assert result["longest_run_frames"] == 12
    assert result["run_displacement_cm"] == 72
    assert result["geometric_pass"]


def test_complete_circle_is_not_net_displacement(monkeypatch):
    centers = np.arange(0, 160, 8)
    path = np.r_[np.arange(20), 0]
    mocked_path(monkeypatch, path)
    result = classify(path, centers, "circular", 160)
    assert result["run_displacement_cm"] == 0
    assert not result["geometric_pass"]


def test_no_frames_is_explicit(monkeypatch):
    mocked_path(monkeypatch, [])
    result = classify([], [0, 4], "linear", 8)
    assert result["failure_reason"] == "no_complete_windows"


def test_unknown_topology_fails():
    with pytest.raises(ValueError):
        classify([], [0, 4], "unknown", 8)


def test_heldout_and_future_do_not_enter_forecast():
    from hipporeplayimm.independent_rejected_forecast import forecast_distributions
    from hipporeplayimm.lagged_neural_prediction import NeuralOperator
    from hipporeplayimm.occupancy_matched_forecast import OccupancyMatchedNull

    operator = NeuralOperator(np.full(2, 0.5), np.array([[0.9, 0.1], [0.1, 0.9]]), np.full(2, 0.5))
    null = OccupancyMatchedNull.from_operator(operator)
    emissions = np.array([[0.8, 0.1], [0.2, 0.9]])
    counts = np.array([[3, 0], [0, 2], [1, 1], [0, 3]])
    original = forecast_distributions(counts, emissions, operator, null, (2,))
    changed = counts.copy()
    changed[1:] = 999
    modified = forecast_distributions(changed, emissions, operator, null, (2,))
    for key in original[2]:
        np.testing.assert_allclose(original[2][key][0], modified[2][key][0])


def test_summary_uses_event_then_session_then_animal():
    import pandas as pd

    from scripts.audit_hc11_coverage_forecasts import BASELINES, summarize

    rows = []
    for animal, event_count in (("a", 1), ("b", 7)):
        for event in range(event_count):
            for split in range(5):
                row = {
                    "dataset": "hc11",
                    "animal": animal,
                    "session": animal,
                    "event_id": event,
                    "split": split,
                    "level": "half",
                    "status": "scored",
                    "n_target_bins": 2,
                    "n_heldout_target_spikes": 2,
                    "lost_with_thinning": True,
                    "lost_supported": True,
                    "full_geometric_pass": True,
                    "score_dynamic": -1 if animal == "a" else -5,
                }
                row.update({"score_" + b: -4 for b in BASELINES})
                rows.append(row)
    data = pd.DataFrame(rows)
    *_, summary = summarize(data)
    primary = summary[(summary.group == "lost_supported") & (summary.contrast == "dynamic_minus_matched_own") & (summary.metric == "delta")]
    assert primary["mean"].item() == 1  # mean of +3 and -1, not event-weighted
    with pytest.raises(ValueError):
        summarize(pd.concat([data, data.iloc[[0]]]))


def test_zero_evaluation_spikes_stay_undefined_per_spike():
    import pandas as pd

    from scripts.audit_hc11_coverage_forecasts import SCORE_COLUMNS, summarize

    row = {
        "dataset": "hc11",
        "animal": "a",
        "session": "s",
        "event_id": 0,
        "split": 0,
        "level": "half",
        "status": "scored",
        "n_target_bins": 2,
        "n_heldout_target_spikes": 0,
        "lost_with_thinning": True,
        "lost_supported": True,
        "full_geometric_pass": True,
        **{c: 0 for c in SCORE_COLUMNS},
    }
    _, events, _, _, summary = summarize(pd.DataFrame([row]))
    assert events.delta.eq(0).all()
    assert events.delta_per_spike.isna().all()
    assert events.informative_splits.eq(0).all()
    assert not summary.metric.eq("delta_per_spike").any()
