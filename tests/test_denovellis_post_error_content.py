import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_denovellis_post_error_content import gates_for, main, marks_for_day, native_events
from scripts.denovellis_post_error_core import (
    build_transitions, contains_event, day_epochs, fit_predictive, near_well_labels,
    normalize_likelihood, reconstruct_visits, route_content, score_visits,
    sequence_test, valid_intervals, weighted_correlation, wilson_upper,
)

WELLS = np.array([[0, 0], [-40, 0], [40, 0]])
PROTOCOL = json.loads((Path(__file__).parents[1] / "docs/denovellis_post_error_protocol.json").read_text())


def visits(order, segments=None):
    return score_visits([dict(well=w, arrival_s=3*i, departure_s=3*i+2, pause_end_s=3*i+2,
                             history_segment=segments[i] if segments else 0, start_index=i, end_index=i+1)
                         for i, w in enumerate(order)], 1, (2, 3))


def test_all_visits_update_task_rule():
    v = visits([2, 1, 2, 3, 1, 2])
    assert v[2]["task_correct"] is False
    assert v[3]["task_kind"] == "inbound" and v[3]["task_correct"] is False
    assert v[5]["required_well"] == 2 and v[5]["task_correct"] is True


def test_numpy_data_attribute_is_not_matlab_epoch_metadata():
    a = np.empty(3, object)
    a[0] = np.array([])
    a[1] = np.array([])
    a[2] = np.array([{}, {"type": "run"}], object)
    assert day_epochs(a, 3)[1]["type"] == "run"
    assert day_epochs(a[2], 3)[1]["type"] == "run"


def test_post_error_correction_and_repeat_chronology():
    v = visits([2, 1, 2, 1, 2, 1, 3])
    rows = build_transitions(v, center=1, outers=(2, 3), immobile_intervals=[[0, 100]])
    assert [r["next_outcome"] for r in rows] == ["repeated_error", "correction"]
    assert [r["consecutive_error_count"] for r in rows] == [1, 2]
    assert rows[0]["correct_alternative_well"] == 3
    assert rows[0]["window_start_s"] == 9
    assert all(r["eligible"] for r in rows)


def test_gaps_do_not_bridge_task_history():
    v = visits([2, 1, 2, 1, 3], [0, 0, 0, 1, 1])
    row = build_transitions(v, center=1, outers=(2, 3), immobile_intervals=[[0, 100]])[0]
    assert not row["eligible"] and row["next_outcome"] == "unclassifiable"
    assert row["exclusion_reason"] == "tracking_gap_or_history_reset"


def test_repeated_well_bouts_keep_first_pause():
    t = np.arange(12) * .1
    xy = WELLS[[1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 2]].astype(float)
    xy[5:7] = [0, 15]
    v, _, _ = reconstruct_visits(t, xy, WELLS)
    assert [r["well"] for r in v] == [2, 1, 3]
    assert v[1]["pause_end_s"] == pytest.approx(.5)
    assert v[1]["departure_s"] == pytest.approx(.9)


def test_missing_position_resets_history():
    t = np.arange(8) * .1
    xy = WELLS[[1, 1, 0, 0, 1, 1, 0, 0]].astype(float)
    xy[3] = np.nan
    v, _, _ = reconstruct_visits(t, xy, WELLS)
    assert v[0]["history_segment"] != v[-1]["history_segment"]
    with pytest.raises(ValueError, match="timestamps"):
        reconstruct_visits([0, 0], WELLS[:2], WELLS)


def test_whole_event_containment_and_gap_boundaries():
    intervals = valid_intervals([0, .1, .2, .3, .4], [True, True, False, True, True])
    assert contains_event(intervals, 0, .1)
    assert not contains_event(intervals, .05, .35)
    assert not contains_event(intervals, .1, .1)
    assert not contains_event(intervals, np.nan, .1)


def test_near_well_overlap_is_unclassifiable():
    labels, _ = near_well_labels([[0, 0], [np.nan, 0]], [[0, 0], [1, 0], [40, 0]], 10)
    assert list(labels) == [-1, -1]


def test_native_duplicate_events_rejected(tmp_path):
    p = tmp_path / "native.csv"
    row = dict(animal="bon", day=3, epoch=2, ripple_number=1, start_time="1s", end_time="2s", actual_speed=0)
    pd.DataFrame([row, row]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        native_events(p)


def test_mark_day_not_confused_with_tetrode_number(tmp_path):
    for name in ["bonmarks03-25.mat", "bonmarks25-03.mat", "bonmarks03.mat"]:
        (tmp_path / name).touch()
    assert [p.name for p in marks_for_day(tmp_path, 3)] == ["bonmarks03-25.mat", "bonmarks03.mat"]


def test_flat_prior_no_temporal_recursion():
    logp = np.array([[0, -2], [-2, 0], [0, -2.]])
    p = normalize_likelihood(logp)
    np.testing.assert_allclose(p[0], p[2])
    np.testing.assert_allclose(p.sum(axis=1), 1)
    np.testing.assert_allclose(p[::-1], normalize_likelihood(logp[::-1]))


def test_ordered_event_and_empty_other_arm():
    p = np.column_stack([np.eye(10), np.zeros((10, 3))])
    routes = [(np.arange(13) < 10, np.arange(10)), (np.arange(13) >= 10, np.arange(3))]
    result = sequence_test(p, np.arange(10), routes, supported_bins=np.ones(10, bool), active_tetrodes=2, seed=20261001, n_shuffles=1000)
    assert result["sequence_passed"] and result["p_value"] <= .05
    assert result == sequence_test(p, np.arange(10), routes, supported_bins=np.ones(10, bool), active_tetrodes=2, seed=20261001, n_shuffles=1000)


def test_stationary_and_unsupported_events_do_not_pass():
    p = np.tile([.1, .2, .2, .2, .3], (10, 1))
    routes = [(np.array([1, 1, 1, 0, 0], bool), np.arange(3)), (np.array([1, 0, 0, 1, 1], bool), np.arange(3))]
    result = sequence_test(p, np.arange(10), routes, supported_bins=np.ones(10, bool), active_tetrodes=2, seed=4)
    assert not result["sequence_passed"]
    result = sequence_test(p, np.arange(10), routes, supported_bins=np.zeros(10, bool), active_tetrodes=1, seed=4)
    assert result["p_value"] is None and result["n_shuffles"] == 0


def test_shared_stem_unassigned_mixed_fractional_content():
    p = np.tile([.4, .1, .2, .3], (10, 1))
    content = route_content(p, [[False, True, False, False], [False, False, True, True]])
    np.testing.assert_allclose(content, [.1, .5])
    with pytest.raises(ValueError, match="disjoint"):
        route_content(p, [[True, False, False, False], [True, False, False, False]])


def test_weighted_correlation_matches_perfect_path():
    assert weighted_correlation(np.eye(8), np.arange(8), np.arange(8)) == pytest.approx(1)
    assert weighted_correlation(np.eye(8)[::-1], np.arange(8), np.arange(8)) == pytest.approx(-1)
    assert wilson_upper(50, 1000) < .075
    assert wilson_upper(75, 1000) > .075


def test_predictive_standardization_never_uses_test_rows():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(120, 2))
    y = (x[:, 0] + rng.normal(size=120) > 0).astype(int)
    a = np.repeat(["a", "b", "c"], 40)
    d = np.repeat(["a1", "b1", "c1"], 40)
    fit = fit_predictive(x, y, a, d)
    np.testing.assert_allclose(fit.mean, x.mean(axis=0))
    design = fit.design([[100, 100]], ["d"], ["unseen"])
    assert not design[0, -len(fit.days):].any()
    np.testing.assert_allclose(fit.mean, x.mean(axis=0))
    with pytest.raises(ValueError, match="both binary"):
        fit_predictive(x, np.ones(120), a, d)


def test_empty_feasibility_cannot_pass():
    inventory = pd.DataFrame([dict(animal="a", status="failed")])
    gates = gates_for(inventory, pd.DataFrame(), pd.DataFrame(), PROTOCOL)
    assert not gates[-1]["passed"]
    assert not next(g for g in gates if g["gate"] == "native_candidates_present")["passed"]


def test_failed_prerequisite_blocks_analysis(tmp_path):
    (tmp_path / "denovellis_post_error_manifest.json").write_text(json.dumps({"protocol": PROTOCOL, "feasibility_passed": False}))
    with pytest.raises(SystemExit) as exc:
        main(["--stage", "analysis", "--prerequisite-dir", str(tmp_path), "--output-dir", str(tmp_path / "never")])
    assert exc.value.code == 2
    assert not (tmp_path / "never").exists()
