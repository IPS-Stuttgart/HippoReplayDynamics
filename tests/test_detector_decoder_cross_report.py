import numpy as np
import pandas as pd
import pytest

from scripts.audit_detector_decoder_cross import recount_independent, truth_independent
from scripts.report_detector_decoder_cross import METRICS, aggregate, detection_truth


def contrast(animal, session, source, value):
    row = {
        "dataset": "d",
        "animal": animal,
        "session": session,
        "source": source,
        "population": "targeted_low",
        "generator": "stationary",
        "peak_gain": 3,
        "likelihood": "poisson",
        "region": 4,
        "family": "targeted",
        "side": "low",
        "status": "complete",
    }
    row.update({m: value for m in METRICS})
    return row


def test_equal_animal_weight_not_event_or_session_weight():
    rows = [contrast("a", "a1", "r0", 0), contrast("a", "a1", "r1", 0), contrast("a", "a2", "r0", 0), contrast("b", "b1", "r0", 1)]
    session, animal, result = aggregate(pd.DataFrame(rows), bootstrap=20)
    assert len(session) == 3 and len(animal) == 2
    assert result.iloc[0].total == pytest.approx(0.5)


def test_duplicate_or_empty_contrasts_rejected():
    row = contrast("a", "s", "r0", 1)
    with pytest.raises(ValueError):
        aggregate(pd.DataFrame([row, row]))
    with pytest.raises(ValueError):
        aggregate(pd.DataFrame())


def test_missing_contrasts_not_imputed_as_zero():
    row = contrast("a", "s", "r0", 1)
    row["status"] = "incomplete"
    with pytest.raises(ValueError, match="no estimable"):
        aggregate(pd.DataFrame([row]))


def test_mixed_truth_windows_not_mislabeled_binary():
    base = {"dataset": "d", "animal": "a", "session": "s", "generator": "moving", "peak_gain": 3, "population": "p", "family": "targeted", "side": "low", "repeat": 0}
    rows = [dict(**base, true_central_occupancy=z, n_gain_peak_detections=n, n_detections=n) for z, n in [(1.0, 1), (0.0, 0), (0.5, 1)]]
    result = detection_truth(pd.DataFrame(rows)).iloc[0]
    assert result.hit_inside == 1 and result.hit_outside == 0
    assert result.mixed_epochs == 1 and result.epochs == 3


def test_independent_audit_reference_reconstruction():
    n = recount_independent(np.array([[0.02, 4], [0.0, 4], [0.01, 7], [0.01, 99]]), [4, 7], np.array([[0.0, 0.02], [0.02, 0.04]]))
    np.testing.assert_array_equal(n, [[1, 1], [1, 0]])
    t = truth_independent(np.array([0, 4, 4, 8]), np.array([[0.0025, 0.0175]]))
    np.testing.assert_allclose(t[0, [0, 4, 8]], [1 / 6, 2 / 3, 1 / 6])


def test_zero_detection_source_records_failure_not_vacuous_pass(tmp_path):
    from scripts.measure_detector_decoder_cross import source_measure

    class Detector:
        def detect_high_mua_in_interval(self, *args, **kwargs):
            return []

    pops = [{"name": "full", "family": "full", "side": "full", "repeat": 0, "indices": [0, 1]}, {"name": "s", "family": "random", "side": "a", "repeat": 0, "indices": [0]}]
    source_measure(
        tmp_path / "empty",
        np.empty((0, 2)),
        None,
        np.array([1, 2]),
        np.ones((2, 2)),
        [0, 4],
        pops,
        Detector(),
        [0.0, 1.0],
        [0.0, 0.0],
        [[0.0, 1.0]],
        {"dataset": "test", "animal": "a", "session": "s", "source": "real", "generator": "real", "peak_gain": 0, "replicate": 0},
    )
    f = pd.read_csv(tmp_path / "empty/factorial_contrasts.csv")
    assert f.status.eq("incomplete").all() and f.total.isna().all()
    s = pd.read_csv(tmp_path / "empty/readout_summary.csv")
    assert s.status.eq("no_events").all() and s.n_windows.eq(0).all()
