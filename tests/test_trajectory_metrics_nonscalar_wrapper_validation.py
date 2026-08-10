import warnings

import numpy as np
import pytest

from hipporeplayimm.trajectory_metrics import trajectory_quality_metrics


def _singleton(value: float) -> np.ndarray:
    return np.array([value], dtype=float)


def test_trajectory_quality_metrics_rejects_nonscalar_log_posterior_wrapper():
    logp = np.empty((1, 2), dtype=object)
    logp[0, 0] = _singleton(np.log(0.5))
    logp[0, 1] = np.array(np.log(0.5))

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(ValueError, match="trajectory_log_posterior.*scalar numeric real values"):
            trajectory_quality_metrics(logp, np.array([0.0, 1.0]))


def test_trajectory_quality_metrics_rejects_nonscalar_bin_center_wrapper():
    centers = np.empty(2, dtype=object)
    centers[0] = _singleton(0.0)
    centers[1] = np.array(1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(ValueError, match="bin_centers.*scalar numeric real values"):
            trajectory_quality_metrics(np.log(np.array([[0.5, 0.5]])), centers)


def test_trajectory_quality_metrics_rejects_nonscalar_timestamp_wrapper():
    times = np.empty(2, dtype=object)
    times[0] = _singleton(0.0)
    times[1] = np.array(1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(ValueError, match="times.*scalar numeric real values"):
            trajectory_quality_metrics(
                np.log(np.array([[0.5, 0.5], [0.4, 0.6]])),
                np.array([0.0, 1.0]),
                times=times,
            )


def test_trajectory_quality_metrics_still_accepts_zero_dimensional_real_wrappers():
    logp = np.empty((1, 2), dtype=object)
    logp[0, 0] = np.array(np.log(0.5))
    logp[0, 1] = np.array(np.log(0.5))
    centers = np.empty(2, dtype=object)
    centers[0] = np.array(0.0)
    centers[1] = np.array(1.0)

    metrics = trajectory_quality_metrics(logp, centers)

    assert metrics["trajectory_time_bins"] == 1
    assert metrics["trajectory_posterior_mean_path_length_cm"] == pytest.approx(0.0)
