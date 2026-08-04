import numpy as np
import pytest

from hipporeplayimm.trajectory_metrics import trajectory_quality_metrics


def _boxed(value: object) -> np.ndarray:
    return np.array(value)


@pytest.mark.parametrize(
    ("bad_value", "message"),
    [
        (_boxed(np.complex128(np.log(0.5) + 0.25j)), "complex"),
        (_boxed(np.bool_(False)), "boolean or text"),
        (_boxed("0.0"), "boolean or text"),
        (_boxed(np.bytes_(b"0.0")), "boolean or text"),
    ],
)
def test_trajectory_quality_metrics_rejects_nested_nonreal_log_posterior_scalars(bad_value, message):
    logp = np.empty((1, 2), dtype=object)
    logp[0, 0] = bad_value
    logp[0, 1] = _boxed(np.float64(np.log(0.5)))

    with pytest.raises(ValueError, match=f"trajectory_log_posterior.*{message}"):
        trajectory_quality_metrics(logp, np.array([0.0, 1.0]))


def test_trajectory_quality_metrics_rejects_nested_complex_bin_centers():
    centers = np.empty(2, dtype=object)
    centers[0] = _boxed(np.complex128(0.0 + 1.0j))
    centers[1] = _boxed(np.float64(1.0))

    with pytest.raises(ValueError, match="bin_centers.*complex"):
        trajectory_quality_metrics(np.log(np.array([[0.5, 0.5]])), centers)


def test_trajectory_quality_metrics_rejects_nested_complex_times():
    times = np.empty(2, dtype=object)
    times[0] = _boxed(np.complex128(0.0 + 1.0j))
    times[1] = _boxed(np.float64(1.0))

    with pytest.raises(ValueError, match="times.*complex"):
        trajectory_quality_metrics(
            np.log(np.array([[0.5, 0.5], [0.4, 0.6]])),
            np.array([0.0, 1.0]),
            times=times,
        )


def test_trajectory_quality_metrics_accepts_nested_real_scalars():
    logp = np.empty((2, 2), dtype=object)
    logp[0, 0] = _boxed(np.float64(np.log(0.75)))
    logp[0, 1] = _boxed(np.float64(np.log(0.25)))
    logp[1, 0] = _boxed(np.float64(np.log(0.25)))
    logp[1, 1] = _boxed(np.float64(np.log(0.75)))
    centers = np.empty(2, dtype=object)
    centers[0] = _boxed(np.float64(0.0))
    centers[1] = _boxed(np.float64(10.0))
    times = np.empty(2, dtype=object)
    times[0] = _boxed(np.float64(0.0))
    times[1] = _boxed(np.float64(0.5))

    metrics = trajectory_quality_metrics(logp, centers, times=times)

    assert metrics["trajectory_posterior_mean_path_length_cm"] == pytest.approx(5.0)
    assert metrics["trajectory_posterior_mean_speed_cm_s"] == pytest.approx(10.0)
