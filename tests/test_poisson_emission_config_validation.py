import numpy as np
import pytest

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.encoding import EncodingConfig, EmissionConfig, EncodingModel, build_emissions


def _session():
    return ReplaySession(
        rat="RatX",
        name="OpenX",
        path=None,
        position=np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
        spikes=np.array([[0.5, 1.0]]),
        tetrode_cell_ids=np.array([[1, 1]]),
        excitatory_neurons=np.array([1]),
        inhibitory_neurons=np.array([]),
        ripple_events=np.array([[0.0, 1.0, 0.5, 0.0, 0.0, 0.0]]),
        run_times=np.array([[0.0, 1.0]]),
        sleep_box_immobile_times=np.empty((0, 2)),
        sleep_times=np.empty((0, 2)),
        rem_times=np.empty((0, 2)),
        well_sequence=None,
        metadata={},
    )


def _encoding():
    return EncodingModel(
        x_edges=np.array([0.0, 1.0, 2.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.5, 0.5], [1.5, 0.5]]),
        rates_hz=np.array([[2.0, 4.0]]),
        occupancy_s=np.ones(2),
        cell_ids=np.array([1]),
        config=EncodingConfig(),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("spike_rate_scale", True),
        ("spike_rate_scale", "1.0"),
        ("spike_rate_scale", np.array([1.0])),
        ("likelihood_temperature", True),
        ("likelihood_temperature", "1.0"),
        ("likelihood_temperature", np.array([1.0])),
        ("negative_binomial_overdispersion", False),
        ("negative_binomial_overdispersion", "0.0"),
        ("negative_binomial_overdispersion", np.array([0.0])),
    ],
)
def test_build_emissions_rejects_lossy_calibration_scalars(field, value):
    with pytest.raises(ValueError, match=field):
        build_emissions(_session(), _encoding(), 0, EmissionConfig(time_bin_s=1.0, **{field: value}))


def test_build_emissions_accepts_numpy_scalar_calibration_values():
    emissions = build_emissions(
        _session(),
        _encoding(),
        0,
        EmissionConfig(
            time_bin_s=1.0,
            spike_rate_scale=np.float64(2.0),
            likelihood_temperature=np.float64(2.0),
            negative_binomial_overdispersion=np.float64(0.0),
        ),
    )
    expected = np.array([2.0, 4.0]) * 2.0
    np.testing.assert_allclose(emissions.log_likelihood[0], (np.log(expected) - expected) / 2.0)
