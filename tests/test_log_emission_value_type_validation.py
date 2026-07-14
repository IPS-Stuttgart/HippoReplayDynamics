import numpy as np
import pytest

from hipporeplayimm.encoding import LogEmissionTensor


def _make_emissions(log_likelihood) -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.array([0.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )


@pytest.mark.parametrize(
    ("log_likelihood", "message"),
    [
        (np.array([[True, False]]), "not boolean"),
        (np.array([["0.0", "-1.0"]]), "not text values"),
        (np.array([[0.0 + 1.0j, -1.0 + 0.0j]]), "not complex values"),
        (np.array([[0.0, np.bool_(False)]], dtype=object), "not boolean"),
        (np.array([[0.0, np.str_("-1.0")]], dtype=object), "not text values"),
        (np.array([[0.0, np.complex128(-1.0 + 0.0j)]], dtype=object), "not complex values"),
    ],
)
def test_log_emission_tensor_rejects_values_silently_reinterpreted_by_float_coercion(
    log_likelihood,
    message,
):
    with pytest.raises(ValueError, match=message):
        _make_emissions(log_likelihood)


def test_log_emission_tensor_still_accepts_real_numeric_object_arrays():
    emissions = _make_emissions(
        np.array([[np.int64(0), np.float32(-1.5)]], dtype=object)
    )

    assert emissions.log_likelihood.dtype == np.dtype(float)
    assert np.allclose(emissions.log_likelihood, [[0.0, -1.5]])
