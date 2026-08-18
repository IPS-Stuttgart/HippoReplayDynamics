import numpy as np
import pytest

from hipporeplayimm import encoding


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_poisson_emissions_reject_boolean_dt(value):
    with pytest.raises(ValueError, match="dt.*boolean"):
        encoding._poisson_log_emissions(
            np.array([[0]]),
            np.array([[1.0]]),
            value,
        )


@pytest.mark.parametrize(
    "parameter",
    [
        "spike_rate_scale",
        "likelihood_temperature",
        "negative_binomial_overdispersion",
        "cell_weights",
    ],
)
@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_poisson_emissions_reject_boolean_numeric_options(parameter, value):
    with pytest.raises(ValueError, match=rf"{parameter}.*boolean"):
        encoding._poisson_log_emissions(
            np.array([[0]]),
            np.array([[1.0]]),
            0.02,
            **{parameter: [value] if parameter == "cell_weights" else value},
        )
