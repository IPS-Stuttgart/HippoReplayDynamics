import numpy as np
import pytest

from hipporeplayimm.clusterless import build_clusterless_mark_emissions
from hipporeplayimm.encoding import EmissionConfig


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (EmissionConfig(spike_rate_scale=True), "spike_rate_scale"),
        (EmissionConfig(likelihood_temperature=True), "likelihood_temperature"),
        (
            EmissionConfig(negative_binomial_overdispersion=False),
            "negative_binomial_overdispersion",
        ),
        (EmissionConfig(spike_rate_scale=np.array([1.0])), "spike_rate_scale"),
    ],
)
def test_clusterless_emission_config_rejects_lossy_numeric_values(config, match):
    with pytest.raises(ValueError, match=match):
        build_clusterless_mark_emissions(object(), object(), 0, config)
