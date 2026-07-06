from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm.encoding import EmissionConfig
from hipporeplayimm.result_improvement_extensions import (
    ReplayEmissionCalibration,
    build_sorted_emissions_with_replay_calibration,
)


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (EmissionConfig(time_bin_s=True), "time_bin_s"),
        (EmissionConfig(spike_rate_scale=np.bool_(True)), "spike_rate_scale"),
        (
            EmissionConfig(likelihood_temperature=np.asarray(True, dtype=object)),
            "likelihood_temperature",
        ),
        (
            EmissionConfig(negative_binomial_overdispersion=True),
            "negative_binomial_overdispersion",
        ),
        (EmissionConfig(cell_weights=np.asarray([True])), "cell_weights"),
        (EmissionConfig(time_bin_s=np.asarray([0.02])), "time_bin_s"),
    ],
)
def test_replay_calibrated_emissions_reject_invalid_config_scalars(config: object, match: str) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match=match):
        build_sorted_emissions_with_replay_calibration(
            object(),
            object(),
            object(),
            config=config,
        )


@pytest.mark.parametrize(
    ("calibration", "match"),
    [
        (ReplayEmissionCalibration(gain_prior_count=True), "gain_prior_count"),
        (
            ReplayEmissionCalibration(gain_prior_count=np.asarray([10.0])),
            "gain_prior_count",
        ),
        (
            ReplayEmissionCalibration(negative_binomial_dispersion=np.bool_(True)),
            "negative_binomial_dispersion",
        ),
        (
            ReplayEmissionCalibration(negative_binomial_dispersion=np.asarray(True, dtype=object)),
            "negative_binomial_dispersion",
        ),
    ],
)
def test_replay_calibrated_emissions_reject_invalid_calibration_scalars(
    calibration: ReplayEmissionCalibration,
    match: str,
) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match=match):
        build_sorted_emissions_with_replay_calibration(
            object(),
            object(),
            object(),
            calibration=calibration,
        )
