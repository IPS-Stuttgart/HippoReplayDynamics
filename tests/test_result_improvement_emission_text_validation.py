from __future__ import annotations

import pytest

import hipporeplayimm
from hipporeplayimm.encoding import EmissionConfig
from hipporeplayimm.result_improvement_extensions import (
    ReplayEmissionCalibration,
    build_sorted_emissions_with_replay_calibration,
)


def test_replay_calibrated_emission_rejects_text_config_scalars_on_import():
    hipporeplayimm.apply_runtime_patches()
    config = EmissionConfig(time_bin_s="0.02")

    with pytest.raises(TypeError, match="time_bin_s"):
        build_sorted_emissions_with_replay_calibration(None, None, 0, config)


def test_replay_calibrated_emission_rejects_text_calibration_scalars_on_import():
    hipporeplayimm.apply_runtime_patches()
    calibration = ReplayEmissionCalibration(gain_prior_count="1.0")

    with pytest.raises(TypeError, match="gain_prior_count"):
        build_sorted_emissions_with_replay_calibration(
            None,
            None,
            0,
            EmissionConfig(),
            calibration,
        )


def test_replay_calibrated_emission_rejects_text_cell_weight_arrays_on_import():
    hipporeplayimm.apply_runtime_patches()
    config = EmissionConfig(cell_weights=["1.0"])

    with pytest.raises(TypeError, match="cell_weights"):
        build_sorted_emissions_with_replay_calibration(None, None, 0, config)
