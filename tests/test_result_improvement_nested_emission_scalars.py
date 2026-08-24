from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipporeplayimm.result_improvement_emission_validation import (
    _validate_replay_calibrated_emission_parameters,
)


def _nested_object_scalar(value: object) -> np.ndarray:
    inner = np.empty((), dtype=object)
    inner[()] = value
    outer = np.empty((), dtype=object)
    outer[()] = inner
    return outer


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        time_bin_s=0.02,
        spike_rate_scale=1.0,
        likelihood_temperature=1.0,
        negative_binomial_overdispersion=0.0,
        cell_weights=None,
    )


def _calibration() -> SimpleNamespace:
    return SimpleNamespace(
        gain_prior_count=10.0,
        max_gain=20.0,
        negative_binomial_dispersion=50.0,
    )


@pytest.mark.parametrize(
    "field",
    (
        "time_bin_s",
        "spike_rate_scale",
        "likelihood_temperature",
        "negative_binomial_overdispersion",
    ),
)
@pytest.mark.parametrize("bad_value", (True, np.bool_(True), "2.0", np.str_("2.0")))
def test_replay_emission_config_rejects_nested_semantic_scalars(
    field: str,
    bad_value: object,
) -> None:
    config = _config()
    setattr(config, field, _nested_object_scalar(bad_value))

    with pytest.raises(TypeError):
        _validate_replay_calibrated_emission_parameters(config, _calibration())


@pytest.mark.parametrize(
    "field",
    ("gain_prior_count", "max_gain", "negative_binomial_dispersion"),
)
@pytest.mark.parametrize("bad_value", (True, np.bool_(True), "2.0", np.str_("2.0")))
def test_replay_emission_calibration_rejects_nested_semantic_scalars(
    field: str,
    bad_value: object,
) -> None:
    calibration = _calibration()
    setattr(calibration, field, _nested_object_scalar(bad_value))

    with pytest.raises(TypeError):
        _validate_replay_calibrated_emission_parameters(_config(), calibration)


@pytest.mark.parametrize("bad_value", (True, np.bool_(True), "0.5", np.str_("0.5")))
def test_replay_emission_cell_weights_reject_nested_semantic_scalars(
    bad_value: object,
) -> None:
    config = _config()
    weights = np.empty(2, dtype=object)
    weights[0] = _nested_object_scalar(bad_value)
    weights[1] = 1.0
    config.cell_weights = weights

    with pytest.raises(TypeError):
        _validate_replay_calibrated_emission_parameters(config, _calibration())


def test_replay_emission_validation_preserves_nested_real_scalars() -> None:
    config = _config()
    config.time_bin_s = _nested_object_scalar(np.float32(0.02))
    config.spike_rate_scale = _nested_object_scalar(np.float64(1.25))
    config.likelihood_temperature = _nested_object_scalar(np.float32(0.75))
    config.negative_binomial_overdispersion = _nested_object_scalar(np.float64(0.1))
    weights = np.empty(2, dtype=object)
    weights[0] = _nested_object_scalar(np.float32(0.5))
    weights[1] = _nested_object_scalar(np.float64(1.5))
    config.cell_weights = weights

    calibration = _calibration()
    calibration.gain_prior_count = _nested_object_scalar(np.float32(5.0))
    calibration.max_gain = _nested_object_scalar(np.float64(10.0))
    calibration.negative_binomial_dispersion = _nested_object_scalar(np.float32(25.0))

    _validate_replay_calibrated_emission_parameters(config, calibration)
