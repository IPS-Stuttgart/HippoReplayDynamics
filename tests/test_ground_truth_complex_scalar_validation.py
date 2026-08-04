from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipporeplayimm.ground_truth_float_metadata import (
    _parse_bool_metadata_value,
    _parse_config_scalar,
    _parse_float_metadata_value,
    _validate_ground_truth_config,
)


def _nested_zero_dimensional_scalar(value: object) -> np.ndarray:
    wrapped = np.empty((), dtype=object)
    wrapped[()] = np.asarray(value)
    return wrapped


def _extended_complex_scalar_cases() -> list[object]:
    return [
        np.clongdouble(1.0 + 2.0j),
        np.clongdouble(1.0 + 0.0j),
        np.array(np.clongdouble(1.0 + 2.0j)),
        _nested_zero_dimensional_scalar(np.clongdouble(1.0 + 2.0j)),
    ]


@pytest.mark.parametrize("value", _extended_complex_scalar_cases())
def test_ground_truth_config_scalar_rejects_extended_complex_values(value: object) -> None:
    with pytest.raises(TypeError, match="visit_radius_cm.*complex"):
        _parse_config_scalar("visit_radius_cm", value)


@pytest.mark.parametrize("value", _extended_complex_scalar_cases())
def test_ground_truth_config_validation_rejects_extended_complex_values(value: object) -> None:
    config = SimpleNamespace(
        well_arrival_window_s=1.0,
        visit_radius_cm=value,
        min_dwell_s=0.2,
        future_horizon_s=30.0,
    )

    with pytest.raises(TypeError, match="visit_radius_cm.*complex"):
        _validate_ground_truth_config(config)


@pytest.mark.parametrize("value", _extended_complex_scalar_cases())
def test_ground_truth_float_metadata_rejects_extended_complex_values(value: object) -> None:
    with pytest.raises(ValueError, match="encoding_bin_size_cm.*real"):
        _parse_float_metadata_value("encoding_bin_size_cm", value)


@pytest.mark.parametrize("value", _extended_complex_scalar_cases())
def test_ground_truth_boolean_metadata_rejects_extended_complex_values(value: object) -> None:
    with pytest.raises(ValueError, match="evidence_comparable.*complex"):
        _parse_bool_metadata_value("evidence_comparable", value)


def test_ground_truth_scalar_parsers_preserve_extended_precision_real_values() -> None:
    real_value = np.longdouble("1.25")
    nested_real = _nested_zero_dimensional_scalar(real_value)

    assert _parse_config_scalar("visit_radius_cm", real_value) == pytest.approx(1.25)
    assert _parse_config_scalar("visit_radius_cm", nested_real) == pytest.approx(1.25)
    assert _parse_float_metadata_value("encoding_bin_size_cm", real_value) == pytest.approx(1.25)
    assert _parse_float_metadata_value("encoding_bin_size_cm", nested_real) == pytest.approx(1.25)
    assert _parse_bool_metadata_value("evidence_comparable", np.longdouble(1.0)) is True
