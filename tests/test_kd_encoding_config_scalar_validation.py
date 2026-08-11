from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.kd_encoding_config_validation import validate_kd_encoding_config
from hipporeplayimm.kd_reference import KDEncodingConfig, fit_kd_place_field_encoding


def _object_scalar(value: object) -> np.ndarray:
    wrapper = np.empty((), dtype=object)
    wrapper[()] = value
    return wrapper


def _nested_object_scalar(value: object) -> np.ndarray:
    return _object_scalar(_object_scalar(value))


@pytest.mark.parametrize(
    "value",
    [
        0,
        1,
        "False",
        "True",
        np.array([True]),
        _nested_object_scalar(1),
        _nested_object_scalar("False"),
    ],
)
def test_kd_encoding_rejects_nonboolean_use_excitatory(value: object) -> None:
    config = KDEncodingConfig(use_excitatory=value)

    with pytest.raises(TypeError, match="use_excitatory"):
        validate_kd_encoding_config(config)


def test_kd_public_encoder_rejects_truthy_text_use_excitatory_before_fitting() -> None:
    config = KDEncodingConfig(use_excitatory="False")

    with pytest.raises(TypeError, match="use_excitatory"):
        fit_kd_place_field_encoding(object(), config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bin_size_cm", _nested_object_scalar(True)),
        ("smoothing_sigma_cm", _nested_object_scalar("2.0")),
        ("min_speed_cm_s", _object_scalar(np.array([1.0]))),
    ],
)
def test_kd_encoding_rejects_nested_malformed_float_scalars(
    field: str,
    value: object,
) -> None:
    config = KDEncodingConfig(**{field: value})

    with pytest.raises(TypeError, match=field):
        validate_kd_encoding_config(config)


def test_kd_encoding_accepts_nested_real_and_boolean_zero_dimensional_scalars() -> None:
    config = KDEncodingConfig(
        bin_size_cm=_nested_object_scalar(np.float64(4.0)),
        use_excitatory=_nested_object_scalar(np.bool_(False)),
    )

    validate_kd_encoding_config(config)


def test_kd_encoding_rejects_cyclic_zero_dimensional_scalar_wrapper() -> None:
    cyclic = np.empty((), dtype=object)
    cyclic[()] = cyclic
    config = KDEncodingConfig(bin_size_cm=cyclic)

    with pytest.raises(TypeError, match="bin_size_cm"):
        validate_kd_encoding_config(config)
