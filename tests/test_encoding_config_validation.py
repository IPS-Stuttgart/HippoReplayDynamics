from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hipporeplayimm.encoding import EncodingConfig, _validate_encoding_config


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bin_size_cm", True),
        ("min_occupancy_s", "0.02"),
        ("rate_floor_hz", np.array([1e-4])),
        ("smoothing_sigma_bins", False),
        ("min_speed_cm_s", "0.0"),
        ("arena_padding_cm", np.array([2.0])),
    ],
)
def test_encoding_config_rejects_lossy_numeric_scalars(field: str, value: object) -> None:
    config = replace(EncodingConfig(), **{field: value})

    with pytest.raises((TypeError, ValueError), match=field):
        _validate_encoding_config(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("use_excitatory", "False"),
        ("exclude_ripple_intervals", 1),
        ("use_excitatory", np.array([True])),
    ],
)
def test_encoding_config_rejects_non_boolean_switch_values(field: str, value: object) -> None:
    config = replace(EncodingConfig(), **{field: value})

    with pytest.raises(TypeError, match=field):
        _validate_encoding_config(config)


def test_encoding_config_accepts_numpy_scalar_values() -> None:
    config = EncodingConfig(
        bin_size_cm=np.array(4.0),
        smoothing_sigma_bins=np.array(0.0),
        min_speed_cm_s=np.float64(5.0),
        min_occupancy_s=np.array(0.02),
        rate_floor_hz=np.float64(1e-4),
        arena_padding_cm=np.array(2.0),
        use_excitatory=np.bool_(False),
        exclude_ripple_intervals=np.array(True),
    )

    _validate_encoding_config(config)
