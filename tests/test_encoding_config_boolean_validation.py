from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.clusterless import ClusterlessMarkConfig, fit_clusterless_mark_encoding
from hipporeplayimm.encoding import EncodingConfig, _validate_encoding_config, fit_place_field_encoding


@pytest.mark.parametrize("name", ["use_excitatory", "exclude_ripple_intervals"])
@pytest.mark.parametrize(
    "value",
    [
        "false",
        "true",
        0,
        1,
        None,
        np.array([False]),
        np.array(False, dtype=object),
    ],
)
def test_encoding_boolean_options_reject_non_boolean_values(name: str, value: object) -> None:
    config = EncodingConfig(**{name: value})

    with pytest.raises(ValueError, match=rf"{name} must be a boolean scalar"):
        _validate_encoding_config(config)


@pytest.mark.parametrize("name", ["use_excitatory", "exclude_ripple_intervals"])
@pytest.mark.parametrize(
    "value",
    [False, True, np.bool_(False), np.bool_(True), np.array(False), np.array(True)],
)
def test_encoding_boolean_options_accept_boolean_scalars(name: str, value: object) -> None:
    _validate_encoding_config(EncodingConfig(**{name: value}))


@pytest.mark.parametrize("name", ["use_excitatory", "exclude_ripple_intervals"])
def test_place_field_fit_rejects_invalid_boolean_before_reading_session(name: str) -> None:
    config = EncodingConfig(**{name: "false"})

    with pytest.raises(ValueError, match=rf"{name} must be a boolean scalar"):
        fit_place_field_encoding(object(), config)


@pytest.mark.parametrize("name", ["use_excitatory", "exclude_ripple_intervals"])
def test_clusterless_fit_uses_patched_nested_encoding_validator(name: str) -> None:
    encoding = EncodingConfig(**{name: "false"})
    config = ClusterlessMarkConfig(encoding=encoding)

    with pytest.raises(ValueError, match=rf"{name} must be a boolean scalar"):
        fit_clusterless_mark_encoding(object(), config)
