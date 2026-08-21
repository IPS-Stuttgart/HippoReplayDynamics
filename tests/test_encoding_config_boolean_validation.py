from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest

from hipporeplayimm.clusterless import ClusterlessMarkConfig, fit_clusterless_mark_encoding
from hipporeplayimm.encoding import EncodingConfig, _validate_encoding_config, fit_place_field_encoding
from hipporeplayimm.encoding_config_boolean_validation import _synchronize_validator_aliases


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

    with pytest.raises(TypeError, match=rf"{name} must be a boolean scalar"):
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

    with pytest.raises(TypeError, match=rf"{name} must be a boolean scalar"):
        fit_place_field_encoding(object(), config)


@pytest.mark.parametrize("name", ["use_excitatory", "exclude_ripple_intervals"])
def test_clusterless_fit_uses_patched_nested_encoding_validator(name: str) -> None:
    encoding = EncodingConfig(**{name: "false"})
    config = ClusterlessMarkConfig(encoding=encoding)

    with pytest.raises(TypeError, match=rf"{name} must be a boolean scalar"):
        fit_clusterless_mark_encoding(object(), config)


def test_encoding_validator_alias_sync_is_limited_to_package_namespace(monkeypatch) -> None:
    external = ModuleType("hipporeplayimm_extension")
    package_probe = ModuleType("hipporeplayimm._encoding_config_alias_probe")

    def previous(config):
        return config

    def patched(config):
        return config

    external._validate_encoding_config = previous
    package_probe._validate_encoding_config = previous
    monkeypatch.setitem(sys.modules, external.__name__, external)
    monkeypatch.setitem(sys.modules, package_probe.__name__, package_probe)

    _synchronize_validator_aliases(previous, patched)

    assert external._validate_encoding_config is previous
    assert package_probe._validate_encoding_config is patched
