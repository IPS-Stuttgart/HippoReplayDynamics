from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.clusterless import ClusterlessMarkConfig, fit_clusterless_mark_encoding
from hipporeplayimm.clusterless_config_validation import _validate_clusterless_mark_config


@pytest.mark.parametrize(
    "value",
    [
        "false",
        "true",
        0,
        1,
        np.array([False]),
        np.array(False, dtype=object),
    ],
)
def test_clusterless_use_excitatory_rejects_truthy_non_booleans(value: object) -> None:
    config = ClusterlessMarkConfig(use_excitatory=value)

    with pytest.raises(ValueError, match="use_excitatory must be a boolean scalar"):
        _validate_clusterless_mark_config(config)


@pytest.mark.parametrize(
    "value",
    [False, True, np.bool_(False), np.bool_(True), np.array(False), np.array(True)],
)
def test_clusterless_use_excitatory_accepts_boolean_scalars(value: object) -> None:
    _validate_clusterless_mark_config(ClusterlessMarkConfig(use_excitatory=value))


def test_clusterless_fit_rejects_invalid_use_excitatory_before_reading_session() -> None:
    config = ClusterlessMarkConfig(use_excitatory="false")

    with pytest.raises(ValueError, match="use_excitatory must be a boolean scalar"):
        fit_clusterless_mark_encoding(object(), config)
