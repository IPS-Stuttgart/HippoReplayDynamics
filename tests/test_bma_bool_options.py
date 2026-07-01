from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.bma_options_patch import _coerce_bool_option


@pytest.mark.parametrize("value", [True, np.bool_(True), 1, 1.0, "1", "1.0", "true", "YES", "on"])
def test_bma_bool_option_accepts_true_values(value: object) -> None:
    assert _coerce_bool_option(value) is True


@pytest.mark.parametrize("value", [False, np.bool_(False), 0, 0.0, "0", "0.0", "false", "No", "off", "", None, np.nan])
def test_bma_bool_option_accepts_false_values(value: object) -> None:
    assert _coerce_bool_option(value) is False


@pytest.mark.parametrize("value", ["maybe", "truthy", "2", [True], np.array([True, False])])
def test_bma_bool_option_rejects_ambiguous_values(value: object) -> None:
    with pytest.raises(ValueError, match="boolean option"):
        _coerce_bool_option(value)
