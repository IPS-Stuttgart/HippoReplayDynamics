from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest

from hipporeplayimm.position_decoding_config_validation import (
    _validated_position_decoding_config,
)
from hipporeplayimm.position_validation import PositionDecodingConfig


def test_position_decoding_preserves_decimal_integer_seed_exactly() -> None:
    seed = Decimal(2**53 + 1)

    normalized = _validated_position_decoding_config(
        PositionDecodingConfig(random_seed=seed)  # type: ignore[arg-type]
    )

    assert normalized.random_seed == int(seed)


def test_position_decoding_rejects_fractional_decimal_integer_controls() -> None:
    value = Decimal("9007199254740992.5")

    for field in (
        "n_folds",
        "max_windows_per_session",
        "random_seed",
        "min_spikes_per_window",
    ):
        config = replace(PositionDecodingConfig(), **{field: value})
        with pytest.raises(ValueError, match=rf"{field} must be an integer"):
            _validated_position_decoding_config(config)


def test_position_decoding_preserves_extended_precision_integer_seed() -> None:
    seed = np.longdouble(str(2**53 + 1))
    if seed == np.longdouble(2**53):
        pytest.skip("numpy.longdouble does not exceed binary64 precision")

    normalized = _validated_position_decoding_config(
        PositionDecodingConfig(random_seed=seed)  # type: ignore[arg-type]
    )

    assert normalized.random_seed == 2**53 + 1


def test_position_decoding_rejects_fractional_extended_precision_seed() -> None:
    seed = np.longdouble(2**53) + np.longdouble("0.5")
    if seed == np.floor(seed):
        pytest.skip("numpy.longdouble does not retain the fractional test value")

    with pytest.raises(ValueError, match="random_seed must be an integer"):
        _validated_position_decoding_config(
            PositionDecodingConfig(random_seed=seed)  # type: ignore[arg-type]
        )
