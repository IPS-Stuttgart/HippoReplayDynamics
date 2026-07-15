from __future__ import annotations

import numpy as np

from hipporeplayimm.position_decoding_config_validation import (
    _validated_position_decoding_config,
)
from hipporeplayimm.position_validation import PositionDecodingConfig


def test_position_decoding_preserves_large_integer_seed_exactly() -> None:
    seed = 2**53 + 1

    normalized = _validated_position_decoding_config(
        PositionDecodingConfig(random_seed=seed)
    )

    assert normalized.random_seed == seed
    assert np.random.default_rng(normalized.random_seed).integers(10) >= 0


def test_position_decoding_preserves_large_numeric_text_seed_exactly() -> None:
    seed = 2**53 + 1

    normalized = _validated_position_decoding_config(
        PositionDecodingConfig(random_seed=str(seed))  # type: ignore[arg-type]
    )

    assert normalized.random_seed == seed


def test_position_decoding_accepts_arbitrary_size_native_integer_seed() -> None:
    seed = 10**400

    normalized = _validated_position_decoding_config(
        PositionDecodingConfig(random_seed=seed)
    )

    assert normalized.random_seed == seed
    assert np.random.default_rng(normalized.random_seed).integers(10) >= 0
