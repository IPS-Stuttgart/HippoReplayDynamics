from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.encoding import EncodingConfig
from hipporeplayimm.score_metadata import encoding_config_for_scores
from hipporeplayimm.score_metadata_bool_validation import _unique_string_from_columns


def _object_frame(column: str, value: object) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series([value], dtype=object)})


@pytest.mark.parametrize(
    "value",
    [
        np.array([b"6.0"]),
        [bytearray(b"6.0")],
        (memoryview(b"6.0"),),
        [np.array([np.float64(6.0)])],
    ],
)
def test_score_metadata_unwraps_singleton_numeric_values(value: object) -> None:
    config = encoding_config_for_scores(
        _object_frame("encoding_bin_size_cm", value),
        EncodingConfig(bin_size_cm=4.0),
    )

    assert config.bin_size_cm == pytest.approx(6.0)


@pytest.mark.parametrize(
    "value, expected",
    [
        (np.array([b"1"]), True),
        ([np.bool_(True)], True),
        ((0.0,), False),
        ([np.array([b"false"])], False),
    ],
)
def test_score_metadata_unwraps_singleton_boolean_values(value: object, expected: bool) -> None:
    config = encoding_config_for_scores(
        _object_frame("encoding_use_excitatory", value),
        EncodingConfig(use_excitatory=not expected),
    )

    assert config.use_excitatory is expected


@pytest.mark.parametrize("value", [np.array([np.nan]), [b"NA"], (pd.NA,)])
def test_score_metadata_unwraps_singleton_missing_values(value: object) -> None:
    fallback = EncodingConfig(bin_size_cm=7.5)

    config = encoding_config_for_scores(
        _object_frame("encoding_bin_size_cm", value),
        fallback,
    )

    assert config.bin_size_cm == pytest.approx(fallback.bin_size_cm)


def test_score_metadata_unwraps_singleton_string_values() -> None:
    frame = _object_frame("state_space_momentum_candidate_source", [np.array([b"emission"])])

    value = _unique_string_from_columns(
        frame,
        ("state_space_momentum_candidate_source",),
        "default",
    )

    assert value == "emission"


@pytest.mark.parametrize(
    "value",
    [
        np.array([b"6.0", b"7.0"]),
        [b"6.0", b"7.0"],
        (),
    ],
)
def test_score_metadata_rejects_nonscalar_containers(value: object) -> None:
    with pytest.raises(ValueError, match="must be scalar"):
        encoding_config_for_scores(
            _object_frame("encoding_bin_size_cm", value),
            EncodingConfig(),
        )
