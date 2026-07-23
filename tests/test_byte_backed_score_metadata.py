from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.encoding import EncodingConfig
from hipporeplayimm.score_metadata import encoding_config_for_scores


@pytest.mark.parametrize(
    "value",
    [
        b"6.0",
        bytearray(b"6.0"),
        memoryview(b"6.0"),
        np.bytes_(b"6.0"),
    ],
)
def test_encoding_metadata_decodes_byte_backed_numeric_scalars(value: object) -> None:
    scores = pd.DataFrame({"encoding_bin_size_cm": [value]})

    config = encoding_config_for_scores(
        scores,
        EncodingConfig(bin_size_cm=4.0),
    )

    assert config.bin_size_cm == pytest.approx(6.0)


@pytest.mark.parametrize(
    "value",
    [
        b"1",
        bytearray(b"1"),
        memoryview(b"1"),
        np.bytes_(b"1"),
    ],
)
def test_encoding_metadata_decodes_byte_backed_boolean_scalars(value: object) -> None:
    scores = pd.DataFrame({"encoding_use_excitatory": [value]})

    config = encoding_config_for_scores(
        scores,
        EncodingConfig(use_excitatory=False),
    )

    assert config.use_excitatory is True


def test_encoding_metadata_skips_byte_backed_missing_markers() -> None:
    fallback = EncodingConfig(bin_size_cm=7.5, use_excitatory=False)
    scores = pd.DataFrame(
        {
            "encoding_bin_size_cm": [memoryview(b"NA")],
            "encoding_use_excitatory": [bytearray(b"null")],
        }
    )

    config = encoding_config_for_scores(scores, fallback)

    assert config.bin_size_cm == pytest.approx(fallback.bin_size_cm)
    assert config.use_excitatory is fallback.use_excitatory
