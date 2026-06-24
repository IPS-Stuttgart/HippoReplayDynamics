from __future__ import annotations

import pandas as pd
import pytest

from hipporeplayimm.encoding import EncodingConfig
from hipporeplayimm.score_metadata import encoding_config_for_scores


def test_score_metadata_bool_rejects_nonbinary_numeric_values() -> None:
    for raw in ("2", "2.0", "-1", "0.5", 2, -1, 0.5):
        with pytest.raises(ValueError, match="cannot parse boolean value"):
            encoding_config_for_scores(
                pd.DataFrame({"encoding_use_excitatory": [raw]}),
                EncodingConfig(),
            )


def test_score_metadata_bool_accepts_binary_numeric_values() -> None:
    for raw in ("1", "1.0", 1, 1.0):
        config = encoding_config_for_scores(
            pd.DataFrame({"encoding_use_excitatory": [raw]}),
            EncodingConfig(use_excitatory=False),
        )
        assert config.use_excitatory is True

    for raw in ("0", "0.0", 0, 0.0):
        config = encoding_config_for_scores(
            pd.DataFrame({"encoding_use_excitatory": [raw]}),
            EncodingConfig(use_excitatory=True),
        )
        assert config.use_excitatory is False


def test_score_metadata_bool_preserves_named_boolean_values() -> None:
    for raw in ("true", "yes", "on", True):
        config = encoding_config_for_scores(
            pd.DataFrame({"encoding_use_excitatory": [raw]}),
            EncodingConfig(use_excitatory=False),
        )
        assert config.use_excitatory is True

    for raw in ("false", "no", "off", False):
        config = encoding_config_for_scores(
            pd.DataFrame({"encoding_use_excitatory": [raw]}),
            EncodingConfig(use_excitatory=True),
        )
        assert config.use_excitatory is False
