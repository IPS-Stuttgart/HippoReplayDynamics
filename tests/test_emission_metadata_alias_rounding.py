from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.encoding import EmissionConfig
from hipporeplayimm.score_metadata import emission_config_for_scores


def test_emission_config_accepts_equivalent_float32_and_float64_aliases() -> None:
    scores = pd.DataFrame(
        {
            "emission_time_bin_s": pd.Series([np.float32(0.1)], dtype=object),
            "time_bin_s": pd.Series([0.1], dtype=object),
            "emission_likelihood_temperature": pd.Series(
                [np.float32(0.8)], dtype=object
            ),
            "likelihood_temperature": pd.Series([0.8], dtype=object),
            "emission_negative_binomial_overdispersion": pd.Series(
                [np.float32(0.2)], dtype=object
            ),
            "negative_binomial_overdispersion": pd.Series([0.2], dtype=object),
        }
    )

    config = emission_config_for_scores(scores, EmissionConfig())

    assert config.time_bin_s == pytest.approx(0.1)
    assert config.likelihood_temperature == pytest.approx(0.8)
    assert config.negative_binomial_overdispersion == pytest.approx(0.2)


def test_emission_config_still_rejects_materially_different_aliases() -> None:
    scores = pd.DataFrame(
        {
            "emission_time_bin_s": [0.1],
            "time_bin_s": [0.11],
        }
    )

    with pytest.raises(ValueError, match="contains multiple values"):
        emission_config_for_scores(scores, EmissionConfig())
