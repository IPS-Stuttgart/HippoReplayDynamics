from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.encoding import EmissionConfig
from hipporeplayimm.score_metadata import emission_config_for_scores


@pytest.mark.parametrize(
    "value",
    [
        b"0.5",
        bytearray(b"0.5"),
        memoryview(b"0.5"),
        np.bytes_(b"0.5"),
    ],
)
def test_emission_metadata_decodes_byte_backed_numeric_scalars(value: object) -> None:
    scores = pd.DataFrame({"emission_spike_rate_scale": [value]})

    config = emission_config_for_scores(
        scores,
        EmissionConfig(spike_rate_scale=1.0),
    )

    assert config.spike_rate_scale == pytest.approx(0.5)


@pytest.mark.parametrize(
    "value",
    [
        b"NA",
        bytearray(b"NA"),
        memoryview(b"NA"),
        np.bytes_(b"NA"),
    ],
)
def test_emission_metadata_skips_byte_backed_missing_markers(value: object) -> None:
    scores = pd.DataFrame({"emission_spike_rate_scale": [value]})

    config = emission_config_for_scores(
        scores,
        EmissionConfig(spike_rate_scale=1.75),
    )

    assert config.spike_rate_scale == pytest.approx(1.75)
