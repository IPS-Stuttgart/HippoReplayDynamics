import pandas as pd
import pytest

from hipporeplayimm.encoding import EmissionConfig
from hipporeplayimm.score_metadata import emission_config_for_scores


def test_emission_metadata_rejects_near_conflicting_time_bins():
    scores = pd.DataFrame(
        {
            "emission_time_bin_s": [0.010000000001],
            "time_bin_s": [0.01],
        }
    )

    with pytest.raises(ValueError, match="emission_time_bin_s"):
        emission_config_for_scores(scores, EmissionConfig())
