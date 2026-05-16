import pandas as pd
import pytest

from hipporeplayimm.encoding import EmissionConfig, EncodingConfig
from hipporeplayimm.ground_truth import _emission_config_for_scores, _encoding_config_for_scores


def test_score_metadata_accepts_model_evidence_column_aliases():
    scores = pd.DataFrame(
        {
            "bin_size_cm": [6.0],
            "smoothing_sigma_bins": [2.25],
            "min_speed_cm_s": [7.5],
            "time_bin_s": [0.015],
        }
    )

    encoding_config = _encoding_config_for_scores(
        scores,
        EncodingConfig(
            bin_size_cm=1.0,
            smoothing_sigma_bins=1.0,
            min_speed_cm_s=1.0,
        ),
    )
    emission_config = _emission_config_for_scores(
        scores,
        EmissionConfig(time_bin_s=0.02),
    )

    assert encoding_config.bin_size_cm == pytest.approx(6.0)
    assert encoding_config.smoothing_sigma_bins == pytest.approx(2.25)
    assert encoding_config.min_speed_cm_s == pytest.approx(7.5)
    assert emission_config.time_bin_s == pytest.approx(0.015)


def test_score_metadata_rejects_conflicting_canonical_and_legacy_values():
    scores = pd.DataFrame(
        {
            "encoding_bin_size_cm": [4.0],
            "bin_size_cm": [6.0],
        }
    )

    with pytest.raises(ValueError, match="encoding_bin_size_cm"):
        _encoding_config_for_scores(scores, EncodingConfig())
