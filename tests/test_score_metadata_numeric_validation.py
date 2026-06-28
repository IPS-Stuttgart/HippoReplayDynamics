from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import hipporeplayimm.clusterless_ground_truth as clusterless_ground_truth
from hipporeplayimm.score_metadata import _unique_float_from_columns, _unique_int_from_columns


def test_score_metadata_rejects_boolean_float_metadata_with_column_context() -> None:
    scores = pd.DataFrame({"encoding_bin_size_cm": [True]})

    with pytest.raises(ValueError, match="encoding_bin_size_cm.*finite numeric"):
        _unique_float_from_columns(scores, ("encoding_bin_size_cm",), default=4.0)


def test_score_metadata_rejects_boolean_integer_metadata_with_column_context() -> None:
    scores = pd.DataFrame({"state_space_momentum_candidate_top_k": [np.bool_(False)]})

    with pytest.raises(ValueError, match="state_space_momentum_candidate_top_k.*finite numeric"):
        _unique_int_from_columns(scores, ("state_space_momentum_candidate_top_k",), default=128)


def test_score_metadata_rejects_malformed_numeric_metadata_with_column_context() -> None:
    scores = pd.DataFrame({"emission_time_bin_s": ["not-a-number"]})

    with pytest.raises(ValueError, match="emission_time_bin_s.*finite numeric"):
        _unique_float_from_columns(scores, ("emission_time_bin_s",), default=0.02)


def test_clusterless_optional_float_rejects_boolean_numeric_metadata_with_column_context() -> None:
    scores = pd.DataFrame({"clusterless_mark_kde_bandwidth": [np.bool_(True)]})

    with pytest.raises(ValueError, match="clusterless_mark_kde_bandwidth.*finite numeric"):
        clusterless_ground_truth._optional_float_from_columns(scores, ("clusterless_mark_kde_bandwidth",), default=None)
