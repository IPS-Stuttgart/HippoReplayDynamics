"""Pairing, animal weighting and explicit missing gradient denominators."""

import numpy as np
import pandas as pd
import pytest

from scripts.report_replay_coverage_map_mismatch import aggregate, gradient_responses, paired_map_effect
from scripts.simulate_replay_coverage_map_mismatch import CONDITION, IDENTITY


def test_equal_animal_weighting_not_event_or_session_weighting():
    rows = [{"dataset": "test", "animal": "many", "session": f"S{i}", "direction": 0, "value": 10.} for i in range(10)]
    rows.append({"dataset": "test", "animal": "one", "session": "S1", "direction": 0, "value": 0.})
    _, animals, summary = aggregate(pd.DataFrame(rows), ["dataset"], ["value"])
    assert len(animals) == 2 and summary["mean"].item() == 5.
    assert summary.animals_available.item() == 2


def test_pair_map_effect_refuses_missing_condition_and_keeps_nan():
    frame = pd.DataFrame({"dataset": ["test"]*4, "animal": ["R1"]*4, "session": ["S1"]*4,
        "direction": [0, 0, 1, 1], "decoder_map": ["generator_known", "independent_RUN_half"]*2,
        "value": [2., 1., np.nan, 3.]})
    pair = paired_map_effect(frame, IDENTITY+["decoder_map"], ["value"])
    assert pair.value.iloc[0] == -1 and np.isnan(pair.value.iloc[1])
    with pytest.raises(ValueError, match="unpaired"):
        paired_map_effect(frame.iloc[:-1], IDENTITY+["decoder_map"], ["value"])


def test_gradient_response_requires_both_sides_and_preserves_denominator():
    identity = dict(zip(IDENTITY, ["test", "R1", "S1", 0], strict=True))
    condition = dict(zip(CONDITION, ["poisson", 1., "generator_known", "poisson", "posterior_mean", "unfiltered"], strict=True))
    rows = []
    for direction in [0, 1]:
        for gradient in [-.5, 0., .5]:
            rows.append({**identity, **condition, "direction": direction, "gradient": gradient,
                "selection": "all", "coordinate": "true_coordinate", "readout": "decoded",
                "normalized_slope": gradient*.4 if direction == 0 or gradient < 0 else np.nan})
    response = gradient_responses(pd.DataFrame(rows))
    assert len(response) == 2 and response.both_gradients_available.sum() == 1
    assert response.gradient_response.iloc[0] == .4
    assert np.isnan(response.gradient_response.iloc[1])
