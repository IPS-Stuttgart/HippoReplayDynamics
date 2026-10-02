"""Known generator and Monte Carlo accounting, not biological calibration."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import stress_run_pair_rate_controls as stress


def protocol():
    return json.loads((Path(__file__).parents[1] / "docs/run_pair_rate_control_stress_protocol.json").read_text())


def test_intensity_depends_on_source_lfp_reference_not_a_spike_derived_phase():
    theta = np.column_stack((np.zeros(4), np.full(4, np.pi)))
    first = stress.known_intensity(theta, [0, 0], "pre", .001, protocol())
    second = stress.known_intensity(theta, [1, 0], "pre", .001, protocol())
    assert second[0, 0] < first[0, 0]
    np.testing.assert_array_equal(first[:, 1], second[:, 1])
    with pytest.raises(ValueError, match="reference"):
        stress.known_intensity(theta, [2, 0], "pre", .001, protocol())


def test_changed_tuning_is_an_intensity_change_not_an_interaction():
    theta = np.zeros((10, 1))
    pre = stress.known_intensity(theta, [0, 0, 0, 0], "pre", .001, protocol())
    post = stress.known_intensity(theta, [0, 0, 0, 0], "post", .001, protocol())
    assert np.all(pre > 0) and np.all(post > 0)
    np.testing.assert_array_equal(pre[:, 1::2], post[:, 1::2])
    assert not np.array_equal(pre[:, ::2], post[:, ::2])


def test_incomplete_monte_carlo_family_is_rejected():
    frame = pd.DataFrame([{"animal": "A", "session": "S", "pause_id": "P", "unit_a": 0,
                           "unit_b": 1, "replicate": 0, "fitted_change": 0., "oracle_change": 0., "discrepancy": 0.}])
    with pytest.raises(ValueError, match="Incomplete"):
        stress.summarize(frame, 2)


def test_zero_oracle_and_large_nuisance_bias_are_separate_summaries():
    frame = pd.DataFrame([{"animal": "A", "session": "S", "pause_id": "P", "unit_a": 0,
                           "unit_b": 1, "replicate": j, "fitted_change": .1,
                           "oracle_change": 0., "discrepancy": .1} for j in range(20)])
    summary = stress.summarize(frame, 20).iloc[0]
    assert summary.systematic_nuisance_bias_detected
    assert not summary.biological_inference and summary.mean_oracle_change == 0
