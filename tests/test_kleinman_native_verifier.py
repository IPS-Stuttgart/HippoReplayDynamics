from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.audit_kleinman_native_adequacy import diagnose, saturated_log_likelihood
from scripts.verify_kleinman_native_adequacy import compare_scores, independent_counts, reference_diagnostic, saturation


def fixture():
    q = np.array([[[0.7, 0.2, 0.1]] * 6, [[0.2, 0.7, 0.1]] * 6, [[0.7, 0.2, 0.1]] * 3 + [[0.2, 0.7, 0.1]] * 3])
    templates = pd.DataFrame({"extent": [0, 0, 1], "start": [0, 1, 0], "end": [0, 1, 1], "direction": [0, 1, 0], "profile": ["static", "static", "linear"]})
    counts = np.array([[7, 2, 1], [9, 0, 1], [6, 3, 1], [1, 8, 1], [3, 6, 1], [2, 7, 1]])
    return counts, np.log(q), templates


def test_histogram_reconstruction_excludes_final_edge():
    trains = [np.array([0, 0.005, 0.01, 0.019, 0.02, 0.03, 0.032])]
    counts = independent_counts(trains, 0, 0.035, 3)
    assert counts[:, 0].tolist() == [2, 2, 1]


def test_scalar_bootstrap_and_crossfit_match_producer():
    counts, logq, templates = fixture()
    expected = diagnose(counts, logq, templates, np.random.default_rng(17))
    ref = reference_diagnostic(counts, logq, expected["best_template_index"], np.random.default_rng(17))
    for key, value in ref.items():
        assert value == pytest.approx(expected[key], abs=1e-9)
    assert saturation(counts) == pytest.approx(saturated_log_likelihood(counts))


def test_checker_detects_changed_heldout_score():
    counts, logq, templates = fixture()
    data = diagnose(counts, logq, templates, np.random.default_rng(17))
    row = SimpleNamespace(animal="a", session="s", **data)
    compare_scores(row, counts, logq, templates, np.random.default_rng(17))
    row.crossfit_template_minus_free += 1
    with pytest.raises(AssertionError):
        compare_scores(row, counts, logq, templates, np.random.default_rng(17))


def test_csv_time_round_trip_preserves_spikes_on_bin_edges():
    import io

    from scripts.verify_kleinman_native_adequacy import read_csv

    start = np.float64("402.80513333333334")
    csv = pd.DataFrame({"start_s": [start]}).to_csv(index=False)
    restored = read_csv(io.StringIO(csv)).start_s.iloc[0]
    assert restored == start
    spikes = [start + np.arange(10) * 0.01]
    np.testing.assert_array_equal(independent_counts(spikes, restored, restored + 0.1, 10), independent_counts(spikes, start, start + 0.1, 10))
