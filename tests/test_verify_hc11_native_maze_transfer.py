import itertools

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from scripts import audit_hc11_native_maze_transfer as producer
from scripts import verify_hc11_native_maze_transfer as verifier


@pytest.mark.parametrize("target", [0, 3, 10, 30])
def test_independently_regenerated_cap(target):
    counts = np.array([[1, 2, 0], [2, 1, 4]])
    assert_array_equal(producer.capped_counts(counts, target, 85)[0], verifier.independent_cap(counts, target, 85))


def test_separate_known_probabilities():
    rates = [np.array([[1.0, 2.0], [3.0, 1.0], [2.0, 8.0]])]
    args = (
        np.array([[1, 2, 1], [0, 2, 1]]),
        np.array([0, 0.02, 0.04]),
        rates,
        rates[0],
        np.array([2.0, 1.0]),
        np.array([1, 2]),
        np.array([1.0, 3.0]),
        np.array([1, 1]),
        np.array([0.0, 2.0, 4.0]),
        np.array([1, 0]),
    )
    a, b = producer.behavior_scores(*args), verifier.known_scores(*args)
    for name in a:
        assert_allclose(a[name], b[name], atol=1e-12)


def test_map_ties_and_rejected_wrong_error():
    q = np.array([[0.5, 0.5], [0.1, 0.9]])
    for error in (0.0, 2.0):
        assert verifier.verify_map_error(q, np.array([0.0, 4.0]), np.array([0.0, 4.0]), "linear", 8.0, error) == 1
    with pytest.raises(ValueError):
        verifier.verify_map_error(q, np.array([0.0, 4.0]), np.array([0.0, 4.0]), "linear", 8.0, 10.0)


def synthetic_scores():
    rows = []
    for regime, count, variant, rat, fold, window, split in itertools.product(
        producer.UNIT_REGIMES, producer.COUNT_REGIMES, producer.frozen.VARIANTS, range(4), range(2), range(2), range(5)
    ):
        value = (rat + 1) / 10 + (fold + window) / 30 + split / 100
        rows.append(
            {
                "unit_regime": regime,
                "count_regime": count,
                "encoding_variant": variant,
                "rat": f"r{rat}",
                "session": f"s{rat}",
                "fold": fold,
                "window_id": window,
                "split": split,
                "n_heldout_spikes": 2,
                "score_global": -20.0,
                "score_iid_position": -19.0,
                "score_first_order_imm": -19.0 + value,
                "score_static_location": -19.5,
                "score_diffusion": -19.0 + value / 2,
                "score_behavior": -18.0,
                "score_behavior_wrong": -25.0,
            }
        )
    return pd.DataFrame(rows)


def test_independent_summary_and_corruption_detection(tmp_path):
    scores = synthetic_scores()
    paired, events = producer.event_contrasts(scores)
    summary, _, _, _ = producer.summaries(events)
    paired.to_csv(tmp_path / "maze_transfer_split_contrasts.csv", index=False)
    events.to_csv(tmp_path / "maze_transfer_window_contrasts.csv", index=False)
    summary.to_csv(tmp_path / "maze_transfer_summary.csv", index=False)
    table, audit = verifier.verify_aggregates(scores, tmp_path)
    assert len(table) == audit["aggregate_intervals"] == 56
    summary.loc[0, "ci_low"] -= 0.1
    summary.to_csv(tmp_path / "maze_transfer_summary.csv", index=False)
    with pytest.raises(ValueError):
        verifier.verify_aggregates(scores, tmp_path)
