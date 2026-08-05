from __future__ import annotations

from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from hipporeplayimm.candidate_pruning_calibration import score_pruning_gap


class _Emissions:
    n_time = 1
    n_bins = 2
    n_spikes = 0


class _SequenceScoreModel:
    def __init__(self, values: list[object]) -> None:
        self._values = iter(values)

    def score(self, emissions, bin_centers, *, candidate_indices):
        del emissions, bin_centers, candidate_indices
        return SimpleNamespace(
            model_name="candidate",
            log_likelihood=next(self._values),
        )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([np.complex128(1.0 + 4.0j), 2.0], "pruned log_likelihood"),
        (
            [1.0, np.array(np.complex128(2.0 + 5.0j), dtype=object)],
            "full-candidate log_likelihood",
        ),
    ],
)
def test_score_pruning_gap_rejects_complex_evidence(values, message):
    model = _SequenceScoreModel(values)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=message):
            score_pruning_gap(
                model,
                _Emissions(),
                np.zeros((2, 1), dtype=float),
                [np.array([0], dtype=int)],
            )


def test_score_pruning_gap_keeps_finite_real_scalars():
    result = score_pruning_gap(
        _SequenceScoreModel([np.float64(1.5), "2.25"]),
        _Emissions(),
        np.zeros((2, 1), dtype=float),
        [np.array([0], dtype=int)],
    )

    assert result["pruned_log_evidence"] == 1.5
    assert result["full_candidate_log_evidence"] == 2.25
    assert result["candidate_pruning_gap"] == 0.75
