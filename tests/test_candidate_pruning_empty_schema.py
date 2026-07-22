from __future__ import annotations

import numpy as np

from hipporeplayimm.candidate_pruning_calibration import score_pruning_gaps
from hipporeplayimm.encoding import LogEmissionTensor

_EXPECTED_COLUMNS = [
    "model",
    "pruned_log_evidence",
    "full_candidate_log_evidence",
    "candidate_pruning_gap",
    "candidate_pruned_runtime_s",
    "candidate_full_runtime_s",
    "candidate_runtime_ratio_full_over_pruned",
    "n_time",
    "n_bins",
    "n_spikes",
    "mean_candidate_count",
]


def test_score_pruning_gaps_preserves_schema_without_candidate_models() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.zeros((1, 2), dtype=float),
        spike_counts=np.zeros((1, 0), dtype=int),
        times=np.array([0.0], dtype=float),
        dt=0.02,
        cell_ids=np.empty(0, dtype=int),
        n_spikes=0,
    )

    result = score_pruning_gaps(
        [object()],
        emissions,
        np.zeros((2, 2), dtype=float),
    )

    assert result.empty
    assert result.columns.tolist() == _EXPECTED_COLUMNS
