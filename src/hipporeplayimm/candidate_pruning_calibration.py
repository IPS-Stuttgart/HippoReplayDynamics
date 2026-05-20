"""Exact-vs-pruned candidate-support calibration utilities."""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import pandas as pd

from .encoding import LogEmissionTensor


def full_grid_candidate_indices(emissions: LogEmissionTensor) -> list[np.ndarray]:
    return [np.arange(emissions.n_bins, dtype=int) for _ in range(emissions.n_time)]


def score_pruning_gap(model, emissions: LogEmissionTensor, bin_centers: np.ndarray, candidate_indices: list[np.ndarray]) -> dict[str, object]:
    """Score a candidate-pruned model against full-grid candidate support."""

    if not hasattr(model, "score"):
        raise TypeError("model must expose a score method")
    start = time.perf_counter()
    pruned = model.score(emissions, bin_centers, candidate_indices=candidate_indices)
    pruned_runtime = time.perf_counter() - start
    start = time.perf_counter()
    full = model.score(emissions, bin_centers, candidate_indices=full_grid_candidate_indices(emissions))
    full_runtime = time.perf_counter() - start
    return {
        "model": pruned.model_name,
        "pruned_log_evidence": float(pruned.log_likelihood),
        "full_candidate_log_evidence": float(full.log_likelihood),
        "candidate_pruning_gap": float(full.log_likelihood - pruned.log_likelihood),
        "candidate_pruned_runtime_s": float(pruned_runtime),
        "candidate_full_runtime_s": float(full_runtime),
        "candidate_runtime_ratio_full_over_pruned": float(full_runtime / pruned_runtime) if pruned_runtime > 0.0 else np.inf,
        "n_time": int(emissions.n_time),
        "n_bins": int(emissions.n_bins),
        "n_spikes": int(emissions.n_spikes),
        "mean_candidate_count": float(np.mean([len(curr) for curr in candidate_indices])),
    }


def score_pruning_gaps(models: Iterable[object], emissions: LogEmissionTensor, bin_centers: np.ndarray) -> pd.DataFrame:
    rows = []
    for model in models:
        if not hasattr(model, "candidate_indices"):
            continue
        try:
            candidates = model.candidate_indices(emissions, bin_centers)
        except TypeError:
            candidates = model.candidate_indices(emissions)
        rows.append(score_pruning_gap(model, emissions, bin_centers, candidates))
    return pd.DataFrame(rows)
