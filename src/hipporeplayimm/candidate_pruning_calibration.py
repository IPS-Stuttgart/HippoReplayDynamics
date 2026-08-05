"""Exact-vs-pruned candidate-support calibration utilities."""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import pandas as pd

from .encoding import LogEmissionTensor
from .result_improvement_extensions import _call_candidate_indices_compat

_CANDIDATE_PRUNING_GAP_COLUMNS = (
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
)


def _finite_real_score(name: str, value: object) -> float:
    """Return a finite real scalar without silently discarding complex parts."""

    current = value
    seen_arrays: set[int] = set()
    while isinstance(current, np.ndarray):
        if current.ndim != 0:
            raise ValueError(f"{name} must be a finite real scalar")
        marker = id(current)
        if marker in seen_arrays:
            raise ValueError(f"{name} must be a finite real scalar")
        seen_arrays.add(marker)
        current = current.item()

    try:
        scalar = np.asarray(current)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if scalar.ndim != 0 or np.issubdtype(scalar.dtype, np.complexfloating):
        raise ValueError(f"{name} must be a finite real scalar")
    item = scalar.item()
    if isinstance(item, (complex, np.complexfloating)):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        numeric = float(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be a finite real scalar")
    return numeric


def full_grid_candidate_indices(emissions: LogEmissionTensor) -> list[np.ndarray]:
    return [np.arange(emissions.n_bins, dtype=int) for _ in range(emissions.n_time)]


def score_pruning_gap(
    model,
    emissions: LogEmissionTensor,
    bin_centers: np.ndarray,
    candidate_indices: list[np.ndarray],
) -> dict[str, object]:
    """Score a candidate-pruned model against full-grid candidate support."""

    if not hasattr(model, "score"):
        raise TypeError("model must expose a score method")
    start = time.perf_counter()
    pruned = model.score(emissions, bin_centers, candidate_indices=candidate_indices)
    pruned_runtime = time.perf_counter() - start
    start = time.perf_counter()
    full = model.score(
        emissions,
        bin_centers,
        candidate_indices=full_grid_candidate_indices(emissions),
    )
    full_runtime = time.perf_counter() - start
    pruned_log_evidence = _finite_real_score(
        "pruned log_likelihood",
        pruned.log_likelihood,
    )
    full_log_evidence = _finite_real_score(
        "full-candidate log_likelihood",
        full.log_likelihood,
    )
    return {
        "model": pruned.model_name,
        "pruned_log_evidence": pruned_log_evidence,
        "full_candidate_log_evidence": full_log_evidence,
        "candidate_pruning_gap": full_log_evidence - pruned_log_evidence,
        "candidate_pruned_runtime_s": float(pruned_runtime),
        "candidate_full_runtime_s": float(full_runtime),
        "candidate_runtime_ratio_full_over_pruned": (
            float(full_runtime / pruned_runtime) if pruned_runtime > 0.0 else np.inf
        ),
        "n_time": int(emissions.n_time),
        "n_bins": int(emissions.n_bins),
        "n_spikes": int(emissions.n_spikes),
        "mean_candidate_count": float(np.mean([len(curr) for curr in candidate_indices])),
    }


def score_pruning_gaps(
    models: Iterable[object],
    emissions: LogEmissionTensor,
    bin_centers: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for model in models:
        if not hasattr(model, "candidate_indices"):
            continue
        candidates = _call_candidate_indices_compat(
            model.candidate_indices,
            emissions,
            bin_centers,
        )
        rows.append(score_pruning_gap(model, emissions, bin_centers, candidates))
    return pd.DataFrame.from_records(rows, columns=_CANDIDATE_PRUNING_GAP_COLUMNS)
