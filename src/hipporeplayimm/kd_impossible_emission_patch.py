"""Make KD evidence recursions robust to impossible emission rows."""

from __future__ import annotations

import numpy as np

_PATCHED_FLAG = "_kd_impossible_emission_patch_applied"


def apply_kd_impossible_emission_patch() -> None:
    """Install guards for all-impossible KD emission rows."""

    from . import kd_reference as kd

    if getattr(kd, _PATCHED_FLAG, False):
        return

    kd._scaled_emission = _scaled_emission
    kd._first_order_separable_log_evidence = _first_order_separable_log_evidence
    kd._second_order_separable_log_evidence = _second_order_separable_log_evidence
    setattr(kd, _PATCHED_FLAG, True)


def _scaled_emission(log_emissions: np.ndarray, time_index: int) -> tuple[np.ndarray, float]:
    row = log_emissions[time_index]
    offset = float(np.max(row))
    if np.isneginf(offset):
        return np.zeros_like(row, dtype=float), offset
    return np.exp(row - offset), offset


def _first_order_separable_log_evidence(log_emissions: np.ndarray, n_bins_x: int, n_bins_y: int, transition: np.ndarray) -> float:
    emission, offset = _scaled_emission(log_emissions, 0)
    alpha = emission.reshape(n_bins_x, n_bins_y) / log_emissions.shape[1]
    conditional = float(alpha.sum())
    if conditional <= 0.0:
        return float("-inf")
    logp = np.log(conditional) + offset
    alpha /= conditional
    for time_index in range(1, log_emissions.shape[0]):
        emission, offset = _scaled_emission(log_emissions, time_index)
        predicted = transition @ alpha @ transition.T
        alpha = predicted * emission.reshape(n_bins_x, n_bins_y)
        conditional = float(alpha.sum())
        if conditional <= 0.0:
            return float("-inf")
        logp += np.log(conditional) + offset
        alpha /= conditional
    return float(logp)


def _second_order_separable_log_evidence(log_emissions: np.ndarray, n_bins: int, initial: np.ndarray, transition: np.ndarray) -> float:
    emission0, offset0 = _scaled_emission(log_emissions, 0)
    alpha0 = emission0.reshape(n_bins, n_bins) / log_emissions.shape[1]
    conditional0 = float(alpha0.sum())
    if conditional0 <= 0.0:
        return float("-inf")
    logp = np.log(conditional0) + offset0
    alpha0 /= conditional0

    emission1, offset1 = _scaled_emission(log_emissions, 1)
    emission1_grid = emission1.reshape(n_bins, n_bins)
    alpha_t = np.einsum("ip,jq,pq,ij->ijpq", initial, initial, alpha0, emission1_grid, optimize=True)
    conditional1 = float(alpha_t.sum())
    if conditional1 <= 0.0:
        return float("-inf")
    logp += np.log(conditional1) + offset1
    alpha_t /= conditional1

    for time_index in range(2, log_emissions.shape[0]):
        emission, offset = _scaled_emission(log_emissions, time_index)
        emission_grid = emission.reshape(n_bins, n_bins)
        y_sum = np.einsum("jbq,abpq->abpj", transition, alpha_t, optimize=True)
        predicted = np.einsum("iap,abpj->ijab", transition, y_sum, optimize=True)
        alpha_t = predicted * emission_grid[:, :, None, None]
        conditional = float(alpha_t.sum())
        if conditional <= 0.0:
            return float("-inf")
        logp += np.log(conditional) + offset
        alpha_t /= conditional
    return float(logp)


__all__ = ["apply_kd_impossible_emission_patch"]
