"""Stabilize common-support selection when emission likelihoods are tied."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_PATCHED_FLAG = "_advanced_result_common_support_tie_patch_applied"
_WRAPPER_FLAG = "_advanced_result_common_support_tie_wrapper"


def apply_advanced_result_common_support_tie_patch() -> None:
    """Install deterministic top-k support selection for advanced diagnostics."""

    from . import advanced_result_diagnostics as diagnostics

    current = diagnostics.common_support_from_emissions
    if getattr(current, _WRAPPER_FLAG, False):
        setattr(diagnostics, _PATCHED_FLAG, True)
        return

    def common_support_from_emissions(
        log_likelihood: np.ndarray,
        *,
        top_k: int = 128,
        extra_candidate_sets: Sequence[Sequence[int]] | None = None,
    ) -> list[np.ndarray]:
        """Build common support with stable index-order tie breaking."""

        values = np.asarray(log_likelihood, dtype=float)
        if values.ndim != 2:
            raise ValueError("log_likelihood must have shape (n_time, n_bins)")
        extras = extra_candidate_sets or ()
        output: list[np.ndarray] = []
        for time_index, row in enumerate(values):
            k = min(max(int(top_k), 1), row.shape[0])
            # Stable descending sort keeps the original (spatial-bin) order for
            # equal likelihoods, unlike argpartition at a tied top-k cutoff.
            selected = np.argsort(-row, kind="stable")[:k]
            support = set(int(index) for index in selected)
            if time_index < len(extras):
                support.update(int(index) for index in extras[time_index])
            output.append(np.asarray(sorted(support), dtype=int))
        return output

    setattr(common_support_from_emissions, _WRAPPER_FLAG, True)
    diagnostics.common_support_from_emissions = common_support_from_emissions
    setattr(diagnostics, _PATCHED_FLAG, True)


__all__ = ["apply_advanced_result_common_support_tie_patch"]
