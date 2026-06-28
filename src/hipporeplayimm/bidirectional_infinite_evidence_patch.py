"""Keep bidirectional replay mixtures finite for impossible directions.

If both directions report ``-inf`` evidence, the equal-prior mixture evidence is
still ``-inf``.  The posterior model weights should nevertheless remain finite
and symmetric so diagnostics and posterior mixtures do not become NaN.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

_PATCHED_FLAG = "_bidirectional_infinite_evidence_patch_applied"


def _equal_prior_logp_and_weights(log_likelihoods: object) -> tuple[float, np.ndarray]:
    """Return equal-prior mixture log evidence and finite component weights."""

    values = np.asarray(log_likelihoods, dtype=float).reshape(-1)
    if values.size == 0:
        return float("nan"), np.empty(0, dtype=float)

    positive_infinite = np.isposinf(values)
    if np.any(positive_infinite):
        weights = np.zeros(values.shape, dtype=float)
        weights[positive_infinite] = 1.0 / float(np.sum(positive_infinite))
        return float("inf"), weights

    finite = np.isfinite(values)
    if np.any(finite):
        normalizer = float(logsumexp(values[finite]))
        weights = np.zeros(values.shape, dtype=float)
        weights[finite] = np.exp(values[finite] - normalizer)
        return float(normalizer - np.log(float(values.size))), weights

    negative_infinite = np.isneginf(values)
    if np.any(negative_infinite):
        weights = np.zeros(values.shape, dtype=float)
        weights[negative_infinite] = 1.0 / float(np.sum(negative_infinite))
        return float("-inf"), weights

    return float("nan"), np.full(values.shape, 1.0 / float(values.size), dtype=float)


def _safe_mixture_log_posterior(log_posteriors: object, weights: np.ndarray) -> np.ndarray | None:
    """Return a posterior mixture without NaNs for impossible posterior slices."""

    weight_values = np.asarray(weights, dtype=float).reshape(-1)
    valid: list[tuple[np.ndarray, float]] = []
    for posterior, weight in zip(log_posteriors, weight_values, strict=False):
        if posterior is None:
            continue
        current = np.asarray(posterior, dtype=float)
        if current.size == 0:
            continue
        valid.append((current, float(weight)))
    if not valid:
        return None

    shape = valid[0][0].shape
    if any(current.shape != shape for current, _ in valid):
        raise ValueError("posterior arrays must have matching shapes")

    stacked = np.stack(
        [current + np.log(max(weight, np.finfo(float).tiny)) for current, weight in valid],
        axis=0,
    )
    mixed = logsumexp(stacked, axis=0)
    normalizer = logsumexp(mixed, axis=-1, keepdims=True)
    normalized = np.full(mixed.shape, -np.inf, dtype=float)
    np.subtract(
        mixed,
        normalizer,
        out=normalized,
        where=np.isfinite(normalizer),
    )
    return normalized


def apply_bidirectional_infinite_evidence_patch() -> None:
    """Patch bidirectional wrappers to avoid NaN weights for all-``-inf`` evidence."""

    from . import result_improvement_extensions as compat
    from . import reverse_models as direct

    if getattr(compat, _PATCHED_FLAG, False) and getattr(direct, _PATCHED_FLAG, False):
        return

    def compat_score(
        self,
        emissions,
        bin_centers,
        *,
        occupancy_s=None,
        candidate_indices=None,
        return_trajectory: bool | None = None,
    ):
        forward = compat.score_replay_model_compat(
            self.forward_model,
            emissions,
            bin_centers,
            occupancy_s=occupancy_s,
            candidate_indices=candidate_indices,
            return_trajectory=return_trajectory,
        )
        reverse_return_trajectory = True if return_trajectory is None or return_trajectory is False else return_trajectory
        reverse = compat.score_replay_model_compat(
            self.reverse_model,
            emissions,
            bin_centers,
            occupancy_s=occupancy_s,
            candidate_indices=candidate_indices,
            return_trajectory=reverse_return_trajectory,
        )
        logp, weights = _equal_prior_logp_and_weights(
            [forward.log_likelihood, reverse.log_likelihood]
        )
        chosen = forward if weights[0] >= weights[1] else reverse
        diagnostics = dict(chosen.diagnostics)
        diagnostics.update(
            {
                "direction_model": "bidirectional",
                "direction_forward_probability": float(weights[0]),
                "direction_reverse_probability": float(weights[1]),
                "direction_forward_log_evidence": float(forward.log_likelihood),
                "direction_reverse_log_evidence": float(reverse.log_likelihood),
            }
        )
        terminal = _safe_mixture_log_posterior(
            [forward.terminal_log_posterior, reverse.terminal_log_posterior],
            weights,
        )
        trajectory = None
        if (
            return_trajectory is not False
            and forward.trajectory_log_posterior is not None
            and reverse.trajectory_log_posterior is not None
        ):
            trajectory = _safe_mixture_log_posterior(
                [forward.trajectory_log_posterior, reverse.trajectory_log_posterior],
                weights,
            )
        if trajectory is not None:
            terminal = np.asarray(trajectory[-1], dtype=float).copy()
        if terminal is not None:
            diagnostics.update(compat._posterior_diagnostics(terminal, bin_centers))
        return compat.EventScore(
            self.name,
            logp,
            emissions.n_time,
            emissions.n_spikes,
            diagnostics=diagnostics,
            terminal_log_posterior=terminal,
            trajectory_log_posterior=trajectory,
        )

    def direct_score(
        self,
        emissions,
        bin_centers,
        *,
        occupancy_s=None,
        candidate_indices=None,
        return_trajectory=None,
    ):
        forward = direct._score_model_with_optional_kwargs(
            self.base_model,
            emissions,
            bin_centers,
            occupancy_s=occupancy_s,
            candidate_indices=candidate_indices,
            return_trajectory=return_trajectory,
        )
        reverse_return_trajectory = True if return_trajectory is False else return_trajectory
        reverse = direct.ReverseTimeReplayModel(self.base_model).score(
            emissions,
            bin_centers,
            occupancy_s=occupancy_s,
            candidate_indices=candidate_indices,
            return_trajectory=reverse_return_trajectory,
        )
        logp, weights = _equal_prior_logp_and_weights(
            [forward.log_likelihood, reverse.log_likelihood]
        )
        terminal = _safe_mixture_log_posterior(
            [forward.terminal_log_posterior, reverse.terminal_log_posterior],
            weights,
        )
        trajectory = None
        if (
            return_trajectory is not False
            and forward.trajectory_log_posterior is not None
            and reverse.trajectory_log_posterior is not None
        ):
            trajectory = _safe_mixture_log_posterior(
                [forward.trajectory_log_posterior, reverse.trajectory_log_posterior],
                weights,
            )
        if trajectory is not None:
            terminal = np.asarray(trajectory[-1], dtype=float).copy()
        diagnostics = {
            "time_direction": "bidirectional-mixture",
            "base_model": str(forward.model_name),
            "forward_model_posterior_probability": float(weights[0]),
            "reverse_model_posterior_probability": float(weights[1]),
        }
        if terminal is not None:
            diagnostics.update(direct._posterior_diagnostics(terminal, bin_centers))
        return direct.EventScore(
            self.name or f"{forward.model_name}-bidirectional",
            logp,
            forward.n_time,
            forward.n_spikes,
            diagnostics=diagnostics,
            terminal_log_posterior=terminal,
            trajectory_log_posterior=trajectory,
        )

    compat.BidirectionalReplayModel.score = compat_score
    direct.BidirectionalReplayModel.score = direct_score
    setattr(compat, _PATCHED_FLAG, True)
    setattr(direct, _PATCHED_FLAG, True)


__all__ = ["apply_bidirectional_infinite_evidence_patch"]
