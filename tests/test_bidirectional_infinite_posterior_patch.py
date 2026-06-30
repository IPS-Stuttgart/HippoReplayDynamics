from __future__ import annotations

import numpy as np

from hipporeplayimm.bidirectional_infinite_evidence_patch import (
    _equal_prior_logp_and_weights,
    _safe_mixture_log_posterior,
    apply_bidirectional_infinite_evidence_patch,
)


def test_safe_bidirectional_posterior_mixture_keeps_all_impossible_terminal_non_nan() -> None:
    logp, weights = _equal_prior_logp_and_weights([-np.inf, -np.inf])

    mixed = _safe_mixture_log_posterior(
        [
            np.array([-np.inf, -np.inf], dtype=float),
            np.array([-np.inf, -np.inf], dtype=float),
        ],
        weights,
    )

    assert logp == -np.inf
    np.testing.assert_allclose(weights, np.array([0.5, 0.5], dtype=float))
    assert mixed is not None
    assert not np.isnan(mixed).any()
    assert np.isneginf(mixed).all()


def test_safe_bidirectional_posterior_mixture_preserves_impossible_trajectory_rows() -> None:
    _, weights = _equal_prior_logp_and_weights([-np.inf, -np.inf])

    mixed = _safe_mixture_log_posterior(
        [
            np.array([[0.0, -np.inf], [-np.inf, -np.inf]], dtype=float),
            np.array([[-np.inf, 0.0], [-np.inf, -np.inf]], dtype=float),
        ],
        weights,
    )

    assert mixed is not None
    assert not np.isnan(mixed).any()
    np.testing.assert_allclose(np.exp(mixed[0]).sum(), 1.0)
    assert np.isneginf(mixed[1]).all()


def test_bidirectional_patch_refreshes_overwritten_score_methods(monkeypatch) -> None:
    from hipporeplayimm import result_improvement_extensions as compat
    from hipporeplayimm import reverse_models as direct

    def stale_score(self, emissions, bin_centers, **kwargs):  # pragma: no cover
        del self, emissions, bin_centers, kwargs
        raise AssertionError("stale bidirectional score method was not refreshed")

    monkeypatch.setattr(compat.BidirectionalReplayModel, "score", stale_score)
    monkeypatch.setattr(direct.BidirectionalReplayModel, "score", stale_score)

    apply_bidirectional_infinite_evidence_patch()

    assert compat.BidirectionalReplayModel.score is not stale_score
    assert direct.BidirectionalReplayModel.score is not stale_score
