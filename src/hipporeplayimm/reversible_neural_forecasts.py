"""Stationary time-reversed and additive reversible forecast operators."""

from __future__ import annotations

import numpy as np

from .conditional_spatial_prediction import identity_likelihood
from .frozen_posterior_prediction import posterior_sha256
from .lagged_neural_prediction import forward_filter, mixture_scores


class ReversibleControls:
    def __init__(self, operator, stationary):
        self.operator = operator
        self.pi = np.asarray(stationary, float).ravel()
        if self.pi.shape != operator.initial.shape or not np.isfinite(self.pi).all() or (self.pi <= 0).any() or abs(self.pi.sum() - 1) > 1e-12:
            raise ValueError("positive aligned normalized stationary distribution required")
        self.equilibrium_error = float(np.max(np.abs(operator.step(self.pi) - self.pi)))
        self.reverse_row_error = float(np.max(np.abs(operator.step(self.pi) / self.pi - 1)))
        if self.equilibrium_error > 1e-12 or self.reverse_row_error > 1e-10:
            raise ValueError("stationary distribution does not match original operator")
        self.neural = hasattr(operator, "transition")

    def adjoint(self, q):
        q = np.asarray(q, float)
        if q.shape[-1] != len(self.pi):
            raise ValueError("unaligned state vectors")
        op = self.operator
        if self.neural:
            return q @ op.transition.T
        shape = q.shape
        values = q.reshape(-1, op.n_modes, op.n_bins)
        back = np.empty_like(values)
        for dest, kernel in enumerate(op.kernels):
            if kernel is None:
                back[:, dest] = values[:, dest].sum(axis=1, keepdims=True) / op.n_bins
            else:
                back[:, dest] = (kernel.T @ values[:, dest].T).T
        return np.einsum("mn,bnx->bmx", op.mode, back).reshape(shape)

    def reverse(self, q):
        return self.adjoint(np.asarray(q, float) / self.pi) * self.pi

    def reversible(self, q):
        return (self.operator.step(q) + self.reverse(q)) / 2


def forecasts(ll, operator, controls):
    filtered = forward_filter(ll, operator)
    dynamic, reverse, reversible = filtered.copy(), filtered.copy(), filtered.copy()
    result = {}
    for h in range(1, 5):
        dynamic = operator.step(dynamic)
        reverse = controls.reverse(reverse)
        reversible = controls.reversible(reversible)
        if h in (1, 2, 4) and h < len(ll):
            result[h] = {name: operator.collapse(q[:-h]).copy() for name, q in (("dynamic", dynamic), ("reverse", reverse), ("reversible", reversible))}
            for q in result[h].values():
                if not np.isfinite(q).all() or (q < 0).any():
                    raise ValueError("invalid forecast probabilities")
                np.testing.assert_allclose(q.sum(axis=1), 1, atol=1e-10, rtol=0)
    return result


def score_model(x, rates, train, held, operator, controls, *, position_order=None):
    ll = identity_likelihood(x[:, train], rates[train])
    if position_order is not None:
        if sorted(position_order) != list(range(rates.shape[1])):
            raise ValueError("invalid frozen position permutation")
        ll = ll[:, position_order]
    predictions = forecasts(ll, operator, controls)
    target = identity_likelihood(x[:, held], rates[held])
    if position_order is not None:
        target = target[:, position_order]
    result = {}
    for h, pred in predictions.items():
        row = {"forecast_sha256": posterior_sha256(pred["dynamic"])}
        for name, q in pred.items():
            values = mixture_scores(q, target[h:])
            values[x[h:, held].sum(axis=1) == 0] = 0
            row["score_" + name] = float(values.sum())
        result[h] = row
    return result
