"""Maximum-entropy transitions with matched dwell, modes and equilibrium."""

from __future__ import annotations

import numpy as np

from .lagged_neural_prediction import NeuralOperator


def stationary_distribution(operator):
    if isinstance(operator, NeuralOperator):
        a = operator.transition.T - np.eye(len(operator.initial))
        a[-1] = 1
        b = np.zeros(len(a))
        b[-1] = 1
        p = np.linalg.solve(a, b)
        iterations = 0
    else:
        p = operator.initial.copy()
        for iterations in range(1, 100001):
            q = operator.step(p)
            q /= q.sum()
            error = np.max(np.abs(q - p))
            p = q
            if error < 1e-15:
                break
        else:
            raise ValueError("stationary distribution did not converge")
    if not np.isfinite(p).all() or (p <= 0).any() or abs(p.sum() - 1) > 1e-12:
        raise ValueError("invalid stationary distribution")
    if np.max(np.abs(operator.step(p) - p)) > 1e-12:
        raise ValueError("stationary distribution failed original operator")
    return p, iterations


class OccupancyMatchedNull:
    def __init__(self, stationary, mode, stay, max_iter=20000):
        self.pi = np.asarray(stationary, float)
        self.mode = np.asarray(mode, float)
        self.stay = np.asarray(stay, float)
        if self.pi.ndim != 2 or self.pi.shape != self.stay.shape:
            raise ValueError("aligned mode-position arrays required")
        self.n_modes, self.n_bins = self.pi.shape
        m, n = self.n_modes, self.n_bins
        if n < 2 or self.mode.shape != (m, m) or not all(np.isfinite(v).all() for v in (self.pi, self.mode, self.stay)):
            raise ValueError("finite aligned multistate parameters required")
        if (self.pi <= 0).any() or (self.mode < 0).any() or (self.stay < 0).any() or (self.stay > 1).any():
            raise ValueError("invalid probability parameters")
        if abs(self.pi.sum() - 1) > 1e-12 or not np.allclose(self.mode.sum(axis=1), 1, atol=1e-12, rtol=0):
            raise ValueError("normalized parameters required")
        fixed = (self.mode.T @ self.pi) * self.stay
        raw_target = self.pi - fixed
        r = (self.pi[:, :, None] * self.mode[:, None, :] * (1 - self.stay.T[None, :, :])).reshape(m * n, m)
        if raw_target.min() < -1e-12:
            raise ValueError("inconsistent fixed incoming mass")
        c = np.maximum(raw_target, 0)
        for dest in range(m):
            total = r[:, dest].sum()
            if abs(c[dest].sum() - total) > 1e-12:
                raise ValueError("inconsistent mode flow totals")
            if total == 0:
                c[dest] = 0
            else:
                if c[dest].sum() == 0:
                    raise ValueError("missing destination mass")
                c[dest] *= total / c[dest].sum()
        self.target_correction = float(np.max(np.abs(c - raw_target)))
        positions = np.tile(np.arange(n), m)
        v = c.copy()

        def rows(v):
            denominator = v.sum(axis=1)[None, :] - v[:, positions].T
            if ((denominator <= 0) & (r > 0)).any():
                raise ValueError("infeasible off-position flow")
            return np.divide(r, denominator, out=np.zeros_like(r), where=r > 0)

        for iteration in range(1, max_iter + 1):
            u = rows(v)
            denominator = u.sum(axis=0)[:, None] - u.reshape(m, n, m).sum(axis=0).T
            if ((denominator <= 0) & (c > 0)).any():
                raise ValueError("infeasible column flow")
            v = np.divide(c, denominator, out=np.zeros_like(c), where=c > 0)
            totals = v.sum(axis=1, keepdims=True)
            v = np.divide(v, totals, out=np.zeros_like(v), where=totals > 0)
            u = rows(v)
            achieved = v * (u.sum(axis=0)[:, None] - u.reshape(m, n, m).sum(axis=0).T)
            error = float(np.max(np.abs(achieved - c)))
            if error < 1e-13 and np.max(np.abs(achieved - c) / self.pi) < 1e-9:
                break
        else:
            raise ValueError("maximum-entropy scaling did not converge")
        self.u, self.v = u, v
        self.iterations = iteration
        self.balance_error = error
        probability = u * (v.sum(axis=1)[None, :] - v[:, positions].T) / self.pi.ravel()[:, None]
        probability += (self.mode[:, None, :] * self.stay.T[None, :, :]).reshape(m * n, m)
        self.mode_error = float(np.max(np.abs(probability - np.repeat(self.mode, n, axis=0))))
        self.equilibrium_error = float(np.max(np.abs(self.step(self.pi.ravel()) - self.pi.ravel())))
        if self.mode_error > 1e-10 or self.equilibrium_error > 1e-11:
            raise ValueError("matched-null constraints failed")

    @classmethod
    def from_operator(cls, operator):
        pi, iterations = stationary_distribution(operator)
        if isinstance(operator, NeuralOperator):
            result = cls(pi[None, :], np.ones((1, 1)), np.diag(operator.transition)[None, :])
        else:
            n = operator.n_bins
            stay = np.array([np.full(n, 1 / n) if k is None else k.diagonal() for k in operator.kernels])
            result = cls(pi.reshape(operator.n_modes, n), operator.mode, stay)
        result.stationary_iterations = iterations
        return result

    def step(self, q):
        q = np.asarray(q, float)
        shape = q.shape
        m, n = self.pi.shape
        if q.shape[-1] != m * n or not np.isfinite(q).all() or (q < 0).any():
            raise ValueError("aligned nonnegative distributions required")
        x = q.reshape(-1, m, n)
        weighted = q.reshape(-1, m * n) / self.pi.ravel()
        flow = weighted[:, :, None] * self.u[None, :, :]
        outside = flow.sum(axis=1)[:, :, None] - flow.reshape(-1, m, n, m).sum(axis=1).transpose(0, 2, 1)
        fixed = np.einsum("ij,bix->bjx", self.mode, x) * self.stay[None, :, :]
        result = fixed + outside * self.v[None, :, :]
        if result.min() < -1e-12:
            raise ValueError("negative null propagation")
        return np.maximum(result, 0).reshape(shape)

    def parameters(self):
        return {"pi": self.pi, "mode": self.mode, "stay": self.stay, "u": self.u, "v": self.v}

    def diagnostics(self):
        return {
            "scaling_iterations": self.iterations,
            "balance_error": self.balance_error,
            "mode_probability_error": self.mode_error,
            "equilibrium_error": self.equilibrium_error,
            "target_mass_correction": self.target_correction,
            "stationary_iterations": self.stationary_iterations,
        }
