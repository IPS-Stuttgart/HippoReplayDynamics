"""Learn a nonspatial, count-conditioned HMM using hmmlearn's EM engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
from scipy.special import logsumexp

from .assembly_predictive_control import multinomial_ll, validate_counts
from .frozen_posterior_prediction import frozen_smoothed_marginal_log_score, posterior_sha256


@dataclass(frozen=True)
class LearnedAssembly:
    probabilities: np.ndarray  # cells x states
    initial: np.ndarray
    transition: np.ndarray  # previous state x next state
    occupancy: np.ndarray
    global_probability: np.ndarray
    objective_trace: np.ndarray
    converged: bool
    restart: int
    n_calibration_bins: int
    n_calibration_events: int
    restart_objectives: tuple[float, ...] = ()
    restart_converged: tuple[bool, ...] = ()
    initialization_with_replacement: bool = False


def validate_sequences(sequences):
    arrays = [validate_counts(x) for x in sequences]
    if not arrays or len({x.shape[1] for x in arrays}) != 1:
        raise ValueError("aligned nonempty event sequences required")
    return arrays


def infer_states(log_likelihood, initial, transition):
    """Exact scaled forward-backward; row centering prevents underflow."""
    from hmmlearn import _hmmc

    ll, pi, a = (np.asarray(x, float) for x in (log_likelihood, initial, transition))
    if (
        ll.ndim != 2 or not len(ll) or pi.shape != (ll.shape[1],)
        or a.shape != (len(pi), len(pi)) or not np.isfinite(ll).all()
        or not np.isfinite(pi).all() or not np.isfinite(a).all()
        or (pi <= 0).any() or (a <= 0).any()
        or not np.isclose(pi.sum(), 1) or not np.allclose(a.sum(axis=1), 1)
    ):
        raise ValueError("finite likelihood and strictly positive stochastic parameters required")
    maximum = ll.max(axis=1, keepdims=True)
    frame = np.exp(ll - maximum)
    logp, forward, scaling = _hmmc.forward_scaling(pi, a, frame)
    backward = _hmmc.backward_scaling(pi, a, frame, scaling)
    posterior = forward * backward
    posterior /= posterior.sum(axis=1, keepdims=True)
    log_posterior = np.full_like(posterior, -np.inf)
    np.log(posterior, out=log_posterior, where=posterior > 0)
    return float(logp + maximum.sum()), log_posterior


def _model_class():
    # Import lazily: the optional research runtime is not a core dependency.
    from hmmlearn import _hmmc
    from hmmlearn.base import BaseHMM
    from hmmlearn.hmm import MultinomialHMM

    class RegularizedMultinomialHMM(MultinomialHMM):
        def _compute_log_likelihood(self, x):
            return multinomial_ll(x, self.emissionprob_.T)

        def _fit_scaling(self, x):
            ll = self._compute_log_likelihood(x)
            maximum = ll.max(axis=1, keepdims=True)
            frame = np.exp(ll - maximum)
            logp, forward, scaling = _hmmc.forward_scaling(self.startprob_, self.transmat_, frame)
            backward = _hmmc.backward_scaling(self.startprob_, self.transmat_, frame, scaling)
            posterior = self._compute_posteriors_scaling(forward, backward)
            return frame, float(logp + maximum.sum()), posterior, forward, backward

        def _do_mstep(self, stats):
            BaseHMM._do_mstep(self, stats)
            counts = stats["obs"] + self.emission_pseudocount * self.reference[None, :]
            self.emissionprob_ = counts / counts.sum(axis=1, keepdims=True)

        def _compute_lower_bound(self, log_prob):
            return float(
                log_prob
                + (np.log(self.startprob_).sum() + np.log(self.transmat_).sum()) / self.n_components
                + self.emission_pseudocount * (self.reference[None, :] * np.log(self.emissionprob_)).sum()
            )

    return RegularizedMultinomialHMM


def fit_learned_assembly(sequences, n_states, seed, *, max_iter=500, restarts=2):
    arrays = validate_sequences(sequences)
    counts = np.concatenate(arrays)
    informative = np.flatnonzero(counts.sum(axis=1) > 0)
    if (
        isinstance(n_states, bool) or not isinstance(n_states, int) or n_states < 1
        or len(informative) == 0 or max_iter < 2 or restarts < 1
    ):
        raise ValueError("insufficient informative calibration bins or invalid fit settings")
    reference = counts.sum(axis=0) + 100.0 / counts.shape[1]
    reference = reference / reference.sum()
    lengths = np.array([len(x) for x in arrays], dtype=int)
    tolerance = 1e-5 * len(counts)
    fits = []
    model_type = _model_class()
    # hmmlearn 0.3.3 otherwise prints a migration warning for every new fit.
    logging.getLogger("hmmlearn.hmm").setLevel(logging.ERROR)
    for restart in range(restarts):
        rng = np.random.default_rng(np.random.SeedSequence([seed, restart]))
        model = model_type(
            n_components=n_states, n_iter=max_iter, tol=tolerance,
            startprob_prior=1 + 1 / n_states, transmat_prior=1 + 1 / n_states,
            init_params="", params="ste", implementation="scaling",
        )
        model.reference = reference
        model.emission_pseudocount = 10.0
        model.startprob_ = np.full(n_states, 1 / n_states)
        model.transmat_ = 0.5 * np.eye(n_states) + 0.5 / n_states
        replacement = len(informative) < n_states
        initial = counts[rng.choice(informative, n_states, replace=replacement)] + 10 * reference
        model.emissionprob_ = initial / initial.sum(axis=1, keepdims=True)
        model.fit(counts, lengths)
        scores, posterior = zip(*[
            infer_states(multinomial_ll(x, model.emissionprob_.T), model.startprob_, model.transmat_) for x in arrays
        ], strict=True)
        final_objective = model._compute_lower_bound(sum(scores))
        trace = np.r_[np.asarray(model.monitor_.history, float), final_objective]
        if not np.isfinite(trace).all() or (np.diff(trace) < -1e-7 * (1 + np.abs(trace[:-1]))).any():
            raise ValueError("nonfinite or decreasing regularized EM objective")
        occupancy = np.exp(np.concatenate(posterior)).sum(axis=0) + 1 / n_states
        occupancy /= occupancy.sum()
        converged = len(trace) >= 3 and 0 <= trace[-2] - trace[-3] < tolerance
        fits.append(LearnedAssembly(
            model.emissionprob_.T.copy(), model.startprob_.copy(), model.transmat_.copy(),
            occupancy, reference, trace, converged, restart, len(counts), len(arrays),
        ))
    return replace(
        max(fits, key=lambda f: f.objective_trace[-1]),
        restart_objectives=tuple(float(f.objective_trace[-1]) for f in fits),
        restart_converged=tuple(bool(f.converged) for f in fits),
        initialization_with_replacement=bool(len(informative) < n_states),
    )


def predict_learned_assembly(counts, train, held, fit):
    """Infer assignments from target training cells, then score held-out cells."""
    counts = validate_counts(counts)
    train, held = np.asarray(train, int), np.asarray(held, int)
    if not len(train) or not len(held) or sorted([*train, *held]) != list(range(counts.shape[1])):
        raise ValueError("disjoint full population partition required")
    if fit.probabilities.shape[0] != counts.shape[1]:
        raise ValueError("model and target cells differ")
    ll = multinomial_ll(counts[:, train], fit.probabilities[train])
    _, posterior = infer_states(ll, fit.initial, fit.transition)
    independent = ll + np.log(fit.occupancy)[None, :]
    independent -= logsumexp(independent, axis=1, keepdims=True)
    states = {"learned_hmm": posterior, "same_emissions_iid": independent}
    hashes = {name: posterior_sha256(q) for name, q in states.items()}
    held_ll = multinomial_ll(counts[:, held], fit.probabilities[held])
    scores = {name: frozen_smoothed_marginal_log_score(q, held_ll).total_log_score for name, q in states.items()}
    scores["nonspatial_global"] = float(multinomial_ll(counts[:, held], fit.global_probability[held, None]).sum())
    if any(posterior_sha256(states[name]) != value for name, value in hashes.items()):
        raise ValueError("held-out observations changed training posterior")
    return scores, hashes
