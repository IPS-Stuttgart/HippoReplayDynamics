"""Cached exact spatial inference for cross-cell count-conditioned prediction."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from scipy.special import gammaln, logsumexp

from . import state_space as ss
from .duration_occupancy import _forward_backward_variable, _mode_transition_matrices, _score_first_order_imm_variable
from .frozen_posterior_prediction import frozen_smoothed_marginal_log_score, posterior_sha256

MODELS = ("iid_position", "static_location", "diffusion", "first_order_imm")


def identity_likelihood(counts, rates):
    counts, rates = np.asarray(counts), np.asarray(rates, float)
    if counts.ndim != 2 or rates.ndim != 2 or counts.shape[1] != rates.shape[0] or not counts.size or not rates.size:
        raise ValueError("nonempty aligned time/cell and cell/position matrices required")
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any() or not np.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError("finite integer counts and positive rates required")
    logp = np.log(rates) - np.log(rates.sum(axis=0, keepdims=True))
    coefficient = gammaln(counts.sum(axis=1) + 1) - gammaln(counts + 1).sum(axis=1)
    return counts @ logp + coefficient[:, None]


class SpatialPredictionContext:
    """Reuse Gaussian matrices, retaining the existing exact IMM inference code."""

    _scaled_emissions = staticmethod(ss._scaled_emissions)
    _as_log_probs = staticmethod(ss._as_log_probs)
    _mode_transition_matrix = staticmethod(ss._mode_transition_matrix)

    def __init__(self, centers):
        self.centers = np.asarray(centers, float)
        if self.centers.ndim != 2 or not len(self.centers) or not np.isfinite(self.centers).all():
            raise ValueError("finite position grid required")
        self.cache = OrderedDict()

    def _gaussian_transition_matrix(self, centers, sigma, cutoff, valid_bin_mask=None):
        if not np.array_equal(centers, self.centers) or valid_bin_mask is not None:
            raise ValueError("context requires its already-restricted spatial support")
        key = (round(float(sigma), 10), float(cutoff))
        if key not in self.cache:
            self.cache[key] = ss._gaussian_transition_matrix(self.centers, key[0], cutoff)
            if len(self.cache) > 128:
                self.cache.popitem(last=False)
        return self.cache[key]

    def infer(self, ll, centers_s):
        ll, times = np.asarray(ll, float), np.asarray(centers_s, float)
        if ll.ndim != 2 or ll.shape[1] != len(self.centers) or not len(ll) or times.shape != (len(ll),) or not np.isfinite(ll).all() or not np.isfinite(times).all():
            raise ValueError("finite aligned emissions and times required")
        durations = np.round(np.diff(times), 9)
        if (durations <= 0).any():
            raise ValueError("time centers must increase")
        diff = [self._gaussian_transition_matrix(self.centers, 60 * np.sqrt(dt), 3.0) for dt in durations]
        modes = _mode_transition_matrices(self, 3, 0.95, 0.06, durations)
        post = {"iid_position": ll - logsumexp(ll, axis=1, keepdims=True)}
        static = ll.sum(axis=0)
        post["static_location"] = np.tile(static - logsumexp(static), (len(ll), 1))
        _, post["diffusion"] = _forward_backward_variable(self, ll, diff)
        _, post["first_order_imm"], mass, _ = _score_first_order_imm_variable(
            self, ll, self.centers, stationary_sigma_cm=2.0, diffusion_transitions=diff, max_step_sigma=3.0, mode_stickiness=0.95, mode_transitions=modes
        )
        return post, mass


def score_event(counts, times, rates, train, held, context, permutation, event_global):
    counts = np.asarray(counts)
    train, held = np.asarray(train, int), np.asarray(held, int)
    if not len(train) or not len(held) or sorted([*train, *held]) != list(range(counts.shape[1])):
        raise ValueError("disjoint full neural partition required")
    if sorted(permutation) != list(range(rates.shape[1])):
        raise ValueError("invalid spatial permutation")
    tr_ll = identity_likelihood(counts[:, train], rates[train])
    inferred = {}
    for name, order in (("real", np.arange(rates.shape[1])), ("population_code_permuted", permutation)):
        post, mass = context.infer(tr_ll[:, order], times)
        inferred[name] = (post, mass, {m: posterior_sha256(q) for m, q in post.items()})
    # Held-out observations are accessed only after every training posterior exists.
    he_ll = identity_likelihood(counts[:, held], rates[held])
    global_scores = {
        "event_global": float(identity_likelihood(counts[:, held], event_global[held, None]).sum()),
        "run_global": float(identity_likelihood(counts[:, held], rates[held].mean(axis=1, keepdims=True)).sum()),
    }
    result = []
    for name, order in (("real", np.arange(rates.shape[1])), ("population_code_permuted", permutation)):
        post, mass, hashes = inferred[name]
        scores = {m: frozen_smoothed_marginal_log_score(q, he_ll[:, order]).total_log_score for m, q in post.items()}
        if any(posterior_sha256(post[m]) != hashes[m] for m in post):
            raise ValueError("held-out spikes changed inference")
        result.append(
            {
                "map": name,
                **{f"score_{m}": v for m, v in (scores | global_scores).items()},
                "mean_training_nonstationary_mass": float(mass[:, 1:].sum(axis=1).mean()),
                "training_imm_posterior_sha256": hashes["first_order_imm"],
                "posterior_unchanged": True,
                "heldout_used_for_inference": False,
            }
        )
    for name in ("iid_position", "static_location"):
        if not np.isclose(result[0][f"score_{name}"], result[1][f"score_{name}"], atol=1e-8):
            raise ValueError("shared spatial permutation changed map-independent model")
    return result
