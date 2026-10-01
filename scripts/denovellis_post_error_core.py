"""Frozen post-error behavior, sequence and predictive-scoring primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.stats import norm


def field(x, key, default=None):
    return x.get(key, default) if isinstance(x, dict) else getattr(x, key, default)


def day_epochs(root, day):
    a = np.asarray(root, dtype=object).reshape(-1)
    if any(field(x, "statematrix") is not None or field(x, "data") is not None or field(x, "type") is not None for x in a):
        return a
    if not 1 <= day <= len(a):
        raise ValueError("Day index absent in MATLAB cell wrapper")
    return np.asarray(a[day - 1], dtype=object).reshape(-1)


def runs(mask):
    mask = np.asarray(mask, bool)
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def near_well_labels(xy, wells, radius=10.0):
    xy, wells = np.asarray(xy, float), np.asarray(wells, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or wells.shape != (3, 2) or not np.all(np.isfinite(wells)):
        raise ValueError("Expected 2D positions and three finite well coordinates")
    distances = np.linalg.norm(xy[:, None] - wells[None], axis=2)
    hits = distances <= radius
    labels = np.where(hits.sum(axis=1) == 1, np.argmax(hits, axis=1) + 1, 0)
    labels[~np.all(np.isfinite(xy), axis=1) | (hits.sum(axis=1) > 1)] = -1
    return labels, distances


def reconstruct_visits(time, xy, wells, *, radius=10.0, max_gap=0.25):
    time = np.asarray(time, float)
    if len(time) < 2 or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError("Nonfinite, duplicate or reversed position timestamps")
    labels, distances = near_well_labels(xy, wells, radius)
    # A broken tracking interval invalidates task history, including within a visit.
    broken = labels < 0
    broken[1:] |= np.diff(time) > max_gap
    segments = np.cumsum(broken)
    visits = []
    for well in (1, 2, 3):
        for start, end in runs((labels == well) & ~broken):
            boundary = end if end < len(time) else end - 1
            visits.append({"well": well, "start_index": start, "end_index": end,
                           "arrival_s": float(time[start]), "departure_s": float(time[boundary]),
                           "pause_end_s": float(time[boundary]), "history_segment": int(segments[start])})
    visits.sort(key=lambda v: v["start_index"])
    merged = []
    for v in visits:
        if merged and v["well"] == merged[-1]["well"] and v["history_segment"] == merged[-1]["history_segment"]:
            merged[-1]["end_index"] = v["end_index"]
            merged[-1]["departure_s"] = v["departure_s"]
        else:
            merged.append(v.copy())
    return merged, labels, distances


def score_visits(visits, center, outers):
    previous = None
    last_outer = None
    segment = None
    output = []
    for i, v in enumerate(visits):
        v = {**v, "visit_index": i + 1, "previous_well": None, "required_well": None, "task_correct": None, "task_kind": "unknown"}
        if v["history_segment"] != segment:
            previous, last_outer = None, None
        if previous is not None:
            v["previous_well"] = previous
            if previous in outers:
                v.update(required_well=center, task_kind="inbound", task_correct=v["well"] == center)
            elif previous == center and last_outer is not None:
                required = next(w for w in outers if w != last_outer)
                v.update(required_well=required, task_kind="outbound", task_correct=v["well"] == required)
        if v["well"] in outers:
            last_outer = v["well"]
        previous, segment = v["well"], v["history_segment"]
        output.append(v)
    return output


def valid_intervals(time, good, max_gap=0.25):
    time, good = np.asarray(time, float), np.asarray(good, bool)
    intervals = []
    for i in range(len(time) - 1):
        if good[i] and good[i + 1] and 0 < time[i + 1] - time[i] <= max_gap:
            if intervals and np.isclose(intervals[-1][1], time[i], rtol=0, atol=1e-8):
                intervals[-1][1] = float(time[i + 1])
            else:
                intervals.append([float(time[i]), float(time[i + 1])])
    return intervals


def clip_intervals(intervals, start, end):
    return [[max(a, start), min(b, end)] for a, b in intervals if min(b, end) > max(a, start)]


def contains_event(intervals, start, end):
    return bool(np.isfinite(start) and np.isfinite(end) and end > start and any(a <= start and end <= b for a, b in intervals))


def build_transitions(visits, *, center, outers, immobile_intervals, max_window=10.0, min_exposure=0.5):
    rows = []
    prior_outcomes = []
    error_run = 0
    previous_segment = None
    for i, v in enumerate(visits):
        if previous_segment != v["history_segment"]:
            prior_outcomes, error_run = [], 0
        previous_segment = v["history_segment"]
        if v["task_kind"] != "outbound" or v["task_correct"] is None:
            continue
        accuracy = (1 + sum(prior_outcomes[-5:])) / (2 + len(prior_outcomes[-5:]))
        prior_outcomes.append(bool(v["task_correct"]))
        error_run = 0 if v["task_correct"] else error_run + 1
        if v["task_correct"]:
            continue
        row = {"error_visit_index": v["visit_index"], "error_arrival_s": v["arrival_s"], "mistaken_well": v["well"],
               "correct_alternative_well": next(w for w in outers if w != v["well"]),
               "prior_five_choice_accuracy": accuracy, "consecutive_error_count": error_run,
               "next_outcome": "unclassifiable", "eligible": False, "exclusion_reason": "missing_subsequent_visits",
               "usable_exposure_s": 0.0, "window_start_s": None, "window_end_s": None, "immobile_intervals": []}
        following = visits[i + 1 : i + 3]
        if len(following) == 2:
            pause, choice = following
            if any(x["history_segment"] != v["history_segment"] for x in following):
                row["exclusion_reason"] = "tracking_gap_or_history_reset"
            elif pause["well"] != center or choice["well"] not in outers or choice["task_kind"] != "outbound":
                row["exclusion_reason"] = "noncanonical_well_sequence"
            else:
                start = pause["arrival_s"]
                end = min(pause["pause_end_s"], start + max_window)
                intervals = clip_intervals(immobile_intervals, start, end)
                exposure = sum(b - a for a, b in intervals)
                row.update(center_visit_index=pause["visit_index"], next_visit_index=choice["visit_index"], next_choice_well=choice["well"],
                           next_choice_arrival_s=choice["arrival_s"], window_start_s=start, window_end_s=end,
                           immobile_intervals=intervals, usable_exposure_s=exposure,
                           next_outcome="correction" if choice["well"] == row["correct_alternative_well"] else "repeated_error",
                           eligible=exposure >= min_exposure, exclusion_reason="" if exposure >= min_exposure else "insufficient_immobile_exposure")
        rows.append(row)
    return rows


def normalize_likelihood(log_likelihood):
    x = np.asarray(log_likelihood, float)
    if x.ndim != 2 or np.any(np.isnan(x)) or np.any(~np.any(np.isfinite(x), axis=1)):
        raise ValueError("Every likelihood row needs finite spatial support")
    return np.exp(x - logsumexp(x, axis=1, keepdims=True))


def weighted_correlation(posterior, time, distance):
    p, t, d = np.asarray(posterior, float), np.asarray(time, float), np.asarray(distance, float)
    if p.shape != (len(t), len(d)) or np.any(~np.isfinite(p)) or np.any(p < 0) or p.sum() <= 0:
        raise ValueError("Invalid posterior/time/graph dimensions or probability mass")
    p = p / p.sum()
    mt, md = np.sum(p * t[:, None]), np.sum(p * d[None, :])
    ct, cd = t[:, None] - mt, d[None, :] - md
    denom = np.sqrt(np.sum(p * ct**2) * np.sum(p * cd**2))
    return float(np.sum(p * ct * cd) / denom) if denom > 0 else 0.0


def sequence_test(posterior, time, routes, *, supported_bins, active_tetrodes, seed, n_shuffles=1000, alpha=0.05):
    p = np.asarray(posterior, float)
    if p.ndim != 2 or len(p) != len(time) or len(p) != len(supported_bins) or np.any(~np.isfinite(p)) or np.any(p < 0) or len(routes) != 2 or not np.isclose(p.sum(axis=1), 1).all():
        raise ValueError("Expected a normalized whole-graph posterior and two routes")
    def statistic(x):
        return max(abs(weighted_correlation(x[:, mask], time, distance)) if x[:, mask].sum() > 0 else 0.0 for mask, distance in routes)
    observed = statistic(p)
    result = {"sequence_statistic": observed, "sequence_passed": False, "p_value": None, "n_shuffles": 0, "reason": "insufficient_spike_bins_or_tetrodes"}
    if np.count_nonzero(supported_bins) < 5 or active_tetrodes < 2:
        return result
    if n_shuffles < 1:
        raise ValueError("Non-vacuous shuffle count required")
    rng = np.random.default_rng(seed)
    null = np.asarray([statistic(p[rng.permutation(len(p))]) for _ in range(n_shuffles)])
    pv = (1 + np.count_nonzero(null >= observed - 1e-12)) / (n_shuffles + 1)
    return {**result, "sequence_passed": bool(pv <= alpha), "p_value": float(pv), "n_shuffles": n_shuffles,
            "reason": "" if pv <= alpha else "order_null_not_rejected"}


def route_content(posterior, unique_masks):
    p = np.asarray(posterior, float)
    a, b = [np.asarray(m, bool) for m in unique_masks]
    if np.any(a & b) or p.ndim != 2 or len(a) != p.shape[1] or len(b) != p.shape[1]:
        raise ValueError("Unique arm masks must be disjoint and match the whole graph")
    return np.asarray([p[:, a].sum(axis=1).mean(), p[:, b].sum(axis=1).mean()])


def wilson_upper(successes, n):
    if not 0 <= successes <= n or n <= 0:
        raise ValueError("Wilson bound needs a positive denominator")
    z = norm.ppf(0.975)
    p = successes / n
    return float((p + z*z/(2*n) + z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n))


@dataclass
class PredictiveFit:
    mean: np.ndarray
    scale: np.ndarray
    animals: tuple
    days: tuple
    coefficients: np.ndarray

    def design(self, x, animal, day):
        numeric = (np.asarray(x, float) - self.mean) / self.scale
        return np.column_stack([np.ones(len(numeric)), numeric,
                                *[np.asarray(animal) == a for a in self.animals],
                                *[np.asarray(day) == d for d in self.days]])

    def predict(self, x, animal, day):
        return expit(self.design(x, animal, day) @ self.coefficients)


def fit_predictive(x, y, animals, days, *, penalty=1.0):
    x, y, animals, days = np.asarray(x, float), np.asarray(y, float), np.asarray(animals), np.asarray(days)
    if x.ndim != 2 or len(x) != len(y) or len(y) != len(animals) or len(y) != len(days) or not np.all(np.isfinite(x)) or set(y) != {0.0, 1.0}:
        raise ValueError("Predictive fit needs complete finite rows and both binary outcomes")
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1
    fit = PredictiveFit(x.mean(axis=0), scale, tuple(sorted(set(animals))), tuple(sorted(set(days))), np.array([]))
    design = fit.design(x, animals, days)
    # Give each animal equal total weight, while retaining a likelihood on the N-row scale.
    weights = np.asarray([len(y) / (len(fit.animals) * np.count_nonzero(animals == a)) for a in animals])
    ridge = np.full(design.shape[1], float(penalty)); ridge[0] = 0
    def objective(beta):
        eta = design @ beta
        loss = np.sum(weights * (np.logaddexp(0, eta) - y * eta)) + np.sum(ridge * beta**2) / 2
        gradient = design.T @ (weights * (expit(eta) - y)) + ridge * beta
        return loss, gradient
    result = minimize(objective, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-7})
    if not result.success:
        raise RuntimeError(f"Predictive optimization failed: {result.message}")
    fit.coefficients = result.x
    return fit


def predictive_log_score(y, probability):
    y, p = np.asarray(y, float), np.clip(np.asarray(probability, float), 1e-12, 1 - 1e-12)
    return y * np.log(p) + (1-y) * np.log1p(-p)
