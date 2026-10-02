"""Animal-balanced, train-only prediction of POST-minus-PRE pair coordination.

This mathematical core does not authorize a biological fit. Source verification,
independent replay validation and full spike-level calibration remain upstream.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class PairData:
    animal: np.ndarray
    session: np.ndarray
    pause: np.ndarray
    unit_a: np.ndarray
    unit_b: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    baseline: np.ndarray
    order: np.ndarray

    @property
    def change(self):
        return self.post - self.pre


def validate(data):
    n = len(data.pre)
    require(n > 0, "A nonempty pair cohort is required")
    arrays = [data.animal, data.session, data.pause, data.unit_a, data.unit_b,
              data.pre, data.post]
    require(all(np.asarray(x).shape == (n,) for x in arrays), "Pair rows are misaligned")
    require(data.baseline.ndim == 2 and data.baseline.shape[0] == n
            and data.baseline.shape[1] > 0, "Baseline columns are required")
    require(data.order.ndim == 2 and data.order.shape[0] == n
            and data.order.shape[1] >= 2, "Original and shuffled order columns are required")
    require(all(np.isfinite(x).all() for x in
                (data.pre, data.post, data.baseline, data.order)), "Missing pair measurements")
    require(np.array_equal(data.baseline[:, 0], data.pre),
            "First baseline column must be the unchanged PRE coordination")
    require(all(np.issubdtype(np.asarray(x).dtype, np.integer)
                for x in (data.unit_a, data.unit_b)), "Unit identities must be integers")
    require(np.all(data.unit_a < data.unit_b), "Pairs must use distinct canonical unit identities")
    keys = list(zip(data.animal, data.session, data.pause, data.unit_a, data.unit_b, strict=True))
    require(len(set(keys)) == n, "Duplicate pause/pair observations")
    for values in (data.animal, data.session, data.pause):
        require(all(isinstance(x, (str, np.str_)) and x.strip() for x in values),
                "Missing animal, recording or pause identity")
    session_owner = {}
    pause_owner = {}
    for animal, session, pause in zip(data.animal, data.session, data.pause, strict=True):
        require(session_owner.setdefault(session, animal) == animal,
                "One recording cannot belong to multiple animals")
        require(pause_owner.setdefault((session, pause), animal) == animal,
                "Pause identity changes animal")
    return data


def hierarchical_weights(animal, session, pause):
    """Equal animal, recording within animal, pause within recording, and pair."""
    animal, session, pause = map(np.asarray, (animal, session, pause))
    require(animal.ndim == 1 and animal.shape == session.shape == pause.shape
            and len(animal) > 0, "Nonempty aligned grouping arrays required")
    answer = np.zeros(len(animal), float)
    animals = np.unique(animal)
    for a in animals:
        animal_rows = animal == a
        sessions = np.unique(session[animal_rows])
        for s in sessions:
            session_rows = animal_rows & (session == s)
            pauses = np.unique(pause[session_rows])
            for p in pauses:
                rows = session_rows & (pause == p)
                answer[rows] = 1 / (len(animals) * len(sessions) * len(pauses) * rows.sum())
    require(np.isclose(answer.sum(), 1) and (answer > 0).all(), "Invalid hierarchy weights")
    return answer


@dataclass(frozen=True)
class Fit:
    coefficient: np.ndarray
    scale: np.ndarray
    rank: int
    feature_count: int

    def predict(self, features):
        x = np.asarray(features, float)
        require(x.ndim == 2 and x.shape[1] == self.feature_count
                and np.isfinite(x).all(), "Invalid prediction features")
        return (x / self.scale) @ self.coefficient


def baseline_support(training, target, weights):
    """Identify controls whose held-out predictions are not training-identifiable."""
    training, target, weights = map(lambda x: np.asarray(x, float), (training, target, weights))
    require(training.ndim == target.ndim == 2 and training.shape[1] == target.shape[1]
            and training.shape[1] > 0 and weights.shape == (len(training),) and len(training) > 0
            and (weights > 0).all() and all(np.isfinite(x).all() for x in (training, target, weights)),
            "Aligned finite baseline support arrays required")
    weights = weights / weights.sum()
    scale = np.sqrt(np.einsum("i,ij,ij->j", weights, training, training))
    scale[scale == 0] = 1
    design = training / scale * np.sqrt(weights[:, None])
    # The full right basis is needed only when there are fewer rows than columns.
    _, singular, right = np.linalg.svd(design, full_matrices=len(design) < design.shape[1])
    tolerance = max(design.shape) * np.finfo(float).eps * singular[0]
    rank = int(np.sum(singular > tolerance))
    target_scaled = target / scale
    loading = np.linalg.norm(target_scaled @ right[rank:].T, axis=1)
    numerical_guard = 100 * np.finfo(float).eps * max(design.shape) * np.maximum(
        1, np.linalg.norm(target_scaled, axis=1))
    unsupported = loading > numerical_guard
    return {"baseline_training_rank": rank, "baseline_columns": training.shape[1],
            "baseline_max_heldout_nullspace_loading": float(loading.max(initial=0)),
            "baseline_unsupported_heldout_rows": int(unsupported.sum()),
            "baseline_prediction_support_complete": not bool(unsupported.any()),
            "scope": "Machine-precision training row-space diagnostic; no observation excluded or prediction changed"}


def fit(features, response, weights, *, order_column=None, order_penalty=1.0):
    x, y, w = map(lambda v: np.asarray(v, float), (features, response, weights))
    require(x.ndim == 2 and x.shape[0] > 0 and y.shape == w.shape == (len(x),),
            "Nonempty aligned fitting arrays required")
    require(np.isfinite(x).all() and np.isfinite(y).all()
            and np.isfinite(w).all() and (w > 0).all(), "Invalid fitting arrays")
    require(np.isfinite(order_penalty) and order_penalty > 0, "Positive fixed order penalty required")
    w = w / w.sum()
    # No intercept or centering: every feature and the response is antisymmetric
    # under exchanging neuron A and B. RMS scaling uses training rows only.
    scale = np.sqrt(np.sum(w[:, None] * x * x, axis=0))
    scale[scale == 0] = 1
    design = (x / scale) * np.sqrt(w[:, None])
    target = y * np.sqrt(w)
    if order_column is not None:
        require(isinstance(order_column, (int, np.integer)) and not isinstance(order_column, bool)
                and 0 <= order_column < x.shape[1], "Invalid order column")
        penalty = np.zeros((1, x.shape[1]))
        penalty[0, order_column] = np.sqrt(order_penalty)
        design = np.vstack((design, penalty))
        target = np.r_[target, 0.0]
    coefficient, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    return Fit(coefficient, scale, int(rank), x.shape[1])


def prediction_check(data, *, order_penalty=1.0):
    """Re-estimate both models for original and every whole-bin control column."""
    validate(data)
    animals = np.unique(data.animal)
    require(len(animals) >= 3, "Leave-animal-out development check requires at least three animals")
    response = data.change
    predictions = np.full(data.order.shape, np.nan)
    baseline_predictions = np.full(data.order.shape, np.nan)
    folds = []
    for animal in animals:
        training = data.animal != animal
        target = ~training
        weights = hierarchical_weights(data.animal[training], data.session[training], data.pause[training])
        support = baseline_support(data.baseline[training], data.baseline[target], weights)
        for condition in range(data.order.shape[1]):
            # The baseline is deliberately refitted inside each condition too.
            base = fit(data.baseline[training], response[training], weights,
                       order_penalty=order_penalty)
            augmented = np.column_stack((data.baseline, data.order[:, condition]))
            model = fit(augmented[training], response[training], weights,
                        order_column=augmented.shape[1] - 1, order_penalty=order_penalty)
            predictions[target, condition] = model.predict(augmented[target])
            baseline_predictions[target, condition] = base.predict(data.baseline[target])
            folds.append({"heldout_animal": str(animal), "condition": condition,
                          "training_animals": len(animals) - 1,
                          "training_rows": int(training.sum()), "heldout_rows": int(target.sum()),
                          "baseline_rank": base.rank, "augmented_rank": model.rank,
                          **support,
                          "order_coefficient_original_units": float(model.coefficient[-1] / model.scale[-1]),
                          "training_feature_rms": model.scale.tolist()})
    require(np.isfinite(predictions).all() and np.isfinite(baseline_predictions).all(),
            "Incomplete held-out predictions")
    gains = (response[:, None] - baseline_predictions) ** 2 - (response[:, None] - predictions) ** 2
    rows = []
    for animal in animals:
        target = data.animal == animal
        weights = hierarchical_weights(data.animal[target], data.session[target], data.pause[target])
        values = weights @ gains[target]
        for condition, value in enumerate(values):
            rows.append({"animal": str(animal), "condition": condition,
                         "heldout_mse_improvement": float(value), "pair_rows": int(target.sum()),
                         "recordings": len(np.unique(data.session[target])),
                         "pauses": len(set(zip(data.session[target], data.pause[target], strict=True)))})
    animal_gain = np.array([[r["heldout_mse_improvement"] for r in rows if r["animal"] == str(a)]
                           for a in animals])
    require(animal_gain.shape == (len(animals), data.order.shape[1]), "Animal denominator changed")
    informative_animals = sum(bool(np.any(data.order[data.animal == a, 0] != 0)) for a in animals)
    summary = {"animals": len(animals), "pair_rows": len(data.pre),
               "animals_with_nonzero_original_order": informative_animals,
               "minimum_informative_animal_coverage_met": informative_animals >= 3,
               "baseline_prediction_support_complete": all(
                   row["baseline_prediction_support_complete"] for row in folds),
               "n_shuffles": data.order.shape[1] - 1,
               "original_mean_animal_gain": float(animal_gain[:, 0].mean()),
               "shuffle_mean_animal_gains": animal_gain[:, 1:].mean(axis=0).tolist(),
               "original_minus_animal_median_shuffle_gain": float(
                   (animal_gain[:, 0] - np.median(animal_gain[:, 1:], axis=1)).mean()),
               "statistical_core_checked": True, "biological_calibration_complete": False,
               "biological_inference_authorized": False, "goal_complete": False}
    return {"summary": summary, "folds": folds, "by_animal": rows,
            "predictions": predictions, "baseline_predictions": baseline_predictions}


def nuisance_features(pre, pre_rate_a, pre_rate_b, event_spikes_a, event_spikes_b, participation):
    """All unsigned count controls enter through orientation-equivariant features."""
    pre, a, b, ea, eb, participation = map(lambda v: np.asarray(v, float),
                                         (pre, pre_rate_a, pre_rate_b, event_spikes_a,
                                          event_spikes_b, participation))
    require(pre.ndim == 1 and all(x.shape == pre.shape for x in (a, b, ea, eb, participation)),
            "Misaligned pair nuisance fields")
    require(all(np.isfinite(x).all() for x in (pre, a, b, ea, eb, participation))
            and all((x >= 0).all() for x in (a, b, ea, eb, participation))
            and (participation <= 1).all(), "Invalid pair rates/counts/participation")
    return np.column_stack((pre, np.log1p(a) - np.log1p(b),
                            pre * np.log1p(a + b), np.log1p(ea) - np.log1p(eb),
                            pre * participation))


def aggregate_event_orders(unit_ids, events, n_shuffles):
    """Average validated events into one observation per frozen pause/pair.

    Callers must independently verify the validation labels and count-preserving
    event controls. This primitive cannot certify those upstream inputs.
    """
    ids = np.asarray(unit_ids)
    require(ids.ndim == 1 and np.issubdtype(ids.dtype, np.integer) and len(ids) >= 2
            and np.all(np.diff(ids) > 0), "Unique increasing frozen unit IDs required")
    require(isinstance(n_shuffles, int) and not isinstance(n_shuffles, bool)
            and n_shuffles >= 1, "Positive shuffle count required")
    matrix = np.zeros((n_shuffles + 1, len(ids), len(ids)))
    participation = np.zeros((len(ids), len(ids)))
    spikes = np.zeros(len(ids))
    identities = set()
    for event in events:
        require(event.get("validated_replay") is True, "Unvalidated candidates cannot enter replay aggregation")
        identity = event.get("event_id")
        require(isinstance(identity, str) and identity and identity not in identities,
                "Missing or duplicate validated-event identity")
        identities.add(identity)
        active = np.asarray(event["unit_ids"])
        totals = np.asarray(event["spike_counts"])
        original, shuffled = np.asarray(event["order"]), np.asarray(event["shuffle_order"])
        require(active.ndim == 1 and np.issubdtype(active.dtype, np.integer)
                and len(active) > 0 and len(np.unique(active)) == len(active), "Invalid active unit identities")
        require(totals.shape == active.shape and np.isfinite(totals).all()
                and (totals > 0).all() and (totals == np.floor(totals)).all(), "Invalid active spike totals")
        require(original.shape == (len(active), len(active))
                and shuffled.shape == (n_shuffles, len(active), len(active)), "Incomplete event-order controls")
        orders = np.concatenate((original[None], shuffled), axis=0)
        require(np.isfinite(orders).all()
                and np.allclose(orders, -orders.swapaxes(1, 2), rtol=0, atol=1e-12)
                and np.max(np.abs(orders)) <= 1 + 1e-12, "Invalid directional event-order matrix")
        index = np.searchsorted(ids, active)
        require(np.all(index < len(ids)) and np.array_equal(ids[np.minimum(index, len(ids) - 1)], active),
                "Event population differs from frozen PRE-selected units")
        for condition in range(n_shuffles + 1):
            matrix[condition][np.ix_(index, index)] += orders[condition]
        participation[np.ix_(index, index)] += 1
        spikes[index] += totals
    if events:
        matrix /= len(events)
        participation /= len(events)
        spikes /= len(events)
    a, b = np.triu_indices(len(ids), 1)
    return {"unit_a": ids[a], "unit_b": ids[b], "order": matrix[:, a, b].T,
            "event_spikes_a": spikes[a], "event_spikes_b": spikes[b],
            "participation": participation[a, b], "validated_events": len(events)}
