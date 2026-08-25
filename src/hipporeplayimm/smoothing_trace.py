"""Auditable exact traces for first-order finite-state smoothing.

The core state-space decoder stores transition matrices in column-stochastic
form: transition[destination, source]. This module makes every forward and
backward quantity explicit so downstream replay analyses cannot silently mix
that convention with the row-stochastic convention used by Bayesian-ACh.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix, diags, issparse

from .state_space_utils import (
    _coerce_valid_bin_mask,
    _scaled_emissions,
    _uniform_probabilities,
)

SMOOTHING_TRACE_SCHEMA_VERSION = "hipporeplayimm.first-order-smoothing-trace.v1"
TRANSITION_CONVENTION = (
    "column-stochastic: transition[destination, source] = "
    "P(x_t=destination | x_(t-1)=source)"
)
_TRANSITION_ATOL = 1e-12


@dataclass(frozen=True, slots=True)
class FirstOrderSmoothingTrace:
    """Complete normalized forward/backward trace.

    predicted_probabilities[t] conditions on emissions strictly before t.
    filtered_probabilities[t] conditions through t and
    smoothed_probabilities[t] conditions on the complete supplied interval.
    Backward messages use the same forward scaling constants and are messages,
    not categorical probabilities.

    Pair matrices are sparse and have axes [source, destination]. The filtering
    pair at step t conditions through emission t + 1; the smoothed pair
    conditions on the complete interval.
    """

    log_evidence: float
    predicted_probabilities: np.ndarray
    filtered_probabilities: np.ndarray
    smoothed_probabilities: np.ndarray
    backward_messages: np.ndarray
    filtering_pair_probabilities: tuple[csr_matrix, ...]
    smoothed_pair_probabilities: tuple[csr_matrix, ...]
    emission_offsets: np.ndarray
    forward_scales: np.ndarray
    log_predictive_probabilities: np.ndarray
    online_surprise: np.ndarray
    schema_version: str = field(default=SMOOTHING_TRACE_SCHEMA_VERSION, init=False)
    transition_convention: str = field(default=TRANSITION_CONVENTION, init=False)

    @property
    def n_time(self) -> int:
        return int(self.filtered_probabilities.shape[0])

    @property
    def n_states(self) -> int:
        return int(self.filtered_probabilities.shape[1])

    def pair_probability_array(self, *, smoothed: bool = True) -> np.ndarray:
        """Materialize pair marginals as [step, source, destination].

        Pair marginals remain sparse in the trace because a dense replay grid
        would require quadratic memory. This helper is intended for small
        diagnostics and golden tests.
        """

        pairs = (
            self.smoothed_pair_probabilities
            if smoothed
            else self.filtering_pair_probabilities
        )
        if not pairs:
            return np.empty((0, self.n_states, self.n_states), dtype=float)
        return np.stack([pair.toarray() for pair in pairs], axis=0)


def _coerce_initial_probabilities(
    initial_probabilities: np.ndarray | None,
    n_states: int,
    valid_bin_mask: np.ndarray | None,
) -> np.ndarray:
    valid_mask = _coerce_valid_bin_mask(valid_bin_mask, n_states)
    if initial_probabilities is None:
        return _uniform_probabilities(n_states, valid_mask)

    initial = np.asarray(initial_probabilities, dtype=float)
    if initial.shape != (n_states,):
        raise ValueError("initial_probabilities must contain one value per state")
    if not np.all(np.isfinite(initial)) or np.any(initial < 0.0):
        raise ValueError("initial_probabilities must be finite and nonnegative")
    total = float(initial.sum())
    if total <= 0.0:
        raise ValueError("initial_probabilities must contain positive mass")
    normalized = initial / total
    if valid_mask is not None and np.any(normalized[~valid_mask] > _TRANSITION_ATOL):
        raise ValueError("initial_probabilities must have zero mass outside valid_bin_mask")
    return normalized


def _coerce_transition(
    value: csr_matrix | np.ndarray,
    n_states: int,
    valid_bin_mask: np.ndarray | None,
) -> csr_matrix:
    try:
        matrix = csr_matrix(value, dtype=float).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError("each transition must be a finite square numeric matrix") from exc
    if matrix.shape != (n_states, n_states):
        raise ValueError(
            "each transition must have shape "
            f"({n_states}, {n_states}); got {matrix.shape}"
        )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if not np.all(np.isfinite(matrix.data)) or np.any(matrix.data < 0.0):
        raise ValueError("transition entries must be finite and nonnegative")
    column_sums = np.asarray(matrix.sum(axis=0), dtype=float).ravel()
    if not np.allclose(column_sums, 1.0, rtol=0.0, atol=_TRANSITION_ATOL):
        raise ValueError(
            "transition must be column-stochastic under "
            "transition[destination, source]"
        )
    valid_mask = _coerce_valid_bin_mask(valid_bin_mask, n_states)
    if valid_mask is not None:
        invalid_mass = np.asarray(matrix[~valid_mask].sum(axis=0), dtype=float).ravel()
        if np.any(invalid_mass > _TRANSITION_ATOL):
            raise ValueError("transition must assign zero mass to invalid destinations")
    return matrix


def _coerce_transitions(
    transitions: csr_matrix | np.ndarray | Sequence[csr_matrix | np.ndarray],
    n_time: int,
    n_states: int,
    valid_bin_mask: np.ndarray | None,
) -> tuple[csr_matrix, ...]:
    expected = max(n_time - 1, 0)
    if issparse(transitions):
        matrix = _coerce_transition(transitions, n_states, valid_bin_mask)
        return tuple(matrix for _ in range(expected))

    if isinstance(transitions, np.ndarray):
        values = np.asarray(transitions)
        if values.ndim == 2:
            matrix = _coerce_transition(values, n_states, valid_bin_mask)
            return tuple(matrix for _ in range(expected))
        if values.ndim == 3:
            raw_sequence: Sequence[csr_matrix | np.ndarray] = [
                values[index] for index in range(values.shape[0])
            ]
        else:
            raise ValueError(
                "transitions must be a square matrix or one matrix per adjacent time pair"
            )
    else:
        raw_sequence = list(transitions)

    if len(raw_sequence) != expected:
        raise ValueError(
            f"transitions must contain {expected} adjacent-time matrices; "
            f"got {len(raw_sequence)}"
        )
    return tuple(
        _coerce_transition(value, n_states, valid_bin_mask)
        for value in raw_sequence
    )


def _normalized_sparse_pair(
    transition: csr_matrix,
    source_factor: np.ndarray,
    destination_factor: np.ndarray,
    *,
    label: str,
) -> csr_matrix:
    source_diagonal = diags(np.asarray(source_factor, dtype=float))
    destination_diagonal = diags(np.asarray(destination_factor, dtype=float))
    pair = (source_diagonal @ transition.T @ destination_diagonal).tocsr()
    total = float(pair.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError(f"{label} pair marginal has no finite positive mass")
    pair = (pair / total).tocsr()
    pair.eliminate_zeros()
    return pair


def first_order_smoothing_trace(
    log_likelihood: np.ndarray,
    transitions: csr_matrix | np.ndarray | Sequence[csr_matrix | np.ndarray],
    *,
    initial_probabilities: np.ndarray | None = None,
    valid_bin_mask: np.ndarray | None = None,
) -> FirstOrderSmoothingTrace:
    """Return an exact first-order filtering and fixed-interval smoothing trace.

    Emission rows may include arbitrary additive log offsets. The returned
    categorical quantities are invariant to those offsets, while log_evidence
    and log_predictive_probabilities restore them exactly.
    """

    values = np.asarray(log_likelihood, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("log_likelihood must have shape (positive time, positive state)")
    n_time, n_states = values.shape
    valid_mask = _coerce_valid_bin_mask(valid_bin_mask, n_states)
    scaled, offsets = _scaled_emissions(values, valid_bin_mask=valid_mask)
    transition_sequence = _coerce_transitions(
        transitions,
        n_time,
        n_states,
        valid_mask,
    )
    initial = _coerce_initial_probabilities(
        initial_probabilities,
        n_states,
        valid_mask,
    )

    predicted = np.empty((n_time, n_states), dtype=float)
    filtered = np.empty_like(predicted)
    scales = np.empty(n_time, dtype=float)

    predicted[0] = initial
    unnormalized = predicted[0] * scaled[0]
    scales[0] = float(unnormalized.sum())
    if not np.isfinite(scales[0]) or scales[0] <= 0.0:
        raise FloatingPointError("the first emission has no finite predictive mass")
    filtered[0] = unnormalized / scales[0]

    for time_index, transition in enumerate(transition_sequence, start=1):
        predicted[time_index] = np.asarray(
            transition @ filtered[time_index - 1],
            dtype=float,
        )
        unnormalized = predicted[time_index] * scaled[time_index]
        scales[time_index] = float(unnormalized.sum())
        if not np.isfinite(scales[time_index]) or scales[time_index] <= 0.0:
            raise FloatingPointError(
                f"emission row {time_index} has no finite predicted mass"
            )
        filtered[time_index] = unnormalized / scales[time_index]

    backward = np.ones_like(filtered)
    for time_index in range(n_time - 2, -1, -1):
        backward[time_index] = np.asarray(
            transition_sequence[time_index].T
            @ (scaled[time_index + 1] * backward[time_index + 1]),
            dtype=float,
        )
        backward[time_index] /= scales[time_index + 1]

    smoothed = filtered * backward
    smoothed_mass = smoothed.sum(axis=1)
    if not np.all(np.isfinite(smoothed_mass)) or np.any(smoothed_mass <= 0.0):
        raise FloatingPointError("at least one smoothed marginal has no finite mass")
    smoothed /= smoothed_mass[:, None]

    filtering_pairs: list[csr_matrix] = []
    smoothed_pairs: list[csr_matrix] = []
    for time_index, transition in enumerate(transition_sequence):
        filtering_pairs.append(
            _normalized_sparse_pair(
                transition,
                filtered[time_index],
                scaled[time_index + 1],
                label=f"filtering step {time_index}",
            )
        )
        smoothed_pairs.append(
            _normalized_sparse_pair(
                transition,
                filtered[time_index],
                scaled[time_index + 1] * backward[time_index + 1],
                label=f"smoothing step {time_index}",
            )
        )

    log_predictive = np.log(scales) + offsets
    return FirstOrderSmoothingTrace(
        log_evidence=float(log_predictive.sum()),
        predicted_probabilities=predicted,
        filtered_probabilities=filtered,
        smoothed_probabilities=smoothed,
        backward_messages=backward,
        filtering_pair_probabilities=tuple(filtering_pairs),
        smoothed_pair_probabilities=tuple(smoothed_pairs),
        emission_offsets=offsets,
        forward_scales=scales,
        log_predictive_probabilities=log_predictive,
        online_surprise=-log_predictive,
    )


__all__ = [
    "FirstOrderSmoothingTrace",
    "SMOOTHING_TRACE_SCHEMA_VERSION",
    "TRANSITION_CONVENTION",
    "first_order_smoothing_trace",
]
