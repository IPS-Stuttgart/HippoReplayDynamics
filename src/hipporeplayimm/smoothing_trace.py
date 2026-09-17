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
from scipy.special import logsumexp

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


def _contains_scalar_kind(
    value: object,
    scalar_types: tuple[type, ...],
    dtype_kinds: set[str],
) -> bool:
    """Inspect semantic scalar leaves through nested object wrappers."""

    if issparse(value):
        return value.dtype.kind in dtype_kinds

    pending = [value]
    seen_arrays: set[int] = set()
    while pending:
        current = pending.pop()
        if isinstance(current, scalar_types):
            return True
        if isinstance(current, np.ndarray):
            marker = id(current)
            if marker in seen_arrays:
                continue
            seen_arrays.add(marker)
        try:
            raw = np.asarray(current)
        except (TypeError, ValueError, OverflowError):
            continue
        if raw.dtype.kind in dtype_kinds:
            return True
        if raw.size == 0 or raw.dtype != object:
            continue
        if raw.ndim == 0:
            try:
                item = raw.item()
            except (TypeError, ValueError):
                continue
            if item is not current:
                pending.append(item)
            continue
        pending.extend(raw.reshape(-1))
    return False


def _has_complex_dtype(value: object) -> bool:
    """Return whether an array-like input contains complex scalar values."""

    return _contains_scalar_kind(
        value,
        (complex, np.complexfloating),
        {"c"},
    )


def _contains_boolean_values(value: object) -> bool:
    """Return whether an array-like input contains boolean scalar values."""

    return _contains_scalar_kind(
        value,
        (bool, np.bool_),
        {"b"},
    )


def _contains_text_values(value: object) -> bool:
    """Return whether an array-like input contains text scalar values."""

    return _contains_scalar_kind(
        value,
        (str, bytes, np.str_, np.bytes_),
        {"U", "S"},
    )


def _reject_coercive_scalar_values(value: object, name: str) -> None:
    """Reject semantic nonnumeric scalars before NumPy can coerce them."""

    if _contains_boolean_values(value):
        raise ValueError(f"{name} must be numeric, not boolean")
    if _contains_text_values(value):
        raise ValueError(f"{name} must be numeric, not text")


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

    if _has_complex_dtype(initial_probabilities):
        raise ValueError("initial_probabilities must be real-valued")
    _reject_coercive_scalar_values(initial_probabilities, "initial_probabilities")
    initial = np.asarray(initial_probabilities, dtype=float)
    if initial.shape != (n_states,):
        raise ValueError("initial_probabilities must contain one value per state")
    if not np.all(np.isfinite(initial)) or np.any(initial < 0.0):
        raise ValueError("initial_probabilities must be finite and nonnegative")
    scale = float(np.max(initial))
    if scale <= 0.0:
        raise ValueError("initial_probabilities must contain positive mass")
    scaled_initial = initial / scale
    normalized = scaled_initial / float(scaled_initial.sum())
    if valid_mask is not None and np.any(normalized[~valid_mask] > _TRANSITION_ATOL):
        raise ValueError("initial_probabilities must have zero mass outside valid_bin_mask")
    return normalized


def _coerce_transition(
    value: csr_matrix | np.ndarray,
    n_states: int,
    valid_bin_mask: np.ndarray | None,
) -> csr_matrix:
    if _has_complex_dtype(value):
        raise ValueError("each transition must be real-valued")
    _reject_coercive_scalar_values(value, "each transition")
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


def _log_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Return exact log probabilities, preserving zero mass as ``-inf``."""

    values = np.asarray(probabilities, dtype=float)
    output = np.full(values.shape, -np.inf, dtype=float)
    positive = values > 0.0
    output[positive] = np.log(values[positive])
    return output


def _log_sparse_matrix(matrix: csr_matrix) -> csr_matrix:
    """Return a CSR matrix whose stored positive probabilities are logged."""

    output = matrix.tocsr(copy=True)
    output.sum_duplicates()
    output.eliminate_zeros()
    output.data = np.log(output.data)
    return output


def _log_sparse_matvec(log_matrix: csr_matrix, values: np.ndarray) -> np.ndarray:
    """Compute ``log(exp(log_matrix) @ exp(values))`` without underflow."""

    vector = np.asarray(values, dtype=float)
    output = np.full(log_matrix.shape[0], -np.inf, dtype=float)
    nonempty = np.diff(log_matrix.indptr) > 0
    if np.any(nonempty):
        terms = log_matrix.data + vector[log_matrix.indices]
        output[nonempty] = np.logaddexp.reduceat(
            terms,
            log_matrix.indptr[:-1][nonempty],
        )
    return output


def _normalized_sparse_log_pair(
    transition: csr_matrix,
    source_log_factor: np.ndarray,
    destination_log_factor: np.ndarray,
    *,
    label: str,
) -> csr_matrix:
    """Return a normalized sparse [source, destination] pair marginal."""

    matrix = transition.tocoo(copy=False)
    if matrix.nnz == 0:
        raise FloatingPointError(f"{label} pair marginal has no finite positive mass")
    log_weights = (
        np.log(matrix.data)
        + np.asarray(source_log_factor, dtype=float)[matrix.col]
        + np.asarray(destination_log_factor, dtype=float)[matrix.row]
    )
    log_total = float(logsumexp(log_weights))
    if not np.isfinite(log_total):
        raise FloatingPointError(f"{label} pair marginal has no finite positive mass")
    probabilities = np.exp(log_weights - log_total)
    pair = csr_matrix(
        (probabilities, (matrix.col, matrix.row)),
        shape=transition.shape,
    )
    pair.sum_duplicates()
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

    The dynamic program is evaluated in log space.  This is required for the
    trace to retain the same exact finite-log-likelihood semantics as the core
    first-order scorer when emission likelihood ratios fall below binary64's
    probability-space range.
    """

    if _has_complex_dtype(log_likelihood):
        raise ValueError("log_likelihood must be real-valued")
    _reject_coercive_scalar_values(log_likelihood, "log_likelihood")
    values = np.asarray(log_likelihood, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("log_likelihood must have shape (positive time, positive state)")
    n_time, n_states = values.shape
    valid_mask = _coerce_valid_bin_mask(valid_bin_mask, n_states)

    # Reuse the shared validation and offsets, but never use its clipped
    # probability representation for inference.  Clipping a finite shifted
    # log-likelihood below -745 changes exact evidence and posterior odds.
    _, offsets = _scaled_emissions(values, valid_bin_mask=valid_mask)
    active_log_likelihood = values.copy()
    if valid_mask is not None:
        active_log_likelihood[:, ~valid_mask] = -np.inf

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

    forward_log_transitions = tuple(
        _log_sparse_matrix(transition) for transition in transition_sequence
    )
    backward_log_transitions = tuple(
        _log_sparse_matrix(transition.T.tocsr()) for transition in transition_sequence
    )

    log_predicted = np.full((n_time, n_states), -np.inf, dtype=float)
    log_filtered = np.full_like(log_predicted, -np.inf)
    log_predictive = np.empty(n_time, dtype=float)

    log_predicted[0] = _log_probabilities(initial)
    for time_index in range(n_time):
        if time_index:
            log_predicted[time_index] = _log_sparse_matvec(
                forward_log_transitions[time_index - 1],
                log_filtered[time_index - 1],
            )
            predicted_mass = float(logsumexp(log_predicted[time_index]))
            if not np.isfinite(predicted_mass):
                raise FloatingPointError(
                    f"prediction at row {time_index} has no finite mass"
                )
            # The transition is stochastic, so this is mathematically zero;
            # normalizing removes only accumulated floating-point roundoff.
            log_predicted[time_index] -= predicted_mass

        joint = log_predicted[time_index] + active_log_likelihood[time_index]
        log_predictive[time_index] = float(logsumexp(joint))
        if not np.isfinite(log_predictive[time_index]):
            label = "the first emission" if time_index == 0 else f"emission row {time_index}"
            raise FloatingPointError(f"{label} has no finite predicted mass")
        log_filtered[time_index] = joint - log_predictive[time_index]

    log_backward = np.zeros_like(log_filtered)
    for time_index in range(n_time - 2, -1, -1):
        continuation = (
            active_log_likelihood[time_index + 1]
            + log_backward[time_index + 1]
        )
        log_backward[time_index] = _log_sparse_matvec(
            backward_log_transitions[time_index],
            continuation,
        ) - log_predictive[time_index + 1]

    log_smoothed = log_filtered + log_backward
    smoothed_norm = logsumexp(log_smoothed, axis=1)
    if not np.all(np.isfinite(smoothed_norm)):
        raise FloatingPointError("at least one smoothed marginal has no finite mass")
    log_smoothed -= smoothed_norm[:, None]

    with np.errstate(over="ignore", under="ignore"):
        predicted = np.exp(log_predicted)
        filtered = np.exp(log_filtered)
        smoothed = np.exp(log_smoothed)
        backward = np.exp(log_backward)

    filtering_pairs: list[csr_matrix] = []
    smoothed_pairs: list[csr_matrix] = []
    for time_index, transition in enumerate(transition_sequence):
        filtering_pairs.append(
            _normalized_sparse_log_pair(
                transition,
                log_filtered[time_index],
                active_log_likelihood[time_index + 1],
                label=f"filtering step {time_index}",
            )
        )
        smoothed_pairs.append(
            _normalized_sparse_log_pair(
                transition,
                log_filtered[time_index],
                active_log_likelihood[time_index + 1]
                + log_backward[time_index + 1],
                label=f"smoothing step {time_index}",
            )
        )

    log_forward_scales = log_predictive - offsets
    with np.errstate(under="ignore"):
        forward_scales = np.exp(log_forward_scales)

    return FirstOrderSmoothingTrace(
        log_evidence=float(log_predictive.sum()),
        predicted_probabilities=predicted,
        filtered_probabilities=filtered,
        smoothed_probabilities=smoothed,
        backward_messages=backward,
        filtering_pair_probabilities=tuple(filtering_pairs),
        smoothed_pair_probabilities=tuple(smoothed_pairs),
        emission_offsets=offsets,
        forward_scales=forward_scales,
        log_predictive_probabilities=log_predictive,
        online_surprise=-log_predictive,
    )


__all__ = [
    "FirstOrderSmoothingTrace",
    "SMOOTHING_TRACE_SCHEMA_VERSION",
    "TRANSITION_CONVENTION",
    "first_order_smoothing_trace",
]
