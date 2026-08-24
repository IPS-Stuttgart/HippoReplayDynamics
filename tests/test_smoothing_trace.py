from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.special import logsumexp

from hipporeplayimm.smoothing_trace import (
    SMOOTHING_TRACE_SCHEMA_VERSION,
    TRANSITION_CONVENTION,
    first_order_smoothing_trace,
)
from hipporeplayimm.state_space import (
    _forward_backward_first_order,
    _gaussian_transition_matrix,
)


def _enumerated_posteriors(
    initial: np.ndarray,
    row_transitions: np.ndarray,
    likelihoods: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    n_time, n_states = likelihoods.shape
    paths = list(itertools.product(range(n_states), repeat=n_time))
    log_weights = np.empty(len(paths), dtype=float)
    for path_index, path in enumerate(paths):
        value = np.log(initial[path[0]]) + np.log(likelihoods[0, path[0]])
        for time_index in range(1, n_time):
            value += np.log(
                row_transitions[
                    time_index - 1,
                    path[time_index - 1],
                    path[time_index],
                ]
            )
            value += np.log(likelihoods[time_index, path[time_index]])
        log_weights[path_index] = value
    log_evidence = float(logsumexp(log_weights))
    state = np.zeros((n_time, n_states), dtype=float)
    pair = np.zeros((n_time - 1, n_states, n_states), dtype=float)
    weights = np.exp(log_weights - log_evidence)
    for path, weight in zip(paths, weights, strict=True):
        for time_index, current in enumerate(path):
            state[time_index, current] += weight
        for time_index in range(n_time - 1):
            pair[time_index, path[time_index], path[time_index + 1]] += weight
    return log_evidence, state, pair


def test_trace_matches_existing_first_order_smoothed_trajectory() -> None:
    log_likelihood = np.array(
        [
            [-0.1, -1.3, -2.0],
            [-1.5, -0.2, -1.1],
            [-2.2, -0.7, -0.05],
            [-0.8, -0.4, -0.9],
        ],
        dtype=float,
    )
    centers = np.array([[0.0], [1.0], [2.0]], dtype=float)
    transition = _gaussian_transition_matrix(centers, 0.8, 8.0)

    expected_log_evidence, expected_log_smoothed = _forward_backward_first_order(
        log_likelihood,
        transition,
    )
    trace = first_order_smoothing_trace(log_likelihood, transition)

    assert trace.schema_version == SMOOTHING_TRACE_SCHEMA_VERSION
    assert trace.transition_convention == TRANSITION_CONVENTION
    np.testing.assert_allclose(trace.log_evidence, expected_log_evidence, atol=1e-13)
    np.testing.assert_allclose(
        trace.smoothed_probabilities,
        np.exp(expected_log_smoothed),
        rtol=1e-12,
        atol=1e-13,
    )


def test_asymmetric_time_varying_trace_matches_exhaustive_paths() -> None:
    initial = np.array([0.65, 0.35], dtype=float)
    row_transitions = np.array(
        [
            [[0.90, 0.10], [0.30, 0.70]],
            [[0.55, 0.45], [0.05, 0.95]],
        ],
        dtype=float,
    )
    likelihoods = np.array(
        [
            [0.60, 0.40],
            [0.15, 0.85],
            [0.92, 0.08],
        ],
        dtype=float,
    )
    column_transitions = np.transpose(row_transitions, (0, 2, 1))

    expected_log_evidence, expected_state, expected_pair = _enumerated_posteriors(
        initial,
        row_transitions,
        likelihoods,
    )
    trace = first_order_smoothing_trace(
        np.log(likelihoods),
        column_transitions,
        initial_probabilities=initial,
    )

    np.testing.assert_allclose(trace.log_evidence, expected_log_evidence, atol=1e-13)
    np.testing.assert_allclose(
        trace.smoothed_probabilities,
        expected_state,
        rtol=1e-12,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        trace.pair_probability_array(),
        expected_pair,
        rtol=1e-12,
        atol=1e-13,
    )
    pair = trace.pair_probability_array()
    np.testing.assert_allclose(pair.sum(axis=2), expected_state[:-1], atol=1e-13)
    np.testing.assert_allclose(pair.sum(axis=1), expected_state[1:], atol=1e-13)


def test_trace_matches_bayesian_ach_row_stochastic_toy_after_transpose() -> None:
    initial = np.array([0.7, 0.2, 0.1], dtype=float)
    row_transition = np.array(
        [
            [0.75, 0.20, 0.05],
            [0.10, 0.70, 0.20],
            [0.15, 0.25, 0.60],
        ],
        dtype=float,
    )
    likelihoods = np.array(
        [
            [0.8, 0.1, 0.2],
            [0.1, 0.9, 0.3],
            [0.2, 0.4, 0.95],
            [0.7, 0.2, 0.1],
        ],
        dtype=float,
    )

    predicted = np.empty_like(likelihoods)
    filtered = np.empty_like(likelihoods)
    scales = np.empty(likelihoods.shape[0], dtype=float)
    predicted[0] = initial
    for time_index in range(likelihoods.shape[0]):
        if time_index:
            predicted[time_index] = filtered[time_index - 1] @ row_transition
        unnormalized = predicted[time_index] * likelihoods[time_index]
        scales[time_index] = float(unnormalized.sum())
        filtered[time_index] = unnormalized / scales[time_index]
    backward = np.ones_like(likelihoods)
    for time_index in range(likelihoods.shape[0] - 2, -1, -1):
        backward[time_index] = row_transition @ (
            likelihoods[time_index + 1] * backward[time_index + 1]
        )
        backward[time_index] /= scales[time_index + 1]
    smoothed = filtered * backward
    smoothed /= smoothed.sum(axis=1, keepdims=True)

    trace = first_order_smoothing_trace(
        np.log(likelihoods),
        row_transition.T,
        initial_probabilities=initial,
    )

    np.testing.assert_allclose(trace.predicted_probabilities, predicted, atol=1e-13)
    np.testing.assert_allclose(trace.filtered_probabilities, filtered, atol=1e-13)
    np.testing.assert_allclose(trace.backward_messages, backward, atol=1e-13)
    np.testing.assert_allclose(trace.smoothed_probabilities, smoothed, atol=1e-13)
    np.testing.assert_allclose(trace.log_evidence, np.log(scales).sum(), atol=1e-13)


def test_log_emission_offsets_change_only_absolute_log_scores() -> None:
    log_likelihood = np.log(
        np.array(
            [
                [0.25, 0.75],
                [0.80, 0.20],
                [0.35, 0.65],
            ],
            dtype=float,
        )
    )
    transition = csr_matrix(np.array([[0.85, 0.30], [0.15, 0.70]], dtype=float))
    offsets = np.array([1000.0, -700.0, 23.5], dtype=float)

    baseline = first_order_smoothing_trace(log_likelihood, transition)
    shifted = first_order_smoothing_trace(
        log_likelihood + offsets[:, None],
        transition,
    )

    for field in (
        "predicted_probabilities",
        "filtered_probabilities",
        "smoothed_probabilities",
        "backward_messages",
        "forward_scales",
    ):
        np.testing.assert_allclose(
            getattr(shifted, field),
            getattr(baseline, field),
            rtol=1e-12,
            atol=1e-13,
        )
    np.testing.assert_allclose(
        shifted.pair_probability_array(),
        baseline.pair_probability_array(),
        atol=1e-13,
    )
    np.testing.assert_allclose(
        shifted.pair_probability_array(smoothed=False),
        baseline.pair_probability_array(smoothed=False),
        atol=1e-13,
    )
    np.testing.assert_allclose(
        shifted.emission_offsets,
        baseline.emission_offsets + offsets,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        shifted.log_predictive_probabilities,
        baseline.log_predictive_probabilities + offsets,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        shifted.log_evidence,
        baseline.log_evidence + offsets.sum(),
        atol=1e-12,
    )


def test_future_emission_cannot_change_prefix_prediction_or_filtering() -> None:
    transition = np.array([[0.8, 0.1], [0.2, 0.9]], dtype=float)
    first = np.log(np.array([[0.7, 0.3], [0.2, 0.8], [0.5, 0.5]]))
    second = first.copy()
    second[-1] = np.log(np.array([0.999, 0.001]))

    baseline = first_order_smoothing_trace(first, transition)
    changed = first_order_smoothing_trace(second, transition)

    np.testing.assert_allclose(
        changed.predicted_probabilities,
        baseline.predicted_probabilities,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        changed.filtered_probabilities[:-1],
        baseline.filtered_probabilities[:-1],
        atol=1e-13,
    )
    assert not np.allclose(
        changed.smoothed_probabilities[:-1],
        baseline.smoothed_probabilities[:-1],
    )


def test_column_stochastic_convention_rejects_untransposed_bayesian_ach_matrix() -> None:
    row_stochastic = np.array([[0.9, 0.1], [0.3, 0.7]], dtype=float)
    log_likelihood = np.zeros((2, 2), dtype=float)

    with pytest.raises(ValueError, match="column-stochastic"):
        first_order_smoothing_trace(log_likelihood, row_stochastic)

    trace = first_order_smoothing_trace(log_likelihood, row_stochastic.T)
    np.testing.assert_allclose(
        trace.predicted_probabilities[1],
        np.array([0.6, 0.4]),
        atol=1e-13,
    )


def test_valid_mask_and_impossible_emissions_retain_exact_support() -> None:
    valid = np.array([True, False, True])
    transition = np.array(
        [
            [0.8, 0.5, 0.2],
            [0.0, 0.0, 0.0],
            [0.2, 0.5, 0.8],
        ],
        dtype=float,
    )
    log_likelihood = np.array(
        [
            [0.0, 100.0, -np.inf],
            [-np.inf, 100.0, 0.0],
            [-0.2, 100.0, -0.3],
        ],
        dtype=float,
    )

    trace = first_order_smoothing_trace(
        log_likelihood,
        transition,
        valid_bin_mask=valid,
    )

    for values in (
        trace.predicted_probabilities,
        trace.filtered_probabilities,
        trace.smoothed_probabilities,
    ):
        np.testing.assert_array_equal(values[:, 1], np.zeros(values.shape[0]))
        np.testing.assert_allclose(values.sum(axis=1), 1.0, atol=1e-13)
    np.testing.assert_array_equal(
        trace.pair_probability_array()[:, 1, :],
        np.zeros((2, 3)),
    )
    np.testing.assert_array_equal(
        trace.pair_probability_array()[:, :, 1],
        np.zeros((2, 3)),
    )


def test_transition_to_invalid_destination_is_rejected() -> None:
    valid = np.array([True, False])
    transition = np.array([[0.9, 0.2], [0.1, 0.8]], dtype=float)

    with pytest.raises(ValueError, match="invalid destinations"):
        first_order_smoothing_trace(
            np.zeros((2, 2), dtype=float),
            transition,
            valid_bin_mask=valid,
        )
