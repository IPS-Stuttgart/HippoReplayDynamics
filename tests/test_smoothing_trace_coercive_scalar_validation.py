from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from hipporeplayimm.smoothing_trace import first_order_smoothing_trace


_IDENTITY = np.eye(2, dtype=float)
_LOG_LIKELIHOOD = np.zeros((2, 2), dtype=float)


def _nested_scalar(value: object) -> np.ndarray:
    wrapped = np.empty((), dtype=object)
    wrapped[()] = value
    return wrapped


def _object_log_likelihood(value: object) -> np.ndarray:
    result = np.empty((2, 2), dtype=object)
    result[:] = 0.0
    result[0, 0] = value
    return result


def _object_initial_probabilities(value: object) -> np.ndarray:
    result = np.empty(2, dtype=object)
    result[:] = 1.0
    result[0] = value
    return result


def _object_transition(value: object) -> np.ndarray:
    result = np.empty((2, 2), dtype=object)
    result[:] = 0.0
    result[0, 0] = value
    result[1, 1] = 1.0
    return result


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (True, "boolean"),
        ("0.0", "text"),
        (_nested_scalar(True), "boolean"),
        (_nested_scalar("0.0"), "text"),
    ],
    ids=["boolean", "text", "nested-boolean", "nested-text"],
)
def test_smoothing_trace_rejects_coercive_log_likelihood_scalars(
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=rf"log_likelihood.*{message}"):
        first_order_smoothing_trace(
            _object_log_likelihood(value),
            _IDENTITY,
        )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (True, "boolean"),
        ("1.0", "text"),
        (_nested_scalar(True), "boolean"),
        (_nested_scalar("1.0"), "text"),
    ],
    ids=["boolean", "text", "nested-boolean", "nested-text"],
)
def test_smoothing_trace_rejects_coercive_initial_probability_scalars(
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=rf"initial_probabilities.*{message}"):
        first_order_smoothing_trace(
            _LOG_LIKELIHOOD,
            _IDENTITY,
            initial_probabilities=_object_initial_probabilities(value),
        )


@pytest.mark.parametrize(
    ("transition", "message"),
    [
        (_object_transition(True), "boolean"),
        (_object_transition("1.0"), "text"),
        (_object_transition(_nested_scalar(True)), "boolean"),
        (_object_transition(_nested_scalar("1.0")), "text"),
        (csr_matrix(np.eye(2, dtype=bool)), "boolean"),
    ],
    ids=["boolean", "text", "nested-boolean", "nested-text", "sparse-boolean"],
)
def test_smoothing_trace_rejects_coercive_transition_scalars(
    transition: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=rf"each transition.*{message}"):
        first_order_smoothing_trace(_LOG_LIKELIHOOD, transition)
