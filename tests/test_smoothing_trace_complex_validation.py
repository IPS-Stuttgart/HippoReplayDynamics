from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from hipporeplayimm.smoothing_trace import first_order_smoothing_trace


_IDENTITY = np.eye(2, dtype=float)
_LOG_LIKELIHOOD = np.zeros((2, 2), dtype=float)


def test_smoothing_trace_rejects_complex_log_likelihood() -> None:
    log_likelihood = np.array(
        [[0.0 + 1.0j, 0.0], [0.0, 0.0]],
        dtype=complex,
    )

    with pytest.raises(ValueError, match="log_likelihood must be real-valued"):
        first_order_smoothing_trace(log_likelihood, _IDENTITY)


def test_smoothing_trace_rejects_complex_initial_probabilities() -> None:
    initial = np.array([1.0 + 1.0j, 1.0], dtype=complex)

    with pytest.raises(ValueError, match="initial_probabilities must be real-valued"):
        first_order_smoothing_trace(
            _LOG_LIKELIHOOD,
            _IDENTITY,
            initial_probabilities=initial,
        )


@pytest.mark.parametrize(
    "transition",
    [
        np.array([[1.0 + 1.0j, 0.0], [0.0, 1.0 - 1.0j]], dtype=complex),
        csr_matrix(
            np.array(
                [[1.0 + 1.0j, 0.0], [0.0, 1.0 - 1.0j]],
                dtype=complex,
            )
        ),
    ],
    ids=["dense", "sparse"],
)
def test_smoothing_trace_rejects_complex_transition(transition: object) -> None:
    with pytest.raises(ValueError, match="each transition must be real-valued"):
        first_order_smoothing_trace(_LOG_LIKELIHOOD, transition)
