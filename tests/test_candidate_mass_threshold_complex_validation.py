from __future__ import annotations

import warnings

import numpy as np
import pytest

from hipporeplayimm.state_space import _mass_retaining_candidate_indices


def _nested_object_scalar(value: object) -> np.ndarray:
    inner = np.empty((), dtype=object)
    inner[()] = value
    outer = np.empty((), dtype=object)
    outer[()] = inner
    return outer


@pytest.mark.parametrize(
    "threshold",
    [
        np.clongdouble(0.85 + 0.125j),
        np.clongdouble(0.85 + 0.0j),
        np.array(np.clongdouble(0.85 + 0.125j), dtype=object),
        _nested_object_scalar(np.clongdouble(0.85 + 0.125j)),
    ],
)
def test_mass_retaining_candidate_support_rejects_complex_threshold(
    threshold: object,
) -> None:
    log_emission = np.log(np.array([0.60, 0.25, 0.10, 0.05]))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TypeError, match="mass_threshold.*real numeric probability"):
            _mass_retaining_candidate_indices(
                log_emission,
                mass_threshold=threshold,
                top_k=1,
                min_k=0,
                max_k=0,
            )


def test_mass_retaining_candidate_support_preserves_real_longdouble_threshold() -> None:
    log_emission = np.log(np.array([0.60, 0.25, 0.10, 0.05]))

    selected = _mass_retaining_candidate_indices(
        log_emission,
        mass_threshold=np.longdouble("0.85"),
        top_k=1,
        min_k=0,
        max_k=0,
    )

    assert selected.tolist() == [0, 1]
