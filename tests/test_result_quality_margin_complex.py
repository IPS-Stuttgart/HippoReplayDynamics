from __future__ import annotations

import warnings

import numpy as np
import pytest

from hipporeplayimm.result_quality_gates import (
    MARGIN_DECISIVE,
    MARGIN_STRONG,
    MARGIN_UNKNOWN,
    evidence_margin_label,
)


def _nested_object_complex(value: complex) -> np.ndarray:
    inner = np.empty((), dtype=object)
    inner[()] = np.complex128(value)
    outer = np.empty((), dtype=object)
    outer[()] = inner
    return outer


@pytest.mark.parametrize(
    "margin",
    [
        np.complex64(12.0 + 5.0j),
        np.complex128(12.0 + 5.0j),
        np.asarray(np.complex128(4.0 + 1.0j), dtype=object),
        np.asarray([np.complex128(4.0 + 1.0j)], dtype=object),
        _nested_object_complex(4.0 + 1.0j),
    ],
)
def test_evidence_margin_label_rejects_complex_values_without_warnings(
    margin: object,
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert evidence_margin_label(margin) == MARGIN_UNKNOWN


def test_evidence_margin_label_keeps_real_numpy_scalars() -> None:
    assert evidence_margin_label(np.float64(4.0)) == MARGIN_STRONG
    assert evidence_margin_label(np.float64(12.0)) == MARGIN_DECISIVE
