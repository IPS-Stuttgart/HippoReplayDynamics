from __future__ import annotations

import warnings

import numpy as np
import pytest

from hipporeplayimm.result_quality_gates import (
    MARGIN_STRONG,
    MARGIN_UNKNOWN,
    evidence_margin_label,
)


def _nested_object_scalar(value: object) -> np.ndarray:
    inner = np.empty((), dtype=object)
    inner[()] = value
    outer = np.empty((), dtype=object)
    outer[()] = inner
    return outer


@pytest.mark.parametrize(
    "margin",
    [
        True,
        False,
        np.bool_(True),
        np.bool_(False),
        np.asarray(True),
        np.asarray([False]),
        np.asarray(np.bool_(True), dtype=object),
        np.asarray([np.bool_(False)], dtype=object),
        _nested_object_scalar(np.bool_(True)),
    ],
)
def test_evidence_margin_label_rejects_boolean_margins(margin: object) -> None:
    # Boolean arrays are pseudo-numeric inputs, not valid evidence margins.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert evidence_margin_label(margin) == MARGIN_UNKNOWN


def test_evidence_margin_label_keeps_nested_real_scalar() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert (
            evidence_margin_label(_nested_object_scalar(np.float64(4.0)))
            == MARGIN_STRONG
        )
