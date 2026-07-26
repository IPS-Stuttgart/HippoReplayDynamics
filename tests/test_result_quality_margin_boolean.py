from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.result_quality_gates import MARGIN_UNKNOWN, evidence_margin_label


@pytest.mark.parametrize(
    "margin",
    [
        True,
        False,
        np.bool_(True),
        np.bool_(False),
        np.asarray(True),
        np.asarray([False]),
    ],
)
def test_evidence_margin_label_rejects_boolean_margins(margin: object) -> None:
    # Boolean arrays are pseudo-numeric inputs, not valid evidence margins.
    assert evidence_margin_label(margin) == MARGIN_UNKNOWN
