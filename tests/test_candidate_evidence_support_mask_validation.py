import numpy as np
import pytest

import hipporeplayimm  # noqa: F401 - applies runtime validation patches
from hipporeplayimm.state_space_model import _candidate_evidence_support_label


def test_candidate_evidence_support_rejects_string_valid_bin_mask() -> None:
    candidates = [np.array([0, 1])]
    valid_mask = np.array(["False", "True"], dtype=object)

    with pytest.raises(ValueError, match="valid_bin_mask entries must be boolean or 0/1"):
        _candidate_evidence_support_label(candidates, n_bins=2, valid_bin_mask=valid_mask)


def test_candidate_evidence_support_accepts_numeric_valid_bin_mask() -> None:
    candidates = [np.array([1])]
    valid_mask = np.array([0, 1], dtype=int)

    label = _candidate_evidence_support_label(candidates, n_bins=2, valid_bin_mask=valid_mask)

    assert label == "exact_full_grid"
