from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm.state_space import _candidate_evidence_support_label


@pytest.mark.parametrize(
    "valid_bin_mask",
    [
        np.array(["yes", ""], dtype=str),
        np.array([1.0 + 0.0j, 0.0 + 0.0j]),
        np.array([2, 0], dtype=int),
    ],
)
def test_candidate_evidence_support_label_rejects_malformed_valid_bin_masks(valid_bin_mask: np.ndarray) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(ValueError, match="valid_bin_mask"):
        _candidate_evidence_support_label(
            [np.array([0], dtype=int)],
            n_bins=2,
            valid_bin_mask=valid_bin_mask,
        )


def test_candidate_evidence_support_label_accepts_binary_valid_bin_mask() -> None:
    hipporeplayimm.apply_runtime_patches()

    assert (
        _candidate_evidence_support_label(
            [np.array([0], dtype=int)],
            n_bins=2,
            valid_bin_mask=np.array([1, 0], dtype=int),
        )
        == "exact_full_grid"
    )
