from __future__ import annotations

import numpy as np

import hipporeplayimm
from hipporeplayimm import result_improvements


def test_result_improvement_count_validator_preserves_large_exact_integers() -> None:
    count = 2**53 + 1

    assert result_improvements._positive_integer_count(count, "n_bootstrap") == count
    assert (
        result_improvements._positive_integer_count(
            np.uint64(count),
            "n_permutations",
        )
        == count
    )


def test_resampling_count_precision_patch_is_idempotent() -> None:
    active = result_improvements._positive_integer_count

    hipporeplayimm.apply_runtime_patches()

    assert result_improvements._positive_integer_count is active
    assert result_improvements._positive_integer_count(2**53 + 1, "n_bootstrap") == 2**53 + 1
