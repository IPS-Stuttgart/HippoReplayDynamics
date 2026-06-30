from __future__ import annotations

import pandas as pd
import pytest

from hipporeplayimm.result_improvements import (
    hierarchical_bootstrap_ci,
    paired_sign_flip_p_value,
)


def _score_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["m", "m"],
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "delta_vs_best_static": [1.0, -0.5],
        }
    )


@pytest.mark.parametrize("bad_count", [0, -1, 1.5, True])
def test_sign_flip_rejects_invalid_permutation_counts(bad_count: object) -> None:
    with pytest.raises(ValueError, match="n_permutations"):
        paired_sign_flip_p_value(
            _score_rows(),
            model="m",
            n_permutations=bad_count,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("bad_count", [0, -1, 1.5, True])
def test_hierarchical_bootstrap_rejects_invalid_resample_counts(bad_count: object) -> None:
    with pytest.raises(ValueError, match="n_bootstrap"):
        hierarchical_bootstrap_ci(
            _score_rows(),
            model="m",
            n_bootstrap=bad_count,  # type: ignore[arg-type]
        )


def test_resampling_count_validators_preserve_positive_integer_counts() -> None:
    p_value = paired_sign_flip_p_value(_score_rows(), model="m", n_permutations=5)
    assert 0.0 <= p_value <= 1.0

    low, high = hierarchical_bootstrap_ci(_score_rows(), model="m", n_bootstrap=5)
    assert low <= high
