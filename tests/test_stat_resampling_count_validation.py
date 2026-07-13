from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.cli_float_values_validation import _positive_integer_count
from hipporeplayimm.result_improvements import hierarchical_bootstrap_ci, paired_sign_flip_p_value


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": ["s1", "s1", "s2", "s2"],
            "model": ["imm", "imm", "imm", "imm"],
            "delta_vs_best_static": [1.0, -0.5, 2.0, 0.25],
        }
    )


@pytest.mark.parametrize(
    "bad_count",
    [0, -1, 1.5, True, [2], "2", b"2", np.str_("2"), np.asarray("2")],
)
def test_hierarchical_bootstrap_ci_rejects_invalid_bootstrap_count(bad_count) -> None:
    with pytest.raises((TypeError, ValueError), match="n_bootstrap"):
        hierarchical_bootstrap_ci(_rows(), model="imm", n_bootstrap=bad_count)


@pytest.mark.parametrize(
    "bad_count",
    [0, -1, 1.5, False, [2], "2", b"2", np.str_("2"), np.asarray("2")],
)
def test_paired_sign_flip_rejects_invalid_permutation_count(bad_count) -> None:
    with pytest.raises((TypeError, ValueError), match="n_permutations"):
        paired_sign_flip_p_value(_rows(), model="imm", n_permutations=bad_count)


@pytest.mark.parametrize(
    "count",
    [
        2**53 + 1,
        np.int64(2**53 + 1),
        np.asarray(2**53 + 1, dtype=np.int64),
        Decimal(str(2**53 + 1)),
    ],
)
def test_resampling_count_validation_preserves_exact_large_integers(count) -> None:
    assert _positive_integer_count("n_bootstrap", count) == 2**53 + 1
