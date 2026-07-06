from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import hipporeplayimm
from hipporeplayimm import result_improvements


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": ["s1", "s1", "s2", "s2"],
            "model": ["imm", "imm", "imm", "imm"],
            "delta_vs_best_static": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_resampling_helpers_reject_boolean_seed() -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(ValueError, match="random_seed"):
        result_improvements.hierarchical_bootstrap_ci(_rows(), model="imm", n_bootstrap=2, random_seed=True)

    with pytest.raises(ValueError, match="random_seed"):
        result_improvements.paired_sign_flip_p_value(_rows(), model="imm", n_permutations=2, random_seed=True)


def test_resampling_helpers_reject_string_seed() -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(ValueError, match="random_seed"):
        result_improvements.hierarchical_bootstrap_ci(_rows(), model="imm", n_bootstrap=2, random_seed="2")

    with pytest.raises(ValueError, match="random_seed"):
        result_improvements.paired_sign_flip_p_value(_rows(), model="imm", n_permutations=2, random_seed=np.array("2"))


def test_resampling_helpers_reject_fractional_seed() -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(ValueError, match="random_seed"):
        result_improvements.hierarchical_bootstrap_ci(_rows(), model="imm", n_bootstrap=2, random_seed=1.5)

    with pytest.raises(ValueError, match="random_seed"):
        result_improvements.paired_sign_flip_p_value(_rows(), model="imm", n_permutations=2, random_seed=1.5)


def test_resampling_helpers_accept_integer_like_seed() -> None:
    hipporeplayimm.apply_runtime_patches()

    lo, hi = result_improvements.hierarchical_bootstrap_ci(_rows(), model="imm", n_bootstrap=4, random_seed=2.0)
    p_value = result_improvements.paired_sign_flip_p_value(_rows(), model="imm", n_permutations=4, random_seed=np.int64(2))

    assert np.isfinite(lo)
    assert np.isfinite(hi)
    assert lo <= hi
    assert 0.0 <= p_value <= 1.0
