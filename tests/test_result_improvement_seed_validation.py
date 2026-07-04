from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import hipporeplayimm
from hipporeplayimm import result_improvements


def _metric_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": ["s1", "s1", "s2", "s2"],
            "model": ["imm", "imm", "imm", "imm"],
            "delta_vs_best_static": [1.0, 2.0, 3.0, 4.0],
        }
    )


@pytest.mark.parametrize("seed", [True, np.bool_(False), 1.5, -1, float("nan"), [1]])
def test_hierarchical_bootstrap_rejects_malformed_random_seed(seed: object) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(ValueError, match="random_seed"):
        result_improvements.hierarchical_bootstrap_ci(
            _metric_rows(),
            model="imm",
            n_bootstrap=2,
            random_seed=seed,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("seed", [True, np.bool_(False), 1.5, -1, float("nan"), [1]])
def test_paired_sign_flip_rejects_malformed_random_seed(seed: object) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(ValueError, match="random_seed"):
        result_improvements.paired_sign_flip_p_value(
            _metric_rows(),
            model="imm",
            n_permutations=2,
            random_seed=seed,  # type: ignore[arg-type]
        )


def test_result_improvement_resampling_accepts_integer_like_random_seed() -> None:
    hipporeplayimm.apply_runtime_patches()

    lo, hi = result_improvements.hierarchical_bootstrap_ci(
        _metric_rows(),
        model="imm",
        n_bootstrap=4,
        random_seed=2.0,  # type: ignore[arg-type]
    )
    p_value = result_improvements.paired_sign_flip_p_value(
        _metric_rows(),
        model="imm",
        n_permutations=4,
        random_seed=np.int64(2),
    )

    assert np.isfinite(lo)
    assert np.isfinite(hi)
    assert lo <= hi
    assert 0.0 <= p_value <= 1.0
