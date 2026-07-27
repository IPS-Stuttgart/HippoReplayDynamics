from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import hipporeplayimm
from hipporeplayimm import advanced_result_diagnostics as diagnostics


def _scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1", "Rat2/Open1", "Rat2/Open1"],
            "model": ["imm", "imm", "imm", "imm"],
            "relative_log_evidence": [1.0, -0.5, 2.0, 0.25],
        }
    )


@pytest.mark.parametrize(
    "bad_count",
    [0, -1, 1.5, True, [2], "2", b"2", np.str_("2"), np.asarray("2")],
)
def test_hierarchical_bootstrap_rejects_invalid_bootstrap_count(bad_count) -> None:
    with pytest.raises(ValueError, match="n_bootstrap"):
        diagnostics.hierarchical_bootstrap(
            _scores(),
            model="imm",
            n_bootstrap=bad_count,
        )


@pytest.mark.parametrize(
    "bad_seed",
    [-1, 1.5, False, [2], "2", b"2", np.str_("2"), np.asarray("2")],
)
def test_hierarchical_bootstrap_rejects_invalid_random_seed(bad_seed) -> None:
    with pytest.raises(ValueError, match="random_seed"):
        diagnostics.hierarchical_bootstrap(
            _scores(),
            model="imm",
            n_bootstrap=8,
            random_seed=bad_seed,
        )


def test_hierarchical_bootstrap_accepts_integral_numeric_scalars() -> None:
    expected = diagnostics.hierarchical_bootstrap(
        _scores(),
        model="imm",
        n_bootstrap=8,
        random_seed=2,
    )
    actual = diagnostics.hierarchical_bootstrap(
        _scores(),
        model="imm",
        n_bootstrap=np.float64(8.0),
        random_seed=np.asarray(2.0),
    )

    assert actual == expected


def test_hierarchical_bootstrap_patch_is_idempotent() -> None:
    patched = diagnostics.hierarchical_bootstrap
    hipporeplayimm.apply_runtime_patches()
    assert diagnostics.hierarchical_bootstrap is patched
