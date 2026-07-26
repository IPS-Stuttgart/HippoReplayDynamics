import numpy as np
import pandas as pd
import pytest

import hipporeplayimm.cli as cli
from hipporeplayimm import apply_runtime_patches
from hipporeplayimm import benchmarks


def test_bootstrap_delta_ci_filters_nonfinite_and_malformed_target_metrics():
    rows = pd.DataFrame(
        {
            "model": ["imm", "imm", "imm", "imm", "imm", "diffusion"],
            "delta_vs_best_static": [1.0, np.inf, "2.0", -np.inf, "bad", np.inf],
        }
    )
    finite_rows = rows.iloc[[0, 2, 5]].copy()

    actual = benchmarks.bootstrap_delta_ci(
        rows,
        model="imm",
        n_bootstrap=128,
        random_seed=7,
    )
    expected = benchmarks.bootstrap_delta_ci(
        finite_rows,
        model="imm",
        n_bootstrap=128,
        random_seed=7,
    )

    assert actual == expected
    assert np.isfinite(actual).all()


@pytest.mark.parametrize("n_bootstrap", [True, 0, -1, 1.5, "10", np.array([10])])
def test_bootstrap_delta_ci_rejects_invalid_resample_counts(n_bootstrap):
    rows = pd.DataFrame(
        {
            "model": ["imm"],
            "delta_vs_best_static": [1.0],
        }
    )

    with pytest.raises(ValueError, match="n_bootstrap must be"):
        benchmarks.bootstrap_delta_ci(rows, n_bootstrap=n_bootstrap)


def test_bootstrap_delta_ci_runtime_patch_refreshes_cli_alias():
    apply_runtime_patches()

    assert cli.bootstrap_delta_ci is benchmarks.bootstrap_delta_ci
