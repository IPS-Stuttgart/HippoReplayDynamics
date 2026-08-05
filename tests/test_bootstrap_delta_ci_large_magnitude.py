from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from hipporeplayimm import benchmarks


def test_flat_bootstrap_handles_large_finite_deltas_without_overflow() -> None:
    rows = pd.DataFrame(
        {
            "model": ["imm", "imm", "imm"],
            "delta_vs_best_static": [1e308, 1e308, -1e308],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        interval = benchmarks.bootstrap_delta_ci(
            rows,
            model="imm",
            n_bootstrap=1_000,
            random_seed=1,
        )

    assert interval == (-1e308, 1e308)
    assert np.isfinite(interval).all()
