from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.result_improvements import hierarchical_bootstrap_ci


def test_hierarchical_bootstrap_retains_partial_missing_group_keys() -> None:
    rows = pd.DataFrame(
        {
            "model": ["target"] * 4,
            "session": ["Rat1", "Rat1", None, None],
            "delta_vs_best_static": [1.0, 3.0, 10.0, 14.0],
        }
    )
    explicit = rows.fillna({"session": "missing-session"})

    observed = hierarchical_bootstrap_ci(
        rows,
        model="target",
        n_bootstrap=128,
        random_seed=7,
    )
    expected = hierarchical_bootstrap_ci(
        explicit,
        model="target",
        n_bootstrap=128,
        random_seed=7,
    )

    assert np.isfinite(observed).all()
    assert observed == pytest.approx(expected)


def test_hierarchical_bootstrap_retains_all_missing_group_keys() -> None:
    rows = pd.DataFrame(
        {
            "model": ["target", "target"],
            "session": [None, None],
            "delta_vs_best_static": [1.0, 3.0],
        }
    )
    explicit = rows.fillna({"session": "missing-session"})

    observed = hierarchical_bootstrap_ci(
        rows,
        model="target",
        n_bootstrap=64,
        random_seed=11,
    )
    expected = hierarchical_bootstrap_ci(
        explicit,
        model="target",
        n_bootstrap=64,
        random_seed=11,
    )

    assert np.isfinite(observed).all()
    assert observed == pytest.approx(expected)
