from __future__ import annotations

import warnings

import pandas as pd

from hipporeplayimm.result_improvements import paired_sign_flip_p_value


def test_result_sign_flip_handles_large_finite_deltas_without_overflow() -> None:
    rows = pd.DataFrame(
        {
            "model": ["momentum", "momentum", "momentum"],
            "delta_vs_best_static": [1e308, 1e308, -1e308],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        p_value = paired_sign_flip_p_value(
            rows,
            model="momentum",
            n_permutations=1_000,
            random_seed=1,
        )

    assert p_value == 1.0
