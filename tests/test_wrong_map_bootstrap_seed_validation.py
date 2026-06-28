from __future__ import annotations

import pandas as pd
import pytest

import hipporeplayimm
from hipporeplayimm import advanced_result_diagnostics as diagnostics


def _wrong_map_deltas() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open2"],
            "statistic": ["fixed_model", "fixed_model"],
            "selected_model": ["stationary", "stationary"],
            "delta_map_log_evidence": [1.0, 2.0],
        }
    )


@pytest.mark.parametrize("seed", [True, 1.5, float("nan"), -1])
def test_wrong_map_rat_bootstrap_rejects_invalid_random_seed(seed: object) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(ValueError, match="random_seed"):
        diagnostics.rat_bootstrap_wrong_map_absolute_evidence_summary(
            _wrong_map_deltas(),
            n_bootstrap=2,
            random_seed=seed,
        )


def test_wrong_map_rat_bootstrap_accepts_integer_like_random_seed() -> None:
    hipporeplayimm.apply_runtime_patches()

    out = diagnostics.rat_bootstrap_wrong_map_absolute_evidence_summary(
        _wrong_map_deltas(),
        n_bootstrap=2,
        random_seed=2.0,
    )

    assert out.loc[0, "random_seed"] == 2
    assert out.loc[0, "observed_rats"] == 1
