from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

import hipporeplayimm
from hipporeplayimm import advanced_result_diagnostics as diagnostics


def _wrong_map_deltas() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat2/Open1"],
            "statistic": ["fixed_model", "fixed_model"],
            "selected_model": ["stationary", "stationary"],
            "delta_map_log_evidence": [1.0, 2.0],
        }
    )


@pytest.mark.parametrize(
    "seed",
    [
        2**53 + 1,
        np.uint64(2**63 + 1),
        Decimal("9007199254740995"),
    ],
)
def test_wrong_map_rat_bootstrap_preserves_exact_large_integer_seed(seed: object) -> None:
    hipporeplayimm.apply_runtime_patches()

    out = diagnostics.rat_bootstrap_wrong_map_absolute_evidence_summary(
        _wrong_map_deltas(),
        n_bootstrap=2,
        random_seed=seed,
    )

    assert int(out.loc[0, "random_seed"]) == int(seed)


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("n_bootstrap", "2"),
        ("n_bootstrap", b"2"),
        ("random_seed", "2"),
        ("random_seed", np.str_("2")),
        ("random_seed", Decimal("9007199254740993.5")),
    ],
)
def test_wrong_map_rat_bootstrap_rejects_lossy_integer_controls(
    argument: str,
    value: object,
) -> None:
    hipporeplayimm.apply_runtime_patches()
    kwargs: dict[str, object] = {"n_bootstrap": 2, "random_seed": 1}
    kwargs[argument] = value

    with pytest.raises(ValueError, match=argument):
        diagnostics.rat_bootstrap_wrong_map_absolute_evidence_summary(
            _wrong_map_deltas(),
            **kwargs,
        )
