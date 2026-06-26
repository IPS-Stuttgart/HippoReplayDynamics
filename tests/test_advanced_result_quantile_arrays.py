from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.advanced_result_diagnostics import _quantile


def test_advanced_diagnostics_quantile_accepts_numpy_arrays():
    assert _quantile(np.array([1.0, 3.0, 5.0]), 0.5) == 3.0


def test_advanced_diagnostics_quantile_empty_inputs_return_nan():
    assert np.isnan(_quantile([], 0.5))
    assert np.isnan(_quantile(np.array([], dtype=float), 0.5))


def test_advanced_diagnostics_quantile_ignores_nonfinite_values():
    assert _quantile(np.array([np.nan, 1.0, np.inf, 3.0]), 0.5) == 2.0


def test_advanced_diagnostics_quantile_all_nonfinite_values_return_nan():
    assert np.isnan(_quantile(np.array([np.nan, np.inf, -np.inf]), 0.5))


def test_advanced_diagnostics_quantile_ignores_pandas_missing_values():
    assert _quantile(pd.array([pd.NA, 1.0, 3.0], dtype="Float64"), 0.5) == 2.0
