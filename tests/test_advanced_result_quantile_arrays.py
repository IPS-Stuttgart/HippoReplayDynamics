from __future__ import annotations

import numpy as np
import pandas as pd

import hipporeplayimm
import hipporeplayimm.advanced_result_diagnostics as diagnostics
from hipporeplayimm.advanced_result_diagnostics import _quantile


def test_advanced_diagnostics_quantile_accepts_numpy_arrays():
    assert _quantile(np.array([1.0, 3.0, 5.0]), 0.5) == 3.0


def test_advanced_diagnostics_quantile_flattens_nested_numpy_arrays():
    values = [np.array([1.0, 3.0]), np.array([5.0])]

    assert _quantile(values, 0.5) == 3.0


def test_advanced_diagnostics_quantile_accepts_iterators():
    values = (value for value in [1.0, 3.0, 5.0])

    assert _quantile(values, 0.5) == 3.0


def test_advanced_diagnostics_quantile_empty_inputs_return_nan():
    assert np.isnan(_quantile([], 0.5))
    assert np.isnan(_quantile(np.array([], dtype=float), 0.5))


def test_advanced_diagnostics_quantile_ignores_nonfinite_values():
    assert _quantile(np.array([np.nan, 1.0, np.inf, 3.0]), 0.5) == 2.0


def test_advanced_diagnostics_quantile_all_nonfinite_values_return_nan():
    assert np.isnan(_quantile(np.array([np.nan, np.inf, -np.inf]), 0.5))


def test_advanced_diagnostics_quantile_ignores_pandas_missing_values():
    assert _quantile(pd.array([pd.NA, 1.0, 3.0], dtype="Float64"), 0.5) == 2.0


def test_advanced_diagnostics_quantile_patch_refreshes_stale_true_flag():
    def stale_quantile(values, q):
        raise RuntimeError("stale quantile helper")

    diagnostics._quantile = stale_quantile
    setattr(diagnostics, "_advanced_result_quantile_array_patch_applied", True)

    hipporeplayimm.apply_runtime_patches()

    assert getattr(diagnostics._quantile, "_advanced_result_quantile_array_wrapper", False)
    values = [np.array([1.0, 3.0]), np.array([5.0])]
    assert diagnostics._quantile(values, 0.5) == 3.0
