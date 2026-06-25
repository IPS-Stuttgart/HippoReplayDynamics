from __future__ import annotations

import numpy as np

from hipporeplayimm.advanced_result_diagnostics import _quantile


def test_advanced_diagnostics_quantile_accepts_numpy_arrays():
    assert _quantile(np.array([1.0, 3.0, 5.0]), 0.5) == 3.0


def test_advanced_diagnostics_quantile_empty_inputs_return_nan():
    assert np.isnan(_quantile([], 0.5))
    assert np.isnan(_quantile(np.array([], dtype=float), 0.5))
