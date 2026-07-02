from __future__ import annotations

from hipporeplayimm.advanced_result_diagnostics import _quantile


def test_mapping_quantile_uses_values_not_keys():
    values = {0: [1.0, 3.0], 1: [5.0]}
    assert _quantile(values, 0.5) == 3.0
