from __future__ import annotations

import json

import numpy as np

from hipporeplayimm.recovery_diagnostics import _json_ready


def test_recovery_diagnostics_json_ready_serializes_numpy_array_metadata() -> None:
    payload = {
        "vector": np.array([1.0, np.nan, np.inf]),
        "scalar": np.array(3, dtype=np.int64),
        "flags": np.array([True, False], dtype=np.bool_),
    }

    ready = _json_ready(payload)

    assert ready == {"vector": [1.0, None, None], "scalar": 3, "flags": [True, False]}
    assert json.loads(json.dumps(ready, sort_keys=True)) == ready
