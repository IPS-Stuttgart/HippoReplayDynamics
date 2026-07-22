from __future__ import annotations

import numpy as np
import pandas as pd


def _wrapper_depth(function, flag: str) -> int:
    depth = 0
    seen: set[int] = set()
    while getattr(function, flag, False):
        assert id(function) not in seen
        seen.add(id(function))
        depth += 1
        function = getattr(function, "__hipporeplayimm_original__")
    return depth


def test_evidence_margin_runtime_refresh_does_not_stack_wrappers() -> None:
    import hipporeplayimm
    from hipporeplayimm import advanced_result_diagnostics as diagnostics
    from hipporeplayimm import advanced_result_evidence_margin_duplicates as patch

    for _ in range(5):
        hipporeplayimm.apply_runtime_patches()

    assert _wrapper_depth(diagnostics.evidence_margin_table, patch._MARGIN_FLAG) == 1
    assert _wrapper_depth(diagnostics._as_bool, patch._BOOL_FLAG) == 1
    assert (
        _wrapper_depth(
            diagnostics.paired_model_margin_threshold_sweep,
            patch._EMPTY_SWEEP_FLAG,
        )
        == 1
    )

    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"] * 3,
            "event_index": [0] * 3,
            "model": ["diffusion", "diffusion", "momentum"],
            "log_evidence": [1.0, 3.0, 2.0],
            "status": ["success"] * 3,
            "evidence_comparable": [True] * 3,
        }
    )

    margin = diagnostics.evidence_margin_table(scores).iloc[0]

    assert margin["best_model_by_evidence"] == "diffusion"
    assert margin["second_best_model_by_evidence"] == "momentum"
    assert margin["models_compared"] == 2
    assert margin["evidence_margin_to_second_best"] == 1.0
    assert diagnostics._as_bool(np.int64(2))
