from __future__ import annotations

import importlib

import pandas as pd

import hipporeplayimm
from hipporeplayimm import shuffle_controls


class _OpaqueIndexScalar:
    """Index-like scalar whose string form deliberately drops integer identity."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __index__(self) -> int:
        return self.value

    def __str__(self) -> str:
        return "opaque-index"


def test_runtime_refresh_restores_exact_shuffle_scope_integer_wrapper() -> None:
    lower = 2**53
    upper = lower + 1

    hipporeplayimm.apply_runtime_patches()
    assert shuffle_controls._scope_label(_OpaqueIndexScalar(lower)) != shuffle_controls._scope_label(
        _OpaqueIndexScalar(upper)
    )

    reloaded = importlib.reload(shuffle_controls)
    assert getattr(reloaded, "_shuffle_scope_exact_integer_patch_applied", False)
    assert reloaded._scope_label(_OpaqueIndexScalar(lower)) == reloaded._scope_label(
        _OpaqueIndexScalar(upper)
    )

    hipporeplayimm.apply_runtime_patches()

    real_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": pd.Series(
                [_OpaqueIndexScalar(lower), _OpaqueIndexScalar(upper)], dtype=object
            ),
            "model": ["imm", "imm"],
            "log_evidence": [10.0, 20.0],
        }
    )
    control_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": pd.Series(
                [_OpaqueIndexScalar(lower), _OpaqueIndexScalar(upper)], dtype=object
            ),
            "model": ["imm", "imm"],
            "log_evidence": [9.0, 30.0],
        }
    )

    annotated = reloaded.add_shuffle_p_values(real_scores, control_scores)

    assert annotated["shuffle_count"].tolist() == [1, 1]
    assert annotated["shuffle_log_evidence_median"].tolist() == [9.0, 30.0]

    live_wrapper = reloaded._numeric_scope_label
    hipporeplayimm.apply_runtime_patches()
    assert reloaded._numeric_scope_label is live_wrapper
