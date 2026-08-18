from __future__ import annotations

import importlib

import numpy as np

import hipporeplayimm
from hipporeplayimm import advanced_result_diagnostics as diagnostics

_WRAPPER_FLAG = "_advanced_result_common_support_tie_wrapper"


def test_common_support_breaks_top_k_ties_by_original_bin_order() -> None:
    hipporeplayimm.apply_runtime_patches()

    flat = np.zeros((1, 6), dtype=float)
    cutoff_tie = np.asarray([[5.0, 4.0, 4.0, 4.0, 1.0]], dtype=float)

    np.testing.assert_array_equal(
        diagnostics.common_support_from_emissions(flat, top_k=3)[0],
        np.asarray([0, 1, 2]),
    )
    np.testing.assert_array_equal(
        diagnostics.common_support_from_emissions(cutoff_tie, top_k=2)[0],
        np.asarray([0, 1]),
    )


def test_common_support_tie_breaking_preserves_explicit_extras() -> None:
    hipporeplayimm.apply_runtime_patches()

    flat = np.zeros((1, 6), dtype=float)
    support = diagnostics.common_support_from_emissions(
        flat,
        top_k=2,
        extra_candidate_sets=[[5]],
    )

    np.testing.assert_array_equal(support[0], np.asarray([0, 1, 5]))


def test_common_support_patch_is_restored_after_diagnostics_reload() -> None:
    hipporeplayimm.apply_runtime_patches()
    flat = np.zeros((1, 6), dtype=float)

    assert getattr(diagnostics.common_support_from_emissions, _WRAPPER_FLAG, False)

    importlib.reload(diagnostics)
    assert not getattr(diagnostics.common_support_from_emissions, _WRAPPER_FLAG, False)

    hipporeplayimm.apply_runtime_patches()

    assert getattr(diagnostics.common_support_from_emissions, _WRAPPER_FLAG, False)
    np.testing.assert_array_equal(
        diagnostics.common_support_from_emissions(flat, top_k=3)[0],
        np.asarray([0, 1, 2]),
    )
