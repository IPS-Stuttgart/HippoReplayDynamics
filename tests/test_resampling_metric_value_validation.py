from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import hipporeplayimm
from hipporeplayimm import result_improvements


def test_hierarchical_bootstrap_filters_nonfinite_metrics_before_resampling() -> None:
    rows = pd.DataFrame(
        {
            "session": ["s1"] * 4 + ["s2"] * 3,
            "model": ["imm"] * 7,
            "delta_vs_best_static": [1.0, 2.0, 100.0, np.inf, 3.0, 4.0, 5.0],
        }
    )
    finite_rows = rows[np.isfinite(rows["delta_vs_best_static"])].copy()

    hipporeplayimm.apply_runtime_patches()
    actual = result_improvements.hierarchical_bootstrap_ci(
        rows,
        model="imm",
        n_bootstrap=10,
        random_seed=1,
    )
    expected = result_improvements.hierarchical_bootstrap_ci(
        finite_rows,
        model="imm",
        n_bootstrap=10,
        random_seed=1,
    )

    assert actual == pytest.approx(expected)
    assert actual[1] > 30.0


def test_sign_flip_returns_nan_when_target_model_has_no_finite_metrics() -> None:
    rows = pd.DataFrame(
        {
            "model": ["imm", "imm", "other"],
            "delta_vs_best_static": [np.inf, -np.inf, 1.0],
        }
    )

    hipporeplayimm.apply_runtime_patches()
    p_value = result_improvements.paired_sign_flip_p_value(
        rows,
        model="imm",
        n_permutations=10,
        random_seed=1,
    )

    assert np.isnan(p_value)


def test_resampling_metric_filter_preserves_numeric_strings() -> None:
    rows = pd.DataFrame(
        {
            "session": ["s1", "s1"],
            "model": ["imm", "imm"],
            "delta_vs_best_static": ["1.0", "2.0"],
        }
    )

    hipporeplayimm.apply_runtime_patches()
    interval = result_improvements.hierarchical_bootstrap_ci(
        rows,
        model="imm",
        n_bootstrap=4,
        random_seed=2,
    )
    p_value = result_improvements.paired_sign_flip_p_value(
        rows,
        model="imm",
        n_permutations=4,
        random_seed=2,
    )

    assert np.isfinite(interval).all()
    assert 0.0 <= p_value <= 1.0


def test_resampling_normalizes_byte_backed_model_labels() -> None:
    rows = pd.DataFrame(
        {
            "session": ["s1", "s1", "s2", "s2"],
            "model": [
                b"imm",
                bytearray(b"imm"),
                memoryview(b"imm"),
                np.bytes_("imm"),
            ],
            "delta_vs_best_static": [1.0, 2.0, 3.0, 4.0],
        }
    )
    text_rows = rows.copy()
    text_rows["model"] = "imm"

    hipporeplayimm.apply_runtime_patches()
    actual_interval = result_improvements.hierarchical_bootstrap_ci(
        rows,
        model="imm",
        n_bootstrap=10,
        random_seed=3,
    )
    expected_interval = result_improvements.hierarchical_bootstrap_ci(
        text_rows,
        model="imm",
        n_bootstrap=10,
        random_seed=3,
    )
    actual_p_value = result_improvements.paired_sign_flip_p_value(
        rows,
        model="imm",
        n_permutations=10,
        random_seed=3,
    )
    expected_p_value = result_improvements.paired_sign_flip_p_value(
        text_rows,
        model="imm",
        n_permutations=10,
        random_seed=3,
    )

    assert actual_interval == pytest.approx(expected_interval)
    assert actual_p_value == pytest.approx(expected_p_value)
