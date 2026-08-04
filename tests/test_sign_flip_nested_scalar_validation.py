from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.sign_flip_report import (
    paired_sign_flip_test,
    score_table_sign_flip_summary,
)


def _nested_zero_dimensional_object(value: object) -> np.ndarray:
    inner = np.empty((), dtype=object)
    inner[()] = value
    outer = np.empty((), dtype=object)
    outer[()] = inner
    return outer


@pytest.mark.parametrize("value", [True, np.bool_(False)])
def test_paired_sign_flip_rejects_nested_boolean_deltas(value: object) -> None:
    values = [1.0, _nested_zero_dimensional_object(value), -1.0]

    with pytest.raises(ValueError, match="not booleans"):
        paired_sign_flip_test(values)


@pytest.mark.parametrize("value", [True, np.bool_(False)])
def test_score_table_sign_flip_rejects_nested_boolean_deltas(value: object) -> None:
    frame = pd.DataFrame(
        {
            "model": ["imm"],
            "delta_vs_best_static": pd.Series(
                [_nested_zero_dimensional_object(value)],
                dtype=object,
            ),
        }
    )

    with pytest.raises(ValueError, match="delta_vs_best_static contains boolean values"):
        score_table_sign_flip_summary(frame)


@pytest.mark.parametrize("value", [1.0 + 2.0j, np.complex128(1.0 + 0.0j)])
def test_paired_sign_flip_rejects_nested_complex_without_lossy_cast(value: object) -> None:
    values = [1.0, _nested_zero_dimensional_object(value), -1.0]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="not complex values"):
            paired_sign_flip_test(values)


@pytest.mark.parametrize("value", [1.0 + 2.0j, np.complex128(1.0 + 0.0j)])
def test_score_table_sign_flip_rejects_nested_complex_without_lossy_cast(value: object) -> None:
    frame = pd.DataFrame(
        {
            "model": ["imm"],
            "delta_vs_best_static": pd.Series(
                [_nested_zero_dimensional_object(value)],
                dtype=object,
            ),
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="delta_vs_best_static contains complex values"):
            score_table_sign_flip_summary(frame)


def test_sign_flip_accepts_nested_real_scalars() -> None:
    positive = _nested_zero_dimensional_object(np.float64(1.0))
    negative = _nested_zero_dimensional_object(np.int64(-1))

    result = paired_sign_flip_test([positive, negative])
    assert result.observed_mean == pytest.approx(0.0)
    assert result.n_observations == 2
    assert result.p_value == pytest.approx(1.0)

    frame = pd.DataFrame(
        {
            "model": ["imm", "imm"],
            "delta_vs_best_static": pd.Series([positive, negative], dtype=object),
        }
    )
    summary = score_table_sign_flip_summary(frame)
    assert summary.loc[0, "observed_mean"] == pytest.approx(0.0)
    assert int(summary.loc[0, "n_observations"]) == 2
    assert summary.loc[0, "p_value"] == pytest.approx(1.0)
