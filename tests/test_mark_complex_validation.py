from __future__ import annotations

import numpy as np


def test_runtime_patch_reinstalls_mark_complex_validation_after_stale_helper(monkeypatch) -> None:
    import hipporeplayimm
    import hipporeplayimm.data as data
    import hipporeplayimm.mark_complex_validation as mark_complex_validation

    def stale_coerce_mark_matrix(value, *, spike_count: int, spike_times: np.ndarray):
        arr = np.asarray(value)
        if arr.ndim == 1 and arr.shape[0] == spike_count:
            return np.real(arr).astype(float, copy=False).reshape(spike_count, 1)
        return None

    spike_times = np.array([0.0, 1.0])
    complex_marks = np.array([1.0 + 1.0j, 2.0 + 0.0j])
    real_complex_marks = np.array([1.0 + 0.0j, 2.0 + 0.0j])

    monkeypatch.setattr(data, "_coerce_mark_matrix", stale_coerce_mark_matrix)
    monkeypatch.setattr(data, mark_complex_validation._PATCHED_FLAG, True, raising=False)

    assert data._coerce_mark_matrix(complex_marks, spike_count=2, spike_times=spike_times) is not None

    hipporeplayimm.apply_runtime_patches()

    assert getattr(data._coerce_mark_matrix, mark_complex_validation._PATCH_WRAPPER_ATTR)
    assert data._coerce_mark_matrix(complex_marks, spike_count=2, spike_times=spike_times) is None
    np.testing.assert_allclose(
        data._coerce_mark_matrix(real_complex_marks, spike_count=2, spike_times=spike_times),
        np.array([[1.0], [2.0]]),
    )


def test_mark_time_column_requires_matching_finite_support() -> None:
    import hipporeplayimm.data as data

    spike_times = np.array([1.0, 2.0, 3.0])
    marks = np.array(
        [
            [1.0, 40.0],
            [np.nan, 50.0],
            [3.0, 60.0],
        ]
    )

    assert not data._looks_like_time_column(marks[:, 0], spike_times)
    np.testing.assert_allclose(
        data._coerce_mark_matrix(marks, spike_count=3, spike_times=spike_times),
        marks,
        equal_nan=True,
    )


def test_runtime_patch_reinstalls_mark_time_column_validation_after_stale_helper(monkeypatch) -> None:
    import hipporeplayimm
    import hipporeplayimm.data as data
    import hipporeplayimm.mark_complex_validation as mark_complex_validation

    def stale_looks_like_time_column(candidate: np.ndarray, spike_times: np.ndarray) -> bool:
        return True

    monkeypatch.setattr(data, "_looks_like_time_column", stale_looks_like_time_column)
    monkeypatch.setattr(data, mark_complex_validation._PATCHED_FLAG, True, raising=False)

    hipporeplayimm.apply_runtime_patches()

    assert getattr(
        data._looks_like_time_column,
        mark_complex_validation._TIME_COLUMN_WRAPPER_ATTR,
    )
    assert not data._looks_like_time_column(
        np.array([1.0, np.nan, 3.0]),
        np.array([1.0, 2.0, 3.0]),
    )


def test_mark_complex_validation_patch_does_not_rewrap_current_wrapper() -> None:
    import hipporeplayimm
    import hipporeplayimm.data as data

    hipporeplayimm.apply_runtime_patches()
    patched_coerce = data._coerce_mark_matrix
    patched_time_column = data._looks_like_time_column

    hipporeplayimm.apply_runtime_patches()

    assert data._coerce_mark_matrix is patched_coerce
    assert data._looks_like_time_column is patched_time_column
