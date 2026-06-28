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


def test_mark_complex_validation_patch_does_not_rewrap_current_wrapper() -> None:
    import hipporeplayimm
    import hipporeplayimm.data as data

    hipporeplayimm.apply_runtime_patches()
    patched = data._coerce_mark_matrix

    hipporeplayimm.apply_runtime_patches()

    assert data._coerce_mark_matrix is patched
