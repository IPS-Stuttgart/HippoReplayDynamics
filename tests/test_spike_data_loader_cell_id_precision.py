from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm import data


def test_spike_data_loader_rejects_boolean_cell_ids_before_float_coercion() -> None:
    raw = np.array([[0.25, True]], dtype=object)

    with pytest.raises(ValueError, match="boolean identifiers"):
        data._as_two_column_array(raw, "Spike_Data")


def test_spike_data_loader_preserves_distinct_large_cell_ids() -> None:
    expected_ids = np.array([2**53, 2**53 + 1], dtype=np.int64)
    if np.iinfo(np.dtype(int)).max < int(expected_ids[-1]):
        pytest.skip("requires platform integers wider than the binary64 exact-integer range")

    raw = np.empty((2, 2), dtype=object)
    raw[:, 0] = [0.25, 0.5]
    raw[:, 1] = expected_ids

    loaded = data._as_two_column_array(raw, "Spike_Data")

    assert loaded.dtype == object
    np.testing.assert_array_equal(np.asarray(loaded[:, 0], dtype=float), np.array([0.25, 0.5]))
    np.testing.assert_array_equal(np.asarray(loaded[:, 1], dtype=int), expected_ids)


def test_spike_data_loader_keeps_float_table_for_ordinary_ids() -> None:
    raw = np.array([[0.25, 1.0], [0.5, 2.0]], dtype=float)

    loaded = data._as_two_column_array(raw, "Spike_Data")

    assert loaded.dtype == np.dtype(float)
    np.testing.assert_array_equal(loaded, raw)


def test_runtime_patch_refreshes_replaced_spike_data_loader(monkeypatch) -> None:
    def lossy_as_two_column_array(value, name):
        del name
        return np.asarray(value, dtype=float)

    monkeypatch.setattr(data, "_as_two_column_array", lossy_as_two_column_array)

    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(ValueError, match="boolean identifiers"):
        data._as_two_column_array(np.array([[0.25, True]], dtype=object), "Spike_Data")
