from __future__ import annotations

import sys
import types
import warnings

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm import data
from hipporeplayimm.data_interval_validation import _synchronize_interval_aliases


def _nested_scalar(value: object) -> np.ndarray:
    inner = np.empty((), dtype=object)
    inner[()] = value
    outer = np.empty((), dtype=object)
    outer[()] = inner
    return outer


def _intervals(start: object, end: object = 2.0) -> np.ndarray:
    intervals = np.empty((1, 2), dtype=object)
    intervals[0, 0] = start
    intervals[0, 1] = end
    return intervals


@pytest.mark.parametrize(
    "start",
    [
        np.bool_(False),
        np.complex128(0.0 + 1.0j),
        "0.0",
        _nested_scalar(np.bool_(False)),
        _nested_scalar(np.complex128(0.0 + 1.0j)),
        np.array([0.0, 1.0], dtype=float),
    ],
)
def test_epoch_interval_loader_rejects_non_real_numeric_bounds(
    start: object,
) -> None:
    hipporeplayimm.apply_runtime_patches()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="Intervals must contain finite real"):
            data._as_intervals(_intervals(start))


def test_epoch_interval_loader_rejects_cyclic_scalar_wrapper() -> None:
    hipporeplayimm.apply_runtime_patches()
    cyclic = np.empty((), dtype=object)
    cyclic[()] = cyclic

    with pytest.raises(ValueError, match="Intervals must contain finite real"):
        data._as_intervals(_intervals(cyclic))


def test_epoch_interval_loader_accepts_nested_real_scalar_bounds() -> None:
    hipporeplayimm.apply_runtime_patches()
    intervals = _intervals(
        _nested_scalar(np.float64(1.25)),
        _nested_scalar(np.int64(3)),
    )

    validated = data._as_intervals(intervals)

    np.testing.assert_array_equal(validated, np.array([[1.25, 3.0]]))


def test_epoch_interval_loader_accepts_extended_precision_real_scalars() -> None:
    hipporeplayimm.apply_runtime_patches()
    intervals = _intervals(
        np.longdouble("1.25"),
        _nested_scalar(np.longdouble("3.5")),
    )

    validated = data._as_intervals(intervals)

    np.testing.assert_array_equal(validated, np.array([[1.25, 3.5]]))


def test_interval_alias_sync_ignores_similarly_named_top_level_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = object()
    replacement = object()
    unrelated = types.ModuleType("hipporeplayimm_extension")
    package_child = types.ModuleType("hipporeplayimm._interval_alias_probe")
    unrelated._as_intervals = original
    package_child._as_intervals = original
    monkeypatch.setitem(sys.modules, unrelated.__name__, unrelated)
    monkeypatch.setitem(sys.modules, package_child.__name__, package_child)

    _synchronize_interval_aliases(original, replacement)

    assert unrelated._as_intervals is original
    assert package_child._as_intervals is replacement
