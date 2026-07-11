from __future__ import annotations

import pytest

import hipporeplayimm
import hipporeplayimm.clusterless as clusterless
import hipporeplayimm.encoding as encoding
import hipporeplayimm.kd_reference as kd_reference
from hipporeplayimm.place_field_run_local_kinematics import (
    _ORIGINAL_ATTR,
    _PATCHED_FLAG,
    _WRAPPER_MARKER,
)


def _run_local_wrapper(function):
    current = function
    seen: set[int] = set()
    while callable(current) and id(current) not in seen:
        if getattr(current, _WRAPPER_MARKER, False):
            return current
        seen.add(id(current))
        current = getattr(current, "__wrapped__", None)
    return None


@pytest.mark.parametrize(
    ("module", "function_name"),
    [
        (encoding, "fit_place_field_encoding"),
        (kd_reference, "fit_kd_place_field_encoding"),
        (clusterless, "fit_clusterless_mark_encoding"),
    ],
)
def test_runtime_patches_restore_replaced_run_local_encoder_wrapper(
    monkeypatch,
    module,
    function_name,
):
    installed = _run_local_wrapper(getattr(module, function_name))
    assert installed is not None
    original = getattr(installed, _ORIGINAL_ATTR)

    monkeypatch.setattr(module, function_name, original)
    monkeypatch.setattr(module, _PATCHED_FLAG, True, raising=False)

    hipporeplayimm.apply_runtime_patches()

    refreshed = _run_local_wrapper(getattr(module, function_name))
    assert refreshed is not None
    assert getattr(refreshed, _ORIGINAL_ATTR) is original
