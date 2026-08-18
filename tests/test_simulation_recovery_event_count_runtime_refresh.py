from __future__ import annotations

import importlib

import hipporeplayimm
import hipporeplayimm.simulation_recovery as simulation_recovery
from hipporeplayimm.simulation_recovery_event_count import (
    _CERTIFIED_EVENT_PATCHED_FLAG,
    _CERTIFIED_EVENT_WRAPPER_FLAG,
    _CERTIFIED_SUMMARY_WRAPPER_FLAG,
    _PATCHED_FLAG,
    _RECOVERY_SUMMARY_WRAPPER_FLAG,
    _wrapper_chain_has_marker,
)


def test_runtime_patches_restore_event_count_wrappers_after_recovery_reload() -> None:
    module = importlib.reload(simulation_recovery)

    # importlib.reload() retains module-dictionary entries that the source does
    # not redefine, so these legacy sentinels survive even though the wrapped
    # functions themselves have just been replaced.
    assert getattr(module, _PATCHED_FLAG, False)
    assert getattr(module, _CERTIFIED_EVENT_PATCHED_FLAG, False)
    assert not _wrapper_chain_has_marker(
        module.recovery_summary,
        _RECOVERY_SUMMARY_WRAPPER_FLAG,
    )
    assert not _wrapper_chain_has_marker(
        module.certified_vs_exact_recovery_summary,
        _CERTIFIED_SUMMARY_WRAPPER_FLAG,
    )
    assert not _wrapper_chain_has_marker(
        module.certified_vs_exact_event_recovery,
        _CERTIFIED_EVENT_WRAPPER_FLAG,
    )

    hipporeplayimm.apply_runtime_patches()

    assert _wrapper_chain_has_marker(
        module.recovery_summary,
        _RECOVERY_SUMMARY_WRAPPER_FLAG,
    )
    assert _wrapper_chain_has_marker(
        module.certified_vs_exact_recovery_summary,
        _CERTIFIED_SUMMARY_WRAPPER_FLAG,
    )
    assert _wrapper_chain_has_marker(
        module.certified_vs_exact_event_recovery,
        _CERTIFIED_EVENT_WRAPPER_FLAG,
    )
