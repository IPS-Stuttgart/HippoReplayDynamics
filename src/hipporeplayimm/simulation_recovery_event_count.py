"""Reload-safe entry point for simulation-recovery event-count patches."""

from __future__ import annotations

from . import simulation_recovery as _recovery
from . import simulation_recovery_negative_infinity as _negative_infinity
from .simulation_recovery_event_count_impl import (
    _distinct_event_count as _distinct_event_count,  # explicit compatibility re-export
    apply_simulation_recovery_event_count_patch as _apply_impl,
)

_PATCHED_FLAG = "_simulation_recovery_session_event_count_patch_applied"
_CERTIFIED_EVENT_PATCHED_FLAG = (
    "_simulation_recovery_certified_event_duplicate_model_patch_applied"
)


def apply_simulation_recovery_event_count_patch() -> None:
    """Install event-count patches and repair stale sentinels after reload.

    ``importlib.reload(simulation_recovery)`` reuses the module dictionary, so
    dynamically added module-level sentinels survive while source functions are
    replaced.  The standard recovery-summary wrapper is the stable canary for a
    full module reload: unlike the certified helpers, it is not intentionally
    replaced by the later status-coercion compatibility layer.
    """

    summary_wrapper_live = bool(
        getattr(_recovery.recovery_summary, _PATCHED_FLAG, False)
    )
    repairing_reload = not summary_wrapper_live

    if repairing_reload:
        # A full recovery-module reload restores ``recovery_summary`` from source
        # but leaves arbitrary dynamic attributes behind.  Clear the stale
        # sentinels together so the established implementation can reinstall its
        # complete wrapper set.  Do not key this decision off the certified
        # helpers: other runtime patches legitimately replace those functions.
        for flag in (_PATCHED_FLAG, _CERTIFIED_EVENT_PATCHED_FLAG):
            if getattr(_recovery, flag, False):
                delattr(_recovery, flag)

    _apply_impl()
    _negative_infinity.apply_simulation_recovery_negative_infinity_patch()

    if repairing_reload:
        # These markers disappear naturally when importlib.reload() restores the
        # source definitions.  Keeping them on the live callables avoids treating
        # a later certified-helper compatibility wrapper as a module reload.
        setattr(_recovery.recovery_summary, _PATCHED_FLAG, True)
        setattr(_recovery.certified_vs_exact_recovery_summary, _PATCHED_FLAG, True)
        setattr(
            _recovery.certified_vs_exact_event_recovery,
            _CERTIFIED_EVENT_PATCHED_FLAG,
            True,
        )


__all__ = ["apply_simulation_recovery_event_count_patch"]
