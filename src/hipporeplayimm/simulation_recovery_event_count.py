"""Reload-safe entry point for simulation-recovery event-count patches."""

from __future__ import annotations

from . import simulation_recovery as _recovery
from .simulation_recovery_event_count_impl import (
    _distinct_event_count,
    apply_simulation_recovery_event_count_patch as _apply_impl,
)

_PATCHED_FLAG = "_simulation_recovery_session_event_count_patch_applied"
_CERTIFIED_EVENT_PATCHED_FLAG = (
    "_simulation_recovery_certified_event_duplicate_model_patch_applied"
)


def apply_simulation_recovery_event_count_patch() -> None:
    """Install event-count patches, repairing stale reload sentinels first.

    ``importlib.reload(simulation_recovery)`` reuses the module dictionary, so
    dynamically added module-level sentinels survive while source functions are
    replaced.  Detect that state from markers on the live wrappers and clear
    only stale sentinels before delegating to the established implementation.
    """

    summary_flag = bool(getattr(_recovery, _PATCHED_FLAG, False))
    summary_wrappers_live = bool(
        getattr(_recovery.recovery_summary, _PATCHED_FLAG, False)
    ) and bool(
        getattr(_recovery.certified_vs_exact_recovery_summary, _PATCHED_FLAG, False)
    )
    if summary_flag and not summary_wrappers_live:
        delattr(_recovery, _PATCHED_FLAG)

    certified_flag = bool(getattr(_recovery, _CERTIFIED_EVENT_PATCHED_FLAG, False))
    certified_wrapper_live = bool(
        getattr(
            _recovery.certified_vs_exact_event_recovery,
            _CERTIFIED_EVENT_PATCHED_FLAG,
            False,
        )
    )
    if certified_flag and not certified_wrapper_live:
        delattr(_recovery, _CERTIFIED_EVENT_PATCHED_FLAG)

    _apply_impl()

    # Mark the actual installed callables.  These markers disappear naturally
    # when importlib.reload() restores the source definitions, unlike dynamic
    # module attributes retained in the reused module dictionary.
    setattr(_recovery.recovery_summary, _PATCHED_FLAG, True)
    setattr(_recovery.certified_vs_exact_recovery_summary, _PATCHED_FLAG, True)
    setattr(
        _recovery.certified_vs_exact_event_recovery,
        _CERTIFIED_EVENT_PATCHED_FLAG,
        True,
    )


__all__ = ["apply_simulation_recovery_event_count_patch"]
