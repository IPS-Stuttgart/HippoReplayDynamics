"""Backward-compatible duration-aware IMM patch shim.

Duration-aware four-mode IMM handling is implemented directly in
``hipporeplayimm.state_space_model`` and ``hipporeplayimm.state_space_candidates``.
This module remains only so older callers that explicitly invoked the former
patch function do not fail.
"""

from __future__ import annotations


def apply_state_space_imm_duration_patch() -> None:
    """Backward-compatible no-op for older duration-patch callers."""

    return None
