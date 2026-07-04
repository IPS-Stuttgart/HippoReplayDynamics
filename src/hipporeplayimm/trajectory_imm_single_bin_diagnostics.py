"""Keep trajectory-IMM single-bin and evidence-only diagnostics well-scoped."""

from __future__ import annotations

import sys
from contextvars import ContextVar
from functools import wraps
from typing import Any

import numpy as np

from .evidence_reporting import DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT

_PATCHED_ATTR = "_trajectory_imm_single_bin_diagnostics_patch_applied"
_ADVANCE_PATCHED_ATTR = "_trajectory_imm_evidence_only_advance_recording_patch"
_ORIGINAL_ATTR = "__hipporeplayimm_original__"
_RECORDED_FORWARD_STATES: ContextVar[list[Any] | None] = ContextVar(
    "trajectory_imm_evidence_only_forward_states",
    default=None,
)


def apply_trajectory_imm_single_bin_diagnostics_patch() -> None:
    """Patch trajectory-family IMM diagnostics for degenerate/evidence-only events."""

    from . import state_space_trajectory_imm as trajectory_imm

    _patch_forward_state_recording(trajectory_imm)

    current = trajectory_imm._score_trajectory_imm_exact_sparse
    if getattr(current, _PATCHED_ATTR, False):
        _refresh_public_alias(current)
        return

    @wraps(current)
    def score_trajectory_imm_exact_sparse(*args: Any, **kwargs: Any) -> Any:
        emissions = args[0] if args else kwargs.get("emissions")
        return_trajectory = bool(kwargs.get("return_trajectory", True))
        token = None
        recorded_states: list[Any] | None = None
        if not return_trajectory:
            token = _RECORDED_FORWARD_STATES.set([])
        try:
            logp, trajectory, terminal, mode_posterior, diagnostics = current(*args, **kwargs)
            if token is not None:
                recorded_states = _RECORDED_FORWARD_STATES.get()
        finally:
            if token is not None:
                _RECORDED_FORWARD_STATES.reset(token)

        if int(getattr(emissions, "n_time", 0)) == 1:
            updated = dict(diagnostics)
            updated.update(_single_bin_diagnostics())
            diagnostics = updated
        if not return_trajectory and recorded_states is not None:
            config = args[2] if len(args) > 2 else kwargs.get("config")
            diagnostics = _evidence_only_mode_diagnostics(
                trajectory_imm,
                config,
                int(getattr(emissions, "n_time", 0)),
                recorded_states,
                diagnostics,
            )
        return logp, trajectory, terminal, mode_posterior, diagnostics

    setattr(score_trajectory_imm_exact_sparse, _PATCHED_ATTR, True)
    setattr(score_trajectory_imm_exact_sparse, _ORIGINAL_ATTR, current)
    trajectory_imm._score_trajectory_imm_exact_sparse = score_trajectory_imm_exact_sparse
    _refresh_public_alias(score_trajectory_imm_exact_sparse)


def _patch_forward_state_recording(trajectory_imm: Any) -> None:
    current = trajectory_imm._advance_state
    if getattr(current, _ADVANCE_PATCHED_ATTR, False):
        return

    @wraps(current)
    def advance_state(*args: Any, **kwargs: Any) -> Any:
        state, entry_counts, momentum_counts = current(*args, **kwargs)
        recorded = _RECORDED_FORWARD_STATES.get()
        if recorded is not None:
            recorded.append(state)
        return state, entry_counts, momentum_counts

    setattr(advance_state, _ADVANCE_PATCHED_ATTR, True)
    setattr(advance_state, _ORIGINAL_ATTR, current)
    trajectory_imm._advance_state = advance_state


def _evidence_only_mode_diagnostics(
    trajectory_imm: Any,
    config: Any,
    n_time: int,
    forward_states: list[Any],
    diagnostics: dict[str, float | int | str],
) -> dict[str, float | int | str]:
    """Update mode summaries from all filtered rows, not just the terminal row."""

    try:
        mode_posterior = _filtered_mode_posterior_history(
            trajectory_imm,
            config,
            n_time,
            forward_states,
        )
    except (AttributeError, TypeError, ValueError):
        return diagnostics

    final_mode = mode_posterior[-1]
    event_mode = mode_posterior.mean(axis=0)
    trajectory_columns = [1, 2, 3]
    updated = dict(diagnostics)
    updated["state_space_trajectory_imm_mode_posterior"] = "filtered_evidence_only_state"
    updated["state_space_trajectory_imm_mean_mode_entropy"] = trajectory_imm._mean_entropy(
        trajectory_imm._as_log_probs(mode_posterior)
    )
    updated["state_space_trajectory_family_terminal_probability"] = float(
        final_mode[trajectory_columns].sum()
    )
    updated["state_space_trajectory_family_event_probability"] = float(
        event_mode[trajectory_columns].sum()
    )
    for mode_index, mode_name in enumerate(trajectory_imm._TRAJECTORY_IMM_MODES):
        key = mode_name.replace("-", "_")
        updated[f"state_space_mode_{key}_terminal_probability"] = float(final_mode[mode_index])
        updated[f"state_space_mode_{key}_event_probability"] = float(event_mode[mode_index])
    return updated


def _filtered_mode_posterior_history(
    trajectory_imm: Any,
    config: Any,
    n_time: int,
    forward_states: list[Any],
) -> np.ndarray:
    if int(n_time) <= 0:
        raise ValueError("n_time must be positive")
    rows = [
        _normalized_mode_row(
            trajectory_imm,
            np.asarray(trajectory_imm._trajectory_imm_mode_prior(config), dtype=float),
        )
    ]
    rows.extend(_filtered_mode_row_from_state(trajectory_imm, state) for state in forward_states)
    if len(rows) != int(n_time):
        raise ValueError("recorded forward-state count does not match emission time bins")
    return np.vstack(rows)


def _filtered_mode_row_from_state(trajectory_imm: Any, state: Any) -> np.ndarray:
    first_order = np.asarray(state.first_order, dtype=float)
    momentum_alpha = np.asarray(state.momentum_alpha, dtype=float)
    if first_order.ndim != 2 or first_order.shape[0] != trajectory_imm._FIRST_ORDER_MODE_COUNT:
        raise ValueError("first-order forward state has unexpected shape")
    mode = np.zeros(len(trajectory_imm._TRAJECTORY_IMM_MODES), dtype=float)
    mode[: trajectory_imm._FIRST_ORDER_MODE_COUNT] = first_order.sum(axis=1)
    mode[trajectory_imm._MOMENTUM_MODE_INDEX] = float(momentum_alpha.sum())
    return _normalized_mode_row(trajectory_imm, mode)


def _normalized_mode_row(trajectory_imm: Any, row: np.ndarray) -> np.ndarray:
    values = np.asarray(row, dtype=float)
    if values.shape != (len(trajectory_imm._TRAJECTORY_IMM_MODES),):
        raise ValueError("mode posterior row has unexpected shape")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("mode posterior row must contain finite nonnegative mass")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("mode posterior row must contain positive mass")
    return values / total


def _refresh_public_alias(patched: Any) -> None:
    state_space = sys.modules.get("hipporeplayimm.state_space")
    if state_space is not None:
        setattr(state_space, "_score_trajectory_imm_exact_sparse", patched)


def _single_bin_diagnostics() -> dict[str, int | str]:
    return {
        "state_space_trajectory_imm_evidence_support": DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT,
        "state_space_trajectory_imm_state_support": "single_bin_fragmented_fallback",
        "state_space_trajectory_imm_transition_support": "none_single_bin",
        "state_space_trajectory_imm_degenerate_reason": "single_time_bin_fragmented_marginal",
        "state_space_trajectory_imm_required_min_time_bins": 2,
        "state_space_momentum_evidence_support": DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT,
        "state_space_momentum_candidate_support": "not_used_single_bin",
        "state_space_momentum_candidate_selection": "none_single_bin",
    }


__all__ = ["apply_trajectory_imm_single_bin_diagnostics_patch"]
