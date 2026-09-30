"""Source-defined Igata route primitives, without treatment comparisons."""

from __future__ import annotations

import re
from itertools import groupby
from pathlib import Path

import numpy as np

FILE_PATTERN = re.compile(r"(?P<date>\d{6})_(?P<block>[^/]+)_trial(?P<trial>\d+)\.npz$")
SPECIAL = {"start": "U", "goal": "E", "old": "S", "new": "G"}


def file_identity(path: Path) -> dict:
    match = FILE_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unrecognized release filename: {path.name}")
    return {"date": match["date"], "recording_block": match["block"], "trial_number": int(match["trial"])}


def compress(labels) -> str:
    return "".join(k for k, _ in groupby(labels))


def grid_labels(xy, offset_mm=0.0) -> np.ndarray:
    """Match the author's first-polygon-wins boundary convention, without executing release code."""
    xy = np.asarray(xy, dtype=float)
    result = np.full(len(xy), "-", dtype="<U1")
    for i in range(25):
        x, y = i % 5 * 200, i // 5 * 200
        # A common lattice translation is a diagnostic of boundary uncertainty, not a new task geometry.
        inside = np.all(np.isfinite(xy), axis=1) & (result == "-")
        inside &= (xy[:, 0] >= x + offset_mm) & (xy[:, 0] <= x + 200 + offset_mm)
        inside &= (xy[:, 1] >= y + offset_mm) & (xy[:, 1] <= y + 200 + offset_mm)
        result[inside] = chr(65 + i)
    return result


def field_string(labels) -> str:
    """Keep unknown positions inside a trajectory; never silently bridge them."""
    s = compress(labels)
    where = [i for i, c in enumerate(s) if "A" <= c <= "Y"]
    return s[where[0] : where[-1] + 1] if where else ""


def rising_edges(values) -> np.ndarray:
    values = np.asarray(values) > 0
    return np.flatnonzero(values & ~np.r_[False, values[:-1]])


def inspect_log(log, max_gap_ms=250.0) -> dict:
    log = np.asarray(log)
    if log.ndim != 2 or log.shape[1] < 14 or len(log) < 2:
        return {"log_status": "invalid_log_shape"}
    t = log[:, 0]
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        return {"log_status": "nonmonotone_or_nonfinite_timestamps"}
    valid = (log[:, 3] > 0) & np.all(np.isfinite(log[:, 1:3]), axis=1)
    labels = grid_labels(log[:, 1:3])
    field = np.flatnonzero(labels != "-")
    cp_edges = rising_edges(log[:, 8])
    goal_edges = rising_edges(log[:, 10])
    cp_labels = labels[cp_edges].tolist()
    cp = cp_labels[0] if len(cp_labels) == 1 else ""
    phase = {"S": "old_checkpoint_active", "G": "new_checkpoint_active"}.get(cp, "unverified_checkpoint")
    gap = float("inf")
    tracked_string = ""
    if len(field):
        first, last = field[0], field[-1]
        good = np.flatnonzero(valid[first : last + 1]) + first
        if len(good):
            # Include endpoint loss and long sample intervals, not just isolated invalid frames.
            gap = float(np.diff(np.r_[t[first], t[good], t[last]]).max(initial=0))
        seen = labels[first : last + 1].copy()
        seen[~valid[first : last + 1]] = "-"
        tracked_string = compress(seen)
    success = len(cp_edges) == 1 and any(i > cp_edges[0] for i in goal_edges) and not np.any(log[:, 9] > 0)
    return {
        "log_status": "ok",
        "n_position_samples": len(log),
        "start_time_ms": float(t[0]),
        "end_time_ms": float(t[-1]),
        "duration_ms": float(t[-1] - t[0]),
        "valid_position_fraction": float(valid.mean()),
        "max_field_tracking_gap_ms": gap,
        "tracking_qc_passed": bool(len(field) and gap <= max_gap_ms),
        "n_checkpoint_rising_edges": len(cp_edges),
        "checkpoint_lattices": "|".join(cp_labels),
        "active_checkpoint_phase": phase,
        "n_goal_port_rising_edges": len(goal_edges),
        "task_success_proxy": bool(success),
        "timeout": bool(np.any(log[:, 9] > 0)),
        "tracked_field_string": tracked_string,
        "full_coordinate_field_string": field_string(labels),
        "undocumented_log_column_count": log.shape[1] - 14,
    }


def classify_route(native_string: str, info: dict, *, total_limit=12, segment_limit=8) -> dict:
    """Audit labels; a checkpoint sensor proves arrival, not consumption or an internal strategy."""
    s = field_string(native_string)
    result = {"field_string": s, "route_label": "unclassifiable", "route_reason": "", "optimized_new_success": False, "old_before_new": False}
    if info.get("log_status") != "ok":
        result["route_reason"] = info.get("log_status", "missing_log")
        return result
    if not info.get("tracking_qc_passed"):
        result["route_reason"] = "tracking_gap"
        return result
    if not s or "-" in s or any(not ("A" <= c <= "Y") for c in s) or s[0] != "U" or s[-1] != "E":
        result["route_reason"] = "incomplete_field_route"
        return result
    if info.get("active_checkpoint_phase") != "new_checkpoint_active":
        result["route_reason"] = "not_verified_new_checkpoint_trial"
        return result
    if not info.get("task_success_proxy"):
        result["route_reason"] = "task_success_or_order_unverified"
        return result
    cp = s.find("G")
    if cp < 0:
        result["route_reason"] = "checkpoint_sensor_route_conflict"
        return result
    old, early_goal = s.find("S"), s.find("E")
    result["old_before_new"] = 0 <= old < cp
    new_segment = s[: cp + 1]
    goal_segment = s[cp:]
    if old < 0 and early_goal > cp and len(s) < total_limit and len(new_segment) < segment_limit and len(goal_segment) < segment_limit:
        result.update(route_label="new", route_reason="source_optimized_route", optimized_new_success=True)
    elif 0 <= old < early_goal < cp and len(s[: old + 1]) < segment_limit and len(s[old : early_goal + 1]) < segment_limit:
        result.update(route_label="obsolete", route_reason="short_old_strategy_to_goal_before_correction")
    else:
        result.update(route_label="other", route_reason="mixed_old_new_without_early_goal" if result["old_before_new"] and early_goal > cp else "nonmatching_or_exploratory")
    return result


def stimulation_alignment(stim, stim_mat, duration_ms) -> dict:
    stim = np.asarray(stim, dtype=float).ravel()
    marks = np.asarray(stim_mat).ravel()
    marks = np.flatnonzero(marks > 0)
    finite = np.all(np.isfinite(stim))
    ordered = finite and (len(stim) < 2 or np.all(np.diff(stim) > 0))
    support = finite and bool(np.all((stim >= 0) & (stim < duration_ms + 2)))
    aligned = len(stim) == len(marks) and finite and bool(np.all((stim - marks >= -1e-6) & (stim - marks < 1.000001)))
    return {
        "n_stimulations": len(stim),
        "stimulation_timestamps_finite_ordered": bool(ordered),
        "stimulation_within_trial_support": bool(support),
        "stimulation_raster_aligned": bool(aligned),
        "stimulation_raster_max_error_ms": float(np.max(np.abs(stim - marks))) if len(stim) == len(marks) and len(stim) and finite else None,
        "stimulation_clock": "trial_relative_ms_consistent_with_1ms_stim_mat" if aligned and len(stim) else "not_demonstrated",
        "online_trigger_latency_verified": False,
    }


def modified_levenshtein(a: str, b: str) -> float:
    """SI p12: substitution lattice distance / 1.13 m; insertion/deletion cost .20."""
    if any(not ("A" <= c <= "Y") for c in a + b):
        raise ValueError("Distance requires open-field lattice strings")
    previous = np.arange(len(b) + 1, dtype=float) * 0.20
    for i, c in enumerate(a, 1):
        current = np.empty(len(b) + 1)
        current[0] = i * 0.20
        ci = ord(c) - 65
        for j, d in enumerate(b, 1):
            di = ord(d) - 65
            cost = np.hypot(ci % 5 - di % 5, ci // 5 - di // 5) * 0.20 / 1.13
            current[j] = min(current[j - 1] + 0.20, previous[j] + 0.20, previous[j - 1] + cost)
        previous = current
    return float(previous[-1])


def adjacent_transitions(rows: list[dict]) -> list[dict]:
    """Never join across resets, missing trials, animals, or recording blocks."""
    ordered = sorted(rows, key=lambda r: (r["animal"], r["date"], r["recording_block"], r["trial_number"]))
    output = []
    for a, b in zip(ordered, ordered[1:]):
        same = all(a[k] == b[k] for k in ("animal", "date", "recording_block"))
        if not same or b["trial_number"] != a["trial_number"] + 1:
            continue
        if b.get("start_time_ms", -np.inf) <= a.get("end_time_ms", np.inf):
            continue
        if not a.get("optimized_new_success"):
            continue
        output.append(
            {
                "animal": a["animal"],
                "released_group": a["released_group"],
                "date": a["date"],
                "recording_block": a["recording_block"],
                "previous_trial_number": a["trial_number"],
                "next_trial_number": b["trial_number"],
                "previous_relative_path": a["relative_path"],
                "next_relative_path": b["relative_path"],
                "next_route_label": b["route_label"],
                "next_route_reason": b["route_reason"],
                "primary_eligible": False,
                "eligibility_reason": "complete_prior_trial_history_and_block_order_unverified",
            }
        )
    return output
