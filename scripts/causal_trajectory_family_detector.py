#!/usr/bin/env python3
"""Causal prefix-evidence detector for trajectory-family replay dynamics.

The detector converts the offline exact-core evidence stack into an online
analysis mode by repeatedly scoring prefixes of an event/window.  Each prefix
uses only spikes observed up to the current prefix end time, then converts the
exact-core log evidence into posterior model probabilities and trajectory-family
claims.  This is intentionally a causal data-access approximation, not yet a
hand-optimized recursive filter implementation.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from aggregate_event_window_sensitivity import DEFAULT_MARGIN_THRESHOLD
from benchmark_model_evidence import _check_session, _events, _postprocess_evidence_scores, _session_path
from benchmark_model_evidence_improved import _family, _models, _run_settings
from hipporeplayimm.clusterless import (
    ClusterlessStateSpaceReplayModel,
    build_clusterless_mark_emissions,
    fit_clusterless_mark_encoding,
)
from hipporeplayimm.data import ReplaySession, load_replay_session
from hipporeplayimm.encoding import EmissionConfig, EncodingConfig, _time_bin_edges, fit_place_field_encoding
from hipporeplayimm.result_improvement_extensions import (
    ReplayEmissionCalibration,
    build_sorted_emissions_with_replay_calibration,
    score_replay_model_compat,
)
from spike_matched_event_window_null import (
    FULL_CORE_REQUIRED_MODELS,
    _add_model_arguments,
    _clusterless_mark_config,
    _spike_count_and_active_cells,
    spike_matched_null_windows,
)


STATIONARY = "sorted-spike-state-space-stationary"
DIFFUSION = "sorted-spike-state-space-diffusion"
FRAGMENTED = "sorted-spike-state-space-fragmented"
FIRST_ORDER_IMM = "sorted-spike-state-space-first-order-imm"
MOMENTUM_EXACT = "sorted-spike-state-space-momentum-exact-sparse"

DEFAULT_CAUSAL_MODELS = " ".join(FULL_CORE_REQUIRED_MODELS)
REQUIRED_CAUSAL_MODELS: tuple[str, ...] = (
    STATIONARY,
    DIFFUSION,
    FRAGMENTED,
    FIRST_ORDER_IMM,
    MOMENTUM_EXACT,
)
TRAJECTORY_MODELS: tuple[str, ...] = (
    DIFFUSION,
    FRAGMENTED,
    FIRST_ORDER_IMM,
    MOMENTUM_EXACT,
)
MODEL_PROBABILITY_COLUMNS = {
    STATIONARY: "p_static",
    DIFFUSION: "p_diffusion",
    FRAGMENTED: "p_fragmented",
    FIRST_ORDER_IMM: "p_first_order_imm",
    MOMENTUM_EXACT: "p_momentum",
}
WINDOW_KEY_COLUMNS = ("session", "event_index", "window_role", "null_index")
PREFIX_KEY_COLUMNS = (*WINDOW_KEY_COLUMNS, "prefix_time_bin_index")
TRAJECTORY_LABEL = "trajectory_family"
NONTRAJECTORY_LABEL = "static_nontrajectory"
AMBIGUOUS_LABEL = "ambiguous"
INCOMPLETE_LABEL = "incomplete_core"

PREFIX_EVIDENCE_OUTPUT = "causal_prefix_event_model_evidence.csv"
TIME_BIN_TABLE_OUTPUT = "causal_replay_detection_time_bin_table.csv"
EVENT_TABLE_OUTPUT = "causal_replay_detection_event_table.csv"
OFFLINE_AGREEMENT_OUTPUT = "causal_offline_agreement_summary.csv"
LATENCY_OUTPUT = "causal_latency_summary.csv"
FALSE_POSITIVE_OUTPUT = "causal_false_positive_summary.csv"
GATE_OUTPUT = "causal_detector_gate_summary.csv"


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        return bool(np.isfinite(numeric) and numeric != 0.0)
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "0.0", "false", "f", "no", "n", "", "nan", "none", "null", "off"}:
        return False
    try:
        numeric = float(text)
    except ValueError:
        return False
    return bool(np.isfinite(numeric) and numeric != 0.0)


def _rat_from_session(session: object) -> str:
    return str(session).split("/", 1)[0]


def _parse_names(value: str | Iterable[str] | None, default: Sequence[str]) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        names = tuple(part.strip() for part in value.replace(",", " ").split() if part.strip())
        return names or tuple(default)
    names = tuple(str(part).strip() for part in value if str(part).strip())
    return names or tuple(default)


def _safe_softmax(log_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(log_values, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return np.full(values.shape, np.nan, dtype=float)
    denominator = logsumexp(values)
    if not np.isfinite(denominator):
        return np.full(values.shape, np.nan, dtype=float)
    return np.exp(values - denominator)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(float(numerator)) or not np.isfinite(float(denominator)) or float(denominator) == 0.0:
        return np.nan
    return float(numerator) / float(denominator)


def _first_value(frame: pd.DataFrame, column: str) -> object:
    if column not in frame.columns:
        return np.nan
    values = frame[column].dropna()
    return values.iloc[0] if not values.empty else np.nan


def _first_numeric(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else np.nan


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _status_ok(frame: pd.DataFrame) -> pd.Series:
    if "status" not in frame.columns:
        return pd.Series(True, index=frame.index)
    return frame["status"].astype(str).eq("success")


def _comparable(frame: pd.DataFrame) -> pd.Series:
    if "evidence_comparable" not in frame.columns:
        return pd.Series(True, index=frame.index)
    return frame["evidence_comparable"].map(_as_bool)


def _window_rows_for_event(
    session: ReplaySession,
    event_id: int,
    *,
    nulls_per_event: int,
    null_random_seed: int,
    spike_count_tolerance_fraction: float,
    active_cell_tolerance: int | None,
    null_candidate_step_s: float | None,
    swr_exclusion_padding_s: float,
    allow_non_run_nulls: bool,
) -> list[dict[str, object]]:
    event = session.ripple(int(event_id))
    real_count, real_active = _spike_count_and_active_cells(
        session.excitatory_spikes(),
        float(event.start),
        float(event.end),
    )
    rows = [
        {
            "window_role": "real",
            "event_window_variant": "core",
            "null_index": -1,
            "matched_null_rank": 0,
            "template_event_index": int(event_id),
            "window_start_s": float(event.start),
            "window_end_s": float(event.end),
            "window_duration_s": float(event.end - event.start),
            "real_event_start_s": float(event.start),
            "real_event_end_s": float(event.end),
            "real_event_duration_s": float(event.end - event.start),
            "real_n_spikes": int(real_count),
            "real_active_cell_count": int(real_active),
            "null_n_spikes": int(real_count),
            "null_active_cell_count": int(real_active),
            "n_spikes_delta": 0,
            "active_cell_count_delta": 0,
            "n_spikes_relative_delta": 0.0,
            "off_swr": False,
        }
    ]
    if int(nulls_per_event) <= 0:
        return rows
    nulls = spike_matched_null_windows(
        session,
        int(event_id),
        nulls_per_event=int(nulls_per_event),
        random_seed=int(null_random_seed),
        spike_count_tolerance_fraction=float(spike_count_tolerance_fraction),
        active_cell_tolerance=active_cell_tolerance,
        candidate_step_s=null_candidate_step_s,
        exclusion_padding_s=float(swr_exclusion_padding_s),
        restrict_to_run_times=not bool(allow_non_run_nulls),
    )
    for row in nulls.to_dict("records"):
        item = dict(row)
        item["window_role"] = "matched_null"
        item["event_window_variant"] = "matched_null"
        rows.append(item)
    return rows


def _prefix_edges(window_start_s: float, window_end_s: float, time_bin_s: float, min_prefix_bins: int, prefix_stride_bins: int) -> list[tuple[int, float]]:
    edges = _time_bin_edges(float(window_start_s), float(window_end_s), float(time_bin_s))
    n_bins = len(edges) - 1
    if n_bins <= 0:
        return []
    start_bin = max(1, int(min_prefix_bins))
    stride = max(1, int(prefix_stride_bins))
    indices = list(range(start_bin, n_bins + 1, stride))
    if n_bins not in indices:
        indices.append(n_bins)
    return [(int(index), float(edges[index])) for index in indices]


def score_causal_prefixes(args: argparse.Namespace) -> pd.DataFrame:
    """Score causal prefixes for SWR events and optional matched off-SWR windows."""

    session_dir = _session_path(args.dataset_root, args.session)
    _check_session(session_dir)
    session = load_replay_session(session_dir)
    event_ids = _events(args.events, session)
    if args.max_events is not None:
        event_ids = event_ids[: args.max_events]

    encoding = fit_place_field_encoding(
        session,
        EncodingConfig(
            bin_size_cm=args.bin_size_cm,
            smoothing_sigma_bins=args.smoothing_sigma_bins,
            min_speed_cm_s=args.min_speed_cm_s,
            min_occupancy_s=args.min_occupancy_s,
            rate_floor_hz=args.rate_floor_hz,
        ),
    )
    models = _models(args, session, encoding=encoding)
    has_clusterless = any(isinstance(model, ClusterlessStateSpaceReplayModel) for model in models.values())
    clusterless_encoding = None
    clusterless_encoding_error = ""
    if has_clusterless:
        try:
            clusterless_encoding = fit_clusterless_mark_encoding(session, _clusterless_mark_config(args))
        except ValueError as exc:
            clusterless_encoding_error = f"{type(exc).__name__}: {exc}"

    emissions_cfg = EmissionConfig(
        time_bin_s=args.time_bin_s,
        spike_rate_scale=args.spike_rate_scale,
        likelihood_temperature=args.emission_likelihood_temperature,
        negative_binomial_overdispersion=args.emission_negative_binomial_overdispersion,
    )
    sorted_calibration = ReplayEmissionCalibration(
        gain_mode=args.replay_gain_mode,
        gain_prior_count=args.replay_gain_prior_count,
        max_gain=args.replay_gain_max_gain,
        emission_model=args.sorted_spike_emission_model,
        negative_binomial_dispersion=args.negative_binomial_dispersion,
    )

    rows: list[dict[str, object]] = []
    for event_id in event_ids:
        windows = _window_rows_for_event(
            session,
            int(event_id),
            nulls_per_event=args.nulls_per_event,
            null_random_seed=args.null_random_seed,
            spike_count_tolerance_fraction=args.spike_count_tolerance_fraction,
            active_cell_tolerance=args.active_cell_tolerance,
            null_candidate_step_s=args.null_candidate_step_s,
            swr_exclusion_padding_s=args.swr_exclusion_padding_s,
            allow_non_run_nulls=args.allow_non_run_nulls,
        )
        for window_index, window in enumerate(windows):
            _score_causal_window_prefixes(
                args,
                session,
                encoding,
                clusterless_encoding,
                clusterless_encoding_error,
                models,
                emissions_cfg,
                sorted_calibration,
                event_id=int(event_id),
                window_index=int(window_index),
                window=window,
                rows=rows,
            )
    return _postprocess_evidence_scores(pd.DataFrame(rows))


def _score_causal_window_prefixes(
    args: argparse.Namespace,
    session: ReplaySession,
    encoding,
    clusterless_encoding,
    clusterless_encoding_error: str,
    models: dict[str, object],
    emissions_cfg: EmissionConfig,
    sorted_calibration: ReplayEmissionCalibration,
    *,
    event_id: int,
    window_index: int,
    window: dict[str, object],
    rows: list[dict[str, object]],
) -> None:
    window_start = float(window["window_start_s"])
    window_end = float(window["window_end_s"])
    for prefix_time_bin_index, prefix_end in _prefix_edges(
        window_start,
        window_end,
        args.time_bin_s,
        args.min_prefix_bins,
        args.prefix_stride_bins,
    ):
        prefix_window = SimpleNamespace(start=window_start, end=float(prefix_end))
        sorted_emissions = build_sorted_emissions_with_replay_calibration(
            session,
            encoding,
            prefix_window,
            emissions_cfg,
            calibration=sorted_calibration,
        )
        if sorted_emissions.n_time == 0:
            continue
        clusterless_emissions = (
            build_clusterless_mark_emissions(session, clusterless_encoding, prefix_window, emissions_cfg)
            if clusterless_encoding is not None
            else None
        )
        prefix_settings = _prefix_settings(
            window,
            window_index=window_index,
            prefix_time_bin_index=prefix_time_bin_index,
            prefix_end_s=float(prefix_end),
        )
        for name, model in models.items():
            start = time.perf_counter()
            use_clusterless = isinstance(model, ClusterlessStateSpaceReplayModel)
            emissions = clusterless_emissions if use_clusterless else sorted_emissions
            bin_centers = clusterless_encoding.bin_centers if use_clusterless and clusterless_encoding is not None else encoding.bin_centers
            occupancy_s = clusterless_encoding.occupancy_s if use_clusterless and clusterless_encoding is not None else encoding.occupancy_s
            if use_clusterless and clusterless_encoding is None:
                rows.append(
                    {
                        "status": "unsupported",
                        "session": session.session_id,
                        "event_index": int(event_id),
                        **prefix_settings,
                        "model": name,
                        "requested_model": name,
                        "model_family": _family(name),
                        "log_evidence": np.nan,
                        "n_time": int(sorted_emissions.n_time),
                        "n_spikes": int(sorted_emissions.n_spikes),
                        "runtime_s": float(time.perf_counter() - start),
                        "error": clusterless_encoding_error or "Clusterless encoding unavailable",
                        "diagnostic_clusterless_observation_model_available": False,
                        "diagnostic_clusterless_observation_model_error": clusterless_encoding_error,
                        **_run_settings(args),
                    }
                )
                continue
            assert emissions is not None
            try:
                result = score_replay_model_compat(model, emissions, bin_centers, occupancy_s=occupancy_s)
                model_name = str(result.model_name)
                row: dict[str, object] = {
                    "status": "success",
                    "session": session.session_id,
                    "event_index": int(event_id),
                    **prefix_settings,
                    "model": model_name,
                    "requested_model": name,
                    "model_family": _family(model_name),
                    "log_evidence": float(result.log_likelihood),
                    "n_time": int(result.n_time),
                    "n_spikes": int(result.n_spikes),
                    "runtime_s": float(time.perf_counter() - start),
                    "error": "",
                    "causal_detector_type": "prefix_evidence",
                    **_run_settings(args),
                    "causal_min_prefix_bins": int(args.min_prefix_bins),
                    "causal_prefix_stride_bins": int(args.prefix_stride_bins),
                    "matched_nulls_per_event": int(args.nulls_per_event),
                    "spike_count_tolerance_fraction": float(args.spike_count_tolerance_fraction),
                    "active_cell_tolerance": "" if args.active_cell_tolerance is None else int(args.active_cell_tolerance),
                    "null_candidate_step_s": "" if args.null_candidate_step_s is None else float(args.null_candidate_step_s),
                    "swr_exclusion_padding_s": float(args.swr_exclusion_padding_s),
                    "allow_non_run_nulls": bool(args.allow_non_run_nulls),
                }
                metadata = getattr(emissions, "metadata", {}) or {}
                row.update({f"emission_{key}": value for key, value in metadata.items()})
                row.update({f"diagnostic_{key}": value for key, value in result.diagnostics.items()})
                rows.append(row)
            except Exception as exc:
                rows.append(
                    {
                        "status": "failure",
                        "session": session.session_id,
                        "event_index": int(event_id),
                        **prefix_settings,
                        "model": name,
                        "requested_model": name,
                        "model_family": _family(name),
                        "log_evidence": np.nan,
                        "n_time": int(emissions.n_time),
                        "n_spikes": int(emissions.n_spikes),
                        "runtime_s": float(time.perf_counter() - start),
                        "error": f"{type(exc).__name__}: {exc}",
                        "causal_detector_type": "prefix_evidence",
                        **_run_settings(args),
                    }
                )
                if not args.continue_on_error:
                    raise


def _prefix_settings(window: dict[str, object], *, window_index: int, prefix_time_bin_index: int, prefix_end_s: float) -> dict[str, object]:
    keys = [
        "window_role",
        "event_window_variant",
        "null_index",
        "matched_null_rank",
        "template_event_index",
        "window_start_s",
        "window_end_s",
        "window_duration_s",
        "real_event_start_s",
        "real_event_end_s",
        "real_event_duration_s",
        "real_n_spikes",
        "real_active_cell_count",
        "null_n_spikes",
        "null_active_cell_count",
        "n_spikes_delta",
        "active_cell_count_delta",
        "n_spikes_relative_delta",
        "off_swr",
        "restrict_to_run_times",
    ]
    out = {key: window.get(key, "") for key in keys}
    out["window_index"] = int(window_index)
    out["prefix_time_bin_index"] = int(prefix_time_bin_index)
    out["prefix_end_s"] = float(prefix_end_s)
    out["prefix_duration_s"] = float(prefix_end_s - float(window["window_start_s"]))
    out["prefix_fraction"] = _safe_ratio(out["prefix_duration_s"], float(window["window_duration_s"]))
    out["latency_s"] = out["prefix_duration_s"]
    return out


def causal_replay_detection_time_bin_table(
    prefix_scores: pd.DataFrame,
    *,
    offline_labels: pd.DataFrame | None = None,
    required_models: Sequence[str] = REQUIRED_CAUSAL_MODELS,
    trajectory_models: Sequence[str] = TRAJECTORY_MODELS,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
) -> pd.DataFrame:
    """Return one causal posterior row per event/window/prefix."""

    if prefix_scores.empty:
        return pd.DataFrame()
    required = tuple(required_models)
    trajectory_set = set(trajectory_models)
    ok = prefix_scores[_status_ok(prefix_scores) & _comparable(prefix_scores)].copy()
    if ok.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    group_cols = [column for column in PREFIX_KEY_COLUMNS if column in ok.columns]
    for key, group in ok.groupby(group_cols, sort=True, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        key_row = {column: value for column, value in zip(group_cols, key_tuple, strict=True)}
        core = group[group["model"].astype(str).isin(required)].dropna(subset=["log_evidence"]).copy()
        present = tuple(model for model in required if model in set(core["model"].astype(str)))
        missing = tuple(model for model in required if model not in set(present))
        by_model = core.drop_duplicates("model", keep="last").set_index("model")
        logz = {model: (float(by_model.loc[model, "log_evidence"]) if model in by_model.index else np.nan) for model in required}
        probabilities = _safe_softmax([logz[model] for model in required]) if not missing else np.full(len(required), np.nan)
        probability_by_model = dict(zip(required, probabilities, strict=True))

        trajectory = core[core["model"].astype(str).isin(trajectory_set)]
        nontrajectory = core[~core["model"].astype(str).isin(trajectory_set)]
        best_trajectory_model = ""
        best_trajectory_log_evidence = np.nan
        best_nontrajectory_model = ""
        best_nontrajectory_log_evidence = np.nan
        trajectory_margin = np.nan
        if not trajectory.empty:
            best_trajectory = trajectory.sort_values(["log_evidence", "model"], ascending=[False, True]).iloc[0]
            best_trajectory_model = str(best_trajectory["model"])
            best_trajectory_log_evidence = float(best_trajectory["log_evidence"])
        if not nontrajectory.empty:
            best_nontrajectory = nontrajectory.sort_values(["log_evidence", "model"], ascending=[False, True]).iloc[0]
            best_nontrajectory_model = str(best_nontrajectory["model"])
            best_nontrajectory_log_evidence = float(best_nontrajectory["log_evidence"])
        if np.isfinite(best_trajectory_log_evidence) and np.isfinite(best_nontrajectory_log_evidence):
            trajectory_margin = float(best_trajectory_log_evidence - best_nontrajectory_log_evidence)

        if missing:
            label = INCOMPLETE_LABEL
        elif trajectory_margin >= float(margin_threshold):
            label = TRAJECTORY_LABEL
        elif trajectory_margin <= -float(margin_threshold):
            label = NONTRAJECTORY_LABEL
        else:
            label = AMBIGUOUS_LABEL

        trajectory_probabilities = [probability_by_model.get(model, np.nan) for model in trajectory_models]
        p_trajectory_family = float(np.sum(trajectory_probabilities)) if not missing and np.all(np.isfinite(trajectory_probabilities)) else np.nan
        row = {
            **key_row,
            "rat": _rat_from_session(_first_value(group, "session")),
            "window_start_s": _first_numeric(group, "window_start_s"),
            "window_end_s": _first_numeric(group, "window_end_s"),
            "window_duration_s": _first_numeric(group, "window_duration_s"),
            "prefix_end_s": _first_numeric(group, "prefix_end_s"),
            "prefix_duration_s": _first_numeric(group, "prefix_duration_s"),
            "prefix_fraction": _first_numeric(group, "prefix_fraction"),
            "n_time": _first_numeric(group, "n_time"),
            "n_spikes": _first_numeric(group, "n_spikes"),
            "off_swr": _as_bool(_first_value(group, "off_swr")),
            "required_models_present": int(len(present)),
            "required_models_total": int(len(required)),
            "required_models_complete": bool(not missing),
            "missing_required_models": " ".join(missing),
            "margin_threshold": float(margin_threshold),
            "best_trajectory_model": best_trajectory_model,
            "best_trajectory_log_evidence": best_trajectory_log_evidence,
            "best_nontrajectory_model": best_nontrajectory_model,
            "best_nontrajectory_log_evidence": best_nontrajectory_log_evidence,
            "trajectory_minus_nontrajectory_log_evidence": trajectory_margin,
            "causal_label": label,
            "causal_trajectory_confident_claim": bool(label == TRAJECTORY_LABEL),
            "causal_nontrajectory_confident_claim": bool(label == NONTRAJECTORY_LABEL),
            "p_trajectory_family": p_trajectory_family,
            "p_nonlocal_replay": p_trajectory_family,
        }
        for model, column in MODEL_PROBABILITY_COLUMNS.items():
            row[column] = float(probability_by_model.get(model, np.nan))
        rows.append(row)

    table = pd.DataFrame(rows)
    if offline_labels is not None and not offline_labels.empty:
        table = _merge_offline_labels(table, offline_labels)
    return table.sort_values(["session", "event_index", "window_role", "null_index", "prefix_time_bin_index"], na_position="last").reset_index(drop=True)


def offline_labels_from_event_model_evidence(
    offline_scores: pd.DataFrame,
    *,
    required_models: Sequence[str] = REQUIRED_CAUSAL_MODELS,
    trajectory_models: Sequence[str] = TRAJECTORY_MODELS,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
) -> pd.DataFrame:
    """Build offline full-window labels for causal agreement summaries."""

    if offline_scores.empty:
        return pd.DataFrame()
    required = tuple(required_models)
    trajectory_set = set(trajectory_models)
    ok = offline_scores[_status_ok(offline_scores) & _comparable(offline_scores)].copy()
    if ok.empty:
        return pd.DataFrame()

    group_cols = ["session", "event_index"]
    if "window_role" in ok.columns and "null_index" in ok.columns:
        group_cols = ["session", "event_index", "window_role", "null_index"]

    rows: list[dict[str, object]] = []
    for key, group in ok.groupby(group_cols, sort=True, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = {column: value for column, value in zip(group_cols, key_tuple, strict=True)}
        core = group[group["model"].astype(str).isin(required)].dropna(subset=["log_evidence"]).copy()
        present = tuple(model for model in required if model in set(core["model"].astype(str)))
        missing = tuple(model for model in required if model not in set(present))
        trajectory = core[core["model"].astype(str).isin(trajectory_set)]
        nontrajectory = core[~core["model"].astype(str).isin(trajectory_set)]
        if trajectory.empty or nontrajectory.empty:
            best_trajectory_model = ""
            best_nontrajectory_model = ""
            best_trajectory_log_evidence = np.nan
            best_nontrajectory_log_evidence = np.nan
            margin = np.nan
            label = INCOMPLETE_LABEL
        else:
            best_trajectory = trajectory.sort_values(["log_evidence", "model"], ascending=[False, True]).iloc[0]
            best_nontrajectory = nontrajectory.sort_values(["log_evidence", "model"], ascending=[False, True]).iloc[0]
            best_trajectory_model = str(best_trajectory["model"])
            best_nontrajectory_model = str(best_nontrajectory["model"])
            best_trajectory_log_evidence = float(best_trajectory["log_evidence"])
            best_nontrajectory_log_evidence = float(best_nontrajectory["log_evidence"])
            margin = best_trajectory_log_evidence - best_nontrajectory_log_evidence
            if missing:
                label = INCOMPLETE_LABEL
            elif margin >= float(margin_threshold):
                label = TRAJECTORY_LABEL
            elif margin <= -float(margin_threshold):
                label = NONTRAJECTORY_LABEL
            else:
                label = AMBIGUOUS_LABEL
        rows.append(
            {
                **row,
                "offline_label": label,
                "offline_required_models_complete": bool(not missing),
                "offline_missing_required_models": " ".join(missing),
                "offline_best_trajectory_model": best_trajectory_model,
                "offline_best_nontrajectory_model": best_nontrajectory_model,
                "offline_best_trajectory_log_evidence": best_trajectory_log_evidence,
                "offline_best_nontrajectory_log_evidence": best_nontrajectory_log_evidence,
                "offline_trajectory_minus_nontrajectory_log_evidence": margin,
            }
        )
    return pd.DataFrame(rows)


def _merge_offline_labels(table: pd.DataFrame, offline_labels: pd.DataFrame) -> pd.DataFrame:
    labels = offline_labels.copy()
    if {"window_role", "null_index"}.issubset(labels.columns):
        return table.merge(labels, on=["session", "event_index", "window_role", "null_index"], how="left")
    real_labels = labels.copy()
    out = table.merge(real_labels, on=["session", "event_index"], how="left")
    is_real = out["window_role"].astype(str).eq("real") if "window_role" in out.columns else pd.Series(True, index=out.index)
    offline_columns = [column for column in labels.columns if column not in {"session", "event_index"}]
    for column in offline_columns:
        out.loc[~is_real, column] = np.nan
    return out


def causal_replay_detection_event_table(time_bin_table: pd.DataFrame) -> pd.DataFrame:
    """Collapse causal prefix rows into one summary row per event/window."""

    if time_bin_table.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for key, group in time_bin_table.groupby(list(WINDOW_KEY_COLUMNS), sort=True, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        key_row = {column: value for column, value in zip(WINDOW_KEY_COLUMNS, key_tuple, strict=True)}
        group = group.sort_values("prefix_time_bin_index")
        final = group.iloc[-1]
        confident = group[group["causal_trajectory_confident_claim"].map(_as_bool)]
        first = confident.iloc[0] if not confident.empty else None
        offline_label = str(final.get("offline_label", ""))
        final_label = str(final["causal_label"])
        final_agreement = (
            bool(final_label == offline_label)
            if offline_label and offline_label != "nan" and final_label != INCOMPLETE_LABEL
            else np.nan
        )
        rows.append(
            {
                **key_row,
                "rat": str(final["rat"]),
                "off_swr": _as_bool(final["off_swr"]),
                "window_start_s": float(final["window_start_s"]),
                "window_end_s": float(final["window_end_s"]),
                "window_duration_s": float(final["window_duration_s"]),
                "causal_detector_type": "prefix_evidence",
                "prefix_rows": int(len(group)),
                "final_prefix_time_bin_index": int(final["prefix_time_bin_index"]),
                "final_prefix_end_s": float(final["prefix_end_s"]),
                "final_required_models_complete": _as_bool(
                    final["required_models_complete"]
                ),
                "final_causal_label": final_label,
                "final_best_trajectory_model": str(final["best_trajectory_model"]),
                "final_best_nontrajectory_model": str(final["best_nontrajectory_model"]),
                "final_trajectory_minus_nontrajectory_log_evidence": float(final["trajectory_minus_nontrajectory_log_evidence"]),
                "final_p_static": float(final["p_static"]),
                "final_p_diffusion": float(final["p_diffusion"]),
                "final_p_fragmented": float(final["p_fragmented"]),
                "final_p_first_order_imm": float(final["p_first_order_imm"]),
                "final_p_momentum": float(final["p_momentum"]),
                "final_p_trajectory_family": float(final["p_trajectory_family"]),
                "final_p_nonlocal_replay": float(final["p_nonlocal_replay"]),
                "has_causal_trajectory_claim": first is not None,
                "latency_to_trajectory_claim_s": float(first["prefix_duration_s"]) if first is not None else np.nan,
                "latency_to_trajectory_claim_fraction": float(first["prefix_fraction"]) if first is not None else np.nan,
                "trajectory_claim_prefix_time_bin_index": int(first["prefix_time_bin_index"]) if first is not None else np.nan,
                "trajectory_claim_best_model": str(first["best_trajectory_model"]) if first is not None else "",
                "trajectory_claim_margin": float(first["trajectory_minus_nontrajectory_log_evidence"]) if first is not None else np.nan,
                "offline_label": offline_label,
                "offline_best_trajectory_model": str(final.get("offline_best_trajectory_model", "")),
                "offline_trajectory_minus_nontrajectory_log_evidence": float(final.get("offline_trajectory_minus_nontrajectory_log_evidence", np.nan)),
                "final_agrees_with_offline_label": final_agreement,
            }
        )
    return pd.DataFrame(rows)


def causal_offline_agreement_summary(event_table: pd.DataFrame) -> pd.DataFrame:
    if event_table.empty or "offline_label" not in event_table.columns:
        return pd.DataFrame(
            [
                {
                    "comparison": "causal_final_vs_offline_full_window",
                    "windows": 0,
                    "offline_labeled_windows": 0,
                    "agreement_windows": 0,
                    "agreement_fraction": np.nan,
                    "offline_trajectory_windows": 0,
                    "causal_detected_offline_trajectory_windows": 0,
                    "offline_trajectory_detection_fraction": np.nan,
                    "offline_nontrajectory_windows": 0,
                    "causal_nontrajectory_agreement_fraction": np.nan,
                }
            ]
        )
    labeled = event_table[event_table["offline_label"].astype(str).isin([TRAJECTORY_LABEL, NONTRAJECTORY_LABEL, AMBIGUOUS_LABEL])].copy()
    agreement = labeled["final_agrees_with_offline_label"].map(_as_bool) if not labeled.empty else pd.Series(dtype=bool)
    offline_trajectory = labeled[labeled["offline_label"].astype(str).eq(TRAJECTORY_LABEL)]
    offline_nontrajectory = labeled[labeled["offline_label"].astype(str).eq(NONTRAJECTORY_LABEL)]
    return pd.DataFrame(
        [
            {
                "comparison": "causal_final_vs_offline_full_window",
                "windows": int(len(event_table)),
                "offline_labeled_windows": int(len(labeled)),
                "agreement_windows": int(agreement.sum()) if not agreement.empty else 0,
                "agreement_fraction": _safe_ratio(int(agreement.sum()) if not agreement.empty else 0, len(labeled)),
                "offline_trajectory_windows": int(len(offline_trajectory)),
                "causal_detected_offline_trajectory_windows": int(offline_trajectory["has_causal_trajectory_claim"].map(_as_bool).sum()) if not offline_trajectory.empty else 0,
                "offline_trajectory_detection_fraction": _safe_ratio(
                    int(offline_trajectory["has_causal_trajectory_claim"].map(_as_bool).sum()) if not offline_trajectory.empty else 0,
                    len(offline_trajectory),
                ),
                "offline_nontrajectory_windows": int(len(offline_nontrajectory)),
                "causal_nontrajectory_agreement_fraction": _safe_ratio(
                    int(offline_nontrajectory["final_causal_label"].astype(str).eq(NONTRAJECTORY_LABEL).sum()) if not offline_nontrajectory.empty else 0,
                    len(offline_nontrajectory),
                ),
            }
        ]
    )


def causal_latency_summary(event_table: pd.DataFrame) -> pd.DataFrame:
    if event_table.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for label, group in (
        ("real_windows", event_table[event_table["window_role"].astype(str).eq("real")]),
        ("offline_trajectory_real_windows", event_table[event_table["window_role"].astype(str).eq("real") & event_table.get("offline_label", pd.Series("", index=event_table.index)).astype(str).eq(TRAJECTORY_LABEL)]),
        ("all_windows", event_table),
    ):
        latencies = pd.to_numeric(group["latency_to_trajectory_claim_s"], errors="coerce").dropna()
        fractions = pd.to_numeric(group["latency_to_trajectory_claim_fraction"], errors="coerce").dropna()
        rows.append(
            {
                "window_set": label,
                "windows": int(len(group)),
                "windows_with_causal_trajectory_claim": int(group["has_causal_trajectory_claim"].map(_as_bool).sum()) if not group.empty else 0,
                "claim_fraction": _safe_ratio(int(group["has_causal_trajectory_claim"].map(_as_bool).sum()) if not group.empty else 0, len(group)),
                "median_latency_s": float(latencies.median()) if not latencies.empty else np.nan,
                "mean_latency_s": float(latencies.mean()) if not latencies.empty else np.nan,
                "min_latency_s": float(latencies.min()) if not latencies.empty else np.nan,
                "max_latency_s": float(latencies.max()) if not latencies.empty else np.nan,
                "median_latency_fraction": float(fractions.median()) if not fractions.empty else np.nan,
                "mean_latency_fraction": float(fractions.mean()) if not fractions.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def causal_false_positive_summary(event_table: pd.DataFrame) -> pd.DataFrame:
    if event_table.empty:
        return pd.DataFrame()
    nulls = event_table[event_table["window_role"].astype(str).eq("matched_null")].copy()
    rows = []
    for label, group in (("matched_off_swr_windows", nulls), ("all_windows", event_table)):
        final_claims = group["final_causal_label"].astype(str).eq(TRAJECTORY_LABEL) if not group.empty else pd.Series(dtype=bool)
        any_claims = group["has_causal_trajectory_claim"].map(_as_bool) if not group.empty else pd.Series(dtype=bool)
        rows.append(
            {
                "window_set": label,
                "windows": int(len(group)),
                "any_prefix_trajectory_claims": int(any_claims.sum()) if not any_claims.empty else 0,
                "any_prefix_false_positive_fraction": _safe_ratio(int(any_claims.sum()) if not any_claims.empty else 0, len(group)),
                "final_prefix_trajectory_claims": int(final_claims.sum()) if not final_claims.empty else 0,
                "final_prefix_false_positive_fraction": _safe_ratio(int(final_claims.sum()) if not final_claims.empty else 0, len(group)),
            }
        )
    return pd.DataFrame(rows)


def causal_detector_gate_summary(event_table: pd.DataFrame, offline_agreement: pd.DataFrame, latency: pd.DataFrame, false_positive: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(gate: str, passed: bool, observed: object, criterion: str, required_for_overall: bool = True) -> None:
        rows.append(
            {
                "gate": gate,
                "passed": bool(passed),
                "observed": observed,
                "criterion": criterion,
                "required_for_overall": bool(required_for_overall),
            }
        )

    add("event_windows_present", len(event_table) > 0, int(len(event_table)), "at least one event/window was summarized")
    complete = int(event_table["final_required_models_complete"].map(_as_bool).sum()) if not event_table.empty else 0
    add("final_prefix_core_complete", complete > 0, f"{complete}/{len(event_table)}", "at least one final prefix has complete exact-core evidence")
    claims = int(event_table["has_causal_trajectory_claim"].map(_as_bool).sum()) if not event_table.empty else 0
    add("causal_trajectory_claims_reported", claims >= 0, claims, "causal trajectory claims are explicitly counted")
    agreement_windows = int(offline_agreement.iloc[0]["offline_labeled_windows"]) if not offline_agreement.empty else 0
    add(
        "offline_agreement_evaluated",
        agreement_windows > 0,
        agreement_windows,
        "offline full-window labels were supplied and agreement was evaluated",
        required_for_overall=False,
    )
    null_windows = int(false_positive[false_positive["window_set"].eq("matched_off_swr_windows")]["windows"].iloc[0]) if not false_positive.empty and false_positive["window_set"].eq("matched_off_swr_windows").any() else 0
    add(
        "off_swr_false_positive_evaluated",
        null_windows > 0,
        null_windows,
        "matched off-SWR windows were supplied for false-positive estimation",
        required_for_overall=False,
    )
    latency_real = latency[latency["window_set"].eq("real_windows")] if not latency.empty else pd.DataFrame()
    add(
        "latency_summary_written",
        not latency_real.empty,
        int(latency_real.iloc[0]["windows"]) if not latency_real.empty else 0,
        "latency summary includes real windows",
    )
    required_rows = [row for row in rows if row["required_for_overall"]]
    rows.append(
        {
            "gate": "overall",
            "passed": all(bool(row["passed"]) for row in required_rows),
            "observed": f"{sum(bool(row['passed']) for row in required_rows)}/{len(required_rows)} required gates passed",
            "criterion": "all required causal-detector infrastructure gates pass",
            "required_for_overall": True,
        }
    )
    return pd.DataFrame(rows)


def write_causal_replay_detection_outputs(
    prefix_scores: pd.DataFrame,
    output: str | Path,
    *,
    offline_event_model_evidence: pd.DataFrame | None = None,
    required_models: Sequence[str] = REQUIRED_CAUSAL_MODELS,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
) -> dict[str, pd.DataFrame]:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    offline_labels = (
        offline_labels_from_event_model_evidence(
            offline_event_model_evidence,
            required_models=required_models,
            margin_threshold=margin_threshold,
        )
        if offline_event_model_evidence is not None
        else pd.DataFrame()
    )
    time_bin_table = causal_replay_detection_time_bin_table(
        prefix_scores,
        offline_labels=offline_labels,
        required_models=required_models,
        margin_threshold=margin_threshold,
    )
    event_table = causal_replay_detection_event_table(time_bin_table)
    offline_agreement = causal_offline_agreement_summary(event_table)
    latency = causal_latency_summary(event_table)
    false_positive = causal_false_positive_summary(event_table)
    outputs = {
        PREFIX_EVIDENCE_OUTPUT: prefix_scores,
        TIME_BIN_TABLE_OUTPUT: time_bin_table,
        EVENT_TABLE_OUTPUT: event_table,
        OFFLINE_AGREEMENT_OUTPUT: offline_agreement,
        LATENCY_OUTPUT: latency,
        FALSE_POSITIVE_OUTPUT: false_positive,
        GATE_OUTPUT: causal_detector_gate_summary(event_table, offline_agreement, latency, false_positive),
    }
    for filename, frame in outputs.items():
        frame.to_csv(out / filename, index=False)
    return outputs


def _add_shared_summary_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--offline-event-model-evidence", default="")
    parser.add_argument("--output", default="results/causal-trajectory-family-detector")
    parser.add_argument("--required-models", default=DEFAULT_CAUSAL_MODELS)
    parser.add_argument("--margin-threshold", type=float, default=DEFAULT_MARGIN_THRESHOLD)


def _add_score_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--events", default="run:0-10")
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--models", default=DEFAULT_CAUSAL_MODELS)
    parser.add_argument("--min-prefix-bins", type=int, default=2)
    parser.add_argument("--prefix-stride-bins", type=int, default=1)
    parser.add_argument("--nulls-per-event", type=int, default=0)
    parser.add_argument("--null-random-seed", type=int, default=1)
    parser.add_argument("--spike-count-tolerance-fraction", type=float, default=0.10)
    parser.add_argument("--active-cell-tolerance", type=int)
    parser.add_argument("--null-candidate-step-s", type=float)
    parser.add_argument("--swr-exclusion-padding-s", type=float, default=0.0)
    parser.add_argument("--allow-non-run-nulls", action="store_true")
    _add_model_arguments(parser)
    _add_shared_summary_arguments(parser)
    parser.add_argument("--continue-on-error", action="store_true")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score", help="Score causal prefix evidence and write detector summaries.")
    _add_score_arguments(score_parser)

    summarize_parser = subparsers.add_parser("summarize", help="Summarize an existing causal-prefix evidence CSV.")
    summarize_parser.add_argument("--causal-prefix-evidence", required=True)
    _add_shared_summary_arguments(summarize_parser)

    args = parser.parse_args()
    required_models = _parse_names(args.required_models, REQUIRED_CAUSAL_MODELS)
    offline = pd.read_csv(args.offline_event_model_evidence) if str(args.offline_event_model_evidence).strip() else None

    if args.command == "score":
        prefix_scores = score_causal_prefixes(args)
        if prefix_scores.empty:
            raise RuntimeError("No causal prefix scores were generated.")
    else:
        prefix_scores = pd.read_csv(args.causal_prefix_evidence)
        if prefix_scores.empty:
            raise RuntimeError("Causal prefix evidence table is empty.")

    outputs = write_causal_replay_detection_outputs(
        prefix_scores,
        args.output,
        offline_event_model_evidence=offline,
        required_models=required_models,
        margin_threshold=args.margin_threshold,
    )
    print("Causal replay detector event table:")
    print(outputs[EVENT_TABLE_OUTPUT].head(25).to_string(index=False))
    print("\nCausal offline agreement summary:")
    print(outputs[OFFLINE_AGREEMENT_OUTPUT].to_string(index=False))
    print("\nCausal latency summary:")
    print(outputs[LATENCY_OUTPUT].to_string(index=False))
    print("\nCausal false-positive summary:")
    print(outputs[FALSE_POSITIVE_OUTPUT].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
