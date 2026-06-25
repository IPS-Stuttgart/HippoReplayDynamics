#!/usr/bin/env python3
"""Time-order shuffle control for clean first-order IMM events.

The control asks whether first-order IMM's paired advantage over the fragmented
model depends on the within-event order of population time bins. In real-scoring
mode it rebuilds sorted-spike emissions for selected Pfeiffer/Foster events,
scores first-order IMM and fragmented on the original order, then scores K
whole-bin shuffled emission orders. In fixture mode it consumes precomputed score
rows with the same schema, which keeps CI lightweight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_DIR), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from hipporeplayimm.data import load_replay_session  # noqa: E402
from hipporeplayimm.encoding import EmissionConfig, EncodingConfig, LogEmissionTensor, fit_place_field_encoding  # noqa: E402
from hipporeplayimm.position_validation import (  # noqa: E402
    VALIDATED_POSITION_BIN_SIZE_CM,
    VALIDATED_POSITION_MIN_SPEED_CM_S,
    VALIDATED_POSITION_SMOOTHING_SIGMA_BINS,
)
from hipporeplayimm.result_improvement_extensions import (  # noqa: E402
    ReplayEmissionCalibration,
    build_sorted_emissions_with_replay_calibration,
    score_replay_model_compat,
)
from hipporeplayimm.sorted_spike_state_space import SortedSpikeStateSpaceReplayModel  # noqa: E402
from hipporeplayimm.state_space import StateSpaceDecoderConfig  # noqa: E402

from scripts.audit_imm_fragmented_hypotheses import (  # noqa: E402
    FIRST_ORDER_IMM,
    FRAGMENTED,
    MOMENTUM_EXACT,
    build_event_table,
)
from scripts.benchmark_model_evidence import _check_session, _session_path  # noqa: E402
from scripts.benchmark_model_evidence_improved import (  # noqa: E402
    DEFAULT_IMPROVED_STATE_SPACE_IMM_SWITCH_TAU_S,
    DEFAULT_IMPROVED_STATE_SPACE_MOMENTUM_CANDIDATE_TOP_K,
    DEFAULT_IMPROVED_STATE_SPACE_MOMENTUM_PREDICTED_CANDIDATE_TOP_K,
    _effective_state_space_imm_stickiness,
    _event_windows,
)

MODEL_ORDER = [FIRST_ORDER_IMM, FRAGMENTED]
EVENT_SCORE_COLUMNS = [
    "status",
    "failure_reason",
    "session",
    "rat",
    "event_index",
    "event_group",
    "score_kind",
    "shuffle_index",
    "model",
    "log_evidence",
    "duration_ms",
    "n_time",
    "n_spikes",
    "n_active_units",
    "runtime_s",
]
DECISION_COLUMNS = [
    "session",
    "rat",
    "event_index",
    "event_group",
    "original_delta_imm_minus_fragmented",
    "median_shuffle_delta_imm_minus_fragmented",
    "mean_shuffle_delta_imm_minus_fragmented",
    "p95_shuffle_delta_imm_minus_fragmented",
    "time_order_advantage",
    "empirical_p_value",
    "original_above_shuffle_median",
    "original_above_shuffle_p95",
    "n_shuffles",
    "duration_ms",
    "n_spikes",
    "n_active_units",
]


@dataclass(frozen=True)
class EventKey:
    session: str
    rat: str
    event_index: int
    event_group: str


def _rat(session: object) -> str:
    return str(session).split("/", 1)[0]


def _status_text(passed: bool) -> str:
    return "pass" if passed else "fail"


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_event_model_evidence(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"session", "event_index", "model", "log_evidence"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"event-model evidence is missing required columns: {missing}")
    if "status" in frame.columns:
        frame = frame[frame["status"].fillna("success").astype(str).eq("success")].copy()
    if "evidence_comparable" not in frame.columns:
        frame["evidence_comparable"] = True
    frame["session"] = frame["session"].astype(str)
    frame["rat"] = frame["session"].map(_rat)
    frame["event_index"] = pd.to_numeric(frame["event_index"], errors="raise").astype(int)
    frame["model"] = frame["model"].astype(str)
    frame["log_evidence"] = pd.to_numeric(frame["log_evidence"], errors="coerce")
    return frame.dropna(subset=["log_evidence"]).copy()


def select_event_groups(
    event_model_evidence: pd.DataFrame,
    *,
    margin_threshold: float,
    max_clean_imm: int,
    max_ambiguous: int,
    max_momentum_like: int,
    seed: int,
) -> pd.DataFrame:
    """Select a capped, approximately balanced set of control events."""

    event_table = build_event_table(event_model_evidence, threshold=margin_threshold)
    if event_table.empty:
        return pd.DataFrame(columns=["session", "rat", "event_index", "event_group"])

    clean = event_table[event_table["delta_imm_minus_fragmented"].ge(margin_threshold)].copy()
    ambiguous = event_table[event_table["delta_imm_minus_fragmented"].abs().lt(margin_threshold)].copy()
    momentum = event_table[
        event_table["within_family_classification"].eq("momentum_like_candidate")
        | event_table["best_exact_core_model"].eq(MOMENTUM_EXACT)
    ].copy()

    parts = [
        _balanced_cap(clean, max_clean_imm, seed=seed + 11).assign(event_group="clean_imm"),
        _balanced_cap(ambiguous, max_ambiguous, seed=seed + 17).assign(event_group="imm_fragmented_ambiguous"),
        _balanced_cap(momentum, max_momentum_like, seed=seed + 23).assign(event_group="momentum_like"),
    ]
    selected = pd.concat([part for part in parts if not part.empty], ignore_index=True)
    if selected.empty:
        return pd.DataFrame(columns=["session", "rat", "event_index", "event_group"])
    selected = selected.drop_duplicates(["session", "event_index"], keep="first")
    return selected[["session", "rat", "event_index", "event_group"]].sort_values(["event_group", "rat", "session", "event_index"]).reset_index(drop=True)


def _balanced_cap(frame: pd.DataFrame, cap: int, *, seed: int) -> pd.DataFrame:
    if cap <= 0 or frame.empty:
        return frame.iloc[0:0].copy()
    rng = np.random.default_rng(seed)
    work = frame.copy()
    work["_tie"] = rng.random(len(work))
    work = work.sort_values(["rat", "session", "_tie", "event_index"]).reset_index(drop=True)
    strata = [group.copy() for _, group in work.groupby(["rat", "session"], sort=True)]
    selected = []
    while strata and len(selected) < cap:
        next_strata = []
        for group in strata:
            if len(selected) >= cap:
                break
            selected.append(group.iloc[[0]])
            if len(group) > 1:
                next_strata.append(group.iloc[1:].copy())
        strata = next_strata
    if not selected:
        return work.iloc[0:0].drop(columns=["_tie"], errors="ignore")
    return pd.concat(selected, ignore_index=True).drop(columns=["_tie"], errors="ignore")


def permute_emission_time_bins(emissions: LogEmissionTensor, permutation: np.ndarray) -> LogEmissionTensor:
    """Return emissions with observation rows permuted but the time grid fixed."""

    permutation = np.asarray(permutation, dtype=int)
    if permutation.shape != (emissions.n_time,):
        raise ValueError(f"permutation must have shape ({emissions.n_time},), got {permutation.shape}")
    if sorted(permutation.tolist()) != list(range(emissions.n_time)):
        raise ValueError("permutation must contain each time-bin index exactly once")
    metadata = dict(getattr(emissions, "metadata", {}) or {})
    metadata["time_order_control"] = "whole_bin_shuffle"
    metadata["time_order_permutation"] = " ".join(str(int(x)) for x in permutation)
    return LogEmissionTensor(
        log_likelihood=np.asarray(emissions.log_likelihood, dtype=float)[permutation].copy(),
        spike_counts=np.asarray(emissions.spike_counts)[permutation].copy(),
        times=np.asarray(emissions.times, dtype=float).copy(),
        dt=float(emissions.dt),
        cell_ids=np.asarray(emissions.cell_ids).copy(),
        n_spikes=int(emissions.n_spikes),
        bin_durations=np.asarray(emissions.bin_durations, dtype=float).copy() if emissions.bin_durations is not None else None,
        transition_durations=np.asarray(emissions.transition_durations, dtype=float).copy() if emissions.transition_durations is not None else None,
        metadata=metadata,
    )


def _state_space_config(args: argparse.Namespace, mode: str) -> StateSpaceDecoderConfig:
    return StateSpaceDecoderConfig(
        mode=mode,
        stationary_sigma_cm=args.state_space_stationary_sigma_cm,
        diffusion_sigma_cm_sqrt_s=args.state_space_diffusion_sigma_cm_sqrt_s,
        max_step_sigma=args.state_space_max_step_sigma,
        imm_mode_stickiness=_effective_state_space_imm_stickiness(args),
        imm_switch_tau_s=args.state_space_imm_switch_tau_s,
        momentum_sigma_cm_sqrt_s=args.state_space_momentum_sigma_cm_sqrt_s,
        momentum_initial_sigma_cm_sqrt_s=args.state_space_momentum_initial_sigma_cm_sqrt_s,
        momentum_velocity_decay=args.state_space_momentum_velocity_decay,
        momentum_velocity_decay_tau_s=args.state_space_momentum_velocity_decay_tau_s,
        momentum_candidate_top_k=args.state_space_momentum_candidate_top_k,
        momentum_candidate_mass_threshold=args.state_space_momentum_candidate_mass_threshold,
        momentum_candidate_min_k=args.state_space_momentum_candidate_min_k,
        momentum_candidate_max_k=args.state_space_momentum_candidate_max_k,
        momentum_predicted_candidate_top_k=args.state_space_momentum_predicted_candidate_top_k,
        momentum_candidate_source=args.state_space_momentum_candidate_source,
        valid_occupancy_threshold_s=args.state_space_valid_occupancy_threshold_s,
    )


def _scoring_models(args: argparse.Namespace) -> dict[str, SortedSpikeStateSpaceReplayModel]:
    return {
        FIRST_ORDER_IMM: SortedSpikeStateSpaceReplayModel(
            mode="first-order-imm",
            config=_state_space_config(args, "first-order-imm"),
            name=FIRST_ORDER_IMM,
        ),
        FRAGMENTED: SortedSpikeStateSpaceReplayModel(
            mode="fragmented",
            config=_state_space_config(args, "fragmented"),
            name=FRAGMENTED,
        ),
    }


def score_selected_events(args: argparse.Namespace, selected: pd.DataFrame) -> pd.DataFrame:
    """Run real original/shuffled rescoring for selected events."""

    if selected.empty:
        return pd.DataFrame(columns=EVENT_SCORE_COLUMNS)
    models = _scoring_models(args)
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(args.seed)

    for session_id, session_events in selected.groupby("session", sort=True):
        session_path = _session_path(args.dataset_root, session_id)
        _check_session(session_path)
        session = load_replay_session(session_path)
        encoding_cfg = EncodingConfig(
            bin_size_cm=args.bin_size_cm,
            smoothing_sigma_bins=args.smoothing_sigma_bins,
            min_speed_cm_s=args.min_speed_cm_s,
            min_occupancy_s=args.min_occupancy_s,
            rate_floor_hz=args.rate_floor_hz,
        )
        encoding = fit_place_field_encoding(session, encoding_cfg)
        emissions_cfg = EmissionConfig(
            time_bin_s=args.time_bin_s,
            spike_rate_scale=args.spike_rate_scale,
            likelihood_temperature=args.emission_likelihood_temperature,
            negative_binomial_overdispersion=args.emission_negative_binomial_overdispersion,
        )
        calibration = ReplayEmissionCalibration(
            gain_mode=args.replay_gain_mode,
            gain_prior_count=args.replay_gain_prior_count,
            max_gain=args.replay_gain_max_gain,
            emission_model=args.sorted_spike_emission_model,
            negative_binomial_dispersion=args.negative_binomial_dispersion,
        )
        for event in session_events.itertuples(index=False):
            windows = _event_windows(args, session.ripple(int(event.event_index)))
            if windows.empty:
                rows.extend(
                    _failure_rows(
                        EventKey(session_id, str(event.rat), int(event.event_index), str(event.event_group)),
                        "original",
                        -1,
                        "no event window passed duration gate",
                    )
                )
                continue
            window = windows.iloc[0]
            event_window = SimpleNamespace(start=float(window["window_start_s"]), end=float(window["window_end_s"]))
            try:
                emissions = build_sorted_emissions_with_replay_calibration(
                    session,
                    encoding,
                    event_window,
                    emissions_cfg,
                    calibration=calibration,
                )
            except Exception as exc:  # pragma: no cover - real-data defensive path
                rows.extend(
                    _failure_rows(
                        EventKey(session_id, str(event.rat), int(event.event_index), str(event.event_group)),
                        "original",
                        -1,
                        f"emission build failed: {exc}",
                    )
                )
                continue
            key = EventKey(session_id, str(event.rat), int(event.event_index), str(event.event_group))
            rows.extend(_score_emissions(models, emissions, encoding, key, score_kind="original", shuffle_index=-1))
            for shuffle_index in range(int(args.n_shuffles)):
                permutation = rng.permutation(emissions.n_time)
                shuffled = permute_emission_time_bins(emissions, permutation)
                rows.extend(_score_emissions(models, shuffled, encoding, key, score_kind="shuffle", shuffle_index=shuffle_index))
    return pd.DataFrame(rows, columns=EVENT_SCORE_COLUMNS)


def _failure_rows(key: EventKey, score_kind: str, shuffle_index: int, reason: str) -> list[dict[str, object]]:
    return [
        {
            "status": "failure",
            "failure_reason": reason,
            "session": key.session,
            "rat": key.rat,
            "event_index": key.event_index,
            "event_group": key.event_group,
            "score_kind": score_kind,
            "shuffle_index": shuffle_index,
            "model": model,
            "log_evidence": np.nan,
            "duration_ms": np.nan,
            "n_time": 0,
            "n_spikes": 0,
            "n_active_units": 0,
            "runtime_s": 0.0,
        }
        for model in MODEL_ORDER
    ]


def _score_emissions(
    models: dict[str, SortedSpikeStateSpaceReplayModel],
    emissions: LogEmissionTensor,
    encoding,
    key: EventKey,
    *,
    score_kind: str,
    shuffle_index: int,
) -> list[dict[str, object]]:
    rows = []
    duration_ms = float(np.sum(emissions.bin_durations) * 1000.0) if emissions.bin_durations is not None else float(emissions.n_time * emissions.dt * 1000.0)
    n_active_units = int(np.count_nonzero(np.asarray(emissions.spike_counts).sum(axis=0) > 0))
    for model_name, model in models.items():
        start = time.perf_counter()
        try:
            result = score_replay_model_compat(model, emissions, encoding.bin_centers, occupancy_s=encoding.occupancy_s)
            rows.append(
                {
                    "status": "success",
                    "failure_reason": "",
                    "session": key.session,
                    "rat": key.rat,
                    "event_index": key.event_index,
                    "event_group": key.event_group,
                    "score_kind": score_kind,
                    "shuffle_index": int(shuffle_index),
                    "model": str(result.model_name),
                    "log_evidence": float(result.log_likelihood),
                    "duration_ms": duration_ms,
                    "n_time": int(result.n_time),
                    "n_spikes": int(result.n_spikes),
                    "n_active_units": n_active_units,
                    "runtime_s": float(time.perf_counter() - start),
                }
            )
        except Exception as exc:  # pragma: no cover - real-data defensive path
            rows.append(
                {
                    "status": "failure",
                    "failure_reason": str(exc),
                    "session": key.session,
                    "rat": key.rat,
                    "event_index": key.event_index,
                    "event_group": key.event_group,
                    "score_kind": score_kind,
                    "shuffle_index": int(shuffle_index),
                    "model": model_name,
                    "log_evidence": np.nan,
                    "duration_ms": duration_ms,
                    "n_time": int(emissions.n_time),
                    "n_spikes": int(emissions.n_spikes),
                    "n_active_units": n_active_units,
                    "runtime_s": float(time.perf_counter() - start),
                }
            )
    return rows


def read_precomputed_scores(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted({"session", "event_index", "event_group", "score_kind", "shuffle_index", "model", "log_evidence"}.difference(frame.columns))
    if missing:
        raise ValueError(f"precomputed event scores are missing required columns: {missing}")
    frame = frame.copy()
    if "status" not in frame.columns:
        frame["status"] = "success"
    if "failure_reason" not in frame.columns:
        frame["failure_reason"] = ""
    if "rat" not in frame.columns:
        frame["rat"] = frame["session"].map(_rat)
    for column in ["duration_ms", "n_time", "n_spikes", "n_active_units", "runtime_s"]:
        if column not in frame.columns:
            frame[column] = np.nan
    frame["session"] = frame["session"].astype(str)
    frame["rat"] = frame["rat"].astype(str)
    frame["event_index"] = pd.to_numeric(frame["event_index"], errors="raise").astype(int)
    frame["shuffle_index"] = pd.to_numeric(frame["shuffle_index"], errors="raise").astype(int)
    frame["model"] = frame["model"].astype(str)
    frame["log_evidence"] = pd.to_numeric(frame["log_evidence"], errors="coerce")
    return frame[EVENT_SCORE_COLUMNS].copy()


def build_decisions(event_scores: pd.DataFrame, *, expected_n_shuffles: int) -> pd.DataFrame:
    if event_scores.empty:
        return pd.DataFrame(columns=DECISION_COLUMNS)
    rows = []
    for (session, event_index), group in event_scores.groupby(["session", "event_index"], sort=True):
        success = group[group["status"].astype(str).eq("success")].copy()
        if success.empty:
            rows.append(_empty_decision(session, event_index, group))
            continue
        rat = str(success["rat"].iloc[0])
        event_group = str(success["event_group"].iloc[0])
        original = _delta_for(success[success["score_kind"].astype(str).eq("original")])
        shuffle_deltas = []
        shuffle_group = success[success["score_kind"].astype(str).eq("shuffle")]
        for _, shuffle_rows in shuffle_group.groupby("shuffle_index", sort=True):
            delta = _delta_for(shuffle_rows)
            if np.isfinite(delta):
                shuffle_deltas.append(delta)
        shuffle = np.asarray(shuffle_deltas, dtype=float)
        median_shuffle = float(np.median(shuffle)) if shuffle.size else np.nan
        mean_shuffle = float(np.mean(shuffle)) if shuffle.size else np.nan
        p95_shuffle = float(np.quantile(shuffle, 0.95)) if shuffle.size else np.nan
        advantage = float(original - median_shuffle) if np.isfinite(original) and np.isfinite(median_shuffle) else np.nan
        empirical_p = float((1 + np.count_nonzero(shuffle >= original)) / (1 + shuffle.size)) if np.isfinite(original) and shuffle.size else np.nan
        rows.append(
            {
                "session": str(session),
                "rat": rat,
                "event_index": int(event_index),
                "event_group": event_group,
                "original_delta_imm_minus_fragmented": original,
                "median_shuffle_delta_imm_minus_fragmented": median_shuffle,
                "mean_shuffle_delta_imm_minus_fragmented": mean_shuffle,
                "p95_shuffle_delta_imm_minus_fragmented": p95_shuffle,
                "time_order_advantage": advantage,
                "empirical_p_value": empirical_p,
                "original_above_shuffle_median": bool(np.isfinite(original) and np.isfinite(median_shuffle) and original > median_shuffle),
                "original_above_shuffle_p95": bool(np.isfinite(original) and np.isfinite(p95_shuffle) and original > p95_shuffle),
                "n_shuffles": int(shuffle.size),
                "duration_ms": _first_numeric(success, "duration_ms"),
                "n_spikes": _first_numeric(success, "n_spikes"),
                "n_active_units": _first_numeric(success, "n_active_units"),
                "expected_n_shuffles": int(expected_n_shuffles),
            }
        )
    return pd.DataFrame(rows).sort_values(["event_group", "rat", "session", "event_index"]).reset_index(drop=True)


def _empty_decision(session: str, event_index: int, group: pd.DataFrame) -> dict[str, object]:
    return {
        "session": str(session),
        "rat": str(group["rat"].iloc[0]) if "rat" in group.columns and len(group) else _rat(session),
        "event_index": int(event_index),
        "event_group": str(group["event_group"].iloc[0]) if "event_group" in group.columns and len(group) else "",
        "original_delta_imm_minus_fragmented": np.nan,
        "median_shuffle_delta_imm_minus_fragmented": np.nan,
        "mean_shuffle_delta_imm_minus_fragmented": np.nan,
        "p95_shuffle_delta_imm_minus_fragmented": np.nan,
        "time_order_advantage": np.nan,
        "empirical_p_value": np.nan,
        "original_above_shuffle_median": False,
        "original_above_shuffle_p95": False,
        "n_shuffles": 0,
        "duration_ms": np.nan,
        "n_spikes": np.nan,
        "n_active_units": np.nan,
        "expected_n_shuffles": 0,
    }


def _delta_for(rows: pd.DataFrame) -> float:
    values = rows.set_index("model")["log_evidence"]
    if FIRST_ORDER_IMM not in values.index or FRAGMENTED not in values.index:
        return np.nan
    return _safe_float(values.loc[FIRST_ORDER_IMM]) - _safe_float(values.loc[FRAGMENTED])


def _first_numeric(rows: pd.DataFrame, column: str) -> float:
    if column not in rows.columns:
        return np.nan
    values = pd.to_numeric(rows[column], errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else np.nan


def build_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"metric": "events", "value": len(decisions)},
        {"metric": "clean_imm_events", "value": int(decisions["event_group"].eq("clean_imm").sum()) if len(decisions) else 0},
        {
            "metric": "ambiguous_control_events",
            "value": int(decisions["event_group"].eq("imm_fragmented_ambiguous").sum()) if len(decisions) else 0,
        },
        {"metric": "momentum_like_control_events", "value": int(decisions["event_group"].eq("momentum_like").sum()) if len(decisions) else 0},
    ]
    for column in [
        "original_delta_imm_minus_fragmented",
        "median_shuffle_delta_imm_minus_fragmented",
        "time_order_advantage",
        "empirical_p_value",
    ]:
        values = pd.to_numeric(decisions[column], errors="coerce").dropna() if column in decisions else pd.Series(dtype=float)
        rows.append({"metric": f"mean_{column}", "value": float(values.mean()) if not values.empty else np.nan})
        rows.append({"metric": f"median_{column}", "value": float(values.median()) if not values.empty else np.nan})
    if len(decisions):
        rows.append({"metric": "original_above_shuffle_median_events", "value": int(decisions["original_above_shuffle_median"].sum())})
        rows.append({"metric": "original_above_shuffle_p95_events", "value": int(decisions["original_above_shuffle_p95"].sum())})
    return pd.DataFrame(rows)


def summarize_by(decisions: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out_columns = columns + [
        "events",
        "median_original_delta_imm_minus_fragmented",
        "median_shuffle_delta_imm_minus_fragmented",
        "median_time_order_advantage",
        "mean_time_order_advantage",
        "original_above_shuffle_median_fraction",
        "original_above_shuffle_p95_count",
    ]
    if decisions.empty:
        return pd.DataFrame(columns=out_columns)
    rows = []
    for keys, group in decisions.groupby(columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(columns, keys, strict=True))
        row.update(
            {
                "events": len(group),
                "median_original_delta_imm_minus_fragmented": _median(group["original_delta_imm_minus_fragmented"]),
                "median_shuffle_delta_imm_minus_fragmented": _median(group["median_shuffle_delta_imm_minus_fragmented"]),
                "median_time_order_advantage": _median(group["time_order_advantage"]),
                "mean_time_order_advantage": _mean(group["time_order_advantage"]),
                "original_above_shuffle_median_fraction": float(group["original_above_shuffle_median"].mean()) if len(group) else np.nan,
                "original_above_shuffle_p95_count": int(group["original_above_shuffle_p95"].sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=out_columns)


def _median(values: Iterable[object]) -> float:
    numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    return float(numeric.median()) if not numeric.empty else np.nan


def _mean(values: Iterable[object]) -> float:
    numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else np.nan


def build_gate_summary(
    event_scores: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    expected_n_shuffles: int,
    manifest_written: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(gate: str, passed: bool, observed: object, criterion: str, gate_type: str = "technical") -> None:
        rows.append(
            {
                "gate": gate,
                "gate_type": gate_type,
                "passed": bool(passed),
                "status": _status_text(bool(passed)),
                "observed": observed,
                "criterion": criterion,
            }
        )

    selected_events = len(decisions)
    success = event_scores[event_scores["status"].astype(str).eq("success")] if not event_scores.empty else event_scores.copy()
    original = success[success["score_kind"].astype(str).eq("original")] if not success.empty else success.copy()
    original_pairs = original.groupby(["session", "event_index"])["model"].apply(lambda x: set(x.astype(str))) if not original.empty else pd.Series(dtype=object)
    original_complete = int(sum(set(MODEL_ORDER).issubset(models) for models in original_pairs)) if len(original_pairs) else 0
    shuffle_complete_events = int(decisions["n_shuffles"].ge(expected_n_shuffles).sum()) if len(decisions) else 0
    add("selected_events_present", selected_events > 0, selected_events, "at least one selected event")
    add(
        "required_models_complete",
        selected_events > 0 and original_complete == selected_events and shuffle_complete_events == selected_events,
        f"original {original_complete}/{selected_events}; shuffles {shuffle_complete_events}/{selected_events}",
        "IMM and fragmented rows exist for every original and expected shuffle",
    )
    add("all_original_scores_present", selected_events > 0 and original_complete == selected_events, f"{original_complete}/{selected_events}", "all selected events have original IMM and fragmented scores")
    add("all_shuffle_scores_present", selected_events > 0 and shuffle_complete_events == selected_events, f"{shuffle_complete_events}/{selected_events}", "all selected events have all expected shuffled paired scores")
    add("no_scoring_failures", len(event_scores) > 0 and bool(event_scores["status"].astype(str).eq("success").all()), int((event_scores["status"].astype(str) != "success").sum()) if len(event_scores) else 0, "zero failed score rows")
    add("n_shuffles_complete", selected_events > 0 and bool(decisions["n_shuffles"].eq(expected_n_shuffles).all()), f"{shuffle_complete_events}/{selected_events}", f"each selected event has {expected_n_shuffles} complete shuffled deltas")
    represented = selected_events > 0 and (decisions["rat"].nunique() >= 2 or decisions["session"].nunique() >= 2)
    add("rats_or_sessions_represented", represented, f"{decisions['rat'].nunique() if len(decisions) else 0} rats; {decisions['session'].nunique() if len(decisions) else 0} sessions", "at least two rats or two sessions represented")
    add("manifest_written", manifest_written, manifest_written, "manifest JSON written")

    clean = decisions[decisions["event_group"].eq("clean_imm")] if len(decisions) else pd.DataFrame()
    ambiguous = decisions[decisions["event_group"].eq("imm_fragmented_ambiguous")] if len(decisions) else pd.DataFrame()
    clean_adv = pd.to_numeric(clean["time_order_advantage"], errors="coerce").dropna() if not clean.empty else pd.Series(dtype=float)
    ambiguous_adv = pd.to_numeric(ambiguous["time_order_advantage"], errors="coerce").dropna() if not ambiguous.empty else pd.Series(dtype=float)
    clean_median = float(clean_adv.median()) if not clean_adv.empty else np.nan
    ambiguous_median = float(ambiguous_adv.median()) if not ambiguous_adv.empty else np.nan
    add(
        "clean_imm_median_time_order_advantage_positive",
        np.isfinite(clean_median) and clean_median > 0.0,
        clean_median,
        "clean IMM median time-order advantage > 0",
        gate_type="interpretation",
    )
    clean_fraction = float(clean["original_above_shuffle_median"].mean()) if len(clean) else np.nan
    add(
        "clean_imm_majority_original_above_shuffle_median",
        np.isfinite(clean_fraction) and clean_fraction > 0.60,
        clean_fraction,
        ">60% clean IMM events above shuffled median",
        gate_type="interpretation",
    )
    clean_p95 = int(clean["original_above_shuffle_p95"].sum()) if len(clean) else 0
    add(
        "clean_imm_at_least_some_original_above_shuffle_p95",
        clean_p95 > 0,
        clean_p95,
        "at least one clean IMM event above shuffled p95",
        gate_type="interpretation",
    )
    add(
        "ambiguous_controls_lower_advantage_than_clean_imm",
        np.isfinite(clean_median) and np.isfinite(ambiguous_median) and ambiguous_median < clean_median,
        f"clean={clean_median}; ambiguous={ambiguous_median}",
        "ambiguous-control median advantage lower than clean IMM median",
        gate_type="interpretation",
    )
    technical_passed = all(row["passed"] for row in rows if row["gate_type"] == "technical")
    interpretation_passed = all(row["passed"] for row in rows if row["gate_type"] == "interpretation")
    rows.append(
        {
            "gate": "technical_overall",
            "gate_type": "technical",
            "passed": technical_passed,
            "status": _status_text(technical_passed),
            "observed": f"{sum(row['passed'] for row in rows if row['gate_type'] == 'technical')}/{sum(row['gate_type'] == 'technical' for row in rows)} technical gates passed",
            "criterion": "all technical gates pass",
        }
    )
    rows.append(
        {
            "gate": "overall",
            "gate_type": "overall",
            "passed": technical_passed and interpretation_passed,
            "status": _status_text(technical_passed and interpretation_passed),
            "observed": f"technical={technical_passed}; interpretation={interpretation_passed}",
            "criterion": "technical and interpretation gates pass",
        }
    )
    return pd.DataFrame(rows)


def write_figures(decisions: pd.DataFrame, event_scores: pd.DataFrame, output: Path) -> list[str]:
    files: list[str] = []
    if decisions.empty:
        return files
    color_map = {"clean_imm": "#1f77b4", "imm_fragmented_ambiguous": "#ff7f0e", "momentum_like": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(6, 5))
    for group, rows in decisions.groupby("event_group", sort=True):
        ax.scatter(
            rows["median_shuffle_delta_imm_minus_fragmented"],
            rows["original_delta_imm_minus_fragmented"],
            label=group,
            alpha=0.85,
            color=color_map.get(str(group), "#7f7f7f"),
        )
    values = pd.to_numeric(pd.concat([decisions["median_shuffle_delta_imm_minus_fragmented"], decisions["original_delta_imm_minus_fragmented"]]), errors="coerce").dropna()
    if not values.empty:
        lo = float(values.min())
        hi = float(values.max())
        pad = max(1.0, 0.05 * (hi - lo))
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linewidth=1, linestyle="--")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Median shuffled ΔIMM-fragmented")
    ax.set_ylabel("Original ΔIMM-fragmented")
    ax.legend(frameon=False)
    fig.tight_layout()
    scatter = output / "clean_imm_time_order_original_vs_shuffled_scatter.png"
    fig.savefig(scatter, dpi=160)
    plt.close(fig)
    files.append(scatter.name)

    fig, ax = plt.subplots(figsize=(7, 4))
    groups = [group for group in ["clean_imm", "imm_fragmented_ambiguous", "momentum_like"] if decisions["event_group"].eq(group).any()]
    data = [pd.to_numeric(decisions.loc[decisions["event_group"].eq(group), "time_order_advantage"], errors="coerce").dropna().to_numpy() for group in groups]
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(groups)
    for idx, values_array in enumerate(data, start=1):
        if values_array.size:
            jitter = np.linspace(-0.12, 0.12, values_array.size)
            ax.scatter(np.full(values_array.size, idx) + jitter, values_array, color="#333333", s=18, alpha=0.75)
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_ylabel("Time-order advantage")
    fig.tight_layout()
    box = output / "clean_imm_time_order_advantage_by_group.png"
    fig.savefig(box, dpi=160)
    plt.close(fig)
    files.append(box.name)

    panel_events = _representative_events(decisions)
    if panel_events:
        fig, axes = plt.subplots(len(panel_events), 1, figsize=(7, max(2.4, 2.1 * len(panel_events))), squeeze=False)
        for ax, (_, row) in zip(axes[:, 0], panel_events, strict=True):
            deltas = _shuffle_deltas_for_event(event_scores, str(row["session"]), int(row["event_index"]))
            original = _safe_float(row["original_delta_imm_minus_fragmented"])
            ax.hist(deltas, bins=min(12, max(3, len(deltas))), color="#b8c8dd", edgecolor="#40566d")
            ax.axvline(original, color="#d62728", linewidth=2, label="original")
            ax.set_title(f"{row['event_group']} {row['session']} event {int(row['event_index'])}  p={_safe_float(row['empirical_p_value']):.3f}")
            ax.set_xlabel("ΔIMM-fragmented")
            ax.set_ylabel("shuffles")
        fig.tight_layout()
        panels = output / "clean_imm_time_order_representative_panels.png"
        fig.savefig(panels, dpi=160)
        plt.close(fig)
        files.append(panels.name)
    return files


def _representative_events(decisions: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    out: list[tuple[str, pd.Series]] = []
    clean = decisions[decisions["event_group"].eq("clean_imm")].copy()
    if not clean.empty:
        out.append(("top_clean", clean.sort_values("time_order_advantage", ascending=False).iloc[0]))
        clean_fail = clean[~clean["original_above_shuffle_median"].astype(bool)]
        if not clean_fail.empty:
            out.append(("clean_fail", clean_fail.iloc[0]))
    for event_index, label in [(540, "event_540"), (550, "event_550")]:
        match = decisions[decisions["event_index"].astype(int).eq(event_index)]
        if not match.empty:
            out.append((label, match.iloc[0]))
    seen = set()
    unique = []
    for label, row in out:
        key = (str(row["session"]), int(row["event_index"]))
        if key not in seen:
            unique.append((label, row))
            seen.add(key)
    return unique[:4]


def _shuffle_deltas_for_event(event_scores: pd.DataFrame, session: str, event_index: int) -> np.ndarray:
    rows = event_scores[
        event_scores["session"].astype(str).eq(session)
        & event_scores["event_index"].astype(int).eq(event_index)
        & event_scores["score_kind"].astype(str).eq("shuffle")
        & event_scores["status"].astype(str).eq("success")
    ]
    deltas = []
    for _, group in rows.groupby("shuffle_index", sort=True):
        delta = _delta_for(group)
        if np.isfinite(delta):
            deltas.append(delta)
    return np.asarray(deltas, dtype=float)


def _sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata() -> dict[str, object]:
    def run(*cmd: str) -> str:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()

    try:
        commit = run("git", "rev-parse", "HEAD")
        branch = run("git", "rev-parse", "--abbrev-ref", "HEAD")
        dirty = bool(run("git", "status", "--porcelain"))
        return {"code_commit": commit, "git_branch": branch, "git_dirty": dirty, "git_error": ""}
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        return {"code_commit": "unavailable", "git_branch": "unavailable", "git_dirty": "", "git_error": str(exc)}


def write_manifest(
    args: argparse.Namespace,
    output: Path,
    *,
    selected_events: int,
    score_rows: int,
    figure_files: list[str],
) -> None:
    input_paths = {
        "event_model_evidence": args.event_model_evidence or "",
        "precomputed_event_scores": args.precomputed_event_scores or "",
    }
    manifest = {
        **_git_metadata(),
        "command_line": " ".join(sys.argv),
        "working_directory": str(Path.cwd()),
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset_root": args.dataset_root,
        "input_paths": input_paths,
        "input_sha256": {key: _sha256(Path(value)) if value else "" for key, value in input_paths.items()},
        "margin_threshold": float(args.margin_threshold),
        "n_shuffles": int(args.n_shuffles),
        "max_clean_imm": int(args.max_clean_imm),
        "max_ambiguous": int(args.max_ambiguous),
        "max_momentum_like": int(args.max_momentum_like),
        "seed": int(args.seed),
        "models": MODEL_ORDER,
        "shuffle_unit": "whole_time_bin_population_vector",
        "selected_events": int(selected_events),
        "score_rows": int(score_rows),
        "figure_files": figure_files,
    }
    (output / "clean_imm_time_order_shuffle_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_outputs(event_scores: pd.DataFrame, output: str | Path, *, expected_n_shuffles: int, args: argparse.Namespace | None = None) -> dict[str, pd.DataFrame]:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    event_scores = event_scores.copy()
    if event_scores.empty:
        event_scores = pd.DataFrame(columns=EVENT_SCORE_COLUMNS)
    decisions = build_decisions(event_scores, expected_n_shuffles=expected_n_shuffles)
    summary = build_summary(decisions)
    by_group = summarize_by(decisions, ["event_group"])
    by_rat = summarize_by(decisions, ["rat"])
    by_session = summarize_by(decisions, ["session"])

    outputs = {
        "clean_imm_time_order_shuffle_event_scores.csv": event_scores[EVENT_SCORE_COLUMNS],
        "clean_imm_time_order_shuffle_decisions.csv": decisions,
        "clean_imm_time_order_shuffle_summary.csv": summary,
        "clean_imm_time_order_shuffle_by_group.csv": by_group,
        "clean_imm_time_order_shuffle_by_rat.csv": by_rat,
        "clean_imm_time_order_shuffle_by_session.csv": by_session,
    }
    for name, frame in outputs.items():
        frame.to_csv(out / name, index=False)

    figure_files = write_figures(decisions, event_scores, out)
    if args is not None:
        write_manifest(args, out, selected_events=len(decisions), score_rows=len(event_scores), figure_files=figure_files)
    manifest_written = (out / "clean_imm_time_order_shuffle_manifest.json").is_file() if args is not None else False
    gates = build_gate_summary(event_scores, decisions, expected_n_shuffles=expected_n_shuffles, manifest_written=manifest_written)
    gates.to_csv(out / "clean_imm_time_order_shuffle_gate_summary.csv", index=False)
    outputs["clean_imm_time_order_shuffle_gate_summary.csv"] = gates
    return outputs


def _selected_from_evidence(args: argparse.Namespace) -> pd.DataFrame:
    if not args.event_model_evidence:
        raise ValueError("--event-model-evidence is required unless --precomputed-event-scores is used")
    evidence = _read_event_model_evidence(args.event_model_evidence)
    selected = select_event_groups(
        evidence,
        margin_threshold=args.margin_threshold,
        max_clean_imm=args.max_clean_imm,
        max_ambiguous=args.max_ambiguous,
        max_momentum_like=args.max_momentum_like,
        seed=args.seed,
    )
    if selected.empty:
        raise RuntimeError("No events selected for clean-IMM time-order shuffle control")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-model-evidence", default="", help="Full-core event-model-evidence CSV used to select event groups")
    parser.add_argument("--precomputed-event-scores", default="", help="Optional precomputed original/shuffle score rows for fixture/debug mode")
    parser.add_argument("--dataset-root", default="data/DataSetFromPfeifferFoster", help="Pfeiffer/Foster dataset root for real rescoring mode")
    parser.add_argument("--output", required=True)
    parser.add_argument("--margin-threshold", type=float, default=5.5)
    parser.add_argument("--max-clean-imm", type=int, default=20)
    parser.add_argument("--max-ambiguous", type=int, default=10)
    parser.add_argument("--max-momentum-like", type=int, default=10)
    parser.add_argument("--n-shuffles", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--state-space-stationary-sigma-cm", type=float, default=2.0)
    parser.add_argument("--state-space-diffusion-sigma-cm-sqrt-s", type=float, default=85.0)
    parser.add_argument("--state-space-max-step-sigma", type=float, default=4.0)
    parser.add_argument("--state-space-valid-occupancy-threshold-s", type=float, default=0.0)
    parser.add_argument("--state-space-imm-mode-stickiness", type=float, default=0.95)
    parser.add_argument("--state-space-imm-switch-tau-s", type=float, default=DEFAULT_IMPROVED_STATE_SPACE_IMM_SWITCH_TAU_S)
    parser.add_argument("--state-space-momentum-sigma-cm-sqrt-s", type=float, default=85.0)
    parser.add_argument("--state-space-momentum-initial-sigma-cm-sqrt-s", type=float, default=85.0)
    parser.add_argument("--state-space-momentum-velocity-decay", type=float, default=0.95)
    parser.add_argument("--state-space-momentum-velocity-decay-tau-s", type=float, default=0.0)
    parser.add_argument("--state-space-momentum-candidate-top-k", type=int, default=DEFAULT_IMPROVED_STATE_SPACE_MOMENTUM_CANDIDATE_TOP_K)
    parser.add_argument("--state-space-momentum-candidate-mass-threshold", type=float)
    parser.add_argument("--state-space-momentum-candidate-min-k", type=int, default=1)
    parser.add_argument("--state-space-momentum-candidate-max-k", type=int, default=0)
    parser.add_argument("--state-space-momentum-predicted-candidate-top-k", type=int, default=DEFAULT_IMPROVED_STATE_SPACE_MOMENTUM_PREDICTED_CANDIDATE_TOP_K)
    parser.add_argument("--state-space-momentum-candidate-source", choices=("emission", "posterior"), default="emission")
    parser.add_argument("--time-bin-s", type=float, default=0.003)
    parser.add_argument("--spike-rate-scale", type=float, default=1.0)
    parser.add_argument("--emission-likelihood-temperature", type=float, default=1.0)
    parser.add_argument("--emission-negative-binomial-overdispersion", type=float, default=0.0)
    parser.add_argument("--sorted-spike-emission-model", choices=("poisson", "negative-binomial", "gamma-poisson"), default="poisson")
    parser.add_argument("--replay-gain-mode", choices=("none", "event", "cell", "event-cell"), default="none")
    parser.add_argument("--replay-gain-prior-count", type=float, default=10.0)
    parser.add_argument("--replay-gain-max-gain", type=float, default=20.0)
    parser.add_argument("--negative-binomial-dispersion", type=float, default=50.0)
    parser.add_argument("--bin-size-cm", type=float, default=VALIDATED_POSITION_BIN_SIZE_CM)
    parser.add_argument("--smoothing-sigma-bins", type=float, default=VALIDATED_POSITION_SMOOTHING_SIGMA_BINS)
    parser.add_argument("--min-speed-cm-s", type=float, default=VALIDATED_POSITION_MIN_SPEED_CM_S)
    parser.add_argument("--min-occupancy-s", type=float, default=EncodingConfig().min_occupancy_s)
    parser.add_argument("--rate-floor-hz", type=float, default=EncodingConfig().rate_floor_hz)
    parser.add_argument("--window-variant-specs", default="")
    parser.add_argument("--window-pre-pads-s", default="0.0")
    parser.add_argument("--window-post-pads-s", default="0.0")
    parser.add_argument("--window-min-duration-s", type=float, default=0.005)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.n_shuffles < 1:
        raise ValueError("--n-shuffles must be at least 1")
    if args.precomputed_event_scores:
        event_scores = read_precomputed_scores(args.precomputed_event_scores)
    else:
        selected = _selected_from_evidence(args)
        event_scores = score_selected_events(args, selected)
    outputs = write_outputs(event_scores, args.output, expected_n_shuffles=args.n_shuffles, args=args)
    gates = outputs["clean_imm_time_order_shuffle_gate_summary.csv"]
    technical = gates[gates["gate"].eq("technical_overall")]
    status = str(technical.iloc[0]["status"]) if not technical.empty else "missing"
    print(f"Wrote clean IMM time-order shuffle control to {args.output}")
    print(f"Technical gate status: {status}")
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
