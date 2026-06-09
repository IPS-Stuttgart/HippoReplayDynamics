#!/usr/bin/env python3
"""Infer compositional replay mode grammars within individual events.

This analysis is deliberately opt-in.  It does not replace full-event model
comparison; it asks a different question: whether an event is better described
as a short sequence of mode segments such as stationary -> momentum ->
fragmented.  Segment scores are computed under stationary, diffusion, momentum,
and fragmented state-space modes, then a semi-Markov Viterbi pass selects an
explicit-duration mode sequence.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from benchmark_model_evidence import _check_session, _events, _session_path
from hipporeplayimm.data import load_replay_session
from hipporeplayimm.encoding import EmissionConfig, EncodingConfig, LogEmissionTensor, fit_place_field_encoding
from hipporeplayimm.position_validation import (
    VALIDATED_POSITION_BIN_SIZE_CM,
    VALIDATED_POSITION_MIN_SPEED_CM_S,
    VALIDATED_POSITION_SMOOTHING_SIGMA_BINS,
)
from hipporeplayimm.result_improvement_extensions import (
    ReplayEmissionCalibration,
    build_sorted_emissions_with_replay_calibration,
    score_replay_model_compat,
)
from hipporeplayimm.sorted_spike_state_space import SortedSpikeStateSpaceReplayModel
from hipporeplayimm.state_space import StateSpaceDecoderConfig

MODE_SEQUENCE_OUTPUT = "replay_mode_sequence.csv"
MOTIF_OUTPUT = "replay_grammar_motifs.csv"
MODE_DURATION_OUTPUT = "mode_duration_summary.csv"
RAT_SUMMARY_OUTPUT = "rat_replay_grammar_summary.csv"

GRAMMAR_MODES = ("stationary", "diffusion", "momentum", "fragmented")
TRAJECTORY_GRAMMAR_MODES = ("diffusion", "momentum")
DEFAULT_MODE_MEAN_DURATION_BINS = {
    "stationary": 3.0,
    "diffusion": 6.0,
    "momentum": 6.0,
    "fragmented": 3.0,
}
_EVENT_KEY_CANDIDATES = (
    "session",
    "event_index",
    "window_role",
    "event_window_variant",
    "window_index",
    "null_index",
    "matched_null_rank",
)


@dataclass(frozen=True)
class GrammarInferenceConfig:
    """Semi-Markov replay grammar settings."""

    max_segments: int = 4
    min_segment_bins: int = 1
    max_segment_bins: int = 0
    duration_prior_log_sd: float = 0.75
    mode_switch_penalty: float = 0.0
    mode_mean_duration_bins: dict[str, float] | None = None

    def means(self) -> dict[str, float]:
        values = dict(DEFAULT_MODE_MEAN_DURATION_BINS)
        if self.mode_mean_duration_bins:
            values.update({str(key): float(value) for key, value in self.mode_mean_duration_bins.items()})
        return values


def parse_mode_mean_duration_bins(value: str | None) -> dict[str, float]:
    """Parse ``mode:mean`` entries separated by spaces or commas."""

    if value is None or not str(value).strip():
        return {}
    means: dict[str, float] = {}
    for raw in str(value).replace(",", " ").split():
        text = raw.strip()
        if not text:
            continue
        if ":" not in text:
            raise ValueError("mode duration means must use mode:mean_bins entries")
        mode, mean = text.split(":", 1)
        mode = mode.strip().lower()
        if mode not in GRAMMAR_MODES:
            raise ValueError(f"unknown grammar mode {mode!r}; expected one of {GRAMMAR_MODES}")
        mean_value = float(mean)
        if not np.isfinite(mean_value) or mean_value <= 0.0:
            raise ValueError("mode mean duration bins must be finite and positive")
        means[mode] = mean_value
    return means


def write_replay_grammar_outputs(
    segment_scores: pd.DataFrame,
    outdir: str | Path,
    *,
    config: GrammarInferenceConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Infer and write replay grammar tables from segment-level mode scores."""

    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    cfg = GrammarInferenceConfig() if config is None else config
    sequence = infer_replay_mode_sequence(segment_scores, config=cfg)
    motifs = replay_grammar_motifs(sequence)
    durations = mode_duration_summary(sequence)
    rat = rat_replay_grammar_summary(motifs)
    tables = {
        MODE_SEQUENCE_OUTPUT: sequence,
        MOTIF_OUTPUT: motifs,
        MODE_DURATION_OUTPUT: durations,
        RAT_SUMMARY_OUTPUT: rat,
    }
    for filename, table in tables.items():
        table.to_csv(output / filename, index=False)
    return tables


def infer_replay_mode_sequence(
    segment_scores: pd.DataFrame,
    *,
    config: GrammarInferenceConfig | None = None,
) -> pd.DataFrame:
    """Infer one semi-Markov mode path per event from candidate segment scores."""

    cfg = GrammarInferenceConfig() if config is None else config
    _validate_grammar_config(cfg)
    if segment_scores.empty:
        return _empty_mode_sequence()
    key_columns = _event_key_columns(segment_scores)
    if not key_columns:
        raise ValueError("segment_scores must include at least one event key column")
    rows: list[dict[str, object]] = []
    for key, group in segment_scores.groupby(key_columns, dropna=False, sort=False):
        event_key = _key_dict(key_columns, key)
        sequence = _infer_event_sequence(group, event_key, cfg)
        rows.extend(sequence)
    if not rows:
        return _empty_mode_sequence()
    return pd.DataFrame(rows).sort_values([*key_columns, "segment_index"]).reset_index(drop=True)


def replay_grammar_motifs(sequence: pd.DataFrame) -> pd.DataFrame:
    """Return one motif row per event from segment-level mode paths."""

    if sequence.empty:
        return _empty_motifs()
    key_columns = _event_key_columns(sequence)
    rows: list[dict[str, object]] = []
    for key, group in sequence.groupby(key_columns, dropna=False, sort=False):
        event_key = _key_dict(key_columns, key)
        ordered = group.sort_values("segment_index")
        modes = ordered["mode"].astype(str).tolist()
        durations = ordered["segment_duration_s"].astype(float)
        total_duration = float(durations.sum())
        fractions = {
            f"{mode}_duration_fraction": _safe_fraction(
                float(ordered.loc[ordered["mode"].eq(mode), "segment_duration_s"].astype(float).sum()),
                total_duration,
            )
            for mode in GRAMMAR_MODES
        }
        mode_durations = ordered.groupby("mode")["segment_duration_s"].sum()
        dominant_mode = str(mode_durations.idxmax()) if not mode_durations.empty else ""
        trajectory_fraction = sum(fractions[f"{mode}_duration_fraction"] for mode in TRAJECTORY_GRAMMAR_MODES)
        motif = "->".join(modes)
        rows.append(
            {
                **event_key,
                "motif": motif,
                "motif_family": _motif_family(modes),
                "segment_count": int(len(modes)),
                "event_grammar_score": float(ordered["event_grammar_score"].iloc[0]),
                "event_duration_s": total_duration,
                "event_n_time": int(ordered["event_n_time"].max()),
                "event_n_spikes": int(ordered["event_n_spikes"].max()),
                "dominant_mode": dominant_mode,
                "compositional_replay": bool(len(modes) > 1),
                "has_trajectory_segment": bool(any(mode in TRAJECTORY_GRAMMAR_MODES for mode in modes)),
                "has_momentum_segment": bool("momentum" in modes),
                "has_stationary_prelude": bool(modes and modes[0] == "stationary" and len(modes) > 1),
                "has_fragmented_endpoint": bool(modes and modes[-1] == "fragmented" and len(modes) > 1),
                "trajectory_duration_fraction": trajectory_fraction,
                **fractions,
            }
        )
    return pd.DataFrame(rows).sort_values(key_columns).reset_index(drop=True)


def mode_duration_summary(sequence: pd.DataFrame) -> pd.DataFrame:
    """Summarize segment durations by grammar mode."""

    columns = [
        "mode",
        "segments",
        "events_with_mode",
        "mean_duration_bins",
        "median_duration_bins",
        "mean_duration_s",
        "median_duration_s",
        "total_duration_s",
        "mean_event_duration_fraction",
    ]
    if sequence.empty:
        return pd.DataFrame(columns=columns)
    key_columns = _event_key_columns(sequence)
    event_mode = sequence.drop_duplicates([*key_columns, "mode"])
    rows = []
    for mode, group in sequence.groupby("mode", sort=False):
        rows.append(
            {
                "mode": str(mode),
                "segments": int(len(group)),
                "events_with_mode": int(event_mode[event_mode["mode"].eq(mode)].shape[0]),
                "mean_duration_bins": float(group["segment_duration_bins"].astype(float).mean()),
                "median_duration_bins": float(group["segment_duration_bins"].astype(float).median()),
                "mean_duration_s": float(group["segment_duration_s"].astype(float).mean()),
                "median_duration_s": float(group["segment_duration_s"].astype(float).median()),
                "total_duration_s": float(group["segment_duration_s"].astype(float).sum()),
                "mean_event_duration_fraction": float(group["event_mode_duration_fraction"].astype(float).mean()),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("mode").reset_index(drop=True)


def rat_replay_grammar_summary(motifs: pd.DataFrame) -> pd.DataFrame:
    """Summarize motif composition at rat level."""

    columns = [
        "rat",
        "events",
        "unique_motifs",
        "most_common_motif",
        "most_common_motif_events",
        "compositional_events",
        "compositional_fraction",
        "events_with_trajectory_segment",
        "trajectory_segment_fraction",
        "events_with_momentum_segment",
        "momentum_segment_fraction",
        "events_with_stationary_prelude",
        "stationary_prelude_fraction",
        "events_with_fragmented_endpoint",
        "fragmented_endpoint_fraction",
        "mean_trajectory_duration_fraction",
    ]
    if motifs.empty:
        return pd.DataFrame(columns=columns)
    frame = motifs.copy()
    frame["rat"] = frame["session"].astype(str).str.split("/", n=1).str[0] if "session" in frame else "unknown"
    rows = []
    for rat, group in frame.groupby("rat", sort=True):
        motif_counts = group["motif"].astype(str).value_counts()
        most_common = str(motif_counts.index[0]) if not motif_counts.empty else ""
        rows.append(
            {
                "rat": str(rat),
                "events": int(len(group)),
                "unique_motifs": int(group["motif"].nunique()),
                "most_common_motif": most_common,
                "most_common_motif_events": int(motif_counts.iloc[0]) if not motif_counts.empty else 0,
                "compositional_events": int(group["compositional_replay"].fillna(False).sum()),
                "compositional_fraction": float(group["compositional_replay"].fillna(False).mean()),
                "events_with_trajectory_segment": int(group["has_trajectory_segment"].fillna(False).sum()),
                "trajectory_segment_fraction": float(group["has_trajectory_segment"].fillna(False).mean()),
                "events_with_momentum_segment": int(group["has_momentum_segment"].fillna(False).sum()),
                "momentum_segment_fraction": float(group["has_momentum_segment"].fillna(False).mean()),
                "events_with_stationary_prelude": int(group["has_stationary_prelude"].fillna(False).sum()),
                "stationary_prelude_fraction": float(group["has_stationary_prelude"].fillna(False).mean()),
                "events_with_fragmented_endpoint": int(group["has_fragmented_endpoint"].fillna(False).sum()),
                "fragmented_endpoint_fraction": float(group["has_fragmented_endpoint"].fillna(False).mean()),
                "mean_trajectory_duration_fraction": float(group["trajectory_duration_fraction"].astype(float).mean()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _infer_event_sequence(
    group: pd.DataFrame,
    event_key: dict[str, object],
    config: GrammarInferenceConfig,
) -> list[dict[str, object]]:
    ok = group[group["status"].fillna("success").eq("success")] if "status" in group else group.copy()
    ok = ok[ok["mode"].astype(str).isin(GRAMMAR_MODES)].copy()
    ok = ok[np.isfinite(ok["segment_log_evidence"].astype(float))]
    if ok.empty:
        return []
    n_time = int(ok["event_n_time"].max()) if "event_n_time" in ok else int(ok["end_bin_exclusive"].max())
    segment_lookup = _segment_score_lookup(ok)
    best: dict[tuple[int, int, str], float] = {}
    back: dict[tuple[int, int, str], dict[str, object]] = {}
    means = config.means()
    max_segment_bins = int(config.max_segment_bins)
    for segment_count in range(1, int(config.max_segments) + 1):
        for end_bin in range(1, n_time + 1):
            start_min = 0 if max_segment_bins <= 0 else max(0, end_bin - max_segment_bins)
            start_max = end_bin - int(config.min_segment_bins)
            for start_bin in range(start_min, start_max + 1):
                if segment_count == 1 and start_bin != 0:
                    continue
                if segment_count > 1 and start_bin == 0:
                    continue
                for mode in GRAMMAR_MODES:
                    segment = segment_lookup.get((start_bin, end_bin, mode))
                    if segment is None:
                        continue
                    duration_prior = _duration_log_prior(
                        end_bin - start_bin,
                        means.get(mode, DEFAULT_MODE_MEAN_DURATION_BINS[mode]),
                        config.duration_prior_log_sd,
                    )
                    local_score = float(segment["segment_log_evidence"]) + duration_prior
                    if segment_count == 1:
                        candidate_score = local_score
                        prev_mode = ""
                    else:
                        prev_candidates = [
                            (prev, best[(segment_count - 1, start_bin, prev)])
                            for prev in GRAMMAR_MODES
                            if prev != mode and (segment_count - 1, start_bin, prev) in best
                        ]
                        if not prev_candidates:
                            continue
                        prev_mode, prev_score = max(prev_candidates, key=lambda item: item[1])
                        candidate_score = prev_score + local_score - float(config.mode_switch_penalty)
                    key = (segment_count, end_bin, mode)
                    if key not in best or candidate_score > best[key]:
                        best[key] = candidate_score
                        back[key] = {
                            **segment,
                            "duration_log_prior": float(duration_prior),
                            "segment_score_with_duration_prior": float(local_score),
                            "prev_mode": prev_mode,
                            "start_bin": int(start_bin),
                            "end_bin_exclusive": int(end_bin),
                        }
    final_candidates = [
        (score, segment_count, mode)
        for (segment_count, end_bin, mode), score in best.items()
        if end_bin == n_time
    ]
    if not final_candidates:
        return []
    event_score, segment_count, mode = max(final_candidates, key=lambda item: item[0])
    segments = _reconstruct_segments(back, segment_count, n_time, mode)
    event_duration_s = float(sum(float(segment["segment_duration_s"]) for segment in segments))
    rows = []
    for index, segment in enumerate(segments):
        mode_duration = sum(float(item["segment_duration_s"]) for item in segments if item["mode"] == segment["mode"])
        rows.append(
            {
                **event_key,
                "segment_index": int(index),
                "mode": str(segment["mode"]),
                "start_bin": int(segment["start_bin"]),
                "end_bin_exclusive": int(segment["end_bin_exclusive"]),
                "segment_duration_bins": int(segment["end_bin_exclusive"]) - int(segment["start_bin"]),
                "segment_start_time_s": float(segment["segment_start_time_s"]),
                "segment_end_time_s": float(segment["segment_end_time_s"]),
                "segment_duration_s": float(segment["segment_duration_s"]),
                "segment_log_evidence": float(segment["segment_log_evidence"]),
                "duration_log_prior": float(segment["duration_log_prior"]),
                "segment_score_with_duration_prior": float(segment["segment_score_with_duration_prior"]),
                "event_grammar_score": float(event_score),
                "event_n_time": int(n_time),
                "event_n_spikes": int(segment["event_n_spikes"]),
                "event_duration_s": event_duration_s,
                "event_mode_duration_fraction": _safe_fraction(mode_duration, event_duration_s),
                "scored_model": str(segment.get("scored_model", "")),
            }
        )
    return rows


def _segment_score_lookup(group: pd.DataFrame) -> dict[tuple[int, int, str], dict[str, object]]:
    lookup: dict[tuple[int, int, str], dict[str, object]] = {}
    for _, row in group.iterrows():
        key = (int(row["start_bin"]), int(row["end_bin_exclusive"]), str(row["mode"]))
        value = float(row["segment_log_evidence"])
        if key in lookup and value <= float(lookup[key]["segment_log_evidence"]):
            continue
        lookup[key] = {
            "mode": str(row["mode"]),
            "segment_log_evidence": value,
            "segment_start_time_s": float(row.get("segment_start_time_s", row["start_bin"])),
            "segment_end_time_s": float(row.get("segment_end_time_s", row["end_bin_exclusive"])),
            "segment_duration_s": float(row.get("segment_duration_s", row["end_bin_exclusive"] - row["start_bin"])),
            "event_n_spikes": int(row.get("event_n_spikes", row.get("n_spikes", 0))),
            "scored_model": str(row.get("scored_model", "")),
        }
    return lookup


def _duration_log_prior(length_bins: int, mean_bins: float, log_sd: float) -> float:
    length = max(float(length_bins), np.finfo(float).eps)
    mean = max(float(mean_bins), np.finfo(float).eps)
    sd = max(float(log_sd), np.finfo(float).eps)
    return float(-0.5 * ((np.log(length) - np.log(mean)) / sd) ** 2)


def _reconstruct_segments(
    back: dict[tuple[int, int, str], dict[str, object]],
    segment_count: int,
    end_bin: int,
    mode: str,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    current_count = int(segment_count)
    current_end = int(end_bin)
    current_mode = str(mode)
    while current_count > 0:
        state = back[(current_count, current_end, current_mode)]
        out.append(state)
        current_end = int(state["start_bin"])
        current_mode = str(state["prev_mode"])
        current_count -= 1
    out.reverse()
    return out


def _motif_family(modes: Sequence[str]) -> str:
    if len(modes) <= 1:
        return "single_mode"
    if modes[0] == "stationary" and modes[-1] == "fragmented" and any(mode in TRAJECTORY_GRAMMAR_MODES for mode in modes):
        return "prelude_trajectory_endpoint"
    if "momentum" in modes:
        return "momentum_compositional"
    if any(mode in TRAJECTORY_GRAMMAR_MODES for mode in modes):
        return "trajectory_compositional"
    return "nontrajectory_compositional"


def _validate_grammar_config(config: GrammarInferenceConfig) -> None:
    if int(config.max_segments) <= 0:
        raise ValueError("max_segments must be positive")
    if int(config.min_segment_bins) <= 0:
        raise ValueError("min_segment_bins must be positive")
    if int(config.max_segment_bins) < 0:
        raise ValueError("max_segment_bins must be nonnegative")
    if int(config.max_segment_bins) and int(config.max_segment_bins) < int(config.min_segment_bins):
        raise ValueError("max_segment_bins must be >= min_segment_bins")
    if float(config.duration_prior_log_sd) <= 0.0:
        raise ValueError("duration_prior_log_sd must be positive")
    if float(config.mode_switch_penalty) < 0.0:
        raise ValueError("mode_switch_penalty must be nonnegative")


def _event_key_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in _EVENT_KEY_CANDIDATES if column in df]


def _key_dict(key_columns: Sequence[str], key: object) -> dict[str, object]:
    if len(key_columns) == 1:
        values = (key,)
    else:
        values = key if isinstance(key, tuple) else (key,)
    return {column: value for column, value in zip(key_columns, values, strict=False)}


def _safe_fraction(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def _empty_mode_sequence() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            *_EVENT_KEY_CANDIDATES,
            "segment_index",
            "mode",
            "start_bin",
            "end_bin_exclusive",
            "segment_duration_bins",
            "segment_start_time_s",
            "segment_end_time_s",
            "segment_duration_s",
            "segment_log_evidence",
            "duration_log_prior",
            "segment_score_with_duration_prior",
            "event_grammar_score",
            "event_n_time",
            "event_n_spikes",
            "event_duration_s",
            "event_mode_duration_fraction",
            "scored_model",
        ]
    )


def _empty_motifs() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            *_EVENT_KEY_CANDIDATES,
            "motif",
            "motif_family",
            "segment_count",
            "event_grammar_score",
            "event_duration_s",
            "event_n_time",
            "event_n_spikes",
            "dominant_mode",
            "compositional_replay",
            "has_trajectory_segment",
            "has_momentum_segment",
            "has_stationary_prelude",
            "has_fragmented_endpoint",
            "trajectory_duration_fraction",
            *[f"{mode}_duration_fraction" for mode in GRAMMAR_MODES],
        ]
    )


def _slice_emissions(emissions: LogEmissionTensor, start_bin: int, end_bin: int) -> LogEmissionTensor:
    start = int(start_bin)
    end = int(end_bin)
    counts = np.asarray(emissions.spike_counts)[start:end].copy()
    out = LogEmissionTensor(
        log_likelihood=np.asarray(emissions.log_likelihood, dtype=float)[start:end].copy(),
        spike_counts=counts,
        times=np.asarray(emissions.times, dtype=float)[start:end].copy(),
        dt=float(emissions.dt),
        cell_ids=np.asarray(emissions.cell_ids).copy(),
        n_spikes=int(counts.sum()),
        bin_durations=np.asarray(emissions.bin_durations, dtype=float)[start:end].copy()
        if emissions.bin_durations is not None
        else None,
        transition_durations=np.asarray(emissions.transition_durations, dtype=float)[start : max(start, end - 1)].copy()
        if emissions.transition_durations is not None and end - start > 1
        else None,
        metadata=dict(getattr(emissions, "metadata", {}) or {}),
    )
    return out


def _segment_times(emissions: LogEmissionTensor, start_bin: int, end_bin: int) -> tuple[float, float, float]:
    durations = np.asarray(emissions.bin_durations, dtype=float)
    times = np.asarray(emissions.times, dtype=float)
    start = int(start_bin)
    end = int(end_bin)
    start_time = float(times[start] - 0.5 * durations[start])
    end_time = float(times[end - 1] + 0.5 * durations[end - 1])
    return start_time, end_time, float(np.sum(durations[start:end]))


def _state_space_config(args: argparse.Namespace, mode: str) -> StateSpaceDecoderConfig:
    return StateSpaceDecoderConfig(
        mode=mode,
        stationary_sigma_cm=float(args.state_space_stationary_sigma_cm),
        diffusion_sigma_cm_sqrt_s=float(args.state_space_diffusion_sigma_cm_sqrt_s),
        max_step_sigma=float(args.state_space_max_step_sigma),
        imm_mode_stickiness=float(args.state_space_imm_mode_stickiness),
        momentum_sigma_cm_sqrt_s=float(args.state_space_momentum_sigma_cm_sqrt_s),
        momentum_initial_sigma_cm_sqrt_s=float(args.state_space_momentum_initial_sigma_cm_sqrt_s),
        momentum_velocity_decay=float(args.state_space_momentum_velocity_decay),
        momentum_velocity_decay_tau_s=float(args.state_space_momentum_velocity_decay_tau_s),
        momentum_candidate_top_k=int(args.state_space_momentum_candidate_top_k),
        momentum_candidate_min_k=int(args.state_space_momentum_candidate_min_k),
        momentum_candidate_max_k=int(args.state_space_momentum_candidate_max_k),
        momentum_predicted_candidate_top_k=int(args.state_space_momentum_predicted_candidate_top_k),
        momentum_candidate_source=str(args.state_space_momentum_candidate_source),
        valid_occupancy_threshold_s=float(args.state_space_valid_occupancy_threshold_s),
    )


def _grammar_models(args: argparse.Namespace) -> dict[str, SortedSpikeStateSpaceReplayModel]:
    return {
        "stationary": SortedSpikeStateSpaceReplayModel(
            mode="stationary",
            config=_state_space_config(args, "stationary"),
            name="replay-grammar-stationary",
        ),
        "diffusion": SortedSpikeStateSpaceReplayModel(
            mode="diffusion",
            config=_state_space_config(args, "diffusion"),
            name="replay-grammar-diffusion",
        ),
        "momentum": SortedSpikeStateSpaceReplayModel(
            mode="momentum-exact-sparse",
            config=_state_space_config(args, "momentum-exact-sparse"),
            name="replay-grammar-momentum",
        ),
        "fragmented": SortedSpikeStateSpaceReplayModel(
            mode="fragmented",
            config=_state_space_config(args, "fragmented"),
            name="replay-grammar-fragmented",
        ),
    }


def score_replay_grammar_segments(args: argparse.Namespace) -> pd.DataFrame:
    """Score all candidate grammar segments for one session."""

    session_dir = _session_path(args.dataset_root, args.session)
    _check_session(session_dir)
    session = load_replay_session(session_dir)
    event_ids = _events(args.events, session)
    if args.max_events is not None:
        event_ids = event_ids[: int(args.max_events)]
    encoding = fit_place_field_encoding(
        session,
        EncodingConfig(
            bin_size_cm=float(args.bin_size_cm),
            smoothing_sigma_bins=float(args.smoothing_sigma_bins),
            min_speed_cm_s=float(args.min_speed_cm_s),
            min_occupancy_s=float(args.min_occupancy_s),
            rate_floor_hz=float(args.rate_floor_hz),
        ),
    )
    emissions_cfg = EmissionConfig(
        time_bin_s=float(args.time_bin_s),
        spike_rate_scale=float(args.spike_rate_scale),
        likelihood_temperature=float(args.emission_likelihood_temperature),
        negative_binomial_overdispersion=float(args.emission_negative_binomial_overdispersion),
    )
    calibration = ReplayEmissionCalibration(
        gain_mode=str(args.replay_gain_mode),
        gain_prior_count=float(args.replay_gain_prior_count),
        max_gain=float(args.replay_gain_max_gain),
        emission_model=str(args.sorted_spike_emission_model),
        negative_binomial_dispersion=float(args.negative_binomial_dispersion),
    )
    models = _grammar_models(args)
    rows: list[dict[str, object]] = []
    for event_id in event_ids:
        event = session.ripple(int(event_id))
        emissions = build_sorted_emissions_with_replay_calibration(
            session,
            encoding,
            event,
            emissions_cfg,
            calibration=calibration,
        )
        if emissions.n_time == 0:
            continue
        rows.extend(_score_event_segments(args, session.session_id, int(event_id), emissions, encoding, models))
    return pd.DataFrame(rows)


def _score_event_segments(
    args: argparse.Namespace,
    session_id: str,
    event_id: int,
    emissions: LogEmissionTensor,
    encoding,
    models: dict[str, SortedSpikeStateSpaceReplayModel],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    max_segment_bins = int(args.max_segment_bins)
    min_segment_bins = int(args.min_segment_bins)
    for start_bin in range(emissions.n_time):
        end_limit = emissions.n_time if max_segment_bins <= 0 else min(emissions.n_time, start_bin + max_segment_bins)
        for end_bin in range(start_bin + min_segment_bins, end_limit + 1):
            segment = _slice_emissions(emissions, start_bin, end_bin)
            start_time, end_time, duration_s = _segment_times(emissions, start_bin, end_bin)
            for mode, model in models.items():
                started = time.perf_counter()
                base = {
                    "session": session_id,
                    "event_index": int(event_id),
                    "mode": mode,
                    "start_bin": int(start_bin),
                    "end_bin_exclusive": int(end_bin),
                    "segment_duration_bins": int(end_bin - start_bin),
                    "segment_start_time_s": start_time,
                    "segment_end_time_s": end_time,
                    "segment_duration_s": duration_s,
                    "event_n_time": int(emissions.n_time),
                    "event_n_spikes": int(emissions.n_spikes),
                    "n_time": int(segment.n_time),
                    "n_spikes": int(segment.n_spikes),
                    "runtime_s": 0.0,
                    **_run_settings(args),
                }
                try:
                    result = score_replay_model_compat(
                        model,
                        segment,
                        encoding.bin_centers,
                        occupancy_s=encoding.occupancy_s,
                    )
                    rows.append(
                        {
                            **base,
                            "status": "success",
                            "scored_model": str(result.model_name),
                            "segment_log_evidence": float(result.log_likelihood),
                            "runtime_s": float(time.perf_counter() - started),
                            "error": "",
                        }
                    )
                    print(
                        f"Scored grammar segment {session_id} event {event_id} bins {start_bin}:{end_bin} mode {mode}",
                        flush=True,
                    )
                except Exception as exc:
                    rows.append(
                        {
                            **base,
                            "status": "failure",
                            "scored_model": str(model.name),
                            "segment_log_evidence": np.nan,
                            "runtime_s": float(time.perf_counter() - started),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    if not args.continue_on_error:
                        raise
    return rows


def _run_settings(args: argparse.Namespace) -> dict[str, object]:
    return {
        "bin_size_cm": float(args.bin_size_cm),
        "smoothing_sigma_bins": float(args.smoothing_sigma_bins),
        "min_speed_cm_s": float(args.min_speed_cm_s),
        "min_occupancy_s": float(args.min_occupancy_s),
        "rate_floor_hz": float(args.rate_floor_hz),
        "time_bin_s": float(args.time_bin_s),
        "spike_rate_scale": float(args.spike_rate_scale),
        "emission_likelihood_temperature": float(args.emission_likelihood_temperature),
        "emission_negative_binomial_overdispersion": float(args.emission_negative_binomial_overdispersion),
        "sorted_spike_emission_model": str(args.sorted_spike_emission_model),
        "replay_gain_mode": str(args.replay_gain_mode),
        "replay_gain_prior_count": float(args.replay_gain_prior_count),
        "replay_gain_max_gain": float(args.replay_gain_max_gain),
        "negative_binomial_dispersion": float(args.negative_binomial_dispersion),
        "state_space_valid_occupancy_threshold_s": float(args.state_space_valid_occupancy_threshold_s),
        "state_space_stationary_sigma_cm": float(args.state_space_stationary_sigma_cm),
        "state_space_diffusion_sigma_cm_sqrt_s": float(args.state_space_diffusion_sigma_cm_sqrt_s),
        "state_space_max_step_sigma": float(args.state_space_max_step_sigma),
        "state_space_momentum_sigma_cm_sqrt_s": float(args.state_space_momentum_sigma_cm_sqrt_s),
        "state_space_momentum_initial_sigma_cm_sqrt_s": float(args.state_space_momentum_initial_sigma_cm_sqrt_s),
        "state_space_momentum_velocity_decay": float(args.state_space_momentum_velocity_decay),
        "state_space_momentum_velocity_decay_tau_s": float(args.state_space_momentum_velocity_decay_tau_s),
        "state_space_momentum_candidate_top_k": int(args.state_space_momentum_candidate_top_k),
        "state_space_momentum_candidate_min_k": int(args.state_space_momentum_candidate_min_k),
        "state_space_momentum_candidate_max_k": int(args.state_space_momentum_candidate_max_k),
        "state_space_momentum_predicted_candidate_top_k": int(args.state_space_momentum_predicted_candidate_top_k),
        "state_space_momentum_candidate_source": str(args.state_space_momentum_candidate_source),
        "max_segments": int(args.max_segments),
        "min_segment_bins": int(args.min_segment_bins),
        "max_segment_bins": int(args.max_segment_bins),
        "duration_prior_log_sd": float(args.duration_prior_log_sd),
        "mode_switch_penalty": float(args.mode_switch_penalty),
        "mode_mean_duration_bins": str(args.mode_mean_duration_bins),
    }


def _grammar_config_from_args(args: argparse.Namespace) -> GrammarInferenceConfig:
    return GrammarInferenceConfig(
        max_segments=int(args.max_segments),
        min_segment_bins=int(args.min_segment_bins),
        max_segment_bins=int(args.max_segment_bins),
        duration_prior_log_sd=float(args.duration_prior_log_sd),
        mode_switch_penalty=float(args.mode_switch_penalty),
        mode_mean_duration_bins=parse_mode_mean_duration_bins(args.mode_mean_duration_bins),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Infer semi-Markov replay grammar motifs within events.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--events", default="run:0-25")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--bin-size-cm", type=float, default=VALIDATED_POSITION_BIN_SIZE_CM)
    parser.add_argument("--smoothing-sigma-bins", type=float, default=VALIDATED_POSITION_SMOOTHING_SIGMA_BINS)
    parser.add_argument("--min-speed-cm-s", type=float, default=VALIDATED_POSITION_MIN_SPEED_CM_S)
    parser.add_argument("--min-occupancy-s", type=float, default=EncodingConfig().min_occupancy_s)
    parser.add_argument("--rate-floor-hz", type=float, default=EncodingConfig().rate_floor_hz)
    parser.add_argument("--time-bin-s", type=float, default=0.003)
    parser.add_argument("--spike-rate-scale", type=float, default=1.0)
    parser.add_argument("--emission-likelihood-temperature", type=float, default=1.0)
    parser.add_argument("--emission-negative-binomial-overdispersion", type=float, default=0.0)
    parser.add_argument(
        "--sorted-spike-emission-model",
        choices=("poisson", "negative-binomial", "gamma-poisson"),
        default="poisson",
    )
    parser.add_argument("--replay-gain-mode", choices=("none", "event", "cell", "event-cell"), default="none")
    parser.add_argument("--replay-gain-prior-count", type=float, default=10.0)
    parser.add_argument("--replay-gain-max-gain", type=float, default=20.0)
    parser.add_argument("--negative-binomial-dispersion", type=float, default=50.0)
    parser.add_argument("--state-space-valid-occupancy-threshold-s", type=float, default=0.0)
    parser.add_argument("--state-space-stationary-sigma-cm", type=float, default=2.0)
    parser.add_argument("--state-space-diffusion-sigma-cm-sqrt-s", type=float, default=85.0)
    parser.add_argument("--state-space-max-step-sigma", type=float, default=4.0)
    parser.add_argument("--state-space-imm-mode-stickiness", type=float, default=0.95)
    parser.add_argument("--state-space-momentum-sigma-cm-sqrt-s", type=float, default=85.0)
    parser.add_argument("--state-space-momentum-initial-sigma-cm-sqrt-s", type=float, default=85.0)
    parser.add_argument("--state-space-momentum-velocity-decay", type=float, default=0.95)
    parser.add_argument("--state-space-momentum-velocity-decay-tau-s", type=float, default=0.0)
    parser.add_argument("--state-space-momentum-candidate-top-k", type=int, default=256)
    parser.add_argument("--state-space-momentum-candidate-min-k", type=int, default=1)
    parser.add_argument("--state-space-momentum-candidate-max-k", type=int, default=0)
    parser.add_argument("--state-space-momentum-predicted-candidate-top-k", type=int, default=16)
    parser.add_argument("--state-space-momentum-candidate-source", choices=("emission", "posterior"), default="emission")
    parser.add_argument("--max-segments", type=int, default=4)
    parser.add_argument("--min-segment-bins", type=int, default=1)
    parser.add_argument("--max-segment-bins", type=int, default=0)
    parser.add_argument("--duration-prior-log-sd", type=float, default=0.75)
    parser.add_argument("--mode-switch-penalty", type=float, default=0.0)
    parser.add_argument(
        "--mode-mean-duration-bins",
        default="",
        help="Optional mode:mean_bins overrides, e.g. stationary:3,diffusion:6,momentum:6,fragmented:3.",
    )
    parser.add_argument("--output", default="results/replay-grammar")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    segment_scores = score_replay_grammar_segments(args)
    if segment_scores.empty:
        raise RuntimeError("No replay grammar segment scores were generated.")
    outputs = write_replay_grammar_outputs(
        segment_scores,
        args.output,
        config=_grammar_config_from_args(args),
    )
    print("\nReplay grammar motifs:")
    print(outputs[MOTIF_OUTPUT].to_string(index=False))
    print("\nMode duration summary:")
    print(outputs[MODE_DURATION_OUTPUT].to_string(index=False))
    print("\nRat replay grammar summary:")
    print(outputs[RAT_SUMMARY_OUTPUT].to_string(index=False))
    print(f"\nSegment score rows: {len(segment_scores)}")
    print(f"Segment score failures: {int(segment_scores['status'].ne('success').sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
