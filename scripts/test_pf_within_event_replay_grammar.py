#!/usr/bin/env python3
"""Test whether Pfeiffer/Foster replay has order-dependent within-event grammar.

Each event is partitioned into fixed, nonoverlapping local windows. The four
predeclared modes are scored in every window, and a duration-penalized
semi-Markov decoder combines those local evidences into an event-level motif.
Whole population-bin permutations preserve each observed population snapshot
while destroying its order and are processed by the identical pipeline.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import hipporeplayimm
from hipporeplayimm.data import load_replay_session
from hipporeplayimm.encoding import EmissionConfig, EncodingConfig, LogEmissionTensor, fit_place_field_encoding
from hipporeplayimm.result_improvement_extensions import (
    ReplayEmissionCalibration,
    build_sorted_emissions_with_replay_calibration,
    score_replay_model_compat,
)
try:
    from _provenance import build_script_provenance, file_sha256, git_metadata
    from benchmark_model_evidence import _check_session, _session_path
    from replay_grammar_analysis import (
        GRAMMAR_MODES,
        GrammarInferenceConfig,
        _grammar_models,
        _slice_emissions,
        infer_replay_mode_sequence,
        replay_grammar_motifs,
    )
except ModuleNotFoundError:  # Imported as scripts.* by tests.
    from scripts._provenance import build_script_provenance, file_sha256, git_metadata
    from scripts.benchmark_model_evidence import _check_session, _session_path
    from scripts.replay_grammar_analysis import (
        GRAMMAR_MODES,
        GrammarInferenceConfig,
        _grammar_models,
        _slice_emissions,
        infer_replay_mode_sequence,
        replay_grammar_motifs,
    )

LOCAL_SCORE_OUTPUT = "pf_within_event_grammar_local_scores.csv"
REPLICATE_OUTPUT = "pf_within_event_grammar_replicates.csv"
SEQUENCE_OUTPUT = "pf_within_event_grammar_sequences.csv"
DECISION_OUTPUT = "pf_within_event_grammar_decisions.csv"
SUMMARY_OUTPUT = "pf_within_event_grammar_summary.csv"
BY_RAT_OUTPUT = "pf_within_event_grammar_by_rat.csv"
GATE_OUTPUT = "pf_within_event_grammar_gate_summary.csv"
MANIFEST_OUTPUT = "pf_within_event_grammar_manifest.json"


def fixed_window_bounds(n_time: int, window_bins: int, *, minimum_bins: int = 2) -> list[tuple[int, int]]:
    """Partition time bins without leaving a one-bin dynamical window."""

    n = int(n_time)
    width = int(window_bins)
    if width <= 0:
        raise ValueError("window_bins must be positive")
    if n < int(minimum_bins):
        return []
    bounds = [(start, min(start + width, n)) for start in range(0, n, width)]
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < int(minimum_bins):
        bounds[-2] = (bounds[-2][0], bounds[-1][1])
        bounds.pop()
    return bounds


def permute_emission_bins(emissions: LogEmissionTensor, permutation: np.ndarray) -> LogEmissionTensor:
    """Return emissions with population bins reordered and timing kept fixed."""

    order = np.asarray(permutation, dtype=int)
    if order.shape != (emissions.n_time,) or set(order.tolist()) != set(range(emissions.n_time)):
        raise ValueError("permutation must contain every emission time-bin index once")
    counts = np.asarray(emissions.spike_counts)[order].copy()
    return LogEmissionTensor(
        log_likelihood=np.asarray(emissions.log_likelihood, dtype=float)[order].copy(),
        spike_counts=counts,
        times=np.asarray(emissions.times, dtype=float).copy(),
        dt=float(emissions.dt),
        cell_ids=np.asarray(emissions.cell_ids).copy(),
        n_spikes=int(counts.sum()),
        bin_durations=np.asarray(emissions.bin_durations, dtype=float).copy()
        if emissions.bin_durations is not None
        else None,
        transition_durations=np.asarray(emissions.transition_durations, dtype=float).copy()
        if emissions.transition_durations is not None
        else None,
        metadata=dict(getattr(emissions, "metadata", {}) or {}),
    )


def cumulative_segment_scores(local_scores: pd.DataFrame) -> pd.DataFrame:
    """Convert fixed-window model scores to additive semi-Markov segments."""

    if local_scores.empty:
        return pd.DataFrame()
    required = {
        "session",
        "rat",
        "event_index",
        "condition",
        "shuffle_index",
        "local_window_index",
        "local_start_time_s",
        "local_end_time_s",
        "local_duration_s",
        "mode",
        "local_log_evidence",
        "event_n_spikes",
    }
    missing = sorted(required.difference(local_scores.columns))
    if missing:
        raise ValueError(f"local scores are missing columns: {missing}")
    rows: list[dict[str, object]] = []
    keys = ["session", "rat", "event_index", "condition", "shuffle_index"]
    for key, event in local_scores.groupby(keys, dropna=False, sort=False):
        key_values = dict(zip(keys, key, strict=True))
        windows = sorted(event["local_window_index"].astype(int).unique())
        if windows != list(range(len(windows))):
            raise ValueError("local window indices must be consecutive from zero")
        n_windows = len(windows)
        for mode in GRAMMAR_MODES:
            mode_rows = event[event["mode"].astype(str).eq(mode)].sort_values("local_window_index")
            if len(mode_rows) != n_windows or mode_rows["status"].astype(str).ne("success").any():
                continue
            values = mode_rows["local_log_evidence"].astype(float).to_numpy()
            cumulative = np.concatenate(([0.0], np.cumsum(values)))
            starts = mode_rows["local_start_time_s"].astype(float).to_numpy()
            ends = mode_rows["local_end_time_s"].astype(float).to_numpy()
            durations = mode_rows["local_duration_s"].astype(float).to_numpy()
            for start in range(n_windows):
                for end in range(start + 1, n_windows + 1):
                    rows.append(
                        {
                            **key_values,
                            "status": "success",
                            "mode": mode,
                            "start_bin": start,
                            "end_bin_exclusive": end,
                            "segment_start_time_s": starts[start],
                            "segment_end_time_s": ends[end - 1],
                            "segment_duration_s": float(durations[start:end].sum()),
                            "segment_log_evidence": float(cumulative[end] - cumulative[start]),
                            "event_n_time": n_windows,
                            "event_n_spikes": int(mode_rows["event_n_spikes"].iloc[0]),
                            "scored_model": f"local-window-{mode}",
                        }
                    )
    return pd.DataFrame(rows)


def infer_local_grammar(
    local_scores: pd.DataFrame,
    *,
    max_segments: int = 4,
    duration_prior_log_sd: float = 0.75,
    mode_switch_penalty: float = 5.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Infer sequence and motif tables from fixed-window local evidence."""

    segments = cumulative_segment_scores(local_scores)
    config = GrammarInferenceConfig(
        max_segments=int(max_segments),
        min_segment_bins=1,
        max_segment_bins=0,
        duration_prior_log_sd=float(duration_prior_log_sd),
        mode_switch_penalty=float(mode_switch_penalty),
        mode_mean_duration_bins={
            "stationary": 1.5,
            "diffusion": 2.5,
            "momentum": 2.5,
            "fragmented": 1.5,
        },
    )
    sequence = infer_replay_mode_sequence(segments, config=config)
    motifs = replay_grammar_motifs(sequence)
    for column in ("rat", "condition", "shuffle_index"):
        values = local_scores[column].dropna().unique()
        if len(values) != 1:
            raise ValueError(f"local grammar input must contain one {column}")
        sequence[column] = values[0]
        motifs[column] = values[0]
    if not motifs.empty:
        motifs["ordered_trajectory_grammar"] = (
            motifs["compositional_replay"].astype(bool)
            & motifs["has_trajectory_segment"].astype(bool)
        )
    return sequence, motifs


def build_grammar_decisions(replicates: pd.DataFrame) -> pd.DataFrame:
    """Return one original-versus-shuffle decision row per event."""

    rows: list[dict[str, object]] = []
    keys = ["session", "rat", "event_index"]
    for key, group in replicates.groupby(keys, sort=True):
        original = group[group["condition"].eq("original")]
        shuffled = group[group["condition"].eq("shuffled")]
        if len(original) != 1 or shuffled.empty:
            continue
        original_row = original.iloc[0]
        null = shuffled["ordered_trajectory_grammar"].astype(float).to_numpy()
        original_value = float(bool(original_row["ordered_trajectory_grammar"]))
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "original_motif": str(original_row["motif"]),
                "original_segment_count": int(original_row["segment_count"]),
                "original_ordered_trajectory_grammar": bool(original_value),
                "shuffle_ordered_trajectory_fraction": float(np.mean(null)),
                "event_ordered_grammar_excess": float(original_value - np.mean(null)),
                "n_shuffles": int(len(null)),
            }
        )
    return pd.DataFrame(rows)


def summarize_grammar_test(replicates: pd.DataFrame, decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize the matched permutation contrast overall and by rat."""

    original = replicates[replicates["condition"].eq("original")]
    shuffled = replicates[replicates["condition"].eq("shuffled")]
    observed = float(original["ordered_trajectory_grammar"].astype(float).mean()) if len(original) else np.nan
    null_by_shuffle = (
        shuffled.groupby("shuffle_index")["ordered_trajectory_grammar"].mean().astype(float)
        if len(shuffled)
        else pd.Series(dtype=float)
    )
    null_median = float(null_by_shuffle.median()) if len(null_by_shuffle) else np.nan
    p_value = (
        float((1 + int((null_by_shuffle >= observed).sum())) / (1 + len(null_by_shuffle)))
        if np.isfinite(observed) and len(null_by_shuffle)
        else np.nan
    )
    summary = pd.DataFrame(
        [
            {
                "hypothesis": "H8_within_event_replay_grammar",
                "events": int(len(original)),
                "rats": int(original["rat"].nunique()) if len(original) else 0,
                "sessions": int(original["session"].nunique()) if len(original) else 0,
                "original_ordered_trajectory_fraction": observed,
                "median_shuffle_ordered_trajectory_fraction": null_median,
                "ordered_trajectory_fraction_excess": observed - null_median,
                "empirical_p_value_one_sided": p_value,
                "events_with_positive_excess": int((decisions["event_ordered_grammar_excess"] > 0).sum())
                if len(decisions)
                else 0,
            }
        ]
    )
    rat_rows: list[dict[str, object]] = []
    for rat, group in replicates.groupby("rat", sort=True):
        real = group[group["condition"].eq("original")]["ordered_trajectory_grammar"].astype(float)
        null = group[group["condition"].eq("shuffled")].groupby("shuffle_index")[
            "ordered_trajectory_grammar"
        ].mean()
        rat_rows.append(
            {
                "rat": rat,
                "events": int(len(real)),
                "original_ordered_trajectory_fraction": float(real.mean()),
                "median_shuffle_ordered_trajectory_fraction": float(null.median()),
                "ordered_trajectory_fraction_excess": float(real.mean() - null.median()),
            }
        )
    return summary, pd.DataFrame(rat_rows)


def build_gate_summary(
    local_scores: pd.DataFrame,
    replicates: pd.DataFrame,
    summary: pd.DataFrame,
    by_rat: pd.DataFrame,
    *,
    expected_events: int,
    n_shuffles: int,
) -> pd.DataFrame:
    """Build technical and interpretation gates without vacuous passes."""

    rows: list[dict[str, object]] = []

    def add(kind: str, gate: str, passed: bool, observed: object, criterion: str) -> None:
        rows.append({"gate_type": kind, "gate": gate, "passed": bool(passed), "observed": observed, "criterion": criterion})

    originals = replicates[replicates["condition"].eq("original")]
    complete_replicates = int(replicates.groupby(["session", "event_index"]).size().eq(n_shuffles + 1).sum()) if len(replicates) else 0
    add("technical", "selected_events_present", expected_events > 0, expected_events, ">0")
    add("technical", "all_events_have_original_motif", len(originals) == expected_events > 0, f"{len(originals)}/{expected_events}", "one original motif per selected event")
    add("technical", "all_shuffle_replicates_complete", complete_replicates == expected_events > 0, f"{complete_replicates}/{expected_events}", f"original plus {n_shuffles} shuffles per event")
    add("technical", "all_local_model_scores_successful", len(local_scores) > 0 and local_scores["status"].eq("success").all(), int(local_scores["status"].ne("success").sum()) if len(local_scores) else 0, "zero failed local model scores")
    add("technical", "all_four_rats_represented", originals["rat"].nunique() == 4 if len(originals) else False, int(originals["rat"].nunique()) if len(originals) else 0, "four rats")
    technical = all(row["passed"] for row in rows if row["gate_type"] == "technical")
    primary = summary.iloc[0] if len(summary) else pd.Series(dtype=object)
    rat_positive = bool(len(by_rat) == 4 and (by_rat["ordered_trajectory_fraction_excess"] > 0).all())
    supported = bool(
        technical
        and float(primary.get("ordered_trajectory_fraction_excess", np.nan)) > 0
        and float(primary.get("empirical_p_value_one_sided", np.nan)) <= 0.05
        and rat_positive
    )
    add("interpretation", "ordered_grammar_exceeds_whole_bin_shuffle", supported, f"excess={primary.get('ordered_trajectory_fraction_excess', np.nan)}; p={primary.get('empirical_p_value_one_sided', np.nan)}", "positive excess, empirical p<=0.05, and positive in all four rats")
    add("overall", "overall", technical and supported, f"technical={technical}; supported={supported}", "technical completion and H8 support gate")
    return pd.DataFrame(rows)


def _number(frame: pd.DataFrame, columns: tuple[str, ...], default: float) -> float:
    for column in columns:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna().unique()
        if len(values):
            return float(values[0])
    return float(default)


def _text(frame: pd.DataFrame, columns: tuple[str, ...], default: str) -> str:
    for column in columns:
        if column in frame and frame[column].notna().any():
            return str(frame[column].dropna().iloc[0])
    return str(default)


def scoring_configuration(evidence: pd.DataFrame) -> tuple[EncodingConfig, EmissionConfig, ReplayEmissionCalibration, SimpleNamespace]:
    """Recover the frozen exact-core scoring configuration from evidence rows."""

    momentum = evidence[evidence["model"].astype(str).str.contains("momentum-exact-sparse", na=False)]
    source = momentum if len(momentum) else evidence
    encoding = EncodingConfig(
        bin_size_cm=_number(evidence, ("bin_size_cm",), 6.0),
        smoothing_sigma_bins=_number(evidence, ("smoothing_sigma_bins",), 2.0),
        min_speed_cm_s=_number(evidence, ("min_speed_cm_s",), 5.0),
        min_occupancy_s=_number(evidence, ("min_occupancy_s",), 0.02),
        rate_floor_hz=_number(evidence, ("rate_floor_hz",), 1e-3),
    )
    emission = EmissionConfig(
        time_bin_s=_number(evidence, ("time_bin_s", "diagnostic_state_space_time_bin_s"), 0.004),
        spike_rate_scale=_number(evidence, ("spike_rate_scale",), 2.0),
        likelihood_temperature=_number(evidence, ("emission_likelihood_temperature",), 0.3),
        negative_binomial_overdispersion=_number(evidence, ("emission_negative_binomial_overdispersion",), 0.0),
    )
    calibration = ReplayEmissionCalibration(
        gain_mode=_text(evidence, ("replay_gain_mode",), "none"),
        gain_prior_count=_number(evidence, ("replay_gain_prior_count",), 10.0),
        max_gain=_number(evidence, ("replay_gain_max_gain",), 20.0),
        emission_model=_text(evidence, ("sorted_spike_emission_model",), "poisson"),
        negative_binomial_dispersion=_number(evidence, ("negative_binomial_dispersion",), 50.0),
    )
    model_args = SimpleNamespace(
        state_space_stationary_sigma_cm=_number(evidence, ("diagnostic_state_space_stationary_sigma_cm",), 2.0),
        state_space_diffusion_sigma_cm_sqrt_s=_number(evidence, ("diagnostic_state_space_diffusion_sigma_cm_sqrt_s",), 60.0),
        state_space_max_step_sigma=_number(evidence, ("diagnostic_state_space_max_step_sigma",), 4.0),
        state_space_imm_mode_stickiness=_number(evidence, ("state_space_imm_mode_stickiness",), 0.95),
        state_space_momentum_sigma_cm_sqrt_s=_number(source, ("diagnostic_state_space_momentum_sigma_cm_sqrt_s", "state_space_momentum_sigma_cm_sqrt_s"), 60.0),
        state_space_momentum_initial_sigma_cm_sqrt_s=_number(source, ("diagnostic_state_space_momentum_initial_sigma_cm_sqrt_s", "state_space_momentum_initial_sigma_cm_sqrt_s"), 60.0),
        state_space_momentum_velocity_decay=_number(source, ("state_space_momentum_velocity_decay", "diagnostic_state_space_momentum_velocity_decay"), 0.95),
        state_space_momentum_velocity_decay_tau_s=_number(source, ("state_space_momentum_velocity_decay_tau_s", "diagnostic_state_space_momentum_velocity_decay_tau_s"), 0.0),
        state_space_momentum_candidate_top_k=int(_number(source, ("state_space_momentum_candidate_top_k",), 256)),
        state_space_momentum_candidate_min_k=int(_number(source, ("state_space_momentum_candidate_min_k",), 1)),
        state_space_momentum_candidate_max_k=int(_number(source, ("state_space_momentum_candidate_max_k",), 0)),
        state_space_momentum_predicted_candidate_top_k=int(_number(source, ("state_space_momentum_predicted_candidate_top_k",), 16)),
        state_space_momentum_candidate_source=_text(source, ("state_space_momentum_candidate_source",), "emission"),
        state_space_valid_occupancy_threshold_s=_number(evidence, ("state_space_valid_occupancy_threshold_s", "diagnostic_state_space_valid_occupancy_threshold_s"), 0.0),
    )
    return encoding, emission, calibration, model_args


def select_balanced_events(evidence: pd.DataFrame, *, events_per_rat: int, seed: int) -> pd.DataFrame:
    """Select a deterministic pre-outcome balanced subset from the event manifest."""

    events = evidence[["session", "event_index"]].drop_duplicates().copy()
    events["rat"] = events["session"].astype(str).str.split("/", n=1).str[0]
    events["selection_hash"] = [
        hashlib.sha256(f"{seed}|{session}|{int(event)}".encode()).hexdigest()
        for session, event in zip(events["session"], events["event_index"], strict=True)
    ]
    selected = (
        events.sort_values(["rat", "selection_hash"])
        .groupby("rat", sort=True, as_index=False)
        .head(int(events_per_rat))
        .sort_values(["rat", "session", "event_index"])
        .reset_index(drop=True)
    )
    return selected


def score_local_windows(
    emissions: LogEmissionTensor,
    encoding,
    models: dict[str, object],
    *,
    session: str,
    rat: str,
    event_index: int,
    condition: str,
    shuffle_index: int,
    local_window_bins: int,
) -> pd.DataFrame:
    """Score all four models in each fixed local window."""

    rows: list[dict[str, object]] = []
    for window_index, (start, end) in enumerate(fixed_window_bounds(emissions.n_time, local_window_bins)):
        local = _slice_emissions(emissions, start, end)
        durations = np.asarray(local.bin_durations, dtype=float)
        times = np.asarray(local.times, dtype=float)
        for mode, model in models.items():
            started = time.perf_counter()
            base = {
                "session": session,
                "rat": rat,
                "event_index": int(event_index),
                "condition": condition,
                "shuffle_index": int(shuffle_index),
                "local_window_index": int(window_index),
                "local_start_bin": int(start),
                "local_end_bin_exclusive": int(end),
                "local_start_time_s": float(times[0] - 0.5 * durations[0]),
                "local_end_time_s": float(times[-1] + 0.5 * durations[-1]),
                "local_duration_s": float(durations.sum()),
                "local_n_spikes": int(local.n_spikes),
                "event_n_spikes": int(emissions.n_spikes),
                "event_n_time": int(emissions.n_time),
                "mode": mode,
            }
            try:
                score = score_replay_model_compat(
                    model,
                    local,
                    encoding.bin_centers,
                    occupancy_s=encoding.occupancy_s,
                )
                rows.append({**base, "status": "success", "local_log_evidence": float(score.log_likelihood), "runtime_s": time.perf_counter() - started, "failure_reason": ""})
            except Exception as exc:  # pragma: no cover - exercised by real-data gate
                rows.append({**base, "status": "failure", "local_log_evidence": np.nan, "runtime_s": time.perf_counter() - started, "failure_reason": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)


def _score_selected_event(
    event_record: dict[str, object],
    *,
    dataset_root: str,
    encoding_cfg: EncodingConfig,
    emission_cfg: EmissionConfig,
    calibration: ReplayEmissionCalibration,
    model_args: SimpleNamespace,
    n_shuffles: int,
    local_window_bins: int,
    max_segments: int,
    duration_prior_log_sd: float,
    mode_switch_penalty: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score one event so event jobs can run in separate processes."""

    session_id = str(event_record["session"])
    event_index = int(event_record["event_index"])
    rat = str(event_record["rat"])
    session_dir = _session_path(dataset_root, session_id)
    _check_session(session_dir)
    session = load_replay_session(session_dir)
    encoding = fit_place_field_encoding(session, encoding_cfg)
    models = _grammar_models(model_args)
    event = session.ripple(event_index)
    original = build_sorted_emissions_with_replay_calibration(
        session,
        encoding,
        event,
        emission_cfg,
        calibration=calibration,
    )
    conditions = [("original", -1, original)]
    for shuffle_index in range(int(n_shuffles)):
        event_seed = int(
            hashlib.sha256(
                f"{seed}|{session_id}|{event_index}|{shuffle_index}".encode()
            ).hexdigest()[:16],
            16,
        )
        permutation = np.random.default_rng(event_seed).permutation(original.n_time)
        conditions.append(
            ("shuffled", shuffle_index, permute_emission_bins(original, permutation))
        )
    local_parts: list[pd.DataFrame] = []
    sequence_parts: list[pd.DataFrame] = []
    motif_parts: list[pd.DataFrame] = []
    for condition, shuffle_index, emissions in conditions:
        local = score_local_windows(
            emissions,
            encoding,
            models,
            session=session_id,
            rat=rat,
            event_index=event_index,
            condition=condition,
            shuffle_index=int(shuffle_index),
            local_window_bins=int(local_window_bins),
        )
        local_parts.append(local)
        if local.empty or local["status"].ne("success").any():
            continue
        sequence, motifs = infer_local_grammar(
            local,
            max_segments=max_segments,
            duration_prior_log_sd=duration_prior_log_sd,
            mode_switch_penalty=mode_switch_penalty,
        )
        sequence_parts.append(sequence)
        motif_parts.append(motifs)
    local_scores = pd.concat(local_parts, ignore_index=True) if local_parts else pd.DataFrame()
    sequences = pd.concat(sequence_parts, ignore_index=True) if sequence_parts else pd.DataFrame()
    replicates = pd.concat(motif_parts, ignore_index=True) if motif_parts else pd.DataFrame()
    print(f"H8 completed {session_id} event {event_index}", flush=True)
    return local_scores, sequences, replicates


def run(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    """Execute the balanced real-data H8 campaign."""

    evidence = pd.read_csv(args.event_evidence)
    selected = select_balanced_events(evidence, events_per_rat=args.events_per_rat, seed=args.seed)
    encoding_cfg, emission_cfg, calibration, model_args = scoring_configuration(evidence)
    local_parts: list[pd.DataFrame] = []
    sequence_parts: list[pd.DataFrame] = []
    motif_parts: list[pd.DataFrame] = []
    task_kwargs = {
        "dataset_root": str(args.dataset_root),
        "encoding_cfg": encoding_cfg,
        "emission_cfg": emission_cfg,
        "calibration": calibration,
        "model_args": model_args,
        "n_shuffles": int(args.n_shuffles),
        "local_window_bins": int(args.local_window_bins),
        "max_segments": int(args.max_segments),
        "duration_prior_log_sd": float(args.duration_prior_log_sd),
        "mode_switch_penalty": float(args.mode_switch_penalty),
        "seed": int(args.seed),
    }
    records = selected.to_dict("records")
    if int(args.workers) <= 1:
        event_outputs = [
            _score_selected_event(record, **task_kwargs) for record in records
        ]
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=int(args.workers)
        ) as executor:
            futures = [
                executor.submit(_score_selected_event, record, **task_kwargs)
                for record in records
            ]
            event_outputs = [future.result() for future in futures]
    for local, sequence, motifs in event_outputs:
        local_parts.append(local)
        sequence_parts.append(sequence)
        motif_parts.append(motifs)
    local_scores = pd.concat(local_parts, ignore_index=True) if local_parts else pd.DataFrame()
    sequences = pd.concat(sequence_parts, ignore_index=True) if sequence_parts else pd.DataFrame()
    replicates = pd.concat(motif_parts, ignore_index=True) if motif_parts else pd.DataFrame()
    decisions = build_grammar_decisions(replicates)
    summary, by_rat = summarize_grammar_test(replicates, decisions)
    gates = build_gate_summary(
        local_scores,
        replicates,
        summary,
        by_rat,
        expected_events=len(selected),
        n_shuffles=args.n_shuffles,
    )
    return {
        LOCAL_SCORE_OUTPUT: local_scores,
        REPLICATE_OUTPUT: replicates,
        SEQUENCE_OUTPUT: sequences,
        DECISION_OUTPUT: decisions,
        SUMMARY_OUTPUT: summary,
        BY_RAT_OUTPUT: by_rat,
        GATE_OUTPUT: gates,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--event-evidence", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--events-per-rat", type=int, default=5)
    parser.add_argument("--n-shuffles", type=int, default=20)
    parser.add_argument("--local-window-bins", type=int, default=5)
    parser.add_argument("--max-segments", type=int, default=4)
    parser.add_argument("--duration-prior-log-sd", type=float, default=0.75)
    parser.add_argument("--mode-switch-penalty", type=float, default=5.5)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = run(args)
    for filename, frame in tables.items():
        frame.to_csv(output / filename, index=False)
    manifest = {
        "analysis": "H8_within_event_replay_grammar",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "event_evidence": str(Path(args.event_evidence).resolve()),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "settings": vars(args),
        "grammar_modes": list(GRAMMAR_MODES),
        "shuffle_preserves": ["population_spike_vector_per_bin", "event_duration", "event_spike_count"],
        "shuffle_destroys": "whole-bin_temporal_order",
        "primary_test": "original ordered-trajectory grammar fraction versus matched whole-bin shuffle fraction",
        "analysis_script_sha256": file_sha256(Path(__file__).resolve()),
        "scoring_package_file": str(Path(hipporeplayimm.__file__).resolve()),
        "scoring_source_git": git_metadata(
            Path(hipporeplayimm.__file__).resolve().parents[2]
        ),
        "provenance": build_script_provenance(
            input_paths={
                "dataset_root": args.dataset_root,
                "event_evidence": args.event_evidence,
            }
        ),
    }
    (output / MANIFEST_OUTPUT).write_text(json.dumps(manifest, indent=2) + "\n")
    print(tables[SUMMARY_OUTPUT].to_string(index=False))
    print(tables[GATE_OUTPUT].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
