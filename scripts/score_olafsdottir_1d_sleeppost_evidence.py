#!/usr/bin/env python3
"""Score frozen Olafsdottir SleepPOST pilot events with a small 1D evidence smoke.

This is a readiness workflow, not a biological claim generator. It uses Track1
linearized position and sorted units to build simple 1D Poisson place-field
emissions, then scores selected SleepPOST candidate events with four exact-core
shaped model rows: stationary, diffusion/Brownian, fragmented, and first-order
IMM. The output contract is intended to answer whether the Olafsdottir 1D data
interface can support replay-evidence scoring before any 1D-vs-2D comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import pandas as pd

from hipporeplayimm.olafsdottir2016 import read_axona_cut


EVENT_MODEL_OUTPUT = "olafsdottir_1d_sleep_event_model_evidence.csv"
DECISION_OUTPUT = "olafsdottir_1d_sleep_model_claim_decisions.csv"
TRAJECTORY_SUMMARY_OUTPUT = "olafsdottir_1d_sleep_trajectory_nontrajectory_summary.csv"
IMM_FRAGMENTED_OUTPUT = "olafsdottir_1d_sleep_imm_fragmented_summary.csv"
PAIR_OUTPUT = "olafsdottir_1d_sleep_by_pair_summary.csv"
ANIMAL_OUTPUT = "olafsdottir_1d_sleep_by_animal_summary.csv"
GATE_OUTPUT = "olafsdottir_1d_sleep_gate_summary.csv"
MANIFEST_OUTPUT = "olafsdottir_1d_sleep_manifest.json"
SUMMARY_OUTPUT = "olafsdottir_1d_sleep_evidence_summary.md"

STATIONARY_MODEL = "stationary"
DIFFUSION_MODEL = "diffusion"
FRAGMENTED_MODEL = "fragmented"
FIRST_ORDER_IMM_MODEL = "first_order_imm"
REQUIRED_MODELS = (STATIONARY_MODEL, DIFFUSION_MODEL, FRAGMENTED_MODEL, FIRST_ORDER_IMM_MODEL)
TRAJECTORY_MODELS = (DIFFUSION_MODEL, FRAGMENTED_MODEL, FIRST_ORDER_IMM_MODEL)

REQUIRED_PAIR_COLUMNS = {
    "animal",
    "date",
    "track_session",
    "sleepPOST_session",
    "hippocampal_tetrodes",
    "usable_pair",
}
REQUIRED_LINEARIZATION_COLUMNS = {
    "animal",
    "date",
    "track_session",
    "sleeppost_session",
    "linearization_status",
}
REQUIRED_DECODER_COLUMNS = {
    "animal",
    "date",
    "track1_session",
    "sleeppost_session",
    "decoder_status",
}
REQUIRED_PILOT_COLUMNS = {
    "selection_tier",
    "animal",
    "date",
    "track1_session",
    "sleeppost_session",
    "event_id",
    "start_time_s",
    "end_time_s",
    "duration_ms",
    "n_spikes",
    "n_active_units",
    "mean_speed_cm_s",
}

EVENT_MODEL_COLUMNS = [
    "animal",
    "date",
    "track1_session",
    "sleeppost_session",
    "pilot_tier",
    "decoder_filter",
    "event_index",
    "event_id",
    "start_time_s",
    "end_time_s",
    "duration_ms",
    "n_spikes",
    "n_active_units",
    "mean_speed_cm_s",
    "decoder_qc_passed",
    "linearization_qc_passed",
    "model",
    "model_family",
    "log_evidence",
    "status",
    "failure_reason",
    "runtime_s",
]

DECISION_COLUMNS = [
    "animal",
    "date",
    "track1_session",
    "sleeppost_session",
    "pilot_tier",
    "decoder_filter",
    "event_index",
    "event_id",
    "start_time_s",
    "end_time_s",
    "duration_ms",
    "n_spikes",
    "n_active_units",
    "mean_speed_cm_s",
    "decoder_qc_passed",
    "linearization_qc_passed",
    "best_model",
    "runner_up_model",
    "best_minus_runner_up_log_evidence",
    "logZ_stationary",
    "logZ_diffusion",
    "logZ_fragmented",
    "logZ_first_order_imm",
    "delta_best_trajectory_minus_stationary",
    "delta_imm_minus_fragmented",
    "trajectory_family_claim",
    "imm_clean_vs_fragmented_claim",
    "fragmented_claim",
    "brownian_diffusion_claim",
    "ambiguous_claim",
]


@dataclass(frozen=True)
class SessionSpikes:
    spike_times_s: np.ndarray
    unit_ids: np.ndarray
    units: tuple[int, ...]


@dataclass(frozen=True)
class PlaceFieldModel:
    unit_ids: tuple[int, ...]
    bin_centers_cm: np.ndarray
    occupancy_s: np.ndarray
    prior: np.ndarray
    rates_hz: np.ndarray


def run_sleep_evidence(
    *,
    dataset_root: str | Path,
    pairs_csv: str | Path,
    linearization_qc: str | Path,
    decoder_qc: str | Path,
    pilot_selection: str | Path,
    pilot_tier: str,
    output_dir: str | Path,
    margin_threshold: float = 5.5,
    position_bin_size_cm: float = 5.0,
    time_bin_s: float = 0.020,
    min_unit_spikes: int = 5,
    min_encoding_units: int = 1,
    smoothing_bins: int = 1,
    diffusion_sigma_cm: float = 12.5,
    stationary_self_transition: float = 0.98,
    imm_mode_persistence: float = 0.92,
) -> dict[str, pd.DataFrame]:
    pairs = load_pairs(pairs_csv)
    linearization = load_linearization_qc(linearization_qc)
    decoder = load_decoder_qc(decoder_qc)
    pilot = load_pilot_selection(pilot_selection)
    selected = pilot[pilot["selection_tier"].astype(str).eq(str(pilot_tier))].copy()
    selected = selected.sort_values(["animal", "date", "sleeppost_session", "event_id"], kind="mergesort").reset_index(drop=True)
    selected.insert(0, "event_index", np.arange(len(selected), dtype=int))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    linearization_root = Path(linearization_qc).resolve().parent
    evidence_rows: list[dict[str, object]] = []
    cache: dict[tuple[str, str, str, str], tuple[PlaceFieldModel, SessionSpikes]] = {}

    for _, event in selected.iterrows():
        event_rows = score_selected_event(
            event,
            dataset_root=Path(dataset_root),
            pairs=pairs,
            linearization=linearization,
            decoder=decoder,
            linearization_root=linearization_root,
            cache=cache,
            margin_threshold=margin_threshold,
            position_bin_size_cm=position_bin_size_cm,
            time_bin_s=time_bin_s,
            min_unit_spikes=min_unit_spikes,
            min_encoding_units=min_encoding_units,
            smoothing_bins=smoothing_bins,
            diffusion_sigma_cm=diffusion_sigma_cm,
            stationary_self_transition=stationary_self_transition,
            imm_mode_persistence=imm_mode_persistence,
        )
        evidence_rows.extend(event_rows)

    evidence = pd.DataFrame(evidence_rows, columns=EVENT_MODEL_COLUMNS)
    decisions = claim_decisions(evidence, margin_threshold=margin_threshold)
    trajectory_summary = trajectory_nontrajectory_summary(decisions, margin_threshold=margin_threshold)
    imm_summary = imm_fragmented_summary(decisions, margin_threshold=margin_threshold)
    by_pair = summarize_by_pair(decisions, evidence, margin_threshold=margin_threshold)
    by_animal = summarize_by_animal(decisions, evidence, margin_threshold=margin_threshold)

    evidence.to_csv(out / EVENT_MODEL_OUTPUT, index=False)
    decisions.to_csv(out / DECISION_OUTPUT, index=False)
    trajectory_summary.to_csv(out / TRAJECTORY_SUMMARY_OUTPUT, index=False)
    imm_summary.to_csv(out / IMM_FRAGMENTED_OUTPUT, index=False)
    by_pair.to_csv(out / PAIR_OUTPUT, index=False)
    by_animal.to_csv(out / ANIMAL_OUTPUT, index=False)

    manifest = build_manifest(
        dataset_root=dataset_root,
        pairs_csv=pairs_csv,
        linearization_qc=linearization_qc,
        decoder_qc=decoder_qc,
        pilot_selection=pilot_selection,
        pilot_tier=pilot_tier,
        selected_events=int(len(selected)),
        required_models=REQUIRED_MODELS,
        margin_threshold=margin_threshold,
        position_bin_size_cm=position_bin_size_cm,
        time_bin_s=time_bin_s,
        min_unit_spikes=min_unit_spikes,
        min_encoding_units=min_encoding_units,
        smoothing_bins=smoothing_bins,
        diffusion_sigma_cm=diffusion_sigma_cm,
        stationary_self_transition=stationary_self_transition,
        imm_mode_persistence=imm_mode_persistence,
    )
    (out / MANIFEST_OUTPUT).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    gates = gate_summary(
        selected=selected,
        decoder=decoder,
        evidence=evidence,
        decisions=decisions,
        output_dir=out,
    )
    gates.to_csv(out / GATE_OUTPUT, index=False)
    (out / SUMMARY_OUTPUT).write_text(
        build_markdown_summary(evidence, decisions, trajectory_summary, imm_summary, gates, manifest),
        encoding="utf-8",
    )
    return {
        "event_model_evidence": evidence,
        "model_claim_decisions": decisions,
        "trajectory_nontrajectory_summary": trajectory_summary,
        "imm_fragmented_summary": imm_summary,
        "by_pair_summary": by_pair,
        "by_animal_summary": by_animal,
        "gate_summary": gates,
    }


def score_selected_event(
    event: pd.Series,
    *,
    dataset_root: Path,
    pairs: pd.DataFrame,
    linearization: pd.DataFrame,
    decoder: pd.DataFrame,
    linearization_root: Path,
    cache: dict[tuple[str, str, str, str], tuple[PlaceFieldModel, SessionSpikes]],
    margin_threshold: float,
    position_bin_size_cm: float,
    time_bin_s: float,
    min_unit_spikes: int,
    min_encoding_units: int,
    smoothing_bins: int,
    diffusion_sigma_cm: float,
    stationary_self_transition: float,
    imm_mode_persistence: float,
) -> list[dict[str, object]]:
    del margin_threshold
    meta = event_metadata(event)
    pair = matching_pair(pairs, meta["animal"], meta["date"], meta["track1_session"], meta["sleeppost_session"])
    lin_pass = linearization_passed(linearization, meta["animal"], meta["date"], meta["track1_session"], meta["sleeppost_session"])
    dec_pass = decoder_ready_for_filter(
        decoder,
        meta["animal"],
        meta["date"],
        meta["track1_session"],
        meta["sleeppost_session"],
        decoder_filter=str(meta["decoder_filter"]),
    )
    meta["decoder_qc_passed"] = bool(dec_pass)
    meta["linearization_qc_passed"] = bool(lin_pass)
    if pair is None:
        return failure_rows(meta, "missing_pair_row")
    if not lin_pass:
        return failure_rows(meta, "linearization_qc_not_passed")
    if not dec_pass:
        return failure_rows(meta, "decoder_qc_not_passed")

    cache_key = (meta["animal"], meta["date"], meta["track1_session"], meta["sleeppost_session"])
    try:
        if cache_key not in cache:
            tetrodes = parse_tetrodes(str(pair["hippocampal_tetrodes"]))
            linearized = load_linearized_position(
                linearization_root=linearization_root,
                animal=meta["animal"],
                date=meta["date"],
            )
            track_spikes = load_session_spikes(
                session_stem(dataset_root, meta["animal"], meta["date"], meta["track1_session"]),
                tetrodes,
            )
            sleep_spikes = load_session_spikes(
                session_stem(dataset_root, meta["animal"], meta["date"], meta["sleeppost_session"]),
                tetrodes,
            )
            place_fields = fit_place_field_model(
                linearized=linearized,
                spikes=track_spikes,
                position_bin_size_cm=position_bin_size_cm,
                min_unit_spikes=min_unit_spikes,
                min_encoding_units=min_encoding_units,
                smoothing_bins=smoothing_bins,
            )
            cache[cache_key] = (place_fields, sleep_spikes)
        place_fields, sleep_spikes = cache[cache_key]
        counts = event_count_matrix(
            sleep_spikes,
            unit_ids=place_fields.unit_ids,
            start_s=float(meta["start_time_s"]),
            end_s=float(meta["end_time_s"]),
            time_bin_s=time_bin_s,
        )
        if counts.shape[0] == 0:
            return failure_rows(meta, "event_has_no_time_bins")
        scores = score_models(
            counts,
            place_fields,
            time_bin_s=time_bin_s,
            diffusion_sigma_cm=diffusion_sigma_cm,
            stationary_self_transition=stationary_self_transition,
            imm_mode_persistence=imm_mode_persistence,
        )
    except Exception as exc:  # noqa: BLE001 - emit explicit failure rows for readiness gates.
        return failure_rows(meta, type(exc).__name__ + ":" + str(exc))

    rows: list[dict[str, object]] = []
    for model in REQUIRED_MODELS:
        start = time.perf_counter()
        value = float(scores.get(model, np.nan))
        runtime_s = max(time.perf_counter() - start, 0.0)
        status = "success" if np.isfinite(value) else "fail"
        rows.append(
            {
                **meta,
                "model": model,
                "model_family": model_family(model),
                "log_evidence": value,
                "status": status,
                "failure_reason": "" if status == "success" else "nonfinite_log_evidence",
                "runtime_s": runtime_s,
            }
        )
    return rows


def fit_place_field_model(
    *,
    linearized: pd.DataFrame,
    spikes: SessionSpikes,
    position_bin_size_cm: float,
    min_unit_spikes: int,
    min_encoding_units: int,
    smoothing_bins: int,
) -> PlaceFieldModel:
    valid = valid_position_mask(linearized)
    times = pd.to_numeric(linearized["time_s"], errors="coerce").to_numpy(dtype=float)
    linear = pd.to_numeric(linearized["linear_position_cm"], errors="coerce").to_numpy(dtype=float)
    if np.count_nonzero(valid) < 2:
        raise ValueError("linearized_position_has_too_few_valid_samples")
    unit_ids = tuple(int(unit_id) for unit_id in spikes.units if np.count_nonzero(spikes.unit_ids == int(unit_id)) >= int(min_unit_spikes))
    if len(unit_ids) < int(min_encoding_units):
        raise ValueError(f"too_few_encoding_units:{len(unit_ids)}")
    edges = position_edges(linear[valid], position_bin_size_cm)
    centers = 0.5 * (edges[:-1] + edges[1:])
    occupancy = occupancy_seconds(linear, times, valid, edges)
    prior = normalised(occupancy + 1e-6)
    rates = np.zeros((len(unit_ids), centers.shape[0]), dtype=float)
    for unit_index, unit_id in enumerate(unit_ids):
        unit_times = spikes.spike_times_s[spikes.unit_ids == int(unit_id)]
        spike_pos = interpolate_position_at_times(unit_times, times, linear, valid)
        counts, _ = np.histogram(spike_pos[np.isfinite(spike_pos)], bins=edges)
        with np.errstate(divide="ignore", invalid="ignore"):
            unit_rates = counts / occupancy
        unit_rates[~np.isfinite(unit_rates)] = 0.0
        rates[unit_index, :] = smooth_1d(unit_rates, smoothing_bins)
    rates = np.maximum(rates, 1e-4)
    return PlaceFieldModel(unit_ids=unit_ids, bin_centers_cm=centers, occupancy_s=occupancy, prior=prior, rates_hz=rates)


def score_models(
    counts: np.ndarray,
    place_fields: PlaceFieldModel,
    *,
    time_bin_s: float,
    diffusion_sigma_cm: float,
    stationary_self_transition: float,
    imm_mode_persistence: float,
) -> dict[str, float]:
    emissions = poisson_log_emissions(counts, place_fields.rates_hz, time_bin_s)
    prior = place_fields.prior
    diffusion = diffusion_log_transition(place_fields.bin_centers_cm, diffusion_sigma_cm)
    stationary = stationary_log_transition(prior, stationary_self_transition)
    fragmented = reset_log_transition(prior)
    return {
        STATIONARY_MODEL: stationary_model_log_evidence(emissions, prior),
        DIFFUSION_MODEL: transition_model_log_evidence(emissions, prior, diffusion),
        FRAGMENTED_MODEL: fragmented_model_log_evidence(emissions, prior),
        FIRST_ORDER_IMM_MODEL: imm_log_evidence(
            emissions,
            prior,
            (stationary, diffusion, fragmented),
            mode_persistence=imm_mode_persistence,
        ),
    }


def poisson_log_emissions(counts: np.ndarray, rates_hz: np.ndarray, dt_s: float) -> np.ndarray:
    safe_rates = np.maximum(np.asarray(rates_hz, dtype=float), 1e-8)
    counts = np.asarray(counts, dtype=float)
    log_rate_dt = np.log(safe_rates * float(dt_s) + 1e-12)
    logp = counts @ log_rate_dt
    logp = logp - float(dt_s) * np.sum(safe_rates, axis=0)
    return logp


def stationary_model_log_evidence(emissions: np.ndarray, prior: np.ndarray) -> float:
    if emissions.shape[0] == 0:
        return np.nan
    return float(logsumexp(np.log(np.maximum(prior, 1e-12)) + np.sum(emissions, axis=0)))


def fragmented_model_log_evidence(emissions: np.ndarray, prior: np.ndarray) -> float:
    log_prior = np.log(np.maximum(prior, 1e-12))
    return float(sum(logsumexp(log_prior + emissions[t]) for t in range(emissions.shape[0])))


def transition_model_log_evidence(emissions: np.ndarray, prior: np.ndarray, log_transition: np.ndarray) -> float:
    if emissions.shape[0] == 0:
        return np.nan
    alpha = np.log(np.maximum(prior, 1e-12)) + emissions[0]
    for t in range(1, emissions.shape[0]):
        alpha = emissions[t] + logsumexp_matrix(alpha[:, None] + log_transition, axis=0)
    return float(logsumexp(alpha))


def imm_log_evidence(
    emissions: np.ndarray,
    prior: np.ndarray,
    log_transitions: Sequence[np.ndarray],
    *,
    mode_persistence: float,
) -> float:
    if emissions.shape[0] == 0:
        return np.nan
    n_modes = len(log_transitions)
    mode_transition = mode_log_transition(n_modes, mode_persistence)
    mode_prior = np.full(n_modes, 1.0 / n_modes, dtype=float)
    alpha = np.log(mode_prior[:, None]) + np.log(np.maximum(prior[None, :], 1e-12)) + emissions[0][None, :]
    for t in range(1, emissions.shape[0]):
        next_alpha = np.full_like(alpha, -np.inf)
        for dest_mode, transition in enumerate(log_transitions):
            incoming = alpha + mode_transition[:, dest_mode][:, None]
            collapsed = logsumexp_matrix(incoming, axis=0)
            next_alpha[dest_mode, :] = emissions[t] + logsumexp_matrix(collapsed[:, None] + transition, axis=0)
        alpha = next_alpha
    return float(logsumexp(alpha.ravel()))


def diffusion_log_transition(centers_cm: np.ndarray, sigma_cm: float) -> np.ndarray:
    centers = np.asarray(centers_cm, dtype=float)
    distances = centers[None, :] - centers[:, None]
    sigma = max(float(sigma_cm), 1e-6)
    logp = -0.5 * (distances / sigma) ** 2
    return row_log_normalise(logp)


def stationary_log_transition(prior: np.ndarray, self_transition: float) -> np.ndarray:
    n_bins = len(prior)
    p_self = min(max(float(self_transition), 0.0), 0.999999)
    reset = (1.0 - p_self) * np.maximum(prior, 1e-12)
    matrix = np.tile(reset[None, :], (n_bins, 1))
    for idx in range(n_bins):
        matrix[idx, idx] += p_self
    return np.log(np.maximum(matrix, 1e-300))


def reset_log_transition(prior: np.ndarray) -> np.ndarray:
    n_bins = len(prior)
    matrix = np.tile(np.maximum(prior, 1e-12)[None, :], (n_bins, 1))
    return np.log(matrix)


def mode_log_transition(n_modes: int, persistence: float) -> np.ndarray:
    p_stay = min(max(float(persistence), 0.0), 0.999999)
    off = (1.0 - p_stay) / max(n_modes - 1, 1)
    matrix = np.full((n_modes, n_modes), off, dtype=float)
    np.fill_diagonal(matrix, p_stay)
    return np.log(np.maximum(matrix, 1e-300))


def row_log_normalise(logp: np.ndarray) -> np.ndarray:
    norm = logsumexp_matrix(logp, axis=1)
    return logp - norm[:, None]


def event_count_matrix(
    spikes: SessionSpikes,
    *,
    unit_ids: Sequence[int],
    start_s: float,
    end_s: float,
    time_bin_s: float,
) -> np.ndarray:
    if not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
        return np.zeros((0, len(unit_ids)), dtype=float)
    edges = np.arange(float(start_s), float(end_s) + float(time_bin_s), float(time_bin_s))
    if edges.size < 2:
        edges = np.asarray([float(start_s), float(end_s)], dtype=float)
    counts = np.zeros((edges.size - 1, len(unit_ids)), dtype=float)
    for unit_index, unit_id in enumerate(unit_ids):
        unit_times = spikes.spike_times_s[spikes.unit_ids == int(unit_id)]
        counts[:, unit_index], _ = np.histogram(unit_times, bins=edges)
    return counts


def load_session_spikes(session_stem: Path, tetrodes: Sequence[int]) -> SessionSpikes:
    spike_times: list[float] = []
    unit_ids: list[int] = []
    units: set[int] = set()
    for tetrode in tetrodes:
        raw_path = session_stem.with_suffix(f".{int(tetrode)}")
        cut_path = session_stem.parent / f"{session_stem.name}_{int(tetrode)}.cut"
        if not raw_path.is_file() or not cut_path.is_file():
            continue
        cut = read_axona_cut(cut_path, tetrode_path=raw_path)
        if cut.spike_times_s is None:
            continue
        labels = np.asarray(cut.labels, dtype=int)
        times = np.asarray(cut.spike_times_s, dtype=float)
        keep = labels > 0
        ids = np.asarray([int(tetrode) * 100 + int(label) for label in labels[keep]], dtype=int)
        spike_times.extend(times[keep].tolist())
        unit_ids.extend(ids.tolist())
        units.update(ids.tolist())
    if not spike_times:
        return SessionSpikes(spike_times_s=np.empty(0, dtype=float), unit_ids=np.empty(0, dtype=int), units=())
    order = np.argsort(np.asarray(spike_times, dtype=float))
    return SessionSpikes(
        spike_times_s=np.asarray(spike_times, dtype=float)[order],
        unit_ids=np.asarray(unit_ids, dtype=int)[order],
        units=tuple(sorted(units)),
    )


def claim_decisions(evidence: pd.DataFrame, *, margin_threshold: float) -> pd.DataFrame:
    if evidence.empty:
        return pd.DataFrame(columns=DECISION_COLUMNS)
    rows: list[dict[str, object]] = []
    group_cols = ["animal", "date", "track1_session", "sleeppost_session", "pilot_tier", "event_index", "event_id"]
    for _, group in evidence.groupby(group_cols, sort=True, dropna=False):
        base = event_base_from_group(group)
        by_model = {str(row.model): float(row.log_evidence) for row in group.itertuples(index=False) if str(row.status) == "success"}
        logz = {model: by_model.get(model, np.nan) for model in REQUIRED_MODELS}
        finite_items = [(model, value) for model, value in logz.items() if np.isfinite(value)]
        finite_items.sort(key=lambda item: item[1], reverse=True)
        best_model = finite_items[0][0] if finite_items else ""
        runner_up = finite_items[1][0] if len(finite_items) > 1 else ""
        best_margin = finite_items[0][1] - finite_items[1][1] if len(finite_items) > 1 else np.nan
        best_traj = max((logz[model] for model in TRAJECTORY_MODELS if np.isfinite(logz[model])), default=np.nan)
        delta_traj_stationary = best_traj - logz[STATIONARY_MODEL] if np.isfinite(best_traj) and np.isfinite(logz[STATIONARY_MODEL]) else np.nan
        delta_imm_frag = logz[FIRST_ORDER_IMM_MODEL] - logz[FRAGMENTED_MODEL] if np.isfinite(logz[FIRST_ORDER_IMM_MODEL]) and np.isfinite(logz[FRAGMENTED_MODEL]) else np.nan
        if np.isfinite(delta_traj_stationary) and delta_traj_stationary >= margin_threshold:
            family_claim = "trajectory_confident"
        elif np.isfinite(delta_traj_stationary) and delta_traj_stationary <= -margin_threshold:
            family_claim = "nontrajectory_confident"
        else:
            family_claim = "ambiguous"
        fragmented_conf = bool(
            best_model == FRAGMENTED_MODEL
            and np.isfinite(best_margin)
            and best_margin >= float(margin_threshold)
        )
        diffusion_conf = bool(
            best_model == DIFFUSION_MODEL
            and np.isfinite(best_margin)
            and best_margin >= float(margin_threshold)
        )
        imm_conf = bool(np.isfinite(delta_imm_frag) and delta_imm_frag >= float(margin_threshold))
        ambiguous = bool(family_claim == "ambiguous" and not imm_conf and not fragmented_conf and not diffusion_conf)
        rows.append(
            {
                **base,
                "best_model": best_model,
                "runner_up_model": runner_up,
                "best_minus_runner_up_log_evidence": best_margin,
                "logZ_stationary": logz[STATIONARY_MODEL],
                "logZ_diffusion": logz[DIFFUSION_MODEL],
                "logZ_fragmented": logz[FRAGMENTED_MODEL],
                "logZ_first_order_imm": logz[FIRST_ORDER_IMM_MODEL],
                "delta_best_trajectory_minus_stationary": delta_traj_stationary,
                "delta_imm_minus_fragmented": delta_imm_frag,
                "trajectory_family_claim": family_claim,
                "imm_clean_vs_fragmented_claim": imm_conf,
                "fragmented_claim": fragmented_conf,
                "brownian_diffusion_claim": diffusion_conf,
                "ambiguous_claim": ambiguous,
            }
        )
    return pd.DataFrame(rows, columns=DECISION_COLUMNS)


def trajectory_nontrajectory_summary(decisions: pd.DataFrame, *, margin_threshold: float) -> pd.DataFrame:
    columns = [
        "scope",
        "events",
        "trajectory_confident_claims",
        "nontrajectory_confident_claims",
        "ambiguous_events",
        "mean_delta_best_trajectory_minus_stationary",
        "median_delta_best_trajectory_minus_stationary",
        "min_delta_best_trajectory_minus_stationary",
        "max_delta_best_trajectory_minus_stationary",
        "margin_threshold",
        "biological_claim_assessed",
    ]
    if decisions.empty:
        return pd.DataFrame([empty_summary_row(columns, scope="overall", margin_threshold=margin_threshold)])
    values = pd.to_numeric(decisions["delta_best_trajectory_minus_stationary"], errors="coerce")
    row = {
        "scope": "overall",
        "events": int(len(decisions)),
        "trajectory_confident_claims": int(decisions["trajectory_family_claim"].astype(str).eq("trajectory_confident").sum()),
        "nontrajectory_confident_claims": int(decisions["trajectory_family_claim"].astype(str).eq("nontrajectory_confident").sum()),
        "ambiguous_events": int(decisions["trajectory_family_claim"].astype(str).eq("ambiguous").sum()),
        "mean_delta_best_trajectory_minus_stationary": finite_mean(values),
        "median_delta_best_trajectory_minus_stationary": finite_median(values),
        "min_delta_best_trajectory_minus_stationary": finite_min(values),
        "max_delta_best_trajectory_minus_stationary": finite_max(values),
        "margin_threshold": float(margin_threshold),
        "biological_claim_assessed": False,
    }
    return pd.DataFrame([row], columns=columns)


def imm_fragmented_summary(decisions: pd.DataFrame, *, margin_threshold: float) -> pd.DataFrame:
    columns = [
        "scope",
        "events",
        "imm_raw_wins",
        "fragmented_raw_wins",
        "imm_confident_wins",
        "fragmented_confident_wins",
        "ambiguous_events",
        "mean_delta_imm_minus_fragmented",
        "median_delta_imm_minus_fragmented",
        "min_delta_imm_minus_fragmented",
        "max_delta_imm_minus_fragmented",
        "margin_threshold",
        "biological_claim_assessed",
    ]
    if decisions.empty:
        return pd.DataFrame([empty_summary_row(columns, scope="overall", margin_threshold=margin_threshold)])
    delta = pd.to_numeric(decisions["delta_imm_minus_fragmented"], errors="coerce")
    row = {
        "scope": "overall",
        "events": int(len(decisions)),
        "imm_raw_wins": int((delta > 0.0).sum()),
        "fragmented_raw_wins": int((delta < 0.0).sum()),
        "imm_confident_wins": int((delta >= float(margin_threshold)).sum()),
        "fragmented_confident_wins": int((delta <= -float(margin_threshold)).sum()),
        "ambiguous_events": int((np.abs(delta) < float(margin_threshold)).sum()),
        "mean_delta_imm_minus_fragmented": finite_mean(delta),
        "median_delta_imm_minus_fragmented": finite_median(delta),
        "min_delta_imm_minus_fragmented": finite_min(delta),
        "max_delta_imm_minus_fragmented": finite_max(delta),
        "margin_threshold": float(margin_threshold),
        "biological_claim_assessed": False,
    }
    return pd.DataFrame([row], columns=columns)


def summarize_by_pair(decisions: pd.DataFrame, evidence: pd.DataFrame, *, margin_threshold: float) -> pd.DataFrame:
    del margin_threshold
    columns = [
        "animal",
        "date",
        "track1_session",
        "sleeppost_session",
        "selected_events",
        "successful_events",
        "model_rows",
        "failed_model_rows",
        "trajectory_confident_claims",
        "imm_clean_vs_fragmented_claims",
        "median_delta_best_trajectory_minus_stationary",
        "median_delta_imm_minus_fragmented",
    ]
    if decisions.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for keys, group in decisions.groupby(["animal", "date", "track1_session", "sleeppost_session"], sort=True):
        animal, date, track, sleep = keys
        egroup = matching_evidence_group(evidence, animal, date, track, sleep)
        status = egroup.groupby(["event_index"])["status"].apply(lambda values: values.astype(str).eq("success").all()) if not egroup.empty else pd.Series(dtype=bool)
        rows.append(
            {
                "animal": animal,
                "date": date,
                "track1_session": track,
                "sleeppost_session": sleep,
                "selected_events": int(len(group)),
                "successful_events": int(status.sum()),
                "model_rows": int(len(egroup)),
                "failed_model_rows": int((~egroup["status"].astype(str).eq("success")).sum()) if not egroup.empty else 0,
                "trajectory_confident_claims": int(group["trajectory_family_claim"].astype(str).eq("trajectory_confident").sum()),
                "imm_clean_vs_fragmented_claims": int(group["imm_clean_vs_fragmented_claim"].map(as_bool).sum()),
                "median_delta_best_trajectory_minus_stationary": finite_median(group["delta_best_trajectory_minus_stationary"]),
                "median_delta_imm_minus_fragmented": finite_median(group["delta_imm_minus_fragmented"]),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def summarize_by_animal(decisions: pd.DataFrame, evidence: pd.DataFrame, *, margin_threshold: float) -> pd.DataFrame:
    del margin_threshold
    columns = [
        "animal",
        "pairs",
        "selected_events",
        "successful_events",
        "model_rows",
        "failed_model_rows",
        "trajectory_confident_claims",
        "imm_clean_vs_fragmented_claims",
        "median_delta_best_trajectory_minus_stationary",
        "median_delta_imm_minus_fragmented",
    ]
    if decisions.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for animal, group in decisions.groupby("animal", sort=True):
        egroup = evidence[evidence["animal"].astype(str).eq(str(animal))] if not evidence.empty else pd.DataFrame(columns=EVENT_MODEL_COLUMNS)
        status = egroup.groupby(["date", "track1_session", "sleeppost_session", "event_index"])["status"].apply(lambda values: values.astype(str).eq("success").all()) if not egroup.empty else pd.Series(dtype=bool)
        rows.append(
            {
                "animal": animal,
                "pairs": int(group[["date", "track1_session", "sleeppost_session"]].drop_duplicates().shape[0]),
                "selected_events": int(len(group)),
                "successful_events": int(status.sum()),
                "model_rows": int(len(egroup)),
                "failed_model_rows": int((~egroup["status"].astype(str).eq("success")).sum()) if not egroup.empty else 0,
                "trajectory_confident_claims": int(group["trajectory_family_claim"].astype(str).eq("trajectory_confident").sum()),
                "imm_clean_vs_fragmented_claims": int(group["imm_clean_vs_fragmented_claim"].map(as_bool).sum()),
                "median_delta_best_trajectory_minus_stationary": finite_median(group["delta_best_trajectory_minus_stationary"]),
                "median_delta_imm_minus_fragmented": finite_median(group["delta_imm_minus_fragmented"]),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def gate_summary(*, selected: pd.DataFrame, decoder: pd.DataFrame, evidence: pd.DataFrame, decisions: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    selected_keys = event_key_set(selected)
    evidence_keys = event_key_set(evidence)
    required_rows_per_event = evidence.groupby(["animal", "date", "track1_session", "sleeppost_session", "event_index"], dropna=False)["model"].apply(lambda models: set(models.astype(str))) if not evidence.empty else pd.Series(dtype=object)
    selected_events_nonempty = bool(selected_keys)
    complete_required_model_events = sum(set(REQUIRED_MODELS).issubset(models) for models in required_rows_per_event)
    decoder_filter = selected_decoder_filter(selected)
    decoder_pass = decoder_ready_rows(decoder, decoder_filter=decoder_filter)
    selected_pairs = pair_key_set(selected)
    decoder_pass_pairs = pair_key_set_from_decoder(decoder_pass)
    models_present = set(evidence["model"].astype(str)) if not evidence.empty else set()
    runtime = pd.to_numeric(evidence["runtime_s"], errors="coerce") if "runtime_s" in evidence else pd.Series(dtype=float)
    gates = [
        gate_row(
            "selected_events_present",
            bool(selected_keys) and selected_keys == evidence_keys,
            f"selected_events={len(selected_keys)}; evidence_events={len(evidence_keys)}",
            "all selected pilot events have evidence rows",
        ),
        gate_row(
            "required_models_complete",
            selected_events_nonempty
            and len(required_rows_per_event) == len(selected_keys)
            and all(set(REQUIRED_MODELS).issubset(models) for models in required_rows_per_event),
            f"complete_events={complete_required_model_events}/{len(selected_keys)}",
            "every selected event has stationary, diffusion, fragmented, and first_order_imm rows",
        ),
        gate_row(
            "no_model_scoring_failures",
            not evidence.empty and evidence["status"].astype(str).eq("success").all(),
            f"failed_model_rows={int((~evidence['status'].astype(str).eq('success')).sum()) if not evidence.empty else 0}",
            "no model rows failed during smoke scoring",
        ),
        gate_row(
            "pairs_represented",
            bool(selected_pairs) and selected_pairs == decoder_pass_pairs,
            f"selected_pairs={len(selected_pairs)}; decoder_pass_pairs={len(decoder_pass_pairs)}",
            f"all {decoder_filter} decoder-ready pairs are represented by the selected pilot tier",
        ),
        gate_row(
            "animals_represented",
            int(evidence["animal"].nunique()) >= 2 if not evidence.empty else False,
            f"animals={int(evidence['animal'].nunique()) if not evidence.empty else 0}",
            "multiple animals are represented",
        ),
        gate_row(
            "stationary_comparator_present",
            selected_events_nonempty and STATIONARY_MODEL in models_present,
            f"models={','.join(sorted(models_present))}",
            "stationary comparator rows are present",
        ),
        gate_row(
            "fragmented_comparator_present",
            selected_events_nonempty and FRAGMENTED_MODEL in models_present,
            f"models={','.join(sorted(models_present))}",
            "fragmented comparator rows are present",
        ),
        gate_row(
            "imm_fragmented_axis_present",
            not decisions.empty and decisions["delta_imm_minus_fragmented"].notna().all(),
            f"finite_axis={int(pd.to_numeric(decisions['delta_imm_minus_fragmented'], errors='coerce').notna().sum()) if not decisions.empty else 0}/{len(decisions)}",
            "IMM-vs-fragmented contrast is populated for every event",
        ),
        gate_row(
            "runtime_recorded",
            not runtime.empty and runtime.notna().all() and np.isfinite(runtime).all() and (runtime >= 0.0).all(),
            f"runtime_rows={int(runtime.notna().sum())}/{len(runtime)}",
            "runtime_s is recorded for every model row",
        ),
        gate_row(
            "manifest_written",
            (output_dir / MANIFEST_OUTPUT).is_file(),
            str(output_dir / MANIFEST_OUTPUT),
            "manifest records exact inputs and smoke parameters",
        ),
    ]
    overall = all(row["passed"] for row in gates)
    gates.append(gate_row("overall", overall, f"passed={sum(row['passed'] for row in gates)}/{len(gates)}", "all readiness gates pass"))
    return pd.DataFrame(gates)


def build_manifest(**kwargs: object) -> dict[str, object]:
    return {
        "analysis": "olafsdottir_1d_sleeppost_evidence_smoke",
        "biological_claim_assessed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": os.environ.get("GITHUB_SHA") or os.environ.get("CI_COMMIT_SHA") or "unknown_without_git",
        **{key: serialisable(value) for key, value in kwargs.items()},
    }


def build_markdown_summary(
    evidence: pd.DataFrame,
    decisions: pd.DataFrame,
    trajectory_summary: pd.DataFrame,
    imm_summary: pd.DataFrame,
    gates: pd.DataFrame,
    manifest: dict[str, object],
) -> str:
    successful_rows = int(evidence["status"].astype(str).eq("success").sum()) if not evidence.empty else 0
    failed_rows = int((~evidence["status"].astype(str).eq("success")).sum()) if not evidence.empty else 0
    lines = [
        "# Olafsdottir 1D SleepPOST Evidence Smoke Summary",
        "",
        "This is a scoring/readiness smoke only. It does not make a biological 1D-vs-2D or trajectory-family claim.",
        "",
        "## Overview",
        "",
        markdown_table(
            ["Metric", "Value"],
            [
                ("pilot tier", manifest.get("pilot_tier", "")),
                ("selected events", manifest.get("selected_events", 0)),
                ("event/model rows", len(evidence)),
                ("successful model rows", successful_rows),
                ("failed model rows", failed_rows),
                ("events with claim decisions", len(decisions)),
                ("models", " ".join(REQUIRED_MODELS)),
            ],
        ),
        "",
        "## Readiness Gates",
        "",
        markdown_table(["Gate", "Status", "Value"], gates[["gate", "status", "value"]].itertuples(index=False, name=None)),
        "",
        "## Trajectory-vs-Stationary Smoke Summary",
        "",
        markdown_table(list(trajectory_summary.columns), trajectory_summary.itertuples(index=False, name=None)),
        "",
        "## IMM-vs-Fragmented Smoke Summary",
        "",
        markdown_table(list(imm_summary.columns), imm_summary.itertuples(index=False, name=None)),
        "",
    ]
    return "\n".join(lines)


def load_pairs(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_PAIR_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"pairs CSV is missing required columns: {missing}")
    frame = frame.copy()
    frame["animal"] = frame["animal"].astype(str).str.upper()
    frame["date"] = frame["date"].astype(str)
    return frame


def load_linearization_qc(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_LINEARIZATION_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"linearization QC is missing required columns: {missing}")
    frame = frame.copy()
    frame["animal"] = frame["animal"].astype(str).str.upper()
    frame["date"] = frame["date"].astype(str)
    return frame


def load_decoder_qc(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_DECODER_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"decoder QC is missing required columns: {missing}")
    frame = frame.copy()
    frame["animal"] = frame["animal"].astype(str).str.upper()
    frame["date"] = frame["date"].astype(str)
    if "decoder_qc_paper_ready" not in frame.columns:
        frame["decoder_qc_paper_ready"] = frame["decoder_status"].map(as_bool)
    return frame


def load_pilot_selection(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_PILOT_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"pilot selection CSV is missing required columns: {missing}")
    frame = frame.copy()
    frame["animal"] = frame["animal"].astype(str).str.upper()
    frame["date"] = frame["date"].astype(str)
    if "decoder_filter" not in frame.columns:
        frame["decoder_filter"] = [decoder_filter_from_tier(tier) for tier in frame["selection_tier"]]
    else:
        frame["decoder_filter"] = [normalize_decoder_filter(value, fallback_tier=tier) for value, tier in zip(frame["decoder_filter"], frame["selection_tier"])]
    return frame


def event_metadata(event: pd.Series) -> dict[str, object]:
    return {
        "animal": str(event["animal"]).upper(),
        "date": str(event["date"]),
        "track1_session": str(event["track1_session"]),
        "sleeppost_session": str(event["sleeppost_session"]),
        "pilot_tier": str(event["selection_tier"]),
        "decoder_filter": normalize_decoder_filter(event.get("decoder_filter", ""), fallback_tier=event["selection_tier"]),
        "event_index": int(event["event_index"]),
        "event_id": int(event["event_id"]),
        "start_time_s": float(event["start_time_s"]),
        "end_time_s": float(event["end_time_s"]),
        "duration_ms": float(event["duration_ms"]),
        "n_spikes": int(event["n_spikes"]),
        "n_active_units": int(event["n_active_units"]),
        "mean_speed_cm_s": float(event["mean_speed_cm_s"]),
        "decoder_qc_passed": False,
        "linearization_qc_passed": False,
    }


def failure_rows(meta: dict[str, object], reason: str) -> list[dict[str, object]]:
    return [
        {
            **meta,
            "model": model,
            "model_family": model_family(model),
            "log_evidence": np.nan,
            "status": "fail",
            "failure_reason": reason,
            "runtime_s": 0.0,
        }
        for model in REQUIRED_MODELS
    ]


def matching_pair(pairs: pd.DataFrame, animal: str, date: str, track: str, sleep: str) -> pd.Series | None:
    rows = pairs[
        pairs["animal"].astype(str).str.upper().eq(str(animal).upper())
        & pairs["date"].astype(str).eq(str(date))
        & pairs["track_session"].astype(str).eq(str(track))
        & pairs["sleepPOST_session"].astype(str).eq(str(sleep))
        & pairs["usable_pair"].map(as_bool)
    ]
    return rows.iloc[0] if not rows.empty else None


def linearization_passed(linearization: pd.DataFrame, animal: str, date: str, track: str, sleep: str) -> bool:
    rows = linearization[
        linearization["animal"].astype(str).str.upper().eq(str(animal).upper())
        & linearization["date"].astype(str).eq(str(date))
        & linearization["track_session"].astype(str).eq(str(track))
        & linearization["sleeppost_session"].astype(str).eq(str(sleep))
    ]
    return bool(not rows.empty and rows.iloc[0]["linearization_status"] == "pass")


def decoder_passed(decoder: pd.DataFrame, animal: str, date: str, track: str, sleep: str) -> bool:
    return decoder_ready_for_filter(decoder, animal, date, track, sleep, decoder_filter="paper_ready")


def decoder_ready_for_filter(decoder: pd.DataFrame, animal: str, date: str, track: str, sleep: str, *, decoder_filter: str) -> bool:
    rows = decoder[
        decoder["animal"].astype(str).str.upper().eq(str(animal).upper())
        & decoder["date"].astype(str).eq(str(date))
        & decoder["track1_session"].astype(str).eq(str(track))
        & decoder["sleeppost_session"].astype(str).eq(str(sleep))
    ]
    if rows.empty:
        return False
    row = rows.iloc[0]
    if decoder_filter == "scoring_available":
        return bool("decoder_qc_scoring_available" in rows.columns and as_bool(row["decoder_qc_scoring_available"]))
    return bool(as_bool(row["decoder_qc_paper_ready"]) if "decoder_qc_paper_ready" in rows.columns else as_bool(row["decoder_status"]))


def decoder_ready_rows(decoder: pd.DataFrame, *, decoder_filter: str) -> pd.DataFrame:
    if decoder.empty:
        return decoder
    if decoder_filter == "scoring_available":
        if "decoder_qc_scoring_available" not in decoder.columns:
            return decoder.iloc[0:0].copy()
        return decoder[decoder["decoder_qc_scoring_available"].map(as_bool)].copy()
    if "decoder_qc_paper_ready" in decoder.columns:
        return decoder[decoder["decoder_qc_paper_ready"].map(as_bool)].copy()
    return decoder[decoder["decoder_status"].map(as_bool)].copy()


def selected_decoder_filter(selected: pd.DataFrame) -> str:
    if selected.empty:
        return "paper_ready"
    if "decoder_filter" in selected.columns:
        values = {normalize_decoder_filter(value, fallback_tier="") for value in selected["decoder_filter"]}
        values.discard("")
        if len(values) == 1:
            return values.pop()
    tiers = {decoder_filter_from_tier(tier) for tier in selected["selection_tier"]} if "selection_tier" in selected.columns else {"paper_ready"}
    tiers.discard("")
    return tiers.pop() if len(tiers) == 1 else "paper_ready"


def normalize_decoder_filter(value: object, *, fallback_tier: object) -> str:
    text = str(value).strip().lower()
    if text in {"paper_ready", "scoring_available"}:
        return text
    return decoder_filter_from_tier(fallback_tier)


def decoder_filter_from_tier(tier: object) -> str:
    text = str(tier).strip().lower()
    if "decoder_available_debug" in text:
        return "scoring_available"
    return "paper_ready"


def load_linearized_position(*, linearization_root: Path, animal: str, date: str) -> pd.DataFrame:
    path = linearization_root / "sessions" / str(animal).upper() / str(date) / "linearized_position.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing linearized_position.csv: {path}")
    return pd.read_csv(path)


def session_stem(dataset_root: Path, animal: str, date: str, session: str) -> Path:
    return dataset_root / str(animal).lower() / str(date) / str(session)


def valid_position_mask(linearized: pd.DataFrame) -> np.ndarray:
    valid = linearized["valid_position"].map(as_bool).to_numpy(dtype=bool) if "valid_position" in linearized else np.zeros(len(linearized), dtype=bool)
    times = pd.to_numeric(linearized.get("time_s", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    linear = pd.to_numeric(linearized.get("linear_position_cm", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    return valid & np.isfinite(times) & np.isfinite(linear)


def position_edges(values: np.ndarray, bin_size_cm: float) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.asarray([0.0, float(bin_size_cm)], dtype=float)
    lo = 0.0
    hi = max(float(np.nanmax(finite)), float(bin_size_cm))
    return np.arange(lo, hi + float(bin_size_cm), float(bin_size_cm))


def occupancy_seconds(linear: np.ndarray, times: np.ndarray, valid: np.ndarray, edges: np.ndarray) -> np.ndarray:
    dt = sample_durations(times)
    keep = np.asarray(valid, dtype=bool) & np.isfinite(linear) & np.isfinite(dt)
    bins = np.searchsorted(edges, linear[keep], side="right") - 1
    bins = np.clip(bins, 0, edges.shape[0] - 2)
    occupancy = np.zeros(edges.shape[0] - 1, dtype=float)
    np.add.at(occupancy, bins, dt[keep])
    return occupancy


def sample_durations(times: np.ndarray) -> np.ndarray:
    arr = np.asarray(times, dtype=float)
    if arr.size == 0:
        return arr
    if arr.size == 1:
        return np.asarray([0.0], dtype=float)
    diffs = np.diff(arr)
    positive = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    default = float(np.nanmedian(positive)) if positive.size else 0.0
    dt = np.diff(arr, append=arr[-1] + default)
    dt[~np.isfinite(dt) | (dt <= 0.0)] = default
    return dt


def interpolate_position_at_times(spike_times: np.ndarray, times: np.ndarray, linear: np.ndarray, valid: np.ndarray) -> np.ndarray:
    keep = np.asarray(valid, dtype=bool) & np.isfinite(times) & np.isfinite(linear)
    if np.count_nonzero(keep) < 2 or spike_times.size == 0:
        return np.full(spike_times.shape, np.nan, dtype=float)
    return np.interp(spike_times, times[keep], linear[keep], left=np.nan, right=np.nan)


def smooth_1d(values: np.ndarray, smoothing_bins: int) -> np.ndarray:
    window = max(int(smoothing_bins), 1)
    arr = np.asarray(values, dtype=float)
    if window <= 1:
        return arr
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(arr, kernel, mode="same")


def normalised(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    total = float(np.nansum(arr))
    if not np.isfinite(total) or total <= 0.0:
        return np.full(arr.shape, 1.0 / max(arr.size, 1), dtype=float)
    return arr / total


def logsumexp(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return -np.inf
    max_value = float(np.nanmax(arr))
    if not np.isfinite(max_value):
        return max_value
    return max_value + math.log(float(np.nansum(np.exp(arr - max_value))))


def logsumexp_matrix(values: np.ndarray, axis: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    max_value = np.nanmax(arr, axis=axis, keepdims=True)
    safe = np.where(np.isfinite(max_value), max_value, 0.0)
    summed = np.nansum(np.exp(arr - safe), axis=axis, keepdims=True)
    out = safe + np.log(np.maximum(summed, 1e-300))
    return np.squeeze(out, axis=axis)


def model_family(model: str) -> str:
    if str(model) == STATIONARY_MODEL:
        return "nontrajectory"
    if str(model) in TRAJECTORY_MODELS:
        return "trajectory"
    return "other"


def event_base_from_group(group: pd.DataFrame) -> dict[str, object]:
    first = group.iloc[0]
    return {column: first[column] for column in DECISION_COLUMNS[: DECISION_COLUMNS.index("best_model")]}


def matching_evidence_group(evidence: pd.DataFrame, animal: object, date: object, track: object, sleep: object) -> pd.DataFrame:
    return evidence[
        evidence["animal"].astype(str).eq(str(animal))
        & evidence["date"].astype(str).eq(str(date))
        & evidence["track1_session"].astype(str).eq(str(track))
        & evidence["sleeppost_session"].astype(str).eq(str(sleep))
    ]


def empty_summary_row(columns: Sequence[str], *, scope: str, margin_threshold: float) -> dict[str, object]:
    row = {column: np.nan for column in columns}
    row["scope"] = scope
    row["events"] = 0
    row["margin_threshold"] = float(margin_threshold)
    row["biological_claim_assessed"] = False
    return row


def event_key_set(frame: pd.DataFrame) -> set[tuple[str, str, str, str, int]]:
    if frame.empty:
        return set()
    required = {"animal", "date", "track1_session", "sleeppost_session", "event_index"}
    if not required.issubset(frame.columns):
        return set()
    return set(
        zip(
            frame["animal"].astype(str),
            frame["date"].astype(str),
            frame["track1_session"].astype(str),
            frame["sleeppost_session"].astype(str),
            pd.to_numeric(frame["event_index"], errors="coerce").fillna(-1).astype(int),
        )
    )


def pair_key_set(frame: pd.DataFrame) -> set[tuple[str, str, str, str]]:
    if frame.empty:
        return set()
    return set(
        zip(
            frame["animal"].astype(str).str.upper(),
            frame["date"].astype(str),
            frame["track1_session"].astype(str),
            frame["sleeppost_session"].astype(str),
        )
    )


def pair_key_set_from_decoder(frame: pd.DataFrame) -> set[tuple[str, str, str, str]]:
    if frame.empty:
        return set()
    return set(
        zip(
            frame["animal"].astype(str).str.upper(),
            frame["date"].astype(str),
            frame["track1_session"].astype(str),
            frame["sleeppost_session"].astype(str),
        )
    )


def gate_row(gate: str, passed: bool, value: str, note: str) -> dict[str, object]:
    return {"gate": gate, "passed": bool(passed), "status": "pass" if passed else "fail", "value": value, "note": note}


def finite_values(values: object) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def finite_mean(values: object) -> float:
    arr = finite_values(values)
    return float(arr.mean()) if arr.size else np.nan


def finite_median(values: object) -> float:
    arr = finite_values(values)
    return float(np.median(arr)) if arr.size else np.nan


def finite_min(values: object) -> float:
    arr = finite_values(values)
    return float(arr.min()) if arr.size else np.nan


def finite_max(values: object) -> float:
    arr = finite_values(values)
    return float(arr.max()) if arr.size else np.nan


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    header = "| " + " | ".join(str(item) for item in headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(format_markdown_value(value) for value in row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def format_markdown_value(value: object) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.6g}"
    return str(value)


def serialisable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def parse_tetrodes(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for item in str(raw).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(int(item))
        except ValueError:
            return ()
    return tuple(values)


def as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--pairs-csv", required=True)
    parser.add_argument("--linearization-qc", required=True)
    parser.add_argument("--decoder-qc", required=True)
    parser.add_argument("--pilot-selection", required=True)
    parser.add_argument("--pilot-tier", default="pilot_20_balanced")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--margin-threshold", type=float, default=5.5)
    parser.add_argument("--position-bin-size-cm", type=float, default=5.0)
    parser.add_argument("--time-bin-s", type=float, default=0.020)
    parser.add_argument("--min-unit-spikes", type=int, default=5)
    parser.add_argument("--min-encoding-units", type=int, default=1)
    parser.add_argument("--smoothing-bins", type=int, default=1)
    parser.add_argument("--diffusion-sigma-cm", type=float, default=12.5)
    parser.add_argument("--stationary-self-transition", type=float, default=0.98)
    parser.add_argument("--imm-mode-persistence", type=float, default=0.92)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_sleep_evidence(
        dataset_root=args.dataset_root,
        pairs_csv=args.pairs_csv,
        linearization_qc=args.linearization_qc,
        decoder_qc=args.decoder_qc,
        pilot_selection=args.pilot_selection,
        pilot_tier=args.pilot_tier,
        output_dir=args.output_dir,
        margin_threshold=args.margin_threshold,
        position_bin_size_cm=args.position_bin_size_cm,
        time_bin_s=args.time_bin_s,
        min_unit_spikes=args.min_unit_spikes,
        min_encoding_units=args.min_encoding_units,
        smoothing_bins=args.smoothing_bins,
        diffusion_sigma_cm=args.diffusion_sigma_cm,
        stationary_self_transition=args.stationary_self_transition,
        imm_mode_persistence=args.imm_mode_persistence,
    )


if __name__ == "__main__":
    main()
