#!/usr/bin/env python3
"""Run multi-session SleepPOST high-spiking candidate-event QC for Olafsdottir2016."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import pandas as pd

from hipporeplayimm.olafsdottir2016 import read_axona_cut, read_axona_pos, read_axona_set


SESSION_OUTPUT = "olafsdottir_sleeppost_event_detection_qc.csv"
EVENT_OUTPUT = "olafsdottir_sleeppost_candidate_events.csv"
ANIMAL_OUTPUT = "olafsdottir_sleeppost_event_detection_by_animal.csv"
GATE_OUTPUT = "olafsdottir_sleeppost_event_detection_gate_summary.csv"
SUMMARY_OUTPUT = "olafsdottir_sleeppost_event_detection_qc_summary.md"
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
    "linearization_status",
}
SESSION_COLUMNS = [
    "animal",
    "date",
    "sleeppost_session",
    "paired_track1_session",
    "sleep_duration_s",
    "n_units_sleep",
    "n_spikes_sleep",
    "mean_mua_rate_hz",
    "raw_candidate_event_count",
    "artifact_flagged_event_count",
    "candidate_event_count",
    "candidate_event_rate_per_min",
    "median_event_duration_ms",
    "median_event_spikes",
    "median_event_active_units",
    "p95_event_spikes",
    "immobile_event_count",
    "running_or_movement_flagged_event_count",
    "event_detection_status",
    "exclusion_reason",
]
EVENT_COLUMNS = [
    "animal",
    "date",
    "session",
    "event_id",
    "start_time_s",
    "end_time_s",
    "duration_ms",
    "n_spikes",
    "n_active_units",
    "mean_mua_rate_hz",
    "peak_mua_rate_hz",
    "mean_speed_cm_s",
    "event_detection_score",
    "candidate_tier",
    "event_qc_status",
    "event_qc_reason",
]


@dataclass(frozen=True)
class SleepSpikes:
    spike_times_s: np.ndarray
    unit_ids: np.ndarray
    unit_count: int


@dataclass(frozen=True)
class SpeedTrace:
    times_s: np.ndarray
    speed_cm_s: np.ndarray


def run_event_detection_qc(
    *,
    dataset_root: str | Path,
    pairs_csv: str | Path,
    linearization_qc: str | Path,
    output_dir: str | Path,
    bin_size_s: float = 0.010,
    smooth_window_s: float = 0.020,
    mua_z_threshold: float = 3.0,
    merge_gap_s: float = 0.020,
    min_duration_ms: float = 20.0,
    max_duration_ms: float = 500.0,
    min_event_spikes: int = 5,
    min_event_active_units: int = 3,
    start_artifact_exclusion_s: float = 0.100,
    max_event_spikes_per_active_unit: float = 10.0,
    immobility_speed_threshold_cm_s: float = 5.0,
    moderate_event_spikes: int = 10,
    strong_event_spikes: int = 25,
    extreme_event_spikes: int = 50,
    min_dataset_candidate_events: int = 10,
    min_dataset_candidate_sessions: int = 3,
    min_paper_candidate_animals: int = 2,
    max_paper_candidate_animal_fraction: float = 0.75,
    max_paper_candidate_session_fraction: float = 0.75,
) -> dict[str, pd.DataFrame]:
    pairs = load_pairs(pairs_csv)
    linearization = load_linearization_qc(linearization_qc)
    usable_pairs = pairs[pairs["usable_pair"].map(_as_bool)].copy()
    linearization_pass = _linearization_pass_keys(linearization)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    session_rows: list[dict[str, object]] = []
    event_rows: list[pd.DataFrame] = []
    for _, pair in usable_pairs.iterrows():
        row, events = summarize_pair(
            pair,
            dataset_root=Path(dataset_root),
            linearization_pass=linearization_pass,
            bin_size_s=bin_size_s,
            smooth_window_s=smooth_window_s,
            mua_z_threshold=mua_z_threshold,
            merge_gap_s=merge_gap_s,
            min_duration_ms=min_duration_ms,
            max_duration_ms=max_duration_ms,
            min_event_spikes=min_event_spikes,
            min_event_active_units=min_event_active_units,
            start_artifact_exclusion_s=start_artifact_exclusion_s,
            max_event_spikes_per_active_unit=max_event_spikes_per_active_unit,
            immobility_speed_threshold_cm_s=immobility_speed_threshold_cm_s,
            moderate_event_spikes=moderate_event_spikes,
            strong_event_spikes=strong_event_spikes,
            extreme_event_spikes=extreme_event_spikes,
        )
        session_rows.append(row)
        event_rows.append(events)

    sessions = pd.DataFrame(session_rows, columns=SESSION_COLUMNS)
    events = pd.concat(event_rows, ignore_index=True) if event_rows else pd.DataFrame(columns=EVENT_COLUMNS)
    animals = summarize_by_animal(sessions, events)
    gates = gate_summary(
        pairs=usable_pairs,
        sessions=sessions,
        events=events,
        immobility_speed_threshold_cm_s=immobility_speed_threshold_cm_s,
        min_dataset_candidate_events=min_dataset_candidate_events,
        min_dataset_candidate_sessions=min_dataset_candidate_sessions,
        min_paper_candidate_animals=min_paper_candidate_animals,
        max_paper_candidate_animal_fraction=max_paper_candidate_animal_fraction,
        max_paper_candidate_session_fraction=max_paper_candidate_session_fraction,
    )

    sessions.to_csv(out / SESSION_OUTPUT, index=False)
    events.to_csv(out / EVENT_OUTPUT, index=False)
    animals.to_csv(out / ANIMAL_OUTPUT, index=False)
    gates.to_csv(out / GATE_OUTPUT, index=False)
    (out / SUMMARY_OUTPUT).write_text(build_markdown_summary(sessions, events, animals, gates), encoding="utf-8")
    return {
        "sessions": sessions,
        "events": events,
        "animals": animals,
        "gates": gates,
    }


def load_pairs(path: str | Path) -> pd.DataFrame:
    pairs = pd.read_csv(path)
    missing = sorted(REQUIRED_PAIR_COLUMNS.difference(pairs.columns))
    if missing:
        raise ValueError(f"pairs CSV is missing required columns: {missing}")
    return pairs


def load_linearization_qc(path: str | Path) -> pd.DataFrame:
    linearization = pd.read_csv(path)
    missing = sorted(REQUIRED_LINEARIZATION_COLUMNS.difference(linearization.columns))
    if missing:
        raise ValueError(f"linearization QC is missing required columns: {missing}")
    return linearization


def summarize_pair(
    pair: pd.Series,
    *,
    dataset_root: Path,
    linearization_pass: set[tuple[str, str, str]],
    bin_size_s: float,
    smooth_window_s: float,
    mua_z_threshold: float,
    merge_gap_s: float,
    min_duration_ms: float,
    max_duration_ms: float,
    min_event_spikes: int,
    min_event_active_units: int,
    start_artifact_exclusion_s: float,
    max_event_spikes_per_active_unit: float,
    immobility_speed_threshold_cm_s: float,
    moderate_event_spikes: int,
    strong_event_spikes: int,
    extreme_event_spikes: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    animal = str(pair["animal"]).upper()
    date = str(pair["date"])
    track_session = str(pair["track_session"])
    sleep_session = str(pair["sleepPOST_session"])
    sleep_stem = _session_stem(dataset_root, animal, date, sleep_session)
    reasons: list[str] = []
    if (animal, date, track_session) not in linearization_pass:
        reasons.append("paired_track1_linearization_not_passed")
    try:
        sleep_duration = sleep_duration_s(sleep_stem)
        spikes = load_sleep_spikes(sleep_stem, _parse_tetrodes(str(pair["hippocampal_tetrodes"])))
        speed = load_sleep_speed(sleep_stem)
        if spikes.spike_times_s.size == 0 or spikes.unit_count == 0:
            reasons.append("missing_sleep_spike_timestamps_or_units")
            row = session_row(
                animal=animal,
                date=date,
                sleep_session=sleep_session,
                track_session=track_session,
                sleep_duration=sleep_duration,
                spikes=spikes,
                events=pd.DataFrame(columns=EVENT_COLUMNS),
                status="fail",
                reasons=reasons,
                immobility_speed_threshold_cm_s=immobility_speed_threshold_cm_s,
            )
            return row, pd.DataFrame(columns=EVENT_COLUMNS)
        if not np.isfinite(sleep_duration) or sleep_duration <= 0.0:
            sleep_duration = float(np.nanmax(spikes.spike_times_s)) if spikes.spike_times_s.size else np.nan
        events = detect_mua_candidate_events(
            animal=animal,
            date=date,
            sleep_session=sleep_session,
            spikes=spikes,
            speed=speed,
            sleep_duration_s=sleep_duration,
            bin_size_s=bin_size_s,
            smooth_window_s=smooth_window_s,
            mua_z_threshold=mua_z_threshold,
            merge_gap_s=merge_gap_s,
            min_duration_ms=min_duration_ms,
            max_duration_ms=max_duration_ms,
            min_event_spikes=min_event_spikes,
            min_event_active_units=min_event_active_units,
            start_artifact_exclusion_s=start_artifact_exclusion_s,
            max_event_spikes_per_active_unit=max_event_spikes_per_active_unit,
            immobility_speed_threshold_cm_s=immobility_speed_threshold_cm_s,
            moderate_event_spikes=moderate_event_spikes,
            strong_event_spikes=strong_event_spikes,
            extreme_event_spikes=extreme_event_spikes,
        )
        status = "pass" if len(qc_pass_events(events)) else "no_candidate_events"
        row = session_row(
            animal=animal,
            date=date,
            sleep_session=sleep_session,
            track_session=track_session,
            sleep_duration=sleep_duration,
            spikes=spikes,
            events=events,
            status=status if not reasons else "fail",
            reasons=reasons,
            immobility_speed_threshold_cm_s=immobility_speed_threshold_cm_s,
        )
        return row, events
    except Exception as exc:  # noqa: BLE001 - QC should continue across sessions.
        reasons.append(type(exc).__name__ + ":" + str(exc))
        row = failed_session_row(animal=animal, date=date, sleep_session=sleep_session, track_session=track_session, reasons=reasons)
        return row, pd.DataFrame(columns=EVENT_COLUMNS)


def detect_mua_candidate_events(
    *,
    animal: str,
    date: str,
    sleep_session: str,
    spikes: SleepSpikes,
    speed: SpeedTrace | None,
    sleep_duration_s: float,
    bin_size_s: float,
    smooth_window_s: float,
    mua_z_threshold: float,
    merge_gap_s: float,
    min_duration_ms: float,
    max_duration_ms: float,
    min_event_spikes: int,
    min_event_active_units: int,
    start_artifact_exclusion_s: float,
    max_event_spikes_per_active_unit: float,
    immobility_speed_threshold_cm_s: float,
    moderate_event_spikes: int,
    strong_event_spikes: int,
    extreme_event_spikes: int,
) -> pd.DataFrame:
    if not np.isfinite(sleep_duration_s) or sleep_duration_s <= 0.0:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    edges = np.arange(0.0, float(sleep_duration_s) + float(bin_size_s), float(bin_size_s))
    if edges.shape[0] < 2:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    counts, _ = np.histogram(spikes.spike_times_s, bins=edges)
    window_bins = max(1, int(round(float(smooth_window_s) / float(bin_size_s))))
    smoothed_counts = _moving_average(counts.astype(float), window_bins)
    mua_rate = smoothed_counts / float(bin_size_s)
    score = _robust_z(mua_rate)
    windows = _threshold_windows(
        score,
        edges,
        threshold=mua_z_threshold,
        merge_gap_s=merge_gap_s,
    )
    rows: list[dict[str, object]] = []
    for window in windows:
        start = float(window["start_time_s"])
        end = float(window["end_time_s"])
        duration_ms = 1000.0 * (end - start)
        if duration_ms < float(min_duration_ms) or duration_ms > float(max_duration_ms):
            continue
        in_event = (spikes.spike_times_s >= start) & (spikes.spike_times_s <= end)
        n_spikes = int(np.count_nonzero(in_event))
        n_active = int(np.unique(spikes.unit_ids[in_event]).shape[0]) if n_spikes else 0
        if n_spikes < int(min_event_spikes) or n_active < int(min_event_active_units):
            continue
        qc_status, qc_reason = event_qc_status(
            start_time_s=start,
            n_spikes=n_spikes,
            n_active_units=n_active,
            start_artifact_exclusion_s=start_artifact_exclusion_s,
            max_event_spikes_per_active_unit=max_event_spikes_per_active_unit,
        )
        bin_start = max(0, int(np.searchsorted(edges, start, side="right") - 1))
        bin_end = min(mua_rate.shape[0], int(np.searchsorted(edges, end, side="left") + 1))
        event_rates = mua_rate[bin_start:bin_end]
        event_score = score[bin_start:bin_end]
        mean_speed = event_mean_speed(speed, start, end)
        rows.append(
            {
                "animal": animal,
                "date": date,
                "session": sleep_session,
                "event_id": len(rows),
                "start_time_s": start,
                "end_time_s": end,
                "duration_ms": duration_ms,
                "n_spikes": n_spikes,
                "n_active_units": n_active,
                "mean_mua_rate_hz": float(n_spikes / max(end - start, 1e-9)),
                "peak_mua_rate_hz": float(np.nanmax(event_rates)) if event_rates.size else np.nan,
                "mean_speed_cm_s": mean_speed,
                "event_detection_score": float(np.nanmax(event_score)) if event_score.size else np.nan,
                "candidate_tier": candidate_tier(
                    n_spikes=n_spikes,
                    score=float(np.nanmax(event_score)) if event_score.size else np.nan,
                    threshold=mua_z_threshold,
                    moderate_event_spikes=moderate_event_spikes,
                    strong_event_spikes=strong_event_spikes,
                    extreme_event_spikes=extreme_event_spikes,
                ),
                "event_qc_status": qc_status,
                "event_qc_reason": qc_reason,
            }
        )
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def session_row(
    *,
    animal: str,
    date: str,
    sleep_session: str,
    track_session: str,
    sleep_duration: float,
    spikes: SleepSpikes,
    events: pd.DataFrame,
    status: str,
    reasons: list[str],
    immobility_speed_threshold_cm_s: float,
) -> dict[str, object]:
    duration_min = float(sleep_duration) / 60.0 if np.isfinite(sleep_duration) and sleep_duration > 0.0 else np.nan
    valid_events = qc_pass_events(events)
    speed = pd.to_numeric(valid_events["mean_speed_cm_s"], errors="coerce") if not valid_events.empty else pd.Series(dtype=float)
    return {
        "animal": animal,
        "date": date,
        "sleeppost_session": sleep_session,
        "paired_track1_session": track_session,
        "sleep_duration_s": float(sleep_duration) if np.isfinite(sleep_duration) else np.nan,
        "n_units_sleep": int(spikes.unit_count),
        "n_spikes_sleep": int(spikes.spike_times_s.shape[0]),
        "mean_mua_rate_hz": float(spikes.spike_times_s.shape[0] / sleep_duration) if np.isfinite(sleep_duration) and sleep_duration > 0 else np.nan,
        "raw_candidate_event_count": int(len(events)),
        "artifact_flagged_event_count": int((events["event_qc_status"].astype(str) != "pass").sum()) if not events.empty else 0,
        "candidate_event_count": int(len(valid_events)),
        "candidate_event_rate_per_min": float(len(valid_events) / duration_min) if np.isfinite(duration_min) and duration_min > 0 else np.nan,
        "median_event_duration_ms": _median(valid_events, "duration_ms"),
        "median_event_spikes": _median(valid_events, "n_spikes"),
        "median_event_active_units": _median(valid_events, "n_active_units"),
        "p95_event_spikes": _percentile(valid_events, "n_spikes", 95.0),
        "immobile_event_count": int((speed <= float(immobility_speed_threshold_cm_s)).sum()) if not valid_events.empty else 0,
        "running_or_movement_flagged_event_count": int((speed > float(immobility_speed_threshold_cm_s)).sum()) if not valid_events.empty else 0,
        "event_detection_status": status,
        "exclusion_reason": ";".join(reasons),
    }


def failed_session_row(*, animal: str, date: str, sleep_session: str, track_session: str, reasons: list[str]) -> dict[str, object]:
    row = {column: np.nan for column in SESSION_COLUMNS}
    row.update(
        {
            "animal": animal,
            "date": date,
            "sleeppost_session": sleep_session,
            "paired_track1_session": track_session,
            "sleep_duration_s": np.nan,
            "n_units_sleep": 0,
            "n_spikes_sleep": 0,
            "mean_mua_rate_hz": np.nan,
            "raw_candidate_event_count": 0,
            "artifact_flagged_event_count": 0,
            "candidate_event_count": 0,
            "candidate_event_rate_per_min": np.nan,
            "immobile_event_count": 0,
            "running_or_movement_flagged_event_count": 0,
            "event_detection_status": "fail",
            "exclusion_reason": ";".join(reasons),
        }
    )
    return row


def load_sleep_spikes(sleep_stem: Path, tetrodes: Sequence[int]) -> SleepSpikes:
    spike_times: list[float] = []
    unit_ids: list[int] = []
    units: set[int] = set()
    for tetrode in tetrodes:
        raw_path = sleep_stem.with_suffix(f".{int(tetrode)}")
        cut_path = sleep_stem.parent / f"{sleep_stem.name}_{int(tetrode)}.cut"
        if not raw_path.is_file() or not cut_path.is_file():
            continue
        cut = read_axona_cut(cut_path, tetrode_path=raw_path)
        if cut.spike_times_s is None:
            continue
        labels = np.asarray(cut.labels, dtype=int)
        times = np.asarray(cut.spike_times_s, dtype=float)
        keep = labels > 0
        cell_ids = np.asarray([int(tetrode) * 100 + int(label) for label in labels[keep]], dtype=int)
        spike_times.extend(times[keep].tolist())
        unit_ids.extend(cell_ids.tolist())
        units.update(cell_ids.tolist())
    if not spike_times:
        return SleepSpikes(spike_times_s=np.empty(0, dtype=float), unit_ids=np.empty(0, dtype=int), unit_count=0)
    order = np.argsort(np.asarray(spike_times, dtype=float))
    return SleepSpikes(
        spike_times_s=np.asarray(spike_times, dtype=float)[order],
        unit_ids=np.asarray(unit_ids, dtype=int)[order],
        unit_count=len(units),
    )


def load_sleep_speed(sleep_stem: Path) -> SpeedTrace | None:
    pos_path = sleep_stem.with_suffix(".pos")
    if not pos_path.is_file():
        return None
    position = read_axona_pos(pos_path)
    xy = np.column_stack([position.x_cm, position.y_cm])
    valid = position.valid & np.isfinite(xy).all(axis=1) & np.isfinite(position.times_s)
    speed = np.full(position.times_s.shape, np.nan, dtype=float)
    if valid.sum() >= 2:
        idx = np.arange(position.times_s.shape[0], dtype=float)
        filled = xy.copy()
        for dim in range(2):
            filled[~valid, dim] = np.interp(idx[~valid], idx[valid], xy[valid, dim])
        dt = np.gradient(position.times_s)
        with np.errstate(divide="ignore", invalid="ignore"):
            vx = np.gradient(filled[:, 0]) / dt
            vy = np.gradient(filled[:, 1]) / dt
        speed[valid] = np.sqrt(vx[valid] * vx[valid] + vy[valid] * vy[valid])
    return SpeedTrace(times_s=position.times_s, speed_cm_s=speed)


def event_mean_speed(speed: SpeedTrace | None, start_s: float, end_s: float) -> float:
    if speed is None or speed.times_s.size == 0:
        return np.nan
    keep = (speed.times_s >= start_s) & (speed.times_s <= end_s) & np.isfinite(speed.speed_cm_s)
    if np.any(keep):
        return float(np.nanmean(speed.speed_cm_s[keep]))
    mid = 0.5 * (float(start_s) + float(end_s))
    valid = np.isfinite(speed.times_s) & np.isfinite(speed.speed_cm_s)
    if valid.sum() < 2:
        return np.nan
    return float(np.interp(mid, speed.times_s[valid], speed.speed_cm_s[valid]))


def summarize_by_animal(sessions: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "animal",
        "sleeppost_sessions",
        "sessions_with_candidates",
        "candidate_event_count",
        "immobile_event_count",
        "total_sleep_spikes",
        "total_sleep_units",
        "median_candidate_events_per_session",
        "event_detection_status",
    ]
    if sessions.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for animal, group in sessions.groupby("animal", sort=True):
        passed_events = qc_pass_events(events)
        animal_events = passed_events[passed_events["animal"].astype(str).eq(str(animal))] if not passed_events.empty else pd.DataFrame(columns=EVENT_COLUMNS)
        statuses = set(group["event_detection_status"].astype(str))
        status = "fail" if "fail" in statuses else ("pass" if len(animal_events) else "no_candidate_events")
        rows.append(
            {
                "animal": animal,
                "sleeppost_sessions": int(len(group)),
                "sessions_with_candidates": int((pd.to_numeric(group["candidate_event_count"], errors="coerce").fillna(0) > 0).sum()),
                "candidate_event_count": int(len(animal_events)),
                "immobile_event_count": int(pd.to_numeric(group["immobile_event_count"], errors="coerce").fillna(0).sum()),
                "total_sleep_spikes": int(pd.to_numeric(group["n_spikes_sleep"], errors="coerce").fillna(0).sum()),
                "total_sleep_units": int(pd.to_numeric(group["n_units_sleep"], errors="coerce").fillna(0).sum()),
                "median_candidate_events_per_session": _median(group, "candidate_event_count"),
                "event_detection_status": status,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def gate_summary(
    *,
    pairs: pd.DataFrame,
    sessions: pd.DataFrame,
    events: pd.DataFrame,
    immobility_speed_threshold_cm_s: float,
    min_dataset_candidate_events: int,
    min_dataset_candidate_sessions: int,
    min_paper_candidate_animals: int,
    max_paper_candidate_animal_fraction: float,
    max_paper_candidate_session_fraction: float,
) -> pd.DataFrame:
    expected_sessions = int(len(pairs))
    passed_events = qc_pass_events(events)
    total_events = int(len(passed_events))
    raw_events = int(len(events))
    artifact_events = raw_events - total_events
    sessions_with_events = int((pd.to_numeric(sessions["candidate_event_count"], errors="coerce").fillna(0) > 0).sum()) if not sessions.empty else 0
    animals_with_events = int(passed_events["animal"].nunique()) if not passed_events.empty else 0
    immobile_events = int((pd.to_numeric(passed_events["mean_speed_cm_s"], errors="coerce") <= float(immobility_speed_threshold_cm_s)).sum()) if not passed_events.empty else 0
    max_animal_fraction = _max_group_fraction(passed_events, "animal")
    max_session_fraction = _max_group_fraction(passed_events, "session")
    status_values = set(sessions["event_detection_status"].astype(str)) if not sessions.empty else set()
    finite_events = _events_finite_and_plausible(passed_events)
    gates = [
        _gate(
            "sleeppost_spike_data_present",
            expected_sessions > 0 and _all_numeric_positive(sessions, "n_spikes_sleep") and _all_numeric_positive(sessions, "n_units_sleep"),
            f"spikes_positive={_count_numeric_positive(sessions, 'n_spikes_sleep')}/{expected_sessions}; units_positive={_count_numeric_positive(sessions, 'n_units_sleep')}/{expected_sessions}",
            "data_integrity",
            "all usable pairs have SleepPOST spike timestamps and unit metadata",
            "SleepPOST spikes are required before any candidate-event screen.",
        ),
        _gate(
            "event_metrics_finite_and_plausible",
            total_events > 0 and finite_events,
            f"qc_candidate_events={total_events}; raw_windows={raw_events}; artifact_flagged={artifact_events}",
            "data_integrity",
            "QC-valid candidate events have finite durations, spike counts, active-unit counts, and MUA rates",
            "Rejects malformed or timestamp-artifact event rows before model evidence.",
        ),
        _gate(
            "no_event_detection_failures",
            "fail" not in status_values and expected_sessions > 0,
            f"statuses={','.join(sorted(status_values))}",
            "data_integrity",
            "no session has event_detection_status=fail",
            "Missing timestamps or unit metadata should stop evidence scaling.",
        ),
        _gate(
            "dataset_usable_candidate_count",
            total_events >= int(min_dataset_candidate_events),
            f"qc_candidate_events={total_events}",
            "dataset_usable",
            f"qc_candidate_events >= {int(min_dataset_candidate_events)}",
            "Enough events exist for a pilot model-evidence run.",
        ),
        _gate(
            "dataset_usable_session_count",
            sessions_with_events >= int(min_dataset_candidate_sessions),
            f"sessions_with_events={sessions_with_events}",
            "dataset_usable",
            f"sessions_with_events >= {int(min_dataset_candidate_sessions)}",
            "Pilot evidence should not come from a single SleepPOST session.",
        ),
        _gate(
            "candidates_detected_in_multiple_animals",
            animals_with_events >= int(min_paper_candidate_animals),
            f"animals_with_events={animals_with_events}",
            "paper_ready",
            f"animals_with_events >= {int(min_paper_candidate_animals)}",
            "Paper-ready scaling needs events beyond one animal.",
        ),
        _gate(
            "candidate_events_not_animal_dominated",
            total_events > 0 and max_animal_fraction <= float(max_paper_candidate_animal_fraction),
            f"max_animal_fraction={max_animal_fraction:.6g}",
            "paper_ready",
            f"max_animal_fraction <= {float(max_paper_candidate_animal_fraction):g}",
            "Avoids a candidate set dominated by one animal.",
        ),
        _gate(
            "candidate_events_not_session_dominated",
            total_events > 0 and max_session_fraction <= float(max_paper_candidate_session_fraction),
            f"max_session_fraction={max_session_fraction:.6g}",
            "paper_ready",
            f"max_session_fraction <= {float(max_paper_candidate_session_fraction):g}",
            "Avoids a candidate set dominated by one SleepPOST session.",
        ),
        _gate(
            "immobile_candidate_subset_exists",
            immobile_events > 0,
            f"immobile_events={immobile_events}",
            "paper_ready",
            f"at least one candidate event has mean_speed_cm_s <= {float(immobility_speed_threshold_cm_s):g}",
            "The replay screen needs an immobile candidate subset rather than only movement-like windows.",
        ),
    ]
    return pd.DataFrame(gates)


def build_markdown_summary(sessions: pd.DataFrame, events: pd.DataFrame, animals: pd.DataFrame, gates: pd.DataFrame) -> str:
    dataset_gates = gates[gates["gate_group"].astype(str).eq("dataset_usable")]
    paper_gates = gates[gates["gate_group"].astype(str).eq("paper_ready")]
    lines = [
        "# Olafsdottir SleepPOST Event-Detection QC Summary",
        "",
        "This is a candidate-event QC checkpoint only. It does not score replay evidence or compare 1D against 2D.",
        "",
        "## Overview",
        "",
        _markdown_table(
            ["Metric", "Value"],
            [
                ("SleepPOST sessions processed", len(sessions)),
                ("Raw high-MUA windows", len(events)),
                ("Artifact-flagged windows", int((events["event_qc_status"].astype(str) != "pass").sum()) if not events.empty else 0),
                ("QC-valid candidate events", len(qc_pass_events(events))),
                ("Sessions with candidates", int((pd.to_numeric(sessions["candidate_event_count"], errors="coerce").fillna(0) > 0).sum()) if not sessions.empty else 0),
                ("Animals with candidates", qc_pass_events(events)["animal"].nunique() if not qc_pass_events(events).empty else 0),
                ("Immobile candidate events", int(pd.to_numeric(sessions["immobile_event_count"], errors="coerce").fillna(0).sum()) if not sessions.empty else 0),
                ("Dataset-usable gates passed", f"{int(dataset_gates['passed'].map(_as_bool).sum())}/{len(dataset_gates)}"),
                ("Paper-ready gates passed", f"{int(paper_gates['passed'].map(_as_bool).sum())}/{len(paper_gates)}"),
            ],
        ),
        "",
        "## Gate Summary",
        "",
        _markdown_table(["Gate", "Group", "Status", "Value"], gates[["gate", "gate_group", "status", "value"]].itertuples(index=False, name=None)),
        "",
        "## Animal Summary",
        "",
        _markdown_table(["Animal", "Sessions", "Candidate events", "Immobile events", "Status"], animals[["animal", "sleeppost_sessions", "candidate_event_count", "immobile_event_count", "event_detection_status"]].itertuples(index=False, name=None)),
        "",
    ]
    return "\n".join(lines)


def sleep_duration_s(sleep_stem: Path) -> float:
    set_path = sleep_stem.with_suffix(".set")
    if not set_path.is_file():
        return np.nan
    header = read_axona_set(set_path)
    return _header_float(header, "duration", default=np.nan)


def candidate_tier(
    *,
    n_spikes: int,
    score: float,
    threshold: float,
    moderate_event_spikes: int,
    strong_event_spikes: int,
    extreme_event_spikes: int,
) -> str:
    if int(n_spikes) >= int(extreme_event_spikes) or (np.isfinite(score) and score >= threshold + 4.0):
        return "extreme"
    if int(n_spikes) >= int(strong_event_spikes) or (np.isfinite(score) and score >= threshold + 2.0):
        return "strong"
    if int(n_spikes) >= int(moderate_event_spikes) or (np.isfinite(score) and score >= threshold + 1.0):
        return "moderate"
    return "weak"


def event_qc_status(
    *,
    start_time_s: float,
    n_spikes: int,
    n_active_units: int,
    start_artifact_exclusion_s: float,
    max_event_spikes_per_active_unit: float,
) -> tuple[str, str]:
    reasons: list[str] = []
    if np.isfinite(start_time_s) and float(start_time_s) < float(start_artifact_exclusion_s):
        reasons.append("recording_start_artifact")
    spikes_per_active = float(n_spikes) / max(float(n_active_units), 1.0)
    if np.isfinite(max_event_spikes_per_active_unit) and spikes_per_active > float(max_event_spikes_per_active_unit):
        reasons.append("implausible_spikes_per_active_unit")
    if reasons:
        return "artifact", ";".join(reasons)
    return "pass", ""


def qc_pass_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty or "event_qc_status" not in events:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return events[events["event_qc_status"].astype(str).eq("pass")].copy()


def _threshold_windows(score: np.ndarray, edges: np.ndarray, *, threshold: float, merge_gap_s: float) -> list[dict[str, float]]:
    above = np.asarray(score, dtype=float) >= float(threshold)
    if above.size == 0:
        return []
    changes = np.diff(above.astype(int), prepend=0, append=0)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    windows = [_window_from_bins(score, edges, int(start), int(end)) for start, end in zip(starts, ends)]
    merged: list[dict[str, float]] = []
    for window in windows:
        if merged and window["start_time_s"] - merged[-1]["end_time_s"] <= float(merge_gap_s):
            merged[-1] = _window_from_bins(score, edges, int(merged[-1]["start_bin"]), int(window["end_bin"]))
        else:
            merged.append(window)
    return merged


def _window_from_bins(score: np.ndarray, edges: np.ndarray, start: int, end: int) -> dict[str, float]:
    end = max(start + 1, end)
    segment = score[start:end]
    peak_offset = int(np.nanargmax(segment)) if segment.size else 0
    return {
        "start_bin": int(start),
        "end_bin": int(end),
        "start_time_s": float(edges[start]),
        "end_time_s": float(edges[end]),
        "peak_time_s": float(0.5 * (edges[start + peak_offset] + edges[start + peak_offset + 1])),
        "peak_score": float(score[start + peak_offset]) if segment.size else np.nan,
    }


def _linearization_pass_keys(linearization: pd.DataFrame) -> set[tuple[str, str, str]]:
    passing = linearization[linearization["linearization_status"].astype(str).eq("pass")]
    return {
        (str(row.animal).upper(), str(row.date), str(row.track_session))
        for row in passing.itertuples(index=False)
    }


def _session_stem(dataset_root: Path, animal: str, date: str, session: str) -> Path:
    return dataset_root / animal.lower() / date / session


def _parse_tetrodes(raw: str) -> tuple[int, ...]:
    return tuple(int(value) for value in re.findall(r"\d+", str(raw)))


def _header_float(header: dict[str, str], key: str, *, default: float) -> float:
    raw = header.get(key)
    if raw is None:
        return float(default)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(raw))
    return float(match.group(0)) if match else float(default)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if int(window) <= 1:
        return np.asarray(values, dtype=float)
    kernel = np.ones(int(window), dtype=float) / float(window)
    return np.convolve(np.asarray(values, dtype=float), kernel, mode="same")


def _robust_z(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    median = float(np.nanmedian(arr))
    mad = float(np.nanmedian(np.abs(arr - median)))
    if np.isfinite(mad) and mad > 0.0:
        return (arr - median) / (1.4826 * mad)
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr))
    if np.isfinite(std) and std > 0.0:
        return (arr - mean) / std
    return np.zeros_like(arr, dtype=float)


def _median(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.median()) if not values.empty else np.nan


def _percentile(frame: pd.DataFrame, column: str, percentile: float) -> float:
    if frame.empty or column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(np.percentile(values, percentile)) if not values.empty else np.nan


def _all_numeric_positive(frame: pd.DataFrame, column: str) -> bool:
    if frame.empty or column not in frame:
        return False
    values = pd.to_numeric(frame[column], errors="coerce")
    return bool(values.notna().all() and (values > 0).all())


def _count_numeric_positive(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame:
        return 0
    return int((pd.to_numeric(frame[column], errors="coerce").fillna(0.0) > 0).sum())


def _events_finite_and_plausible(events: pd.DataFrame) -> bool:
    if events.empty:
        return False
    for column in ("duration_ms", "n_spikes", "n_active_units", "mean_mua_rate_hz", "peak_mua_rate_hz", "event_detection_score"):
        values = pd.to_numeric(events[column], errors="coerce")
        if values.isna().any():
            return False
    return bool((events["duration_ms"] > 0).all() and (events["n_spikes"] > 0).all() and (events["n_active_units"] > 0).all())


def _max_group_fraction(events: pd.DataFrame, column: str) -> float:
    if events.empty or column not in events:
        return np.nan
    counts = events.groupby(column).size()
    return float(counts.max() / counts.sum()) if counts.sum() else np.nan


def _gate(gate: str, passed: bool, value: str, gate_group: str, requirement: str, note: str) -> dict[str, object]:
    return {
        "gate": gate,
        "gate_group": gate_group,
        "passed": bool(passed),
        "status": "pass" if passed else "fail",
        "value": value,
        "requirement": requirement,
        "note": note,
    }


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--linearization-qc", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/olafsdottir-sleeppost-event-qc"))
    parser.add_argument("--bin-size-s", type=float, default=0.010)
    parser.add_argument("--smooth-window-s", type=float, default=0.020)
    parser.add_argument("--mua-z-threshold", type=float, default=3.0)
    parser.add_argument("--merge-gap-s", type=float, default=0.020)
    parser.add_argument("--min-duration-ms", type=float, default=20.0)
    parser.add_argument("--max-duration-ms", type=float, default=500.0)
    parser.add_argument("--min-event-spikes", type=int, default=5)
    parser.add_argument("--min-event-active-units", type=int, default=3)
    parser.add_argument("--start-artifact-exclusion-s", type=float, default=0.100)
    parser.add_argument("--max-event-spikes-per-active-unit", type=float, default=10.0)
    parser.add_argument("--immobility-speed-threshold-cm-s", type=float, default=5.0)
    parser.add_argument("--moderate-event-spikes", type=int, default=10)
    parser.add_argument("--strong-event-spikes", type=int, default=25)
    parser.add_argument("--extreme-event-spikes", type=int, default=50)
    parser.add_argument("--min-dataset-candidate-events", type=int, default=10)
    parser.add_argument("--min-dataset-candidate-sessions", type=int, default=3)
    parser.add_argument("--min-paper-candidate-animals", type=int, default=2)
    parser.add_argument("--max-paper-candidate-animal-fraction", type=float, default=0.75)
    parser.add_argument("--max-paper-candidate-session-fraction", type=float, default=0.75)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tables = run_event_detection_qc(
        dataset_root=args.dataset_root,
        pairs_csv=args.pairs_csv,
        linearization_qc=args.linearization_qc,
        output_dir=args.output_dir,
        bin_size_s=args.bin_size_s,
        smooth_window_s=args.smooth_window_s,
        mua_z_threshold=args.mua_z_threshold,
        merge_gap_s=args.merge_gap_s,
        min_duration_ms=args.min_duration_ms,
        max_duration_ms=args.max_duration_ms,
        min_event_spikes=args.min_event_spikes,
        min_event_active_units=args.min_event_active_units,
        start_artifact_exclusion_s=args.start_artifact_exclusion_s,
        max_event_spikes_per_active_unit=args.max_event_spikes_per_active_unit,
        immobility_speed_threshold_cm_s=args.immobility_speed_threshold_cm_s,
        moderate_event_spikes=args.moderate_event_spikes,
        strong_event_spikes=args.strong_event_spikes,
        extreme_event_spikes=args.extreme_event_spikes,
        min_dataset_candidate_events=args.min_dataset_candidate_events,
        min_dataset_candidate_sessions=args.min_dataset_candidate_sessions,
        min_paper_candidate_animals=args.min_paper_candidate_animals,
        max_paper_candidate_animal_fraction=args.max_paper_candidate_animal_fraction,
        max_paper_candidate_session_fraction=args.max_paper_candidate_session_fraction,
    )
    print(tables["sessions"].to_string(index=False))
    print()
    print(tables["gates"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
