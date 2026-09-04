#!/usr/bin/env python3
"""Test surprise-gated retrospective replay in the Denovellis W-track data.

The analysis is deliberately downstream of the published Denovellis replay classification table. It does not decode or reclassify neural data. Behavioral
surprise is estimated causally from choices made before each outbound trial,
never from replay. Where DIO reward-pump data can be identified, a separate
reward-outcome surprise sensitivity is reported.

This is not, by itself, a Bayesian-smoothing test: the published summary table
does not retain enough path detail to compare replay with a formal filtered-to-
smoothed latent-state revision or to distinguish past-route from future-route
content at the W-track well. The script makes that boundary machine-readable.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

try:
    from scripts._provenance import build_script_provenance
except ModuleNotFoundError:  # Direct execution from scripts/.
    from _provenance import build_script_provenance


ANIMAL_DIRECTORIES = {
    "bon": "Bond",
    "cha": "Chapati",
    "con": "Conley",
    "cor": "Corriander",
    "dav": "Dave",
    "dud": "Dudley",
    "egy": "Egypt",
    "fra": "Frank",
    "gov": "Government",
    "remy": "Remy",
}

CONTINUOUS_STATE_FLAGS = ("Continuous", "Hover-Continuous-Mix", "Fragmented-Continuous-Mix")
CONTINUOUS_STATE_FRACTIONS = tuple(f"{state}_fraction_of_time" for state in CONTINUOUS_STATE_FLAGS)
EVENT_CONTENT_METRICS = (
    "continuous_component_fraction",
    "replay_total_distance",
    "replay_total_displacement",
    "replay_velocity_actual_position",
    "replay_velocity_toward_center_well",
    "replay_distance_from_actual_position",
)
EVENT_QUALITY_METRICS = (
    "event_duration_s",
    "n_total_spikes",
    "n_unique_spiking",
    "max_ripple_consensus_trace_zscore",
    "actual_speed",
)

TRIAL_OUTPUT = "denovellis_outbound_trials.csv"
EVENT_OUTPUT = "denovellis_post_choice_replay_events.csv"
EXTRACTION_AUDIT_OUTPUT = "denovellis_trial_extraction_audit.csv"
REWARD_MAPPING_OUTPUT = "denovellis_reward_pump_mapping.csv"
REWARD_VALIDATION_OUTPUT = "denovellis_inferred_outcome_validation.csv"
ASSOCIATION_OUTPUT = "denovellis_surprise_replay_associations.csv"
SENSITIVITY_OUTPUT = "denovellis_surprise_memory_sensitivity.csv"
BY_ANIMAL_OUTPUT = "denovellis_surprise_replay_by_animal.csv"
LOAO_OUTPUT = "denovellis_surprise_leave_one_animal_out.csv"
NULL_OUTPUT = "denovellis_surprise_permutation_null.csv"
GATE_OUTPUT = "denovellis_surprise_replay_gate_summary.csv"
MANIFEST_OUTPUT = "denovellis_surprise_replay_manifest.json"
REPORT_OUTPUT = "denovellis_surprise_replay_report.md"
RATE_FIGURE = "denovellis_choice_surprise_rate.png"
EFFECT_FIGURE = "denovellis_surprise_effect_forest.png"


@dataclass(frozen=True)
class AssociationSpec:
    analysis_scope: str
    endpoint: str
    endpoint_kind: str
    expected_direction: str
    trajectory_only: bool = False


PRIMARY_SPECS = (
    AssociationSpec("choice_surprise", "all_event_rate_hz", "rate", "positive"),
    AssociationSpec("choice_surprise", "trajectory_event_rate_hz", "rate", "positive"),
    AssociationSpec("choice_surprise", "continuous_component_fraction", "content", "positive"),
    AssociationSpec("choice_surprise", "replay_total_distance", "content", "positive", trajectory_only=True),
    AssociationSpec("choice_surprise", "replay_total_displacement", "content", "positive", trajectory_only=True),
    AssociationSpec("choice_surprise", "replay_velocity_actual_position", "content", "positive", trajectory_only=True),
    AssociationSpec("choice_surprise", "replay_velocity_toward_center_well", "content", "positive", trajectory_only=True),
    AssociationSpec("choice_surprise", "replay_distance_from_actual_position", "content", "positive", trajectory_only=True),
)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _day_epochs(root: Any, day: int) -> np.ndarray:
    array = np.asarray(root, dtype=object)
    if array.ndim == 0:
        return np.asarray([array.item()], dtype=object)
    try:
        day_value = array.reshape(-1)[int(day) - 1]
    except IndexError:
        day_value = array
    return np.asarray(day_value, dtype=object).reshape(-1)


def _linpos_day_epochs(root: Any, day: int) -> np.ndarray:
    """Return linpos epochs from either day-wrapped or direct-epoch files."""

    flat = np.asarray(root, dtype=object).reshape(-1)
    if any(_field(value, "statematrix") is not None for value in flat):
        return flat
    return _day_epochs(root, day)


def _find_day_file(dataset_root: Path, animal: str, file_type: str, day: int) -> Path | None:
    directory = dataset_root / ANIMAL_DIRECTORIES.get(animal.lower(), animal)
    matches = sorted(directory.glob(f"*{file_type}{int(day):02d}.mat"))
    return matches[0] if matches else None


def _final_near_well_run_start(distance: np.ndarray, radius_cm: float) -> int | None:
    near = np.isfinite(distance) & (distance <= float(radius_cm))
    indices = np.flatnonzero(near)
    if not len(indices):
        return None
    start = int(indices[-1])
    while start > 0 and near[start - 1]:
        start -= 1
    return start


def extract_outbound_trials_from_epoch(
    linpos_epoch: Any,
    *,
    animal: str,
    day: int,
    epoch: int,
    well_radius_cm: float = 10.0,
    max_window_s: float = 10.0,
) -> pd.DataFrame:
    """Return center-to-outer-well choices and post-arrival dwell exposure."""

    statematrix = _field(linpos_epoch, "statematrix")
    trajwells = np.asarray(_field(linpos_epoch, "trajwells", np.empty((0, 2))), dtype=float)
    if statematrix is None or trajwells.size == 0:
        return pd.DataFrame()
    trajwells = trajwells.reshape(-1, 2)
    time = np.asarray(_field(statematrix, "time", []), dtype=float).reshape(-1)
    wells = np.asarray(_field(statematrix, "wellExitEnter", []), dtype=float)
    distances = np.asarray(_field(statematrix, "linearDistanceToWells", []), dtype=float)
    if wells.ndim != 2 or wells.shape[1] < 2 or distances.ndim != 2:
        return pd.DataFrame()
    if len(time) != len(wells) or len(time) != len(distances):
        return pd.DataFrame()

    valid_center = trajwells[:, 0][np.isfinite(trajwells[:, 0])]
    if not len(valid_center):
        return pd.DataFrame()
    center_well = int(pd.Series(valid_center.astype(int)).mode().iloc[0])
    outer_wells = {int(value) for value in trajwells[:, 1] if np.isfinite(value)}
    changes = np.r_[True, np.any(wells[1:, :2] != wells[:-1, :2], axis=1)]
    starts = np.flatnonzero(changes)
    ends = np.r_[starts[1:], len(time)]

    rows: list[dict[str, object]] = []
    previous_outer: int | None = None
    outbound_index = 0
    session = f"{animal}-{int(day):02d}-{int(epoch):02d}"
    for start, end in zip(starts, ends, strict=True):
        exit_well, enter_well = (int(value) for value in wells[start, :2])
        if exit_well != center_well or enter_well not in outer_wells:
            continue
        if enter_well < 1 or enter_well > distances.shape[1]:
            continue
        near_start = _final_near_well_run_start(distances[start:end, enter_well - 1], well_radius_cm)
        if near_start is None:
            continue
        arrival_time = float(time[start + near_start])
        departure_time = float(time[end - 1])
        dwell = max(0.0, departure_time - arrival_time)
        outbound_index += 1
        alternation_consistent: bool | None = None
        if previous_outer is not None:
            alternation_consistent = bool(enter_well != previous_outer)
        rows.append(
            {
                "animal": animal.lower(),
                "day": int(day),
                "epoch": int(epoch),
                "session": session,
                "trial_index": outbound_index,
                "center_well": center_well,
                "destination_well": enter_well,
                "previous_outer_well": previous_outer,
                "alternation_consistent": alternation_consistent,
                "arrival_time_s": arrival_time,
                "departure_time_s": departure_time,
                "post_arrival_dwell_s": dwell,
                "choice_analysis_start_s": arrival_time,
                "choice_analysis_exposure_s": min(float(max_window_s), dwell),
                "well_radius_cm": float(well_radius_cm),
            }
        )
        previous_outer = enter_well

    frame = pd.DataFrame(rows)
    if not frame.empty:
        denominator = max(len(frame) - 1, 1)
        frame["trial_progress"] = (frame["trial_index"] - 1) / denominator
        frame["trial_id"] = frame["session"] + "-outbound-" + frame["trial_index"].astype(str)
    return frame


def extract_all_outbound_trials(
    dataset_root: str | Path,
    replay_info: pd.DataFrame,
    *,
    well_radius_cm: float = 10.0,
    max_window_s: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract every W-track session represented in the replay table."""

    root = Path(dataset_root)
    frames: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    keys = replay_info[["animal", "day", "epoch"]].drop_duplicates()
    for row in keys.itertuples(index=False):
        animal, day, epoch = str(row.animal).lower(), int(row.day), int(row.epoch)
        path = _find_day_file(root, animal, "linpos", day)
        status = "pass"
        reason = ""
        frame = pd.DataFrame()
        if path is None:
            status, reason = "fail", "missing_linpos"
        else:
            try:
                epochs = _linpos_day_epochs(loadmat(path, squeeze_me=True, struct_as_record=False)["linpos"], day)
                frame = extract_outbound_trials_from_epoch(
                    epochs[epoch - 1],
                    animal=animal,
                    day=day,
                    epoch=epoch,
                    well_radius_cm=well_radius_cm,
                    max_window_s=max_window_s,
                )
                if frame.empty:
                    status, reason = "fail", "no_outbound_trials"
                else:
                    frames.append(frame)
            except (IndexError, KeyError, OSError, TypeError, ValueError) as exc:
                status, reason = "fail", f"linpos_parse_error:{type(exc).__name__}"
        audit.append(
            {
                "animal": animal,
                "day": day,
                "epoch": epoch,
                "linpos_path": "" if path is None else str(path),
                "n_outbound_trials": int(len(frame)),
                "status": status,
                "failure_reason": reason,
            }
        )
    trials = (
        pd.concat([frame.dropna(axis=1, how="all") for frame in frames], ignore_index=True)
        if frames
        else pd.DataFrame()
    )
    return trials, pd.DataFrame(audit)


def assign_causal_binary_surprise(
    frame: pd.DataFrame,
    *,
    outcome_col: str,
    decay: float = 0.95,
    prefix: str = "choice",
    context_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """Add a discounted Beta-Bernoulli prediction using past trials only."""

    if not 0.0 < float(decay) <= 1.0:
        raise ValueError("decay must be in (0, 1]")
    output = frame.copy()
    probability = pd.Series(np.nan, index=output.index, dtype=float)
    surprise = pd.Series(np.nan, index=output.index, dtype=float)
    history_n = pd.Series(0, index=output.index, dtype=int)
    group_cols = ["session", *context_cols]
    grouped = output.groupby(group_cols, sort=False, dropna=False) if group_cols else [((), output)]
    for _, group in grouped:
        alpha = 1.0
        beta = 1.0
        n_seen = 0
        for index in group.sort_values("trial_index").index:
            alpha = 1.0 + float(decay) * (alpha - 1.0)
            beta = 1.0 + float(decay) * (beta - 1.0)
            value = output.at[index, outcome_col]
            if pd.isna(value):
                continue
            p_one = alpha / (alpha + beta)
            observed = bool(value)
            p_observed = p_one if observed else 1.0 - p_one
            probability.at[index] = p_observed
            surprise.at[index] = -math.log(max(p_observed, np.finfo(float).tiny))
            history_n.at[index] = n_seen
            alpha += float(observed)
            beta += float(not observed)
            n_seen += 1
    tag = str(float(decay)).replace(".", "p")
    output[f"{prefix}_probability_decay_{tag}"] = probability
    output[f"{prefix}_surprise_nats_decay_{tag}"] = surprise
    output[f"{prefix}_history_n_decay_{tag}"] = history_n
    return output


def load_replay_info(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = {"Animal ID", "day", "epoch", "start_time", "end_time", "ripple_number"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"replay-info table is missing columns: {sorted(missing)}")
    frame = frame.rename(columns={"Animal ID": "animal"}).copy()
    frame["animal"] = frame["animal"].astype(str).str.lower()
    frame["start_time_s"] = pd.to_timedelta(frame["start_time"]).dt.total_seconds()
    frame["end_time_s"] = pd.to_timedelta(frame["end_time"]).dt.total_seconds()
    frame["event_duration_s"] = frame["end_time_s"] - frame["start_time_s"]
    frame["replay_velocity_toward_center_well"] = -pd.to_numeric(
        frame.get("replay_velocity_center_well", pd.Series(np.nan, index=frame.index)), errors="coerce"
    )
    if "duration" in frame.columns:
        parsed_duration = pd.to_timedelta(frame["duration"], errors="coerce").dt.total_seconds()
        frame["event_duration_s"] = parsed_duration.fillna(frame["event_duration_s"])
    for column in CONTINUOUS_STATE_FLAGS:
        frame[column] = frame.get(column, False).map(_as_bool)
    fractions = []
    for column in CONTINUOUS_STATE_FRACTIONS:
        values = pd.to_numeric(frame.get(column, 0.0), errors="coerce").fillna(0.0)
        fractions.append(values)
    frame["continuous_component_fraction"] = np.clip(sum(fractions), 0.0, 1.0)
    frame["trajectory_component_present"] = frame[list(CONTINUOUS_STATE_FLAGS)].any(axis=1)
    return frame


def link_replay_events(
    trials: pd.DataFrame,
    replay_info: pd.DataFrame,
    *,
    start_col: str = "choice_analysis_start_s",
    exposure_col: str = "choice_analysis_exposure_s",
    scope: str = "choice",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Link events to non-overlapping trial dwell windows."""

    updated = trials.copy()
    events: list[pd.DataFrame] = []
    n_events = pd.Series(0, index=updated.index, dtype=int)
    n_trajectory = pd.Series(0, index=updated.index, dtype=int)
    grouped = {
        key: group.sort_values("start_time_s")
        for key, group in replay_info.groupby(["animal", "day", "epoch"], sort=False)
    }
    for index, trial in updated.iterrows():
        exposure = float(trial.get(exposure_col, 0.0))
        start = float(trial.get(start_col, np.nan))
        if not np.isfinite(start) or not np.isfinite(exposure) or exposure <= 0.0:
            continue
        key = (str(trial["animal"]), int(trial["day"]), int(trial["epoch"]))
        candidates = grouped.get(key)
        if candidates is None:
            continue
        selected = candidates[(candidates["start_time_s"] >= start) & (candidates["start_time_s"] <= start + exposure)].copy()
        n_events.at[index] = len(selected)
        n_trajectory.at[index] = int(selected["trajectory_component_present"].sum())
        if selected.empty:
            continue
        for column, value in trial.items():
            selected[column] = value
        selected["analysis_scope"] = scope
        selected["time_from_window_start_s"] = selected["start_time_s"] - start
        events.append(selected)
    updated[f"{scope}_n_events"] = n_events
    updated[f"{scope}_n_trajectory_events"] = n_trajectory
    updated[f"{scope}_all_event_rate_hz"] = n_events / pd.to_numeric(updated[exposure_col], errors="coerce")
    updated[f"{scope}_trajectory_event_rate_hz"] = n_trajectory / pd.to_numeric(updated[exposure_col], errors="coerce")
    return updated, pd.concat(events, ignore_index=True) if events else pd.DataFrame()


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "pass"}
    if pd.isna(value):
        return False
    return bool(value)


def _candidate_pump_times(dio_epoch: Any) -> dict[int, np.ndarray]:
    candidates: dict[int, np.ndarray] = {}
    if not isinstance(dio_epoch, np.ndarray):
        return candidates
    for pin, value in enumerate(np.asarray(dio_epoch, dtype=object).reshape(-1), start=1):
        pulse_times = np.asarray(_field(value, "pulsetimes", []), dtype=float)
        if pulse_times.ndim == 1 and pulse_times.size == 2:
            pulse_times = pulse_times.reshape(1, 2)
        if pulse_times.ndim != 2 or pulse_times.shape[1] < 2 or len(pulse_times) < 2:
            continue
        durations = pulse_times[:, 1] - pulse_times[:, 0]
        median = float(np.nanmedian(durations))
        stable_fraction = float(np.mean(np.abs(durations - median) <= max(3.0, 0.05 * abs(median))))
        if stable_fraction >= 0.60 and 100.0 <= median <= 10_000.0:
            candidates[pin] = pulse_times[:, 1] / 10_000.0
    return candidates


def add_dio_reward_observations(
    trials: pd.DataFrame,
    dataset_root: str | Path,
    *,
    min_mapping_hits: int = 3,
    mapping_purity: float = 0.90,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Map stable DIO pump channels to wells and annotate observed rewards."""

    root = Path(dataset_root)
    output = trials.copy()
    output["reward_mapping_complete"] = False
    output["reward_observed"] = pd.Series(pd.NA, index=output.index, dtype="boolean")
    for column in ("reward_time_s", "reward_outcome_time_s", "reward_analysis_exposure_s"):
        output[column] = np.nan
    mapping_rows: list[dict[str, object]] = []

    for (animal, day, epoch), group in output.groupby(["animal", "day", "epoch"], sort=False):
        animal = str(animal)
        dio_path = root / ANIMAL_DIRECTORIES.get(animal, animal) / "DIO"
        matches = sorted(dio_path.glob(f"*DIO{int(day):02d}.mat")) if dio_path.exists() else []
        if not matches:
            continue
        try:
            epochs = _day_epochs(loadmat(matches[0], squeeze_me=True, struct_as_record=False)["DIO"], int(day))
            dio_epoch = epochs[int(epoch) - 1]
        except (IndexError, KeyError, OSError, TypeError, ValueError):
            continue
        candidate_times = _candidate_pump_times(dio_epoch)
        pin_to_well: dict[int, int] = {}
        for pin, times in candidate_times.items():
            hit_wells: list[int] = []
            for pulse_time in times:
                hits = group[
                    (pd.to_numeric(group["arrival_time_s"], errors="coerce") <= pulse_time)
                    & (pd.to_numeric(group["departure_time_s"], errors="coerce") + 0.1 >= pulse_time)
                ]
                if not hits.empty:
                    hit_wells.append(int(hits.sort_values("arrival_time_s").iloc[-1]["destination_well"]))
            counts = pd.Series(hit_wells, dtype=int).value_counts()
            hits = int(counts.sum())
            purity = float(counts.iloc[0] / hits) if hits else 0.0
            mapped_well = int(counts.index[0]) if hits >= min_mapping_hits and purity >= mapping_purity else -1
            if mapped_well > 0:
                pin_to_well[pin] = mapped_well
            mapping_rows.append(
                {
                    "animal": animal,
                    "day": int(day),
                    "epoch": int(epoch),
                    "dio_path": str(matches[0]),
                    "pin": pin,
                    "n_pulses": len(times),
                    "n_trial_window_hits": hits,
                    "mapping_purity": purity,
                    "mapped_well": mapped_well if mapped_well > 0 else np.nan,
                    "mapping_status": "pass" if mapped_well > 0 else "fail",
                }
            )
        well_to_times = {well: candidate_times[pin] for pin, well in pin_to_well.items()}
        required_wells = set(group["destination_well"].astype(int))
        complete = required_wells.issubset(well_to_times)
        output.loc[group.index, "reward_mapping_complete"] = complete
        if not complete:
            continue
        latencies: list[float] = []
        trial_rewards: dict[int, tuple[bool, float]] = {}
        for index, trial in group.iterrows():
            times = well_to_times[int(trial["destination_well"])]
            valid = times[
                (times >= float(trial["arrival_time_s"]))
                & (times <= float(trial["departure_time_s"]) + 0.1)
            ]
            observed = bool(len(valid))
            reward_time = float(valid[0]) if observed else np.nan
            trial_rewards[index] = (observed, reward_time)
            if observed:
                latencies.append(reward_time - float(trial["arrival_time_s"]))
        expected_latency = float(np.nanmedian(latencies)) if latencies else 1.0
        for index, (observed, reward_time) in trial_rewards.items():
            arrival = float(output.at[index, "arrival_time_s"])
            departure = float(output.at[index, "departure_time_s"])
            outcome_time = reward_time if observed else arrival + expected_latency
            output.at[index, "reward_observed"] = observed
            output.at[index, "reward_time_s"] = reward_time
            output.at[index, "reward_outcome_time_s"] = outcome_time
            output.at[index, "reward_analysis_exposure_s"] = min(10.0, max(0.0, departure - outcome_time))

    mapping = pd.DataFrame(mapping_rows)
    validation = build_reward_validation(output)
    return output, mapping, validation


def build_reward_validation(trials: pd.DataFrame) -> pd.DataFrame:
    valid = trials[
        trials["reward_mapping_complete"].map(_as_bool)
        & trials["alternation_consistent"].notna()
        & trials["reward_observed"].notna()
    ].copy()
    rows: list[dict[str, object]] = []
    if not valid.empty:
        for animal, group in valid.groupby("animal", sort=True):
            rows.append(
                {
                    "scope": "animal",
                    "animal": animal,
                    "n_trials": len(group),
                    "n_sessions": group["session"].nunique(),
                    "agreement_fraction": float((group["alternation_consistent"].map(_as_bool) == group["reward_observed"].map(_as_bool)).mean()),
                    "reward_fraction": float(group["reward_observed"].map(_as_bool).mean()),
                }
            )
        rows.append(
            {
                "scope": "pooled",
                "animal": "all",
                "n_trials": len(valid),
                "n_sessions": valid["session"].nunique(),
                "agreement_fraction": float((valid["alternation_consistent"].map(_as_bool) == valid["reward_observed"].map(_as_bool)).mean()),
                "reward_fraction": float(valid["reward_observed"].map(_as_bool).mean()),
            }
        )
    return pd.DataFrame(rows)


def _rank(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").rank(method="average", pct=True).to_numpy(dtype=float)


def _within_group_residuals(values: np.ndarray, groups: pd.Series) -> np.ndarray:
    frame = pd.DataFrame({"value": values, "group": groups.astype(str).to_numpy()})
    return (frame["value"] - frame.groupby("group", sort=False)["value"].transform("mean")).to_numpy(dtype=float)


def partial_rank_effect(
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    outcome_col: str,
    control_cols: Sequence[str],
) -> tuple[float, int]:
    """Partial Spearman effect with session-by-outcome fixed intercepts."""

    columns = ["session", outcome_col, x_col, y_col, *control_cols]
    data = frame[columns].copy()
    for column in (x_col, y_col, *control_cols):
        data[column] = pd.to_numeric(data[column], errors="coerce")
        data.loc[~np.isfinite(data[column]), column] = np.nan
    data = data.dropna()
    if len(data) < 5:
        return np.nan, len(data)
    groups = data["session"].astype(str) + "|" + data[outcome_col].astype(str)
    x = _within_group_residuals(_rank(data[x_col]), groups)
    y = _within_group_residuals(_rank(data[y_col]), groups)
    controls = []
    for column in control_cols:
        control = _within_group_residuals(_rank(data[column]), groups)
        if np.nanstd(control) > 0.0:
            controls.append(control)
    if controls:
        design = np.column_stack(controls)
        x = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
        y = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return np.nan, len(data)
    return float(np.corrcoef(x, y)[0, 1]), len(data)


def _bootstrap_effects(
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    outcome_col: str,
    control_cols: Sequence[str],
    n_bootstraps: int,
    seed: int,
) -> np.ndarray:
    if n_bootstraps <= 0:
        return np.array([], dtype=float)
    rng = np.random.default_rng(seed)
    animals = np.asarray(sorted(frame["animal"].dropna().astype(str).unique()))
    values: list[float] = []
    for _ in range(int(n_bootstraps)):
        pieces = []
        for draw, animal in enumerate(rng.choice(animals, size=len(animals), replace=True)):
            piece = frame[frame["animal"].astype(str).eq(animal)].copy()
            piece["animal"] = f"draw{draw}:{animal}"
            piece["session"] = f"draw{draw}:" + piece["session"].astype(str)
            pieces.append(piece)
        sample = pd.concat(pieces, ignore_index=True)
        effect, _ = partial_rank_effect(sample, x_col=x_col, y_col=y_col, outcome_col=outcome_col, control_cols=control_cols)
        if np.isfinite(effect):
            values.append(effect)
    return np.asarray(values, dtype=float)


def _permutation_effects(
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    outcome_col: str,
    control_cols: Sequence[str],
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    if n_permutations <= 0:
        return np.array([], dtype=float)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    groups = frame.groupby(["session", outcome_col], sort=False, dropna=False).groups
    for _ in range(int(n_permutations)):
        shuffled = frame.copy()
        for indices in groups.values():
            source = shuffled.loc[list(indices), x_col].to_numpy(copy=True)
            shuffled.loc[list(indices), x_col] = rng.permutation(source)
        effect, _ = partial_rank_effect(shuffled, x_col=x_col, y_col=y_col, outcome_col=outcome_col, control_cols=control_cols)
        if np.isfinite(effect):
            values.append(effect)
    return np.asarray(values, dtype=float)


def _trial_content_table(events: pd.DataFrame, trials: pd.DataFrame, *, trajectory_only: bool) -> pd.DataFrame:
    selected = events.copy()
    if trajectory_only:
        selected = selected[selected["trajectory_component_present"].map(_as_bool)]
    if selected.empty:
        return pd.DataFrame()
    numeric = [*EVENT_CONTENT_METRICS, *EVENT_QUALITY_METRICS]
    for column in numeric:
        selected[column] = pd.to_numeric(selected.get(column), errors="coerce")
    aggregates = {column: "median" for column in numeric}
    content = selected.groupby("trial_id", as_index=False).agg(aggregates)
    metadata_columns = [
        "trial_id",
        "animal",
        "session",
        "trial_index",
        "trial_progress",
        "alternation_consistent",
        "reward_observed",
        "choice_surprise_nats",
        "reward_surprise_nats",
    ]
    metadata = trials[[column for column in metadata_columns if column in trials.columns]].drop_duplicates("trial_id")
    return metadata.merge(content, on="trial_id", how="inner", validate="one_to_one")


def estimate_association(
    frame: pd.DataFrame,
    *,
    analysis_scope: str,
    endpoint: str,
    endpoint_kind: str,
    surprise_col: str,
    outcome_col: str,
    control_cols: Sequence[str],
    expected_direction: str,
    n_bootstraps: int,
    n_permutations: int,
    seed: int,
) -> tuple[dict[str, object], np.ndarray]:
    effect, n = partial_rank_effect(frame, x_col=surprise_col, y_col=endpoint, outcome_col=outcome_col, control_cols=control_cols)
    bootstrap = _bootstrap_effects(
        frame,
        x_col=surprise_col,
        y_col=endpoint,
        outcome_col=outcome_col,
        control_cols=control_cols,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )
    null = _permutation_effects(
        frame,
        x_col=surprise_col,
        y_col=endpoint,
        outcome_col=outcome_col,
        control_cols=control_cols,
        n_permutations=n_permutations,
        seed=seed + 1,
    )
    ci = np.nanpercentile(bootstrap, [2.5, 97.5]) if len(bootstrap) else [np.nan, np.nan]
    p_positive = (1.0 + float(np.sum(null >= effect))) / (1.0 + len(null)) if len(null) and np.isfinite(effect) else np.nan
    p_two = (1.0 + float(np.sum(np.abs(null) >= abs(effect)))) / (1.0 + len(null)) if len(null) and np.isfinite(effect) else np.nan
    finite = np.isfinite(pd.to_numeric(frame[surprise_col], errors="coerce")) & np.isfinite(
        pd.to_numeric(frame[endpoint], errors="coerce")
    )
    animals = int(frame.loc[finite, "animal"].nunique())
    row = {
        "analysis_scope": analysis_scope,
        "predictor": surprise_col,
        "endpoint": endpoint,
        "endpoint_kind": endpoint_kind,
        "expected_direction": expected_direction,
        "n_observations": n,
        "n_animals": animals,
        "partial_spearman": effect,
        "animal_bootstrap_ci95_low": float(ci[0]),
        "animal_bootstrap_ci95_high": float(ci[1]),
        "permutation_p_positive": p_positive,
        "permutation_p_two_sided": p_two,
        "n_bootstraps": len(bootstrap),
        "n_permutations": len(null),
        "positive_supported": bool(np.isfinite(effect) and effect > 0.0 and ci[0] > 0.0 and p_positive < 0.05),
    }
    return row, null


def _bh_qvalues(p_values: pd.Series) -> pd.Series:
    values = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    result = np.full(values.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return pd.Series(result, index=p_values.index)
    order = finite[np.argsort(values[finite])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.minimum(ranked, 1.0)
    return pd.Series(result, index=p_values.index)


def build_analysis_tables(
    trials: pd.DataFrame,
    events: pd.DataFrame,
    replay_info: pd.DataFrame,
    *,
    n_bootstraps: int,
    n_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    primary_decay = "0p95"
    choice_surprise = f"choice_surprise_nats_decay_{primary_decay}"
    trials = trials.copy()
    trials["choice_surprise_nats"] = trials[choice_surprise]
    trials["all_event_rate_hz"] = trials["choice_all_event_rate_hz"]
    trials["trajectory_event_rate_hz"] = trials["choice_trajectory_event_rate_hz"]
    events = events.copy()
    events["choice_surprise_nats"] = events[choice_surprise]

    for index, spec in enumerate(PRIMARY_SPECS):
        if spec.endpoint_kind == "rate":
            analysis = trials[trials["choice_analysis_exposure_s"] > 0.0].copy()
            controls = ("trial_progress", "choice_analysis_exposure_s")
        else:
            analysis = _trial_content_table(events, trials, trajectory_only=spec.trajectory_only)
            controls = ("trial_progress", *EVENT_QUALITY_METRICS)
        if analysis.empty or spec.endpoint not in analysis:
            continue
        row, null = estimate_association(
            analysis,
            analysis_scope=spec.analysis_scope,
            endpoint=spec.endpoint,
            endpoint_kind=spec.endpoint_kind,
            surprise_col="choice_surprise_nats",
            outcome_col="alternation_consistent",
            control_cols=controls,
            expected_direction=spec.expected_direction,
            n_bootstraps=n_bootstraps,
            n_permutations=n_permutations,
            seed=seed + 100 * index,
        )
        rows.append(row)
        null_rows.extend(
            {
                "analysis_scope": spec.analysis_scope,
                "endpoint": spec.endpoint,
                "permutation_index": i,
                "null_partial_spearman": value,
            }
            for i, value in enumerate(null)
        )

    reward_trials = trials[
        trials["reward_mapping_complete"].map(_as_bool)
        & trials["reward_surprise_nats"].notna()
        & (trials["reward_analysis_exposure_s"] > 0.0)
    ].copy()
    if not reward_trials.empty:
        reward_trials, reward_linked = link_replay_events(
            reward_trials,
            replay_info=replay_info,
            start_col="reward_outcome_time_s",
            exposure_col="reward_analysis_exposure_s",
            scope="reward",
        )
        reward_trials["all_event_rate_hz"] = reward_trials["reward_all_event_rate_hz"]
        reward_trials["trajectory_event_rate_hz"] = reward_trials["reward_trajectory_event_rate_hz"]
        if not reward_linked.empty:
            reward_linked["reward_surprise_nats"] = reward_linked["reward_surprise_nats"]
        reward_specs = [AssociationSpec("reward_outcome_surprise", spec.endpoint, spec.endpoint_kind, spec.expected_direction, spec.trajectory_only) for spec in PRIMARY_SPECS]
        for offset, spec in enumerate(reward_specs, start=len(rows)):
            if spec.endpoint_kind == "rate":
                analysis = reward_trials
                controls = ("trial_progress", "reward_analysis_exposure_s", "alternation_consistent")
            else:
                analysis = _trial_content_table(reward_linked, reward_trials, trajectory_only=spec.trajectory_only)
                controls = ("trial_progress", "alternation_consistent", *EVENT_QUALITY_METRICS)
            if analysis.empty or spec.endpoint not in analysis:
                continue
            row, null = estimate_association(
                analysis,
                analysis_scope=spec.analysis_scope,
                endpoint=spec.endpoint,
                endpoint_kind=spec.endpoint_kind,
                surprise_col="reward_surprise_nats",
                outcome_col="reward_observed",
                control_cols=controls,
                expected_direction=spec.expected_direction,
                n_bootstraps=n_bootstraps,
                n_permutations=n_permutations,
                seed=seed + 100 * offset,
            )
            rows.append(row)
            null_rows.extend(
                {
                    "analysis_scope": spec.analysis_scope,
                    "endpoint": spec.endpoint,
                    "permutation_index": i,
                    "null_partial_spearman": value,
                }
                for i, value in enumerate(null)
            )

    update_trials = trials[
        trials["alternation_consistent"].eq(False)
        & trials["next_alternation_consistent"].notna()
        & (trials["choice_analysis_exposure_s"] > 0.0)
    ].copy()
    if not update_trials.empty:
        update_trials["next_choice_correction"] = update_trials["next_alternation_consistent"].map(_as_bool).astype(float)
        for offset, predictor in enumerate(("all_event_rate_hz", "trajectory_event_rate_hz"), start=len(rows)):
            row, null = estimate_association(
                update_trials,
                analysis_scope="exploratory_post_error_behavioral_update",
                endpoint="next_choice_correction",
                endpoint_kind="behavioral_update",
                surprise_col=predictor,
                outcome_col="alternation_consistent",
                control_cols=("choice_surprise_nats", "trial_progress", "choice_analysis_exposure_s"),
                expected_direction="positive",
                n_bootstraps=n_bootstraps,
                n_permutations=n_permutations,
                seed=seed + 100 * offset,
            )
            rows.append(row)
            null_rows.extend(
                {
                    "analysis_scope": "exploratory_post_error_behavioral_update",
                    "endpoint": "next_choice_correction",
                    "predictor": predictor,
                    "permutation_index": i,
                    "null_partial_spearman": value,
                }
                for i, value in enumerate(null)
            )

    associations = pd.DataFrame(rows)
    if not associations.empty:
        associations["fdr_q_two_sided"] = np.nan
        for _, indices in associations.groupby("analysis_scope", sort=False).groups.items():
            associations.loc[list(indices), "fdr_q_two_sided"] = _bh_qvalues(
                associations.loc[list(indices), "permutation_p_two_sided"]
            )

    sensitivity_rows: list[dict[str, object]] = []
    for tag in ("1p0", "0p95", "0p8"):
        column = f"choice_surprise_nats_decay_{tag}"
        for endpoint in ("all_event_rate_hz", "trajectory_event_rate_hz"):
            effect, n = partial_rank_effect(
                trials[trials["choice_analysis_exposure_s"] > 0.0],
                x_col=column,
                y_col=endpoint,
                outcome_col="alternation_consistent",
                control_cols=("trial_progress", "choice_analysis_exposure_s"),
            )
            sensitivity_rows.append({"surprise_decay": tag.replace("p", "."), "endpoint": endpoint, "n_observations": n, "partial_spearman": effect})

    by_animal_rows: list[dict[str, object]] = []
    loao_rows: list[dict[str, object]] = []
    for endpoint in ("all_event_rate_hz", "trajectory_event_rate_hz"):
        for animal, group in trials.groupby("animal", sort=True):
            effect, n = partial_rank_effect(
                group,
                x_col="choice_surprise_nats",
                y_col=endpoint,
                outcome_col="alternation_consistent",
                control_cols=("trial_progress", "choice_analysis_exposure_s"),
            )
            by_animal_rows.append({"animal": animal, "endpoint": endpoint, "n_trials": n, "partial_spearman": effect})
        for animal in sorted(trials["animal"].unique()):
            retained = trials[trials["animal"] != animal]
            effect, n = partial_rank_effect(
                retained,
                x_col="choice_surprise_nats",
                y_col=endpoint,
                outcome_col="alternation_consistent",
                control_cols=("trial_progress", "choice_analysis_exposure_s"),
            )
            loao_rows.append({"excluded_animal": animal, "endpoint": endpoint, "n_trials": n, "partial_spearman": effect})
    return associations, pd.DataFrame(sensitivity_rows), pd.DataFrame(by_animal_rows), pd.DataFrame(loao_rows), pd.DataFrame(null_rows)


def build_gate_summary(
    trials: pd.DataFrame,
    events: pd.DataFrame,
    validation: pd.DataFrame,
    associations: pd.DataFrame,
    extraction_audit: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(gate: str, passed: bool, observed: object, criterion: str, gate_type: str = "technical") -> None:
        rows.append({"gate": gate, "passed": bool(passed), "observed": observed, "criterion": criterion, "gate_type": gate_type})

    usable = trials[trials["alternation_consistent"].notna() & (trials["choice_analysis_exposure_s"] >= 0.5)]
    n_animals = usable["animal"].nunique()
    n_events = len(events)
    dio_pooled = validation[validation["scope"].eq("pooled")] if not validation.empty else pd.DataFrame()
    dio_animals = int(validation[validation["scope"].eq("animal")]["animal"].nunique()) if not validation.empty else 0
    agreement = float(dio_pooled.iloc[0]["agreement_fraction"]) if len(dio_pooled) else np.nan
    n_parsed = int(extraction_audit["status"].eq("pass").sum()) if len(extraction_audit) else 0
    parse_fraction = n_parsed / len(extraction_audit) if len(extraction_audit) else 0.0
    add(
        "linpos_sessions_parse",
        parse_fraction >= 0.90,
        f"{n_parsed}/{len(extraction_audit)} ({parse_fraction:.1%})",
        ">=90% of represented run epochs yield outbound trials",
    )
    add("multiple_animals_represented", n_animals >= 5, n_animals, ">=5 animals with analyzable outbound trials")
    add("trial_count_adequate", len(usable) >= 1000, len(usable), ">=1000 outbound trials with >=0.5 s exposure")
    add("linked_replay_events_present", n_events >= 500, n_events, ">=500 published SWRs linked to post-arrival windows")
    event_speed = pd.to_numeric(events.get("actual_speed", pd.Series(dtype=float)), errors="coerce")
    immobile_fraction = float((event_speed <= 4.0).mean()) if len(event_speed) else np.nan
    add(
        "linked_events_are_immobile",
        np.isfinite(immobile_fraction) and immobile_fraction >= 0.99,
        immobile_fraction,
        ">=99% of linked published SWRs have actual speed <=4 cm/s",
    )
    add("dio_reward_validation_animals", dio_animals >= 3, dio_animals, ">=3 animals with complete reward-pump mapping")
    add("alternation_label_reward_agreement", np.isfinite(agreement) and agreement >= 0.80, agreement, "inferred alternation consistency agrees with observed reward on >=80% of mapped trials")
    technical = all(row["passed"] for row in rows)
    add("technical_overall", technical, technical, "all technical gates pass")

    choice = associations[associations["analysis_scope"].eq("choice_surprise")] if not associations.empty else pd.DataFrame()
    for endpoint in ("trajectory_event_rate_hz", "replay_total_distance"):
        match = choice[choice["endpoint"].eq(endpoint)]
        supported = bool(len(match) and match.iloc[0]["positive_supported"])
        effect = float(match.iloc[0]["partial_spearman"]) if len(match) else np.nan
        add(f"choice_surprise_{endpoint}_positive", supported, effect, "positive effect, animal-bootstrap CI > 0, and one-sided trial-label permutation p < 0.05", "interpretation")
    support = bool(
        len(choice)
        and choice[choice["endpoint"].isin(["trajectory_event_rate_hz", "replay_total_distance"])]["positive_supported"].all()
        and len(choice[choice["endpoint"].isin(["trajectory_event_rate_hz", "replay_total_distance"])]) == 2
    )
    add("surprise_gated_retrospective_replay_supported", support, support, "both predeclared rate and extent endpoints support a positive association", "interpretation")
    add("bayesian_smoothing_identified", False, False, "requires explicit filtered-vs-smoothed latent revision and past-vs-future path content, unavailable in the summary artifact", "claim_boundary")
    return pd.DataFrame(rows)


def _plot_rate_tertiles(trials: pd.DataFrame, path: Path) -> None:
    data = trials[
        trials["alternation_consistent"].notna()
        & trials["choice_surprise_nats"].notna()
        & (trials["choice_analysis_exposure_s"] > 0.0)
    ].copy()
    data["surprise_tertile"] = data.groupby("alternation_consistent", group_keys=False)["choice_surprise_nats"].transform(
        lambda values: pd.qcut(values.rank(method="first"), 3, labels=["low", "middle", "high"])
    )
    aggregate = data.groupby(["animal", "alternation_consistent", "surprise_tertile"], observed=True).agg(
        events=("choice_n_events", "sum"),
        trajectory_events=("choice_n_trajectory_events", "sum"),
        exposure=("choice_analysis_exposure_s", "sum"),
    ).reset_index()
    aggregate["all_rate_min"] = 60.0 * aggregate["events"] / aggregate["exposure"]
    aggregate["trajectory_rate_min"] = 60.0 * aggregate["trajectory_events"] / aggregate["exposure"]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharex=True)
    order = ["low", "middle", "high"]
    colors = {False: "#C44E52", True: "#4C72B0"}
    for axis, metric, title in zip(axes, ["all_rate_min", "trajectory_rate_min"], ["All SWRs", "Trajectory-component SWRs"], strict=True):
        for outcome, group in aggregate.groupby("alternation_consistent"):
            summary = group.groupby("surprise_tertile", observed=True)[metric].agg(["mean", "sem"]).reindex(order)
            label = "alternation-consistent" if outcome else "alternation-inconsistent"
            axis.errorbar(np.arange(3), summary["mean"], yerr=summary["sem"], marker="o", capsize=3, color=colors[bool(outcome)], label=label)
        axis.set_title(title)
        axis.set_xticks(np.arange(3), order)
        axis.set_xlabel("causal choice-surprise tertile")
        axis.axhline(0.0, color="0.75", linewidth=0.8)
    axes[0].set_ylabel("events / min of post-arrival dwell")
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_effect_forest(associations: pd.DataFrame, path: Path) -> None:
    data = associations[associations["analysis_scope"].eq("choice_surprise")].copy()
    if data.empty:
        return
    data = data.iloc[::-1]
    y = np.arange(len(data))
    figure, axis = plt.subplots(figsize=(8.2, max(3.5, 0.5 * len(data) + 1.2)))
    effect = data["partial_spearman"].to_numpy(dtype=float)
    low = data["animal_bootstrap_ci95_low"].to_numpy(dtype=float)
    high = data["animal_bootstrap_ci95_high"].to_numpy(dtype=float)
    axis.errorbar(effect, y, xerr=np.maximum(0.0, np.vstack([effect - low, high - effect])), fmt="o", color="#365C8D", capsize=3)
    axis.axvline(0.0, color="black", linewidth=0.9)
    axis.set_yticks(y, data["endpoint"].str.replace("_", " "))
    axis.set_xlabel("quality-adjusted partial Spearman effect of surprise")
    axis.set_title("Denovellis post-choice replay: animal-bootstrap 95% CI")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _report_markdown(
    trials: pd.DataFrame,
    events: pd.DataFrame,
    associations: pd.DataFrame,
    gates: pd.DataFrame,
    validation: pd.DataFrame,
) -> str:
    technical = _gate_value(gates, "technical_overall")
    support = _gate_value(gates, "surprise_gated_retrospective_replay_supported")
    verdict = "technical-fail"
    if technical:
        verdict = "positive" if support else "technical-pass / hypothesis-not-supported"
    usable = trials[trials["alternation_consistent"].notna() & (trials["choice_analysis_exposure_s"] >= 0.5)]
    lines = [
        "# Denovellis Surprise-Gated Retrospective Replay",
        "",
        f"**Verdict:** {verdict}",
        "",
        "## Design",
        "",
        "Outbound W-track choices were reconstructed from `wellExitEnter`. Arrival was the start of the final <=10 cm visit to the destination well, and replay opportunity was measured during up to 10 s of the subsequent dwell. Choice surprise was a strictly causal discounted Beta-Bernoulli prediction of whether the animal would alternate, computed before the current outcome and never from replay. Published Denovellis SWR classifications supplied replay endpoints; no neural data were re-decoded.",
        "",
        "## Coverage",
        "",
        f"- {len(usable)} analyzable outbound choices across {usable['animal'].nunique()} animals and {usable['session'].nunique()} run epochs.",
        f"- {len(events)} published SWRs linked to post-arrival windows; {int(events['trajectory_component_present'].sum()) if len(events) else 0} contained a continuous component.",
    ]
    pooled = validation[validation["scope"].eq("pooled")] if not validation.empty else pd.DataFrame()
    if len(pooled):
        row = pooled.iloc[0]
        lines.append(f"- Alternation labels agreed with reward-pump observations on {100 * float(row['agreement_fraction']):.1f}% of {int(row['n_trials'])} mapped trials.")
    lines.extend(["", "## Primary Readout", ""])
    for row in associations[associations["analysis_scope"].eq("choice_surprise")].itertuples(index=False):
        lines.append(
            f"- `{row.endpoint}`: partial Spearman {row.partial_spearman:+.3f}, animal-bootstrap 95% CI [{row.animal_bootstrap_ci95_low:+.3f}, {row.animal_bootstrap_ci95_high:+.3f}], one-sided permutation p={row.permutation_p_positive:.4f}."
        )
    update = associations[associations["analysis_scope"].eq("exploratory_post_error_behavioral_update")]
    if not update.empty:
        lines.extend(
            [
                "",
                "## Exploratory Behavioral Update",
                "",
                "This post hoc diagnostic asks whether replay after an alternation error predicts correction on the next outbound choice. It is not part of the predeclared support gate.",
            ]
        )
        for row in update.itertuples(index=False):
            lines.append(
                f"- `{row.predictor}` versus next-choice correction: partial Spearman {row.partial_spearman:+.3f}, animal-bootstrap 95% CI [{row.animal_bootstrap_ci95_low:+.3f}, {row.animal_bootstrap_ci95_high:+.3f}], one-sided permutation p={row.permutation_p_positive:.4f}."
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A positive result requires both more trajectory-component events per unit dwell exposure and greater decoded trajectory extent as causal surprise rises. Rate, content, and direction-compatible endpoints remain separate; a positive result on one is not promoted to the others.",
            "",
            "This analysis can support or reject **surprise-gated retrospective replay** under the stated behavioral proxy. It cannot identify Bayesian smoothing: the published event table does not retain a formal filtered-versus-smoothed behavioral belief revision, nor enough path-resolved content to distinguish the just-traversed arm from the future arm at every well.",
            "",
            "## Gates",
            "",
            _markdown_table(gates),
            "",
        ]
    )
    return "\n".join(lines)


def _gate_value(gates: pd.DataFrame, gate: str) -> bool:
    row = gates[gates["gate"].eq(gate)]
    return bool(len(row) and _as_bool(row.iloc[0]["passed"]))


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def run_analysis(
    *,
    dataset_root: str | Path,
    replay_info_path: str | Path,
    output_dir: str | Path,
    well_radius_cm: float = 10.0,
    max_window_s: float = 10.0,
    min_exposure_s: float = 0.5,
    n_bootstraps: int = 500,
    n_permutations: int = 1000,
    seed: int = 4,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    replay = load_replay_info(replay_info_path)
    trials, extraction_audit = extract_all_outbound_trials(
        dataset_root,
        replay,
        well_radius_cm=well_radius_cm,
        max_window_s=max_window_s,
    )
    if trials.empty:
        raise RuntimeError("no outbound W-track trials could be reconstructed")
    for decay in (1.0, 0.95, 0.8):
        trials = assign_causal_binary_surprise(trials, outcome_col="alternation_consistent", decay=decay, prefix="choice")
    trials, pump_mapping, validation = add_dio_reward_observations(trials, dataset_root)
    reward_subset = trials[trials["reward_mapping_complete"].map(_as_bool)].copy()
    if not reward_subset.empty:
        reward_subset = assign_causal_binary_surprise(
            reward_subset,
            outcome_col="reward_observed",
            decay=0.95,
            prefix="reward",
            context_cols=("alternation_consistent",),
        )
        reward_columns = [column for column in reward_subset.columns if column.startswith("reward_")]
        trials.loc[reward_subset.index, reward_columns] = reward_subset[reward_columns]
    trials["choice_surprise_nats"] = trials["choice_surprise_nats_decay_0p95"]
    trials["reward_surprise_nats"] = trials.get("reward_surprise_nats_decay_0p95", np.nan)
    trials["next_alternation_consistent"] = trials.groupby("session", sort=False)[
        "alternation_consistent"
    ].shift(-1)
    trials = trials[trials["choice_analysis_exposure_s"] >= float(min_exposure_s)].copy()
    trials, events = link_replay_events(trials, replay, scope="choice")
    associations, sensitivity, by_animal, loao, null = build_analysis_tables(
        trials,
        events,
        replay,
        n_bootstraps=n_bootstraps,
        n_permutations=n_permutations,
        seed=seed,
    )
    gates = build_gate_summary(trials, events, validation, associations, extraction_audit)

    paths = {
        "trials": output / TRIAL_OUTPUT,
        "events": output / EVENT_OUTPUT,
        "extraction_audit": output / EXTRACTION_AUDIT_OUTPUT,
        "reward_mapping": output / REWARD_MAPPING_OUTPUT,
        "reward_validation": output / REWARD_VALIDATION_OUTPUT,
        "associations": output / ASSOCIATION_OUTPUT,
        "sensitivity": output / SENSITIVITY_OUTPUT,
        "by_animal": output / BY_ANIMAL_OUTPUT,
        "loao": output / LOAO_OUTPUT,
        "null": output / NULL_OUTPUT,
        "gates": output / GATE_OUTPUT,
        "manifest": output / MANIFEST_OUTPUT,
        "report": output / REPORT_OUTPUT,
        "rate_figure": output / RATE_FIGURE,
        "effect_figure": output / EFFECT_FIGURE,
    }
    trials.to_csv(paths["trials"], index=False)
    events.to_csv(paths["events"], index=False)
    extraction_audit.to_csv(paths["extraction_audit"], index=False)
    pump_mapping.to_csv(paths["reward_mapping"], index=False)
    validation.to_csv(paths["reward_validation"], index=False)
    associations.to_csv(paths["associations"], index=False)
    sensitivity.to_csv(paths["sensitivity"], index=False)
    by_animal.to_csv(paths["by_animal"], index=False)
    loao.to_csv(paths["loao"], index=False)
    null.to_csv(paths["null"], index=False)
    gates.to_csv(paths["gates"], index=False)
    _plot_rate_tertiles(trials, paths["rate_figure"])
    _plot_effect_forest(associations, paths["effect_figure"])
    paths["report"].write_text(_report_markdown(trials, events, associations, gates, validation), encoding="utf-8")
    manifest = {
        "analysis": "denovellis_surprise_gated_retrospective_replay",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "well_radius_cm": well_radius_cm,
            "max_window_s": max_window_s,
            "min_exposure_s": min_exposure_s,
            "choice_surprise_decay": 0.95,
            "choice_surprise_decay_sensitivity": [1.0, 0.95, 0.8],
            "n_bootstraps": n_bootstraps,
            "n_permutations": n_permutations,
            "seed": seed,
        },
        "counts": {
            "outbound_trials": len(trials),
            "linked_events": len(events),
            "animals": trials["animal"].nunique(),
            "sessions": trials["session"].nunique(),
        },
        "claim_boundary": {
            "tests": "surprise-gated retrospective replay rate and summary-content associations",
            "does_not_test": "formal Bayesian smoothing or unambiguous past-route versus future-route content",
            "replay_source": "published Denovellis state-classification summary; no rescoring",
        },
        "provenance": build_script_provenance(
            input_paths={"replay_info": replay_info_path, "dataset_root": dataset_root},
            cwd=Path.cwd(),
        ),
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--replay-info", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--well-radius-cm", type=float, default=10.0)
    parser.add_argument("--max-window-s", type=float, default=10.0)
    parser.add_argument("--min-exposure-s", type=float, default=0.5)
    parser.add_argument("--n-bootstraps", type=int, default=500)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = run_analysis(
        dataset_root=args.dataset_root,
        replay_info_path=args.replay_info,
        output_dir=args.output_dir,
        well_radius_cm=args.well_radius_cm,
        max_window_s=args.max_window_s,
        min_exposure_s=args.min_exposure_s,
        n_bootstraps=args.n_bootstraps,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )
    print(paths["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
