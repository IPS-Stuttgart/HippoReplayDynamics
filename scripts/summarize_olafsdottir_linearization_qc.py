#!/usr/bin/env python3
"""Run multi-session Track1 linearization QC for Olafsdottir2016 pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hipporeplayimm.olafsdottir2016 import read_axona_cut, read_axona_pos
from linearize_olafsdottir_ztrack import (
    linearize_pos_file,
    project_points_to_centerline,
    smooth_positions,
)


SESSION_OUTPUT = "olafsdottir_track1_linearization_qc.csv"
ANIMAL_OUTPUT = "olafsdottir_track1_linearization_by_animal.csv"
GATE_OUTPUT = "olafsdottir_track1_linearization_gate_summary.csv"
FIGURE_OUTPUT = "olafsdottir_track1_linearization_figures_manifest.csv"
SUMMARY_OUTPUT = "olafsdottir_track1_linearization_qc_summary.md"
REQUIRED_PAIR_COLUMNS = {
    "animal",
    "date",
    "track_session",
    "sleepPOST_session",
    "hippocampal_tetrodes",
    "usable_pair",
}
SESSION_COLUMNS = [
    "animal",
    "date",
    "track_session",
    "sleeppost_session",
    "n_position_samples",
    "position_duration_s",
    "valid_position_fraction",
    "track_length_cm",
    "linearized_position_min_cm",
    "linearized_position_max_cm",
    "linearized_position_span_cm",
    "occupancy_nonzero_bins",
    "occupancy_nonzero_fraction",
    "median_off_track_distance_cm",
    "p95_off_track_distance_cm",
    "median_speed_cm_s",
    "p95_speed_cm_s",
    "n_spikes_track1",
    "n_units_track1",
    "track_spike_position_overlap_s",
    "orientation_rule",
    "reversal_applied",
    "linearization_status",
    "exclusion_reason",
]


def run_linearization_qc(
    *,
    dataset_root: str | Path,
    pairs_csv: str | Path,
    output_dir: str | Path,
    min_occupancy_nonzero_fraction: float = 0.25,
    min_valid_position_fraction: float = 0.50,
    occupancy_bin_size_cm: float = 5.0,
    smoothing_window_samples: int = 5,
) -> dict[str, pd.DataFrame]:
    pairs = load_pairs(pairs_csv)
    usable_pairs = pairs[pairs["usable_pair"].map(_as_bool)].copy()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    session_rows: list[dict[str, object]] = []
    figure_rows: list[dict[str, object]] = []
    for _, pair in usable_pairs.iterrows():
        session_dir = out / "sessions" / str(pair["animal"]).upper() / str(pair["date"])
        row, figures = summarize_pair(
            pair,
            dataset_root=Path(dataset_root),
            output_dir=session_dir,
            min_occupancy_nonzero_fraction=min_occupancy_nonzero_fraction,
            min_valid_position_fraction=min_valid_position_fraction,
            occupancy_bin_size_cm=occupancy_bin_size_cm,
            smoothing_window_samples=smoothing_window_samples,
        )
        session_rows.append(row)
        figure_rows.extend(figures)

    sessions = pd.DataFrame(session_rows, columns=SESSION_COLUMNS)
    animals = summarize_by_animal(sessions)
    gates = gate_summary(
        pairs=usable_pairs,
        sessions=sessions,
        min_occupancy_nonzero_fraction=min_occupancy_nonzero_fraction,
        min_valid_position_fraction=min_valid_position_fraction,
    )
    figures = pd.DataFrame(
        figure_rows,
        columns=["animal", "date", "track_session", "figure_type", "figure_path"],
    )

    sessions.to_csv(out / SESSION_OUTPUT, index=False)
    animals.to_csv(out / ANIMAL_OUTPUT, index=False)
    gates.to_csv(out / GATE_OUTPUT, index=False)
    figures.to_csv(out / FIGURE_OUTPUT, index=False)
    (out / SUMMARY_OUTPUT).write_text(build_markdown_summary(sessions, animals, gates), encoding="utf-8")
    return {
        "sessions": sessions,
        "animals": animals,
        "gates": gates,
        "figures": figures,
    }


def load_pairs(path: str | Path) -> pd.DataFrame:
    pairs = pd.read_csv(path)
    missing = sorted(REQUIRED_PAIR_COLUMNS.difference(pairs.columns))
    if missing:
        raise ValueError(f"pairs CSV is missing required columns: {missing}")
    return pairs


def summarize_pair(
    pair: pd.Series,
    *,
    dataset_root: Path,
    output_dir: Path,
    min_occupancy_nonzero_fraction: float,
    min_valid_position_fraction: float,
    occupancy_bin_size_cm: float,
    smoothing_window_samples: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    animal = str(pair["animal"]).upper()
    date = str(pair["date"])
    track_session = str(pair["track_session"])
    sleep_session = str(pair["sleepPOST_session"])
    track_stem = _track_stem_path(dataset_root, animal, date, track_session)
    pos_path = track_stem.with_suffix(".pos")
    orientation_rule = "inferred_occupied_bin_diameter"
    reversal_applied = _r2142_reversal_applied(animal, str(pair.get("hippocampal_tetrodes", "")))
    reasons: list[str] = []
    figures: list[dict[str, object]] = []

    try:
        if not pos_path.is_file():
            raise FileNotFoundError(f"missing Track1 .pos file: {pos_path}")
        linearize_pos_file(
            pos_path,
            output_dir,
            smoothing_window_samples=smoothing_window_samples,
            occupancy_bin_size_cm=occupancy_bin_size_cm,
        )
        linearized = pd.read_csv(output_dir / "linearized_position.csv")
        diagnostics = pd.read_csv(output_dir / "linearization_diagnostics.csv")
        geometry = json.loads((output_dir / "track_geometry.json").read_text(encoding="utf-8"))
        position = read_axona_pos(pos_path)
        projection_error = recompute_projection_error(
            position,
            np.asarray(geometry["centerline_points_cm"], dtype=float),
            smoothing_window_samples=smoothing_window_samples,
        )
        n_spikes, n_units, spike_overlap = track_spike_summary(
            track_stem,
            _parse_tetrodes(str(pair.get("hippocampal_tetrodes", ""))),
            float(linearized["time_s"].min()) if not linearized.empty else np.nan,
            float(linearized["time_s"].max()) if not linearized.empty else np.nan,
        )
        row = session_qc_row(
            animal=animal,
            date=date,
            track_session=track_session,
            sleep_session=sleep_session,
            linearized=linearized,
            diagnostics=diagnostics,
            projection_error=projection_error,
            n_spikes=n_spikes,
            n_units=n_units,
            spike_overlap=spike_overlap,
            orientation_rule=orientation_rule,
            reversal_applied=reversal_applied,
            min_occupancy_nonzero_fraction=min_occupancy_nonzero_fraction,
            min_valid_position_fraction=min_valid_position_fraction,
            reasons=reasons,
        )
        figures = write_figures(
            animal=animal,
            date=date,
            track_session=track_session,
            output_dir=output_dir,
            linearized=linearized,
            diagnostics=diagnostics,
            projection_error=projection_error,
            centerline=np.asarray(geometry["centerline_points_cm"], dtype=float),
        )
    except Exception as exc:  # noqa: BLE001 - QC should keep processing remaining sessions.
        reasons.append(type(exc).__name__ + ":" + str(exc))
        row = failed_session_row(
            animal=animal,
            date=date,
            track_session=track_session,
            sleep_session=sleep_session,
            orientation_rule=orientation_rule,
            reversal_applied=reversal_applied,
            reasons=reasons,
        )
    return row, figures


def recompute_projection_error(position, centerline: np.ndarray, *, smoothing_window_samples: int) -> np.ndarray:
    xy_raw = np.column_stack([position.x_cm, position.y_cm])
    xy = smooth_positions(xy_raw, position.valid, window_samples=smoothing_window_samples)
    _linear, projection_error = project_points_to_centerline(xy, position.valid, centerline)
    return projection_error


def session_qc_row(
    *,
    animal: str,
    date: str,
    track_session: str,
    sleep_session: str,
    linearized: pd.DataFrame,
    diagnostics: pd.DataFrame,
    projection_error: np.ndarray,
    n_spikes: int,
    n_units: int,
    spike_overlap: float,
    orientation_rule: str,
    reversal_applied: bool,
    min_occupancy_nonzero_fraction: float,
    min_valid_position_fraction: float,
    reasons: list[str],
) -> dict[str, object]:
    valid = linearized["valid_position"].map(_as_bool) if "valid_position" in linearized else pd.Series(dtype=bool)
    linear = pd.to_numeric(linearized.get("linear_position_cm", pd.Series(dtype=float)), errors="coerce")
    speed = pd.to_numeric(linearized.get("speed_cm_s", pd.Series(dtype=float)), errors="coerce")
    finite_linear = linear[valid & np.isfinite(linear)]
    finite_speed = speed[valid & np.isfinite(speed)]
    finite_error = np.asarray(projection_error, dtype=float)
    finite_error = finite_error[np.isfinite(finite_error)]
    occupancy = diagnostics[diagnostics["metric"].astype(str).eq("occupancy_by_linear_bin")]
    occupancy_values = pd.to_numeric(occupancy.get("value", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    occupancy_nonzero_bins = int((occupancy_values > 0.0).sum())
    occupancy_nonzero_fraction = float(occupancy_nonzero_bins / len(occupancy_values)) if len(occupancy_values) else 0.0
    valid_position_fraction = float(valid.mean()) if len(valid) else 0.0
    position_duration = _position_duration(linearized)
    if len(linearized) <= 0:
        reasons.append("no_position_samples")
    if finite_linear.empty:
        reasons.append("no_finite_linearized_positions")
    if not np.isfinite(_diag_value(diagnostics, "track_length_cm")) or _diag_value(diagnostics, "track_length_cm") <= 0.0:
        reasons.append("nonpositive_track_length")
    if not finite_linear.empty and float(finite_linear.max() - finite_linear.min()) <= 0.0:
        reasons.append("nonpositive_linearized_position_span")
    if valid_position_fraction < min_valid_position_fraction:
        reasons.append("low_valid_position_fraction")
    if occupancy_nonzero_fraction < min_occupancy_nonzero_fraction:
        reasons.append("low_occupancy_coverage")
    if n_spikes <= 0 or n_units <= 0:
        reasons.append("no_track1_spikes")
    if not np.isfinite(spike_overlap) or spike_overlap <= 0.0:
        reasons.append("no_spike_position_temporal_overlap")
    if animal == "R2142" and not reversal_applied:
        reasons.append("r2142_reversal_not_applied")

    return {
        "animal": animal,
        "date": date,
        "track_session": track_session,
        "sleeppost_session": sleep_session,
        "n_position_samples": int(len(linearized)),
        "position_duration_s": position_duration,
        "valid_position_fraction": valid_position_fraction,
        "track_length_cm": _diag_value(diagnostics, "track_length_cm"),
        "linearized_position_min_cm": float(finite_linear.min()) if not finite_linear.empty else np.nan,
        "linearized_position_max_cm": float(finite_linear.max()) if not finite_linear.empty else np.nan,
        "linearized_position_span_cm": float(finite_linear.max() - finite_linear.min()) if not finite_linear.empty else np.nan,
        "occupancy_nonzero_bins": occupancy_nonzero_bins,
        "occupancy_nonzero_fraction": occupancy_nonzero_fraction,
        "median_off_track_distance_cm": float(np.nanmedian(finite_error)) if finite_error.size else np.nan,
        "p95_off_track_distance_cm": float(np.nanpercentile(finite_error, 95.0)) if finite_error.size else np.nan,
        "median_speed_cm_s": float(np.nanmedian(finite_speed)) if not finite_speed.empty else np.nan,
        "p95_speed_cm_s": float(np.nanpercentile(finite_speed, 95.0)) if not finite_speed.empty else np.nan,
        "n_spikes_track1": int(n_spikes),
        "n_units_track1": int(n_units),
        "track_spike_position_overlap_s": float(spike_overlap) if np.isfinite(spike_overlap) else 0.0,
        "orientation_rule": orientation_rule,
        "reversal_applied": bool(reversal_applied),
        "linearization_status": "pass" if not reasons else "fail",
        "exclusion_reason": ";".join(reasons),
    }


def failed_session_row(
    *,
    animal: str,
    date: str,
    track_session: str,
    sleep_session: str,
    orientation_rule: str,
    reversal_applied: bool,
    reasons: list[str],
) -> dict[str, object]:
    row = {column: np.nan for column in SESSION_COLUMNS}
    row.update(
        {
            "animal": animal,
            "date": date,
            "track_session": track_session,
            "sleeppost_session": sleep_session,
            "n_position_samples": 0,
            "n_spikes_track1": 0,
            "n_units_track1": 0,
            "track_spike_position_overlap_s": 0.0,
            "orientation_rule": orientation_rule,
            "reversal_applied": bool(reversal_applied),
            "linearization_status": "fail",
            "exclusion_reason": ";".join(reasons),
        }
    )
    return row


def track_spike_summary(track_stem: Path, tetrodes: Sequence[int], position_start_s: float, position_end_s: float) -> tuple[int, int, float]:
    spike_times: list[float] = []
    units: set[tuple[int, int]] = set()
    for tetrode in tetrodes:
        raw_path = track_stem.with_suffix(f".{int(tetrode)}")
        cut_path = track_stem.parent / f"{track_stem.name}_{int(tetrode)}.cut"
        if not raw_path.is_file() or not cut_path.is_file():
            continue
        cut = read_axona_cut(cut_path, tetrode_path=raw_path)
        if cut.spike_times_s is None:
            continue
        labels = np.asarray(cut.labels, dtype=int)
        times = np.asarray(cut.spike_times_s, dtype=float)
        keep = labels > 0
        spike_times.extend(times[keep].tolist())
        units.update((int(tetrode), int(label)) for label in np.unique(labels[keep]) if int(label) > 0)
    if not spike_times:
        return 0, 0, 0.0
    times = np.asarray(spike_times, dtype=float)
    overlap = _interval_overlap(position_start_s, position_end_s, float(np.nanmin(times)), float(np.nanmax(times)))
    return int(times.shape[0]), int(len(units)), float(overlap)


def summarize_by_animal(sessions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "animal",
        "track1_sessions",
        "qc_pass_sessions",
        "median_valid_position_fraction",
        "median_track_length_cm",
        "median_occupancy_nonzero_fraction",
        "total_track1_spikes",
        "total_track1_units",
        "animal_retained_after_linearization",
    ]
    if sessions.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for animal, group in sessions.groupby("animal", sort=True):
        pass_rows = group[group["linearization_status"].astype(str).eq("pass")]
        rows.append(
            {
                "animal": animal,
                "track1_sessions": int(len(group)),
                "qc_pass_sessions": int(len(pass_rows)),
                "median_valid_position_fraction": _median(group, "valid_position_fraction"),
                "median_track_length_cm": _median(group, "track_length_cm"),
                "median_occupancy_nonzero_fraction": _median(group, "occupancy_nonzero_fraction"),
                "total_track1_spikes": int(pd.to_numeric(group["n_spikes_track1"], errors="coerce").fillna(0).sum()),
                "total_track1_units": int(pd.to_numeric(group["n_units_track1"], errors="coerce").fillna(0).sum()),
                "animal_retained_after_linearization": bool(len(pass_rows) > 0),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def gate_summary(
    *,
    pairs: pd.DataFrame,
    sessions: pd.DataFrame,
    min_occupancy_nonzero_fraction: float,
    min_valid_position_fraction: float,
) -> pd.DataFrame:
    expected_pairs = int(len(pairs))
    pass_rows = sessions[sessions["linearization_status"].astype(str).eq("pass")] if not sessions.empty else sessions
    input_animals = set(pairs["animal"].astype(str).str.upper()) if not pairs.empty else set()
    retained_animals = set(pass_rows["animal"].astype(str).str.upper()) if not pass_rows.empty else set()
    r2142_rows = sessions[sessions["animal"].astype(str).str.upper().eq("R2142")] if not sessions.empty else sessions
    gates = [
        _gate(
            "all_usable_pairs_processed",
            len(sessions) == expected_pairs,
            f"sessions={len(sessions)}; usable_pairs={expected_pairs}",
            "one QC row for every usable manifest pair",
            "The linearization QC should cover the full manifest-level usable set.",
        ),
        _gate(
            "track1_position_samples_present",
            expected_pairs > 0 and _all_numeric_positive(sessions, "n_position_samples"),
            f"positive={_count_numeric_positive(sessions, 'n_position_samples')}/{expected_pairs}",
            "all usable pairs have Track1 position samples",
            "Position samples are required for any 1D coordinate system.",
        ),
        _gate(
            "finite_linearized_positions_present",
            expected_pairs > 0 and _all_numeric_positive(sessions, "valid_position_fraction"),
            f"positive={_count_numeric_positive(sessions, 'valid_position_fraction')}/{expected_pairs}",
            "all usable pairs have finite linearized positions",
            "At least some finite linearized positions must exist for every Track1 session.",
        ),
        _gate(
            "linearized_track_span_positive",
            expected_pairs > 0
            and _all_numeric_positive(sessions, "track_length_cm")
            and _all_numeric_positive(sessions, "linearized_position_span_cm"),
            (
                f"positive_track_length={_count_numeric_positive(sessions, 'track_length_cm')}/{expected_pairs}; "
                f"positive_span={_count_numeric_positive(sessions, 'linearized_position_span_cm')}/{expected_pairs}"
            ),
            "all usable pairs have positive track length and positive linearized span",
            "The inferred 1D coordinate must not collapse to a degenerate point or segment.",
        ),
        _gate(
            "valid_position_fraction_acceptable",
            expected_pairs > 0 and _all_numeric_at_least(sessions, "valid_position_fraction", min_valid_position_fraction),
            f"passing={_count_numeric_at_least(sessions, 'valid_position_fraction', min_valid_position_fraction)}/{expected_pairs}",
            f"all valid_position_fraction >= {min_valid_position_fraction:g}",
            "This catches sessions where the coordinate is technically finite but mostly unusable.",
        ),
        _gate(
            "spike_position_temporal_overlap_present",
            expected_pairs > 0 and _all_numeric_positive(sessions, "track_spike_position_overlap_s"),
            f"positive={_count_numeric_positive(sessions, 'track_spike_position_overlap_s')}/{expected_pairs}",
            "all usable pairs have Track1 spike/position temporal overlap",
            "Encoding spikes need to overlap the Track1 position epoch.",
        ),
        _gate(
            "meaningful_track_occupancy",
            expected_pairs > 0 and _all_numeric_at_least(sessions, "occupancy_nonzero_fraction", min_occupancy_nonzero_fraction),
            f"passing={_count_numeric_at_least(sessions, 'occupancy_nonzero_fraction', min_occupancy_nonzero_fraction)}/{expected_pairs}",
            f"all occupancy_nonzero_fraction >= {min_occupancy_nonzero_fraction:g}",
            "Occupancy should cover a meaningful fraction of the inferred linear track.",
        ),
        _gate(
            "no_animal_lost_after_linearization",
            bool(input_animals) and input_animals == retained_animals,
            f"input_animals={len(input_animals)}; retained_animals={len(retained_animals)}",
            "every manifest-level animal has at least one passing linearization QC row",
            "The geometry QC should not silently collapse to one animal.",
        ),
        _gate(
            "r2142_reversal_explicit",
            r2142_rows.empty or bool(r2142_rows["reversal_applied"].map(_as_bool).all()),
            f"r2142_rows={len(r2142_rows)}; reversal_applied={bool(r2142_rows['reversal_applied'].map(_as_bool).all()) if not r2142_rows.empty else 'not_present'}",
            "R2142 rows explicitly carry reversal_applied=true",
            "The known R2142 hippocampus/MEC tetrode reversal remains visible in QC.",
        ),
    ]
    return pd.DataFrame(gates)


def write_figures(
    *,
    animal: str,
    date: str,
    track_session: str,
    output_dir: Path,
    linearized: pd.DataFrame,
    diagnostics: pd.DataFrame,
    projection_error: np.ndarray,
    centerline: np.ndarray,
) -> list[dict[str, object]]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    rows.append(_write_2d_position_figure(figures_dir, animal, date, track_session, linearized, centerline))
    rows.append(_write_linear_position_figure(figures_dir, animal, date, track_session, linearized))
    rows.append(_write_occupancy_figure(figures_dir, animal, date, track_session, diagnostics))
    rows.append(_write_histogram_figure(figures_dir, animal, date, track_session, projection_error, "off_track_distance_histogram", "Off-track distance (cm)"))
    speed = pd.to_numeric(linearized.get("speed_cm_s", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    rows.append(_write_histogram_figure(figures_dir, animal, date, track_session, speed, "speed_histogram", "Speed (cm/s)"))
    return rows


def _write_2d_position_figure(figures_dir: Path, animal: str, date: str, track_session: str, linearized: pd.DataFrame, centerline: np.ndarray) -> dict[str, object]:
    path = figures_dir / "position_2d_centerline.png"
    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    ax.plot(linearized["x_cm"], linearized["y_cm"], ".", ms=1.5, alpha=0.4, label="position")
    ax.plot(centerline[:, 0], centerline[:, 1], "-", lw=2.0, label="centerline")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_title(f"{animal} {date} Track1")
    ax.axis("equal")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return _figure_row(animal, date, track_session, "position_2d_centerline", path)


def _write_linear_position_figure(figures_dir: Path, animal: str, date: str, track_session: str, linearized: pd.DataFrame) -> dict[str, object]:
    path = figures_dir / "linear_position_over_time.png"
    fig, ax = plt.subplots(figsize=(6, 3), dpi=120)
    ax.plot(linearized["time_s"], linearized["linear_position_cm"], "-", lw=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("linear position (cm)")
    ax.set_title(f"{animal} {date} linearized Track1")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return _figure_row(animal, date, track_session, "linear_position_over_time", path)


def _write_occupancy_figure(figures_dir: Path, animal: str, date: str, track_session: str, diagnostics: pd.DataFrame) -> dict[str, object]:
    path = figures_dir / "linear_occupancy.png"
    occupancy = diagnostics[diagnostics["metric"].astype(str).eq("occupancy_by_linear_bin")]
    fig, ax = plt.subplots(figsize=(6, 3), dpi=120)
    if not occupancy.empty:
        starts = pd.to_numeric(occupancy["bin_start_cm"], errors="coerce")
        values = pd.to_numeric(occupancy["value"], errors="coerce")
        width = float(np.nanmedian(np.diff(starts))) if len(starts) > 1 else 5.0
        ax.bar(starts, values, width=width, align="edge")
    ax.set_xlabel("linear position (cm)")
    ax.set_ylabel("occupancy (s)")
    ax.set_title(f"{animal} {date} occupancy")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return _figure_row(animal, date, track_session, "linear_occupancy", path)


def _write_histogram_figure(figures_dir: Path, animal: str, date: str, track_session: str, values: np.ndarray, figure_type: str, xlabel: str) -> dict[str, object]:
    path = figures_dir / f"{figure_type}.png"
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    fig, ax = plt.subplots(figsize=(4, 3), dpi=120)
    if finite.size:
        ax.hist(finite, bins=40)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("samples")
    ax.set_title(f"{animal} {date}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return _figure_row(animal, date, track_session, figure_type, path)


def build_markdown_summary(sessions: pd.DataFrame, animals: pd.DataFrame, gates: pd.DataFrame) -> str:
    pass_sessions = int(sessions["linearization_status"].astype(str).eq("pass").sum()) if not sessions.empty else 0
    total_sessions = int(len(sessions))
    pass_animals = int(animals["animal_retained_after_linearization"].map(_as_bool).sum()) if not animals.empty else 0
    total_animals = int(len(animals))
    gate_passes = int(gates["passed"].map(_as_bool).sum()) if not gates.empty else 0
    lines = [
        "# Olafsdottir Track1 Linearization QC Summary",
        "",
        "This is a geometry/coverage checkpoint only. It does not compare 1D evidence with the 2D paper result.",
        "",
        "## Overview",
        "",
        _markdown_table(
            ["Metric", "Value"],
            [
                ("Track1 sessions processed", total_sessions),
                ("Track1 sessions passing QC", pass_sessions),
                ("Animals processed", total_animals),
                ("Animals retained after linearization", pass_animals),
                ("Readiness gates passed", f"{gate_passes}/{len(gates)}"),
            ],
        ),
        "",
        "## Gate Summary",
        "",
        _markdown_table(["Gate", "Status", "Value"], gates[["gate", "status", "value"]].itertuples(index=False, name=None)),
        "",
        "## Animal Summary",
        "",
        _markdown_table(["Animal", "Sessions", "Passing", "Median valid position", "Median occupancy fraction"], _animal_summary_rows(animals)),
        "",
    ]
    return "\n".join(lines)


def _track_stem_path(dataset_root: Path, animal: str, date: str, track_session: str) -> Path:
    return dataset_root / animal.lower() / date / track_session


def _parse_tetrodes(raw: str) -> tuple[int, ...]:
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


def _r2142_reversal_applied(animal: str, hpc_tetrodes: str) -> bool:
    return animal.upper() == "R2142" and _parse_tetrodes(hpc_tetrodes) == tuple(range(1, 9))


def _position_duration(linearized: pd.DataFrame) -> float:
    if "time_s" not in linearized or linearized.empty:
        return np.nan
    times = pd.to_numeric(linearized["time_s"], errors="coerce").dropna()
    if times.empty:
        return np.nan
    return float(times.max() - times.min())


def _diag_value(diagnostics: pd.DataFrame, metric: str) -> float:
    rows = diagnostics[diagnostics["metric"].astype(str).eq(metric)]
    if rows.empty:
        return np.nan
    value = pd.to_numeric(rows["value"], errors="coerce").dropna()
    return float(value.iloc[0]) if not value.empty else np.nan


def _interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    values = np.asarray([a_start, a_end, b_start, b_end], dtype=float)
    if not np.all(np.isfinite(values)):
        return 0.0
    return max(0.0, min(float(a_end), float(b_end)) - max(float(a_start), float(b_start)))


def _median(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.median()) if not values.empty else np.nan


def _all_numeric_positive(frame: pd.DataFrame, column: str) -> bool:
    if frame.empty or column not in frame:
        return False
    values = pd.to_numeric(frame[column], errors="coerce")
    return bool(values.notna().all() and (values > 0).all())


def _count_numeric_positive(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame:
        return 0
    return int((pd.to_numeric(frame[column], errors="coerce").fillna(0.0) > 0.0).sum())


def _all_numeric_at_least(frame: pd.DataFrame, column: str, threshold: float) -> bool:
    if frame.empty or column not in frame:
        return False
    values = pd.to_numeric(frame[column], errors="coerce")
    return bool(values.notna().all() and (values >= float(threshold)).all())


def _count_numeric_at_least(frame: pd.DataFrame, column: str, threshold: float) -> int:
    if frame.empty or column not in frame:
        return 0
    return int((pd.to_numeric(frame[column], errors="coerce").fillna(-np.inf) >= float(threshold)).sum())


def _gate(gate: str, passed: bool, value: str, requirement: str, note: str) -> dict[str, object]:
    return {
        "gate": gate,
        "passed": bool(passed),
        "status": "pass" if passed else "fail",
        "value": value,
        "requirement": requirement,
        "note": note,
    }


def _figure_row(animal: str, date: str, track_session: str, figure_type: str, path: Path) -> dict[str, object]:
    return {
        "animal": animal,
        "date": date,
        "track_session": track_session,
        "figure_type": figure_type,
        "figure_path": str(path),
    }


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _animal_summary_rows(animals: pd.DataFrame) -> list[tuple[object, ...]]:
    if animals.empty:
        return []
    rows: list[tuple[object, ...]] = []
    for row in animals.itertuples(index=False):
        rows.append(
            (
                row.animal,
                int(row.track1_sessions),
                int(row.qc_pass_sessions),
                f"{float(row.median_valid_position_fraction):.3g}" if np.isfinite(float(row.median_valid_position_fraction)) else "nan",
                f"{float(row.median_occupancy_nonzero_fraction):.3g}" if np.isfinite(float(row.median_occupancy_nonzero_fraction)) else "nan",
            )
        )
    return rows


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
    parser.add_argument("--output-dir", type=Path, default=Path("results/olafsdottir-linearization-qc"))
    parser.add_argument("--min-occupancy-nonzero-fraction", type=float, default=0.25)
    parser.add_argument("--min-valid-position-fraction", type=float, default=0.50)
    parser.add_argument("--occupancy-bin-size-cm", type=float, default=5.0)
    parser.add_argument("--smoothing-window-samples", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tables = run_linearization_qc(
        dataset_root=args.dataset_root,
        pairs_csv=args.pairs_csv,
        output_dir=args.output_dir,
        min_occupancy_nonzero_fraction=args.min_occupancy_nonzero_fraction,
        min_valid_position_fraction=args.min_valid_position_fraction,
        occupancy_bin_size_cm=args.occupancy_bin_size_cm,
        smoothing_window_samples=args.smoothing_window_samples,
    )
    print(tables["sessions"].to_string(index=False))
    print()
    print(tables["gates"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
