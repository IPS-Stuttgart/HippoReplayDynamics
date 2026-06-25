#!/usr/bin/env python3
"""Plot diagnostic panels for selected Olafsdottir 1D SleepPOST events.

This script is intentionally an inspection helper, not a scorer. It reads existing
Olafsdottir 1D evidence outputs, selects a small set of interesting positive and
negative events, and renders place-field-sorted spike rasters, simple decoded
posterior heatmaps, model-evidence bars, and per-event markdown summaries.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _provenance import build_script_provenance
import score_olafsdottir_1d_sleeppost_evidence as scorer


DECISION_INPUT = "olafsdottir_1d_sleep_model_claim_decisions.csv"
MANIFEST_INPUT = "olafsdottir_1d_sleep_manifest.json"
OUTPUT_MANIFEST = "olafsdottir_1d_event_diagnostic_manifest.csv"
OUTPUT_RUN_MANIFEST = "olafsdottir_1d_event_diagnostic_run_manifest.json"

DEFAULT_EVIDENCE_DIRS = {
    "balanced_debug": "results/olafsdottir-1d-sleeppost-evidence-pilot20-debug",
    "high_information_debug": "results/olafsdottir-1d-sleeppost-evidence-pilot20-high-information-debug",
    "holdout19_debug": "results/olafsdottir-1d-sleeppost-evidence-pilot20-high-information-holdout19-debug",
}

PAIR_KEYS = ["animal", "date", "track1_session", "sleeppost_session"]
EVENT_ID_KEYS = [*PAIR_KEYS, "event_id", "start_time_s", "end_time_s"]
MODEL_LOGZ = {
    "stationary": "logZ_stationary",
    "diffusion": "logZ_diffusion",
    "fragmented": "logZ_fragmented",
    "first_order_imm": "logZ_first_order_imm",
}
SELECTION_COLUMNS = [
    "event_slug",
    "selection_reasons",
    "source_tier_labels",
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
    "best_model",
    "delta_best_trajectory_minus_stationary",
    "delta_imm_minus_fragmented",
    "trajectory_family_claim",
    "imm_clean_vs_fragmented_claim",
    "raster_path",
    "posterior_heatmap_path",
    "model_evidence_bars_path",
    "summary_path",
]


def run_event_diagnostics(
    *,
    dataset_root: str | Path,
    pairs_csv: str | Path,
    linearization_qc: str | Path,
    evidence_dirs: Sequence[str],
    output_dir: str | Path,
    max_events: int = 12,
    time_bin_s: float | None = None,
    position_bin_size_cm: float | None = None,
    min_unit_spikes: int | None = None,
    min_encoding_units: int | None = None,
    smoothing_bins: int | None = None,
    skip_data_panels: bool = False,
) -> pd.DataFrame:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sources = load_evidence_sources(evidence_dirs)
    all_decisions = pd.concat([source["decisions"] for source in sources], ignore_index=True)
    selected = select_diagnostic_events(all_decisions, max_events=max_events)
    if selected.empty:
        selected = pd.DataFrame(columns=SELECTION_COLUMNS)
        selected.to_csv(out / OUTPUT_MANIFEST, index=False)
        return selected

    pairs = scorer.load_pairs(pairs_csv)
    linearization_root = Path(linearization_qc).resolve().parent
    cache: dict[tuple[str, str, str, str], tuple[scorer.PlaceFieldModel, scorer.SessionSpikes]] = {}
    rendered_rows: list[dict[str, object]] = []

    for _, event in selected.iterrows():
        event = event.copy()
        slug = str(event["event_slug"])
        raster_path = out / f"event_{slug}_raster_placefield_sorted.png"
        posterior_path = out / f"event_{slug}_posterior_heatmap.png"
        bars_path = out / f"event_{slug}_model_evidence_bars.png"
        summary_path = out / f"event_{slug}_summary.md"
        place_fields = None
        counts = None
        sleep_spikes = None
        render_error = ""
        if not skip_data_panels:
            try:
                place_fields, sleep_spikes = load_pair_data(
                    event,
                    dataset_root=Path(dataset_root),
                    pairs=pairs,
                    linearization_root=linearization_root,
                    cache=cache,
                    position_bin_size_cm=first_non_none(position_bin_size_cm, event.get("position_bin_size_cm"), 5.0),
                    min_unit_spikes=int(first_non_none(min_unit_spikes, event.get("min_unit_spikes"), 5)),
                    min_encoding_units=int(first_non_none(min_encoding_units, event.get("min_encoding_units"), 1)),
                    smoothing_bins=int(first_non_none(smoothing_bins, event.get("smoothing_bins"), 1)),
                )
                dt = float(first_non_none(time_bin_s, event.get("time_bin_s"), 0.02))
                counts = scorer.event_count_matrix(
                    sleep_spikes,
                    unit_ids=place_fields.unit_ids,
                    start_s=float(event["start_time_s"]),
                    end_s=float(event["end_time_s"]),
                    time_bin_s=dt,
                )
                write_raster_panel(event, place_fields, sleep_spikes, raster_path)
                write_posterior_heatmap(event, place_fields, counts, dt, posterior_path)
            except Exception as exc:  # noqa: BLE001 - diagnostic artifact records rendering failures.
                render_error = f"{type(exc).__name__}: {exc}"
                write_placeholder_figure(raster_path, "Raster unavailable", render_error)
                write_placeholder_figure(posterior_path, "Posterior unavailable", render_error)
        else:
            write_placeholder_figure(raster_path, "Raster skipped", "--skip-data-panels was used")
            write_placeholder_figure(posterior_path, "Posterior skipped", "--skip-data-panels was used")
        write_model_evidence_bars(event, bars_path)
        write_event_summary(event, summary_path, render_error=render_error)
        row = event.to_dict()
        row.update(
            {
                "raster_path": str(raster_path),
                "posterior_heatmap_path": str(posterior_path),
                "model_evidence_bars_path": str(bars_path),
                "summary_path": str(summary_path),
                "render_error": render_error,
            }
        )
        rendered_rows.append(row)

    rendered = pd.DataFrame(rendered_rows)
    for col in SELECTION_COLUMNS:
        if col not in rendered.columns:
            rendered[col] = ""
    rendered.to_csv(out / OUTPUT_MANIFEST, index=False)
    run_manifest = {
        "analysis": "olafsdottir_1d_event_diagnostic_panels",
        "dataset_root": str(dataset_root),
        "pairs_csv": str(pairs_csv),
        "linearization_qc": str(linearization_qc),
        "evidence_dirs": list(evidence_dirs),
        "output_dir": str(out),
        "max_events": int(max_events),
        "skip_data_panels": bool(skip_data_panels),
        "selected_events": int(len(rendered)),
        **build_script_provenance(
            input_paths={
                "pairs_csv": pairs_csv,
                "linearization_qc": linearization_qc,
                **{f"evidence_dir_{idx}": Path(parse_labeled_path(raw)[1]) / DECISION_INPUT for idx, raw in enumerate(evidence_dirs)},
            }
        ),
    }
    (out / OUTPUT_RUN_MANIFEST).write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rendered


def load_evidence_sources(raw_sources: Sequence[str]) -> list[dict[str, object]]:
    if not raw_sources:
        raw_sources = [f"{label}={path}" for label, path in DEFAULT_EVIDENCE_DIRS.items() if Path(path).is_dir()]
    sources: list[dict[str, object]] = []
    for raw in raw_sources:
        label, path = parse_labeled_path(raw)
        root = Path(path)
        decisions_path = root / DECISION_INPUT
        if not decisions_path.is_file():
            raise FileNotFoundError(decisions_path)
        decisions = pd.read_csv(decisions_path)
        manifest = read_json(root / MANIFEST_INPUT)
        decisions["source_tier_label"] = label
        decisions["source_evidence_dir"] = str(root)
        for key, default in {
            "time_bin_s": 0.02,
            "position_bin_size_cm": 5.0,
            "min_unit_spikes": 5,
            "min_encoding_units": 1,
            "smoothing_bins": 1,
        }.items():
            decisions[key] = manifest.get(key, default)
        sources.append({"label": label, "path": root, "decisions": decisions, "manifest": manifest})
    if not sources:
        raise ValueError("no evidence directories provided or found")
    return sources


def parse_labeled_path(raw: str) -> tuple[str, str]:
    if "=" in str(raw):
        label, path = str(raw).split("=", 1)
        return safe_token(label), path
    path = str(raw)
    return safe_token(Path(path).name), path


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def select_diagnostic_events(decisions: pd.DataFrame, *, max_events: int) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame(columns=SELECTION_COLUMNS)
    data = decisions.copy()
    for column in ["delta_best_trajectory_minus_stationary", "delta_imm_minus_fragmented", "start_time_s", "end_time_s"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data["event_key"] = data.apply(event_key, axis=1)
    rows: dict[str, pd.Series] = {}
    reasons: dict[str, list[str]] = {}

    def add(row: pd.Series, reason: str) -> None:
        key = str(row["event_key"])
        if key not in rows:
            rows[key] = row.copy()
            reasons[key] = []
        if reason not in reasons[key]:
            reasons[key].append(reason)

    for _, row in data.sort_values("delta_best_trajectory_minus_stationary", ascending=False, kind="mergesort").head(4).iterrows():
        add(row, "top_trajectory_minus_stationary")
    for _, row in data.sort_values("delta_imm_minus_fragmented", ascending=False, kind="mergesort").head(5).iterrows():
        add(row, "top_imm_minus_fragmented")
    for _, row in data.sort_values("delta_best_trajectory_minus_stationary", ascending=True, kind="mergesort").head(4).iterrows():
        add(row, "most_stationary_favored")

    positive = data[
        data["trajectory_family_claim"].astype(str).eq("trajectory_confident")
        | data["imm_clean_vs_fragmented_claim"].map(as_bool)
    ].copy()
    for _, row in positive[positive["animal"].astype(str).eq("R2192")].sort_values(
        ["delta_best_trajectory_minus_stationary", "delta_imm_minus_fragmented"], ascending=False, kind="mergesort"
    ).head(4).iterrows():
        add(row, "R2192_positive")
    for _, row in positive[~positive["animal"].astype(str).eq("R2192")].sort_values(
        ["delta_best_trajectory_minus_stationary", "delta_imm_minus_fragmented"], ascending=False, kind="mergesort"
    ).head(4).iterrows():
        add(row, "representative_non_R2192_positive")
    negative = data[~data["event_key"].isin(rows.keys())].copy()
    if not negative.empty:
        negative = negative.sort_values("delta_best_trajectory_minus_stationary", ascending=True, kind="mergesort")
        for _, row in negative.head(4).iterrows():
            add(row, "representative_negative")

    selected_rows = []
    for key, row in rows.items():
        row = row.copy()
        row["selection_reasons"] = ";".join(reasons[key])
        selected_rows.append(row)
    selected = pd.DataFrame(selected_rows)
    if selected.empty:
        return pd.DataFrame(columns=SELECTION_COLUMNS)
    selected = collapse_duplicate_event_sources(selected, reasons)
    selected = selected.sort_values(
        ["selection_priority", "animal", "date", "sleeppost_session", "event_id"],
        kind="mergesort",
    ).head(int(max_events))
    selected["event_slug"] = selected.apply(make_event_slug, axis=1)
    return selected.reset_index(drop=True)


def collapse_duplicate_event_sources(selected: pd.DataFrame, reasons: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for key, group in selected.groupby("event_key", sort=False):
        group = group.copy()
        best = group.sort_values(
            ["delta_best_trajectory_minus_stationary", "delta_imm_minus_fragmented"], ascending=False, kind="mergesort"
        ).iloc[0].copy()
        best["selection_reasons"] = ";".join(reasons.get(str(key), []))
        best["source_tier_labels"] = ";".join(sorted({str(value) for value in group["source_tier_label"].dropna().unique()}))
        best["selection_priority"] = selection_priority(str(best["selection_reasons"]))
        rows.append(best)
    return pd.DataFrame(rows)


def selection_priority(reasons: str) -> int:
    order = [
        "top_trajectory_minus_stationary",
        "top_imm_minus_fragmented",
        "R2192_positive",
        "representative_non_R2192_positive",
        "most_stationary_favored",
        "representative_negative",
    ]
    parts = set(str(reasons).split(";"))
    for idx, reason in enumerate(order):
        if reason in parts:
            return idx
    return len(order)


def load_pair_data(
    event: pd.Series,
    *,
    dataset_root: Path,
    pairs: pd.DataFrame,
    linearization_root: Path,
    cache: dict[tuple[str, str, str, str], tuple[scorer.PlaceFieldModel, scorer.SessionSpikes]],
    position_bin_size_cm: float,
    min_unit_spikes: int,
    min_encoding_units: int,
    smoothing_bins: int,
) -> tuple[scorer.PlaceFieldModel, scorer.SessionSpikes]:
    animal = str(event["animal"])
    date = str(event["date"])
    track = str(event["track1_session"])
    sleep = str(event["sleeppost_session"])
    key = (animal, date, track, sleep)
    if key in cache:
        return cache[key]
    pair = scorer.matching_pair(pairs, animal, date, track, sleep)
    if pair is None:
        raise ValueError(f"missing pair row for {key}")
    tetrodes = scorer.parse_tetrodes(str(pair["hippocampal_tetrodes"]))
    linearized = scorer.load_linearized_position(linearization_root=linearization_root, animal=animal, date=date)
    track_spikes = scorer.load_session_spikes(scorer.session_stem(dataset_root, animal, date, track), tetrodes)
    sleep_spikes = scorer.load_session_spikes(scorer.session_stem(dataset_root, animal, date, sleep), tetrodes)
    place_fields = scorer.fit_place_field_model(
        linearized=linearized,
        spikes=track_spikes,
        position_bin_size_cm=float(position_bin_size_cm),
        min_unit_spikes=int(min_unit_spikes),
        min_encoding_units=int(min_encoding_units),
        smoothing_bins=int(smoothing_bins),
    )
    cache[key] = (place_fields, sleep_spikes)
    return place_fields, sleep_spikes


def write_raster_panel(event: pd.Series, place_fields: scorer.PlaceFieldModel, sleep_spikes: scorer.SessionSpikes, path: Path) -> None:
    start = float(event["start_time_s"])
    end = float(event["end_time_s"])
    unit_ids = np.asarray(place_fields.unit_ids, dtype=int)
    peaks = place_fields.bin_centers_cm[np.argmax(place_fields.rates_hz, axis=1)]
    order = np.argsort(peaks)
    sorted_units = unit_ids[order]
    y_by_unit = {int(unit): idx for idx, unit in enumerate(sorted_units)}
    keep = (sleep_spikes.spike_times_s >= start) & (sleep_spikes.spike_times_s <= end) & np.isin(sleep_spikes.unit_ids, sorted_units)
    times_ms = (sleep_spikes.spike_times_s[keep] - start) * 1000.0
    ys = np.asarray([y_by_unit.get(int(unit), -1) for unit in sleep_spikes.unit_ids[keep]], dtype=float)
    valid = ys >= 0
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ax.scatter(times_ms[valid], ys[valid], s=9, color="black", linewidths=0)
    ax.set_title(f"{event['animal']} {event['date']} event {event['event_id']} spike raster")
    ax.set_xlabel("time from event start (ms)")
    ax.set_ylabel("units sorted by Track1 place-field peak")
    ax.set_ylim(-1, max(len(sorted_units), 1))
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_posterior_heatmap(event: pd.Series, place_fields: scorer.PlaceFieldModel, counts: np.ndarray, dt_s: float, path: Path) -> None:
    if counts is None or counts.shape[0] == 0:
        write_placeholder_figure(path, "Posterior unavailable", "event has no time bins")
        return
    emissions = scorer.poisson_log_emissions(counts, place_fields.rates_hz, dt_s)
    log_post = emissions + np.log(np.maximum(place_fields.prior[None, :], 1e-12))
    log_post = log_post - scorer.logsumexp_matrix(log_post, axis=1)[:, None]
    posterior = np.exp(log_post)
    centers = np.asarray(place_fields.bin_centers_cm, dtype=float)
    map_path = centers[np.argmax(posterior, axis=1)]
    mean_path = posterior @ centers
    time_ms = (np.arange(posterior.shape[0]) + 0.5) * float(dt_s) * 1000.0
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    extent = [0, posterior.shape[0] * float(dt_s) * 1000.0, float(np.nanmin(centers)), float(np.nanmax(centers))]
    image = ax.imshow(posterior.T, aspect="auto", origin="lower", extent=extent, cmap="magma")
    ax.plot(time_ms, map_path, color="cyan", linewidth=1.3, label="MAP")
    ax.plot(time_ms, mean_path, color="white", linewidth=1.1, label="posterior mean")
    ax.set_title(f"{event['animal']} {event['date']} event {event['event_id']} decoded posterior")
    ax.set_xlabel("time from event start (ms)")
    ax.set_ylabel("linearized position (cm)")
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    fig.colorbar(image, ax=ax, label="posterior probability")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_model_evidence_bars(event: pd.Series, path: Path) -> None:
    values = []
    labels = []
    stationary = float(event.get("logZ_stationary", np.nan))
    for model, column in MODEL_LOGZ.items():
        value = float(event.get(column, np.nan))
        labels.append(model.replace("first_order_", "1st-order "))
        values.append(value - stationary if np.isfinite(stationary) and np.isfinite(value) else np.nan)
    fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
    colors = ["#777777", "#348ABD", "#A60628", "#467821"]
    ax.bar(labels, values, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(5.5, color="#444444", linestyle="--", linewidth=0.9, label="5.5 threshold")
    ax.axhline(-5.5, color="#444444", linestyle="--", linewidth=0.9)
    ax.set_ylabel("log evidence relative to stationary")
    ax.set_title(f"{event['animal']} {event['date']} event {event['event_id']} model evidence")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_placeholder_figure(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.5), constrained_layout=True)
    ax.text(0.5, 0.55, title, ha="center", va="center", fontsize=14)
    ax.text(0.5, 0.40, message, ha="center", va="center", wrap=True, fontsize=9)
    ax.set_axis_off()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_event_summary(event: pd.Series, path: Path, *, render_error: str = "") -> None:
    start_s = float(event.get("start_time_s", np.nan))
    end_s = float(event.get("end_time_s", np.nan))
    duration_ms = float(event.get("duration_ms", np.nan))
    delta_family = float(event.get("delta_best_trajectory_minus_stationary", np.nan))
    delta_imm = float(event.get("delta_imm_minus_fragmented", np.nan))
    source_tiers = event.get("source_tier_labels", event.get("source_tier_label", ""))
    lines = [
        f"# Olafsdottir Event {event['event_id']} Diagnostics\n\n",
        f"- selection reasons: {event.get('selection_reasons', '')}\n",
        f"- source tiers: {source_tiers}\n",
        f"- animal/date: {event['animal']} / {event['date']}\n",
        f"- Track1/SleepPOST: {event['track1_session']} / {event['sleeppost_session']}\n",
        f"- window: {start_s:.6g}-{end_s:.6g} s ({duration_ms:.6g} ms)\n",
        f"- spikes / active units: {event.get('n_spikes', '')} / {event.get('n_active_units', '')}\n",
        f"- best model: {event.get('best_model', '')}\n",
        f"- trajectory-family claim: {event.get('trajectory_family_claim', '')}\n",
        f"- IMM clean vs fragmented claim: {event.get('imm_clean_vs_fragmented_claim', '')}\n",
        f"- delta trajectory minus stationary: {delta_family:.6g}\n",
        f"- delta IMM minus fragmented: {delta_imm:.6g}\n",
    ]
    if render_error:
        lines.append(f"- render warning: {render_error}\n")
    path.write_text("".join(lines), encoding="utf-8")

def event_key(row: pd.Series) -> str:
    parts = []
    for column in EVENT_ID_KEYS:
        value = row.get(column, "")
        if column in {"start_time_s", "end_time_s"}:
            try:
                value = f"{float(value):.6f}"
            except Exception:  # noqa: BLE001
                value = str(value)
        parts.append(str(value))
    return "|".join(parts)


def make_event_slug(row: pd.Series) -> str:
    raw = f"{row['animal']}_{row['date']}_{row['sleeppost_session']}_event_{int(row['event_id'])}"
    return safe_token(raw)


def safe_token(value: object) -> str:
    text = str(value).strip().replace("/", "_")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unnamed"


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def first_non_none(*values: object) -> object:
    for value in values:
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            return value
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--pairs-csv", required=True)
    parser.add_argument("--linearization-qc", required=True)
    parser.add_argument("--evidence-dir", action="append", default=[], help="LABEL=PATH evidence directory. Defaults to known debug tiers if omitted.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-events", type=int, default=12)
    parser.add_argument("--time-bin-s", type=float, default=None)
    parser.add_argument("--position-bin-size-cm", type=float, default=None)
    parser.add_argument("--min-unit-spikes", type=int, default=None)
    parser.add_argument("--min-encoding-units", type=int, default=None)
    parser.add_argument("--smoothing-bins", type=int, default=None)
    parser.add_argument("--skip-data-panels", action="store_true", help="Only write selection manifest, model-evidence bars, placeholders, and summaries.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_event_diagnostics(
        dataset_root=args.dataset_root,
        pairs_csv=args.pairs_csv,
        linearization_qc=args.linearization_qc,
        evidence_dirs=args.evidence_dir,
        output_dir=args.output_dir,
        max_events=args.max_events,
        time_bin_s=args.time_bin_s,
        position_bin_size_cm=args.position_bin_size_cm,
        min_unit_spikes=args.min_unit_spikes,
        min_encoding_units=args.min_encoding_units,
        smoothing_bins=args.smoothing_bins,
        skip_data_panels=args.skip_data_panels,
    )


if __name__ == "__main__":
    main()
