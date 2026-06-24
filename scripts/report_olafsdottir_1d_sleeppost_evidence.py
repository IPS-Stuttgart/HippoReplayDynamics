#!/usr/bin/env python3
"""Summarize an existing Olafsdottir 1D SleepPOST evidence smoke run.

This reporter never rescored events. It consumes the CSV/JSON artifacts emitted
by ``score_olafsdottir_1d_sleeppost_evidence.py`` and produces compact debug
tables for deciding whether a pilot run is ready to scale, needs event/decoder
triage, or is technically blocked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _provenance import build_script_provenance


EVENT_MODEL_INPUT = "olafsdottir_1d_sleep_event_model_evidence.csv"
DECISION_INPUT = "olafsdottir_1d_sleep_model_claim_decisions.csv"
GATE_INPUT = "olafsdottir_1d_sleep_gate_summary.csv"
MANIFEST_INPUT = "olafsdottir_1d_sleep_manifest.json"

QUALITY_OUTPUT = "olafsdottir_1d_sleep_debug_quality_table.csv"
MODEL_RANK_OUTPUT = "olafsdottir_1d_sleep_model_rank_summary.csv"
TRAJECTORY_MARGIN_OUTPUT = "olafsdottir_1d_sleep_trajectory_margin_summary.csv"
IMM_FRAGMENTED_AUDIT_OUTPUT = "olafsdottir_1d_sleep_imm_fragmented_audit.csv"
PAIR_DEBUG_OUTPUT = "olafsdottir_1d_sleep_by_pair_debug_summary.csv"
ANIMAL_DEBUG_OUTPUT = "olafsdottir_1d_sleep_by_animal_debug_summary.csv"
REPORT_OUTPUT = "olafsdottir_1d_sleep_debug_report.md"

TRAJECTORY_MARGIN_FIGURE = "olafsdottir_1d_sleep_trajectory_minus_stationary.png"
IMM_FRAGMENTED_FIGURE = "olafsdottir_1d_sleep_imm_minus_fragmented.png"
MODEL_WINNER_FIGURE = "olafsdottir_1d_sleep_model_winner_counts.png"
ANIMAL_MARGIN_FIGURE = "olafsdottir_1d_sleep_by_animal_margins.png"

EVENT_KEYS = ["animal", "date", "track1_session", "sleeppost_session", "pilot_tier", "event_id"]
PAIR_KEYS = ["animal", "date", "track1_session", "sleeppost_session"]
MODEL_LOGZ_COLUMNS = ["logZ_stationary", "logZ_diffusion", "logZ_fragmented", "logZ_first_order_imm"]
QUALITY_COLUMNS = [
    "animal",
    "date",
    "track1_session",
    "sleeppost_session",
    "pilot_tier",
    "decoder_filter",
    "event_id",
    "duration_ms",
    "n_spikes",
    "n_active_units",
    "mean_speed_cm_s",
    "mean_mua_rate_hz",
    "peak_mua_rate_hz",
    "candidate_tier",
    "best_model",
    "runner_up_model",
    "best_minus_runner_up_log_evidence",
    "delta_best_trajectory_minus_stationary",
    "delta_imm_minus_fragmented",
    "trajectory_family_claim",
    "imm_clean_vs_fragmented_claim",
    "fragmented_claim",
    "brownian_diffusion_claim",
    "ambiguous_claim",
    "decoder_qc_passed",
    "linearization_qc_passed",
    "decoder_status",
    "decoder_qc_paper_ready",
    "decoder_qc_scoring_available",
    "encoding_units_passing_qc",
    "posterior_mean_error_cm_median",
    "map_error_cm_median",
    "posterior_coverage_fraction",
    *MODEL_LOGZ_COLUMNS,
]
CORRELATION_COLUMNS = [
    "n_spikes",
    "n_active_units",
    "duration_ms",
    "mean_mua_rate_hz",
    "posterior_mean_error_cm_median",
    "map_error_cm_median",
    "posterior_coverage_fraction",
]


def run_report(
    *,
    evidence_dir: str | Path,
    output_dir: str | Path | None = None,
    margin_threshold: float | None = None,
    write_figures: bool = True,
) -> dict[str, pd.DataFrame | dict[str, object]]:
    evidence_root = Path(evidence_dir)
    out = Path(output_dir) if output_dir is not None else evidence_root
    out.mkdir(parents=True, exist_ok=True)

    evidence = read_required_csv(evidence_root / EVENT_MODEL_INPUT)
    decisions = read_required_csv(evidence_root / DECISION_INPUT)
    gates = read_optional_csv(evidence_root / GATE_INPUT)
    manifest = read_optional_json(evidence_root / MANIFEST_INPUT)
    threshold = float(margin_threshold if margin_threshold is not None else manifest.get("margin_threshold", 5.5))

    decoder = read_manifest_table(manifest, "decoder_qc", evidence_root)
    pilot = read_manifest_table(manifest, "pilot_selection", evidence_root)
    quality = build_quality_table(decisions, decoder=decoder, pilot=pilot)
    model_rank = build_model_rank_summary(evidence, decisions, margin_threshold=threshold)
    trajectory_margin = build_margin_summary(quality, margin_column="delta_best_trajectory_minus_stationary", margin_threshold=threshold)
    imm_audit = build_imm_fragmented_audit(quality, margin_threshold=threshold)
    by_pair = build_group_debug_summary(quality, PAIR_KEYS, margin_threshold=threshold)
    by_animal = build_group_debug_summary(quality, ["animal"], margin_threshold=threshold)
    classification = classify_run(quality, gates, margin_threshold=threshold)

    quality.to_csv(out / QUALITY_OUTPUT, index=False)
    model_rank.to_csv(out / MODEL_RANK_OUTPUT, index=False)
    trajectory_margin.to_csv(out / TRAJECTORY_MARGIN_OUTPUT, index=False)
    imm_audit.to_csv(out / IMM_FRAGMENTED_AUDIT_OUTPUT, index=False)
    by_pair.to_csv(out / PAIR_DEBUG_OUTPUT, index=False)
    by_animal.to_csv(out / ANIMAL_DEBUG_OUTPUT, index=False)
    (out / REPORT_OUTPUT).write_text(
        build_markdown_report(
            quality=quality,
            model_rank=model_rank,
            trajectory_margin=trajectory_margin,
            imm_audit=imm_audit,
            by_pair=by_pair,
            by_animal=by_animal,
            gates=gates,
            manifest=manifest,
            classification=classification,
            margin_threshold=threshold,
        ),
        encoding="utf-8",
    )
    if write_figures:
        write_debug_figures(quality, out, margin_threshold=threshold)

    provenance = {
        "analysis": "olafsdottir_1d_sleeppost_evidence_debug_report",
        "evidence_dir": str(evidence_root),
        "output_dir": str(out),
        "margin_threshold": threshold,
        **classification,
        **build_script_provenance(
            input_paths={
                "event_model_evidence": evidence_root / EVENT_MODEL_INPUT,
                "model_claim_decisions": evidence_root / DECISION_INPUT,
                "gate_summary": evidence_root / GATE_INPUT,
                "manifest": evidence_root / MANIFEST_INPUT,
            }
        ),
    }
    (out / "olafsdottir_1d_sleep_debug_report_manifest.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "quality": quality,
        "model_rank": model_rank,
        "trajectory_margin": trajectory_margin,
        "imm_fragmented_audit": imm_audit,
        "by_pair": by_pair,
        "by_animal": by_animal,
        "classification": classification,
    }


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def read_optional_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_manifest_table(manifest: dict[str, object], key: str, evidence_dir: Path) -> pd.DataFrame:
    value = manifest.get(key)
    if not isinstance(value, str) or not value:
        return pd.DataFrame()
    path = resolve_input_path(value, evidence_dir)
    return pd.read_csv(path) if path is not None and path.is_file() else pd.DataFrame()


def resolve_input_path(value: str, evidence_dir: Path) -> Path | None:
    path = Path(value)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([Path.cwd() / path, evidence_dir / path, evidence_dir.parent / path])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def build_quality_table(decisions: pd.DataFrame, *, decoder: pd.DataFrame, pilot: pd.DataFrame) -> pd.DataFrame:
    quality = decisions.copy()
    if not decoder.empty:
        decoder_cols = [column for column in decoder.columns if column in PAIR_KEYS or column in QUALITY_COLUMNS]
        decoder_table = decoder[dedupe(decoder_cols)].drop_duplicates(PAIR_KEYS, keep="first")
        quality = quality.merge(decoder_table, on=PAIR_KEYS, how="left", suffixes=("", "_decoder"))
    if not pilot.empty:
        pilot = filter_pilot_metadata_to_scored_tier(pilot, quality)
        pilot_cols = [
            column
            for column in pilot.columns
            if column in {"animal", "date", "track1_session", "sleeppost_session", "event_id"}
            or column not in quality.columns
        ]
        event_merge_keys = ["animal", "date", "track1_session", "sleeppost_session", "event_id"]
        if set(event_merge_keys).issubset(pilot_cols):
            pilot_table = pilot[dedupe(pilot_cols)].drop_duplicates(event_merge_keys, keep="first")
            quality = quality.merge(
                pilot_table,
                on=event_merge_keys,
                how="left",
                suffixes=("", "_pilot"),
            )
    for column in QUALITY_COLUMNS:
        if column not in quality.columns:
            quality[column] = np.nan
    return quality[QUALITY_COLUMNS]


def filter_pilot_metadata_to_scored_tier(pilot: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    if pilot.empty or quality.empty or "selection_tier" not in pilot.columns or "pilot_tier" not in quality.columns:
        return pilot
    tiers = {str(value) for value in quality["pilot_tier"].dropna().unique()}
    if not tiers:
        return pilot
    filtered = pilot[pilot["selection_tier"].astype(str).isin(tiers)].copy()
    return filtered if not filtered.empty else pilot


def build_model_rank_summary(evidence: pd.DataFrame, decisions: pd.DataFrame, *, margin_threshold: float) -> pd.DataFrame:
    if evidence.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "model_family",
                "model_rows",
                "successful_rows",
                "raw_best_events",
                "confident_best_events",
                "mean_log_evidence",
                "median_log_evidence",
                "mean_model_rank",
                "median_model_rank",
            ]
        )
    data = evidence.copy()
    data["log_evidence"] = pd.to_numeric(data["log_evidence"], errors="coerce")
    success = data[data["status"].astype(str).eq("success") & data["log_evidence"].notna()].copy()
    if success.empty:
        success["model_rank"] = np.nan
    else:
        group_cols = [column for column in EVENT_KEYS if column in success.columns]
        success["model_rank"] = success.groupby(group_cols, dropna=False)["log_evidence"].rank(method="first", ascending=False)
    raw_best = decisions["best_model"].value_counts(dropna=False) if "best_model" in decisions.columns else pd.Series(dtype=int)
    confident = decisions[
        pd.to_numeric(decisions.get("best_minus_runner_up_log_evidence", pd.Series(dtype=float)), errors="coerce") >= margin_threshold
    ]
    confident_best = confident["best_model"].value_counts(dropna=False) if "best_model" in confident.columns else pd.Series(dtype=int)
    rows = []
    for model, group in data.groupby("model", sort=True, dropna=False):
        success_group = success[success["model"].astype(str).eq(str(model))]
        rows.append(
            {
                "model": model,
                "model_family": first_nonempty(group.get("model_family", pd.Series(dtype=object))),
                "model_rows": int(len(group)),
                "successful_rows": int(group["status"].astype(str).eq("success").sum()),
                "raw_best_events": int(raw_best.get(model, 0)),
                "confident_best_events": int(confident_best.get(model, 0)),
                "mean_log_evidence": finite_mean(success_group["log_evidence"]),
                "median_log_evidence": finite_median(success_group["log_evidence"]),
                "mean_model_rank": finite_mean(success_group["model_rank"]),
                "median_model_rank": finite_median(success_group["model_rank"]),
            }
        )
    return pd.DataFrame(rows)


def build_margin_summary(quality: pd.DataFrame, *, margin_column: str, margin_threshold: float) -> pd.DataFrame:
    groups: list[tuple[str, list[str]]] = [("overall", []), ("animal", ["animal"]), ("pair", PAIR_KEYS)]
    rows: list[dict[str, object]] = []
    for scope, group_cols in groups:
        if group_cols:
            iterator = quality.groupby(group_cols, sort=True, dropna=False)
        else:
            iterator = [((), quality)]
        for key, group in iterator:
            row = base_group_row(scope, group_cols, key)
            values = pd.to_numeric(group[margin_column], errors="coerce")
            row.update(
                {
                    "events": int(len(group)),
                    "trajectory_confident_claims": int(group["trajectory_family_claim"].astype(str).eq("trajectory_confident").sum()),
                    "nontrajectory_confident_claims": int(group["trajectory_family_claim"].astype(str).eq("nontrajectory_confident").sum()),
                    "ambiguous_events": int(group["trajectory_family_claim"].astype(str).eq("ambiguous").sum()),
                    "positive_margin_events": int((values > 0.0).sum()),
                    "strong_stationary_events": int((values <= -float(margin_threshold)).sum()),
                    "mean_delta_best_trajectory_minus_stationary": finite_mean(values),
                    "median_delta_best_trajectory_minus_stationary": finite_median(values),
                    "min_delta_best_trajectory_minus_stationary": finite_min(values),
                    "max_delta_best_trajectory_minus_stationary": finite_max(values),
                    "margin_threshold": float(margin_threshold),
                    **correlation_columns(group, margin_column, prefix="corr_delta_trajectory_with"),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_imm_fragmented_audit(quality: pd.DataFrame, *, margin_threshold: float) -> pd.DataFrame:
    audit = quality.copy()
    delta = pd.to_numeric(audit["delta_imm_minus_fragmented"], errors="coerce")
    audit["imm_raw_win"] = delta > 0.0
    audit["fragmented_raw_win"] = delta < 0.0
    audit["imm_confident_win_at_5p5"] = delta >= float(margin_threshold)
    audit["fragmented_confident_win_at_5p5"] = delta <= -float(margin_threshold)
    audit["imm_fragmented_ambiguous"] = delta.abs() < float(margin_threshold)
    cols = [
        "animal",
        "date",
        "track1_session",
        "sleeppost_session",
        "pilot_tier",
        "decoder_filter",
        "event_id",
        "duration_ms",
        "n_spikes",
        "n_active_units",
        "mean_speed_cm_s",
        "candidate_tier",
        "best_model",
        "runner_up_model",
        "delta_imm_minus_fragmented",
        "imm_raw_win",
        "fragmented_raw_win",
        "imm_confident_win_at_5p5",
        "fragmented_confident_win_at_5p5",
        "imm_fragmented_ambiguous",
        "trajectory_family_claim",
        "posterior_mean_error_cm_median",
        "map_error_cm_median",
        "posterior_coverage_fraction",
    ]
    return audit[cols]


def build_group_debug_summary(quality: pd.DataFrame, group_cols: Sequence[str], *, margin_threshold: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in quality.groupby(list(group_cols), sort=True, dropna=False):
        row = base_group_row("group", list(group_cols), key)
        traj = pd.to_numeric(group["delta_best_trajectory_minus_stationary"], errors="coerce")
        imm = pd.to_numeric(group["delta_imm_minus_fragmented"], errors="coerce")
        row.update(
            {
                "selected_events": int(len(group)),
                "trajectory_confident_claims": int(group["trajectory_family_claim"].astype(str).eq("trajectory_confident").sum()),
                "nontrajectory_confident_claims": int(group["trajectory_family_claim"].astype(str).eq("nontrajectory_confident").sum()),
                "ambiguous_trajectory_static_events": int(group["trajectory_family_claim"].astype(str).eq("ambiguous").sum()),
                "imm_raw_wins": int((imm > 0.0).sum()),
                "fragmented_raw_wins": int((imm < 0.0).sum()),
                "imm_confident_wins": int((imm >= float(margin_threshold)).sum()),
                "fragmented_confident_wins": int((imm <= -float(margin_threshold)).sum()),
                "imm_fragmented_ambiguous_events": int((imm.abs() < float(margin_threshold)).sum()),
                "median_delta_best_trajectory_minus_stationary": finite_median(traj),
                "median_delta_imm_minus_fragmented": finite_median(imm),
                "mean_n_spikes": finite_mean(pd.to_numeric(group["n_spikes"], errors="coerce")),
                "mean_n_active_units": finite_mean(pd.to_numeric(group["n_active_units"], errors="coerce")),
                "median_duration_ms": finite_median(pd.to_numeric(group["duration_ms"], errors="coerce")),
                "median_decoder_posterior_error_cm": finite_median(
                    pd.to_numeric(group["posterior_mean_error_cm_median"], errors="coerce")
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def classify_run(quality: pd.DataFrame, gates: pd.DataFrame, *, margin_threshold: float) -> dict[str, object]:
    del margin_threshold
    technical_pass = bool(
        not gates.empty and as_bool(gates.set_index("gate").get("passed", pd.Series(dtype=bool)).get("overall", False))
    )
    if not technical_pass:
        return {
            "technical_classification": "technical-fail",
            "biological_classification": "not-assessed",
            "run_classification": "technical-fail",
        }
    if quality.empty:
        biological = "biological-ambiguous"
    else:
        trajectory_claims = quality["trajectory_family_claim"].astype(str).eq("trajectory_confident").sum()
        median_margin = finite_median(pd.to_numeric(quality["delta_best_trajectory_minus_stationary"], errors="coerce"))
        biological = "biological-positive" if trajectory_claims > len(quality) / 2 and median_margin > 0.0 else "biological-ambiguous"
    return {
        "technical_classification": "technical-pass",
        "biological_classification": biological,
        "run_classification": f"technical-pass; {biological}",
    }


def build_markdown_report(
    *,
    quality: pd.DataFrame,
    model_rank: pd.DataFrame,
    trajectory_margin: pd.DataFrame,
    imm_audit: pd.DataFrame,
    by_pair: pd.DataFrame,
    by_animal: pd.DataFrame,
    gates: pd.DataFrame,
    manifest: dict[str, object],
    classification: dict[str, object],
    margin_threshold: float,
) -> str:
    del model_rank, trajectory_margin, by_pair
    events = int(len(quality))
    trajectory_confident = int(quality["trajectory_family_claim"].astype(str).eq("trajectory_confident").sum()) if events else 0
    nontrajectory_confident = int(quality["trajectory_family_claim"].astype(str).eq("nontrajectory_confident").sum()) if events else 0
    ambiguous = int(quality["trajectory_family_claim"].astype(str).eq("ambiguous").sum()) if events else 0
    traj_delta = pd.to_numeric(quality["delta_best_trajectory_minus_stationary"], errors="coerce") if events else pd.Series(dtype=float)
    imm_delta = pd.to_numeric(quality["delta_imm_minus_fragmented"], errors="coerce") if events else pd.Series(dtype=float)
    imm_confident = int((imm_delta >= float(margin_threshold)).sum())
    imm_ambiguous = int((imm_delta.abs() < float(margin_threshold)).sum())
    technical_gates = gates[["gate", "status", "value"]] if not gates.empty else pd.DataFrame(columns=["gate", "status", "value"])
    lines = [
        "# Olafsdottir 1D SleepPOST Evidence Debug Report",
        "",
        "This report summarizes an existing evidence smoke output directory; it does not rescore events.",
        "",
        "## Classification",
        "",
        f"- technical: {classification['technical_classification']}",
        f"- biological: {classification['biological_classification']}",
        f"- combined: {classification['run_classification']}",
        "",
        "## Provenance",
        "",
        f"- source pilot tier: {manifest.get('pilot_tier', '')}",
        f"- source code commit: {manifest.get('code_commit', '')}",
        f"- margin threshold: {margin_threshold:g}",
        "",
        "## Technical Gates",
        "",
        dataframe_to_markdown(technical_gates),
        "",
        "## Model Interpretation Summary",
        "",
        f"- events: {events}",
        f"- trajectory-confident events: {trajectory_confident}/{events}",
        f"- nontrajectory-confident events: {nontrajectory_confident}/{events}",
        f"- trajectory/static ambiguous events: {ambiguous}/{events}",
        f"- median trajectory-minus-stationary margin: {finite_median(traj_delta):.6g}",
        f"- IMM-confident-over-fragmented events: {imm_confident}/{events}",
        f"- IMM/fragmented ambiguous events: {imm_ambiguous}/{events}",
        f"- median IMM-minus-fragmented margin: {finite_median(imm_delta):.6g}",
        "",
        "## Diagnostic Questions",
        "",
        f"- strongly stationary events at threshold: {int((traj_delta <= -float(margin_threshold)).sum())}",
        f"- positive trajectory margins: {int((traj_delta > 0.0).sum())}/{events}",
        f"- clean IMM events concentrated by animal: {compact_positive_counts(by_animal, 'imm_confident_wins')}",
        f"- trajectory-confident events concentrated by animal: {compact_positive_counts(by_animal, 'trajectory_confident_claims')}",
        f"- best model counts: {compact_counts(quality['best_model']) if events else ''}",
        f"- IMM/fragmented audit rows: {len(imm_audit)}",
        "",
        "## Claim Boundary",
        "",
        claim_boundary_text(classification),
        "",
    ]
    return "\n".join(lines) + "\n"


def write_debug_figures(quality: pd.DataFrame, output_dir: Path, *, margin_threshold: float) -> None:
    if quality.empty:
        return
    plot_histogram(
        pd.to_numeric(quality["delta_best_trajectory_minus_stationary"], errors="coerce"),
        output_dir / TRAJECTORY_MARGIN_FIGURE,
        xlabel="Best trajectory minus stationary log evidence",
        title="Trajectory-family margin",
    )
    plot_histogram(
        pd.to_numeric(quality["delta_imm_minus_fragmented"], errors="coerce"),
        output_dir / IMM_FRAGMENTED_FIGURE,
        xlabel="IMM minus fragmented log evidence",
        title="IMM vs fragmented margin",
    )
    counts = quality["best_model"].astype(str).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color="#4C78A8")
    ax.set_ylabel("Events")
    ax.set_title("Best model counts")
    fig.tight_layout()
    fig.savefig(output_dir / MODEL_WINNER_FIGURE, dpi=160)
    plt.close(fig)

    animal = build_group_debug_summary(quality, ["animal"], margin_threshold=margin_threshold)
    if not animal.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(animal["animal"].astype(str), animal["median_delta_best_trajectory_minus_stationary"], color="#59A14F")
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_ylabel("Median trajectory margin")
        ax.set_title("By-animal trajectory margins")
        fig.tight_layout()
        fig.savefig(output_dir / ANIMAL_MARGIN_FIGURE, dpi=160)
        plt.close(fig)


def plot_histogram(values: pd.Series, path: Path, *, xlabel: str, title: str) -> None:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    if not finite.empty:
        ax.hist(finite, bins=min(12, max(3, len(finite))), color="#4C78A8", edgecolor="white")
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Events")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def correlation_columns(group: pd.DataFrame, margin_column: str, *, prefix: str) -> dict[str, float]:
    values = pd.to_numeric(group[margin_column], errors="coerce")
    out: dict[str, float] = {}
    for column in CORRELATION_COLUMNS:
        label = f"{prefix}_{column}"
        out[label] = pearson(values, pd.to_numeric(group[column], errors="coerce")) if column in group.columns else np.nan
    return out


def pearson(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return np.nan
    return float(np.corrcoef(frame["x"], frame["y"])[0, 1])


def base_group_row(scope: str, group_cols: Sequence[str], key: object) -> dict[str, object]:
    row: dict[str, object] = {"scope": scope}
    if not group_cols:
        return row
    values = key if isinstance(key, tuple) else (key,)
    row.update({column: value for column, value in zip(group_cols, values)})
    return row


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows available._"
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in frame.columns) + " |")
    return "\n".join(lines)


def claim_boundary_text(classification: dict[str, object]) -> str:
    if classification["technical_classification"] == "technical-fail":
        return "Technical gates failed, so model rankings should not be interpreted."
    if classification["biological_classification"] == "biological-positive":
        return "This run is descriptively positive, but still requires the planned audit and sensitivity checks before any cross-dataset claim."
    return "The scorer interface works, but this run does not establish a robust trajectory-family or clean-IMM biological claim."


def compact_counts(values: pd.Series) -> str:
    counts = values.astype(str).value_counts().sort_index()
    return "; ".join(f"{key}={int(value)}" for key, value in counts.items())


def compact_positive_counts(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame.columns:
        return "none"
    positives = frame[pd.to_numeric(frame[column], errors="coerce") > 0]
    if positives.empty:
        return "none"
    key = "animal" if "animal" in positives.columns else positives.columns[0]
    return "; ".join(f"{row[key]}={int(row[column])}" for _, row in positives.iterrows())


def finite_mean(values: Iterable[object]) -> float:
    finite = finite_values(values)
    return float(np.mean(finite)) if finite.size else np.nan


def finite_median(values: Iterable[object]) -> float:
    finite = finite_values(values)
    return float(np.median(finite)) if finite.size else np.nan


def finite_min(values: Iterable[object]) -> float:
    finite = finite_values(values)
    return float(np.min(finite)) if finite.size else np.nan


def finite_max(values: Iterable[object]) -> float:
    finite = finite_values(values)
    return float(np.max(finite)) if finite.size else np.nan


def finite_values(values: Iterable[object]) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def first_nonempty(values: pd.Series) -> object:
    for value in values:
        if pd.notna(value) and str(value) != "":
            return value
    return ""


def dedupe(columns: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for column in columns:
        if column not in seen:
            seen.add(column)
            out.append(column)
    return out


def as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "pass"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True, help="Existing score_olafsdottir output directory")
    parser.add_argument("--output-dir", help="Directory for report outputs; defaults to --evidence-dir")
    parser.add_argument("--margin-threshold", type=float, help="Override margin threshold; defaults to manifest value")
    parser.add_argument("--no-figures", action="store_true", help="Skip optional PNG debug figures")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_report(
        evidence_dir=args.evidence_dir,
        output_dir=args.output_dir,
        margin_threshold=args.margin_threshold,
        write_figures=not args.no_figures,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
