#!/usr/bin/env python3
"""Create figures for the IMM-vs-fragmented hypothesis audit.

The figures are deliberately evidence-first: they show where first-order IMM
cleanly beats fragmented, where cases remain ambiguous, and which representative
events should or should not be used as examples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_COLUMNS = [
    ("stationary", "logZ_stationary"),
    ("diffusion", "logZ_diffusion"),
    ("fragmented", "logZ_fragmented"),
    ("first-order IMM", "logZ_first_order_imm"),
    ("momentum", "logZ_momentum_exact_sparse"),
]


def _read_audit_table(audit_dir: str | Path) -> pd.DataFrame:
    root = Path(audit_dir)
    for filename in [
        "trajectory_taxonomy_event_table.csv",
        "imm_fragmented_head_to_head_event_table.csv",
    ]:
        path = root / filename
        if path.is_file():
            table = pd.read_csv(path)
            table["session"] = table["session"].astype(str)
            table["event_index"] = pd.to_numeric(table["event_index"], errors="raise").astype(int)
            return table
    raise FileNotFoundError(
        f"Could not find trajectory_taxonomy_event_table.csv or "
        f"imm_fragmented_head_to_head_event_table.csv under {root}"
    )


def _safe_name(value: object) -> str:
    return str(value).replace("/", "_").replace(" ", "_")


def _write_taxonomy_bar(table: pd.DataFrame, output: Path) -> None:
    counts = table["within_family_classification"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ax.barh(counts.index.astype(str), counts.to_numpy())
    ax.set_xlabel("events")
    ax.set_title("Trajectory-family taxonomy from full-core evidence")
    for index, value in enumerate(counts.to_numpy()):
        ax.text(value + 0.5, index, str(int(value)), va="center")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _write_delta_histogram(table: pd.DataFrame, output: Path, threshold: float) -> None:
    values = pd.to_numeric(table["delta_imm_minus_fragmented"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.hist(values.to_numpy(), bins=30)
    ax.axvline(threshold, linestyle="--", label=f"clean IMM threshold +{threshold:g}")
    ax.axvline(-threshold, linestyle="--", label=f"fragmented threshold -{threshold:g}")
    ax.axvline(0.0, linestyle=":", label="tie")
    ax.set_xlabel("log evidence: first-order IMM minus fragmented")
    ax.set_ylabel("events")
    ax.set_title("Does first-order IMM beat fragmented?")
    ax.legend()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _write_delta_scatter(table: pd.DataFrame, output: Path, threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    for label, group in table.groupby("within_family_classification"):
        ax.scatter(
            pd.to_numeric(group["delta_imm_minus_momentum"], errors="coerce"),
            pd.to_numeric(group["delta_imm_minus_fragmented"], errors="coerce"),
            label=label,
            alpha=0.8,
            s=28,
        )
    ax.axhline(threshold, linestyle="--")
    ax.axhline(-threshold, linestyle="--")
    ax.axhline(0.0, linestyle=":")
    ax.axvline(0.0, linestyle=":")
    ax.set_xlabel("log evidence: first-order IMM minus momentum")
    ax.set_ylabel("log evidence: first-order IMM minus fragmented")
    ax.set_title("IMM superiority axes")
    ax.legend(fontsize=7, loc="best")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _event_title(row: pd.Series) -> str:
    return (
        f"{row['session']} event {int(row['event_index'])}\n"
        f"{row['within_family_classification']} | "
        f"ΔIMM-frag={float(row['delta_imm_minus_fragmented']):.1f}"
    )


def _plot_event_bars(rows: pd.DataFrame, output: Path, *, title: str) -> None:
    if rows.empty:
        return
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 1, figsize=(9.5, max(3.2, 2.4 * n_rows)), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    for axis, (_, row) in zip(axes, rows.iterrows(), strict=False):
        labels = []
        values = []
        for label, column in MODEL_COLUMNS:
            value = float(row[column])
            if np.isfinite(value):
                labels.append(label)
                values.append(value)
        relative = np.asarray(values) - np.max(values)
        axis.barh(labels, relative)
        axis.axvline(0.0, linestyle=":")
        axis.set_xlabel("relative log evidence (best model = 0)")
        axis.set_title(_event_title(row), fontsize=9)
    fig.suptitle(title)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _select_examples(table: pd.DataFrame, max_examples: int) -> pd.DataFrame:
    pieces = []
    clean = table[table["within_family_classification"].eq("clean_imm_switching_candidate")]
    pieces.append(clean.sort_values("delta_imm_minus_fragmented", ascending=False).head(max_examples))
    ambiguous = table[table["within_family_classification"].eq("imm_fragmented_ambiguous")]
    pieces.append(ambiguous.sort_values("delta_imm_minus_fragmented", key=lambda s: s.abs()).head(2))
    momentum = table[table["within_family_classification"].eq("momentum_like_candidate")]
    pieces.append(momentum.sort_values("delta_momentum_minus_diffusion", ascending=False).head(2))
    examples = pd.concat([piece for piece in pieces if not piece.empty], ignore_index=True)
    return examples.drop_duplicates(subset=["session", "event_index"]).reset_index(drop=True)


def _write_focus_examples(table: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    focus = table[table["event_index"].isin([540, 550])].copy()
    if not focus.empty:
        focus = focus.sort_values(["event_index", "session"])
        _plot_event_bars(focus, output_dir / "event_540_550_evidence_bars.png", title="Collaborator focus events")
    return focus


def write_figures(audit_dir: str | Path, output: str | Path, threshold: float = 5.5, max_examples: int = 6) -> dict[str, object]:
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = _read_audit_table(audit_dir)

    figures: list[str] = []
    taxonomy_path = output_dir / "taxonomy_counts.png"
    _write_taxonomy_bar(table, taxonomy_path)
    figures.append(taxonomy_path.name)

    hist_path = output_dir / "imm_minus_fragmented_histogram.png"
    _write_delta_histogram(table, hist_path, threshold)
    figures.append(hist_path.name)

    scatter_path = output_dir / "imm_superiority_scatter.png"
    _write_delta_scatter(table, scatter_path, threshold)
    figures.append(scatter_path.name)

    clean_examples = table[table["within_family_classification"].eq("clean_imm_switching_candidate")]
    clean_examples = clean_examples.sort_values("delta_imm_minus_fragmented", ascending=False).head(max_examples)
    clean_path = output_dir / "top_clean_imm_evidence_bars.png"
    _plot_event_bars(clean_examples, clean_path, title="Top clean first-order IMM examples")
    if clean_path.exists():
        figures.append(clean_path.name)

    examples = _select_examples(table, max_examples=max_examples)
    gallery_path = output_dir / "taxonomy_example_evidence_bars.png"
    _plot_event_bars(examples, gallery_path, title="Representative taxonomy examples")
    if gallery_path.exists():
        figures.append(gallery_path.name)

    focus = _write_focus_examples(table, output_dir)
    if (output_dir / "event_540_550_evidence_bars.png").exists():
        figures.append("event_540_550_evidence_bars.png")

    example_columns = [
        "session",
        "rat",
        "event_index",
        "within_family_classification",
        "best_exact_core_model",
        "delta_imm_minus_fragmented",
        "delta_imm_minus_momentum",
        "delta_momentum_minus_diffusion",
        "delta_trajectory_minus_stationary",
    ]
    examples[example_columns].to_csv(output_dir / "example_event_manifest.csv", index=False)
    focus[example_columns].to_csv(output_dir / "focus_event_manifest.csv", index=False)

    summary = {
        "events": int(len(table)),
        "clean_imm_switching_candidates": int(table["within_family_classification"].eq("clean_imm_switching_candidate").sum()),
        "imm_fragmented_ambiguous": int(table["within_family_classification"].eq("imm_fragmented_ambiguous").sum()),
        "fragmented_candidates": int(table["within_family_classification"].eq("fragmented_candidate").sum()),
        "mean_delta_imm_minus_fragmented": float(pd.to_numeric(table["delta_imm_minus_fragmented"], errors="coerce").mean()),
        "median_delta_imm_minus_fragmented": float(pd.to_numeric(table["delta_imm_minus_fragmented"], errors="coerce").median()),
        "threshold": float(threshold),
        "figures": figures,
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--margin-threshold", type=float, default=5.5)
    parser.add_argument("--max-examples", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = write_figures(
        args.audit_dir,
        args.output,
        threshold=args.margin_threshold,
        max_examples=args.max_examples,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
