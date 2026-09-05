#!/usr/bin/env python3
"""Non-decoding detector/window availability report for the coverage study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts._provenance import build_script_provenance, file_sha256


def summarize(root):
    meta = json.loads((root / "coverage_event_definition_manifest.json").read_text())
    audit = json.loads((root / "coverage_event_definition_reconstruction_audit.json").read_text())
    if meta["status"] not in {"complete", "complete_with_ripple_unavailable"} or audit["status"] != "pass":
        raise ValueError("successful preparation and reconstruction audit required")
    if audit["input_file_sha256"]["preparation_manifest"] != file_sha256(root / "coverage_event_definition_manifest.json") or audit["sessions"] != meta["sessions"] or audit["window_rows_verified"] != meta["windows"]:
        raise ValueError("audit does not cover this preparation artifact")
    for name, digest in meta["output_sha256"].items():
        if file_sha256(root / name) != digest:
            raise ValueError(f"changed preparation output: {name}")
    windows = pd.read_csv(root / "coverage_event_definition_windows.csv")
    counts = pd.read_csv(root / "coverage_event_definition_counts.csv")
    sessions = pd.read_csv(root / "coverage_event_definition_sessions.csv")
    counts = counts.merge(sessions[["dataset", "animal", "session", "immobile_duration_s", "ripple_status"]], on=["dataset", "animal", "session"], validate="many_to_one")
    counts["eligible_rate_per_immobile_minute"] = counts.eligible_windows / (counts.immobile_duration_s / 60)
    keys = ["dataset", "detector", "window_variant", "peak_threshold_z"]
    animal = counts.groupby(keys + ["animal"], as_index=False).agg(
        sessions=("session", "size"), detector_available_sessions=("detector_available", "sum"),
        eligible_windows=("eligible_windows", lambda x: x.sum(min_count=1)),
        mean_session_rate_per_immobile_minute=("eligible_rate_per_immobile_minute", "mean"))
    total = counts.groupby(keys, as_index=False).agg(sessions=("session", "size"), available_sessions=("detector_available", "sum"),
        source_windows=("source_windows", lambda x: x.sum(min_count=1)), eligible_windows=("eligible_windows", lambda x: x.sum(min_count=1)))
    rates = animal.groupby(keys, as_index=False).agg(animals=("animal", "size"), equal_animal_mean_rate_per_immobile_minute=("mean_session_rate_per_immobile_minute", "mean"))
    total = total.merge(rates, on=keys, validate="one_to_one")
    overlap = counts[counts.detector.eq("source_high_mua")].copy()
    overlap["fraction_also_ripple_detected"] = overlap.other_detector_overlap_at_primary_z3 / overlap.eligible_windows.replace(0, np.nan)
    # Unknown ripple detection never enters the overlap denominator.
    overlap.loc[~overlap.overlap_available, "fraction_also_ripple_detected"] = np.nan
    return windows, counts, animal, total, overlap, meta


def run(root, out):
    windows, counts, animal, total, overlap, meta = summarize(root)
    out.mkdir(parents=True, exist_ok=False)
    for name, frame in [("session_counts", counts), ("animal_counts", animal), ("dataset_counts", total), ("mua_overlap", overlap)]:
        frame.to_csv(out / f"coverage_event_definition_{name}.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = {"source_high_mua": "#176b8e", "native_ripple_table": "#b94748", "lfp_ripple_detected": "#b94748"}
    for ax, dataset, title in zip(axes[0], ["pfeiffer_foster", "tanni2022"], ["Pfeiffer/Foster: native ripple tables", "Tanni: CA1 LFP ripple-like detector"], strict=True):
        local = total[total.dataset.eq(dataset) & total.peak_threshold_z.isin([0, 3])].sort_values(["detector", "window_variant"])
        x = np.arange(len(local))
        ax.bar(x, local.eligible_windows, color=[colors[v] for v in local.detector])
        labels = [("MUA" if row.detector == "source_high_mua" else "Ripple") + ("\ncore" if row.window_variant == "detected_core" else "\n200 ms") for row in local.itertuples()]
        ax.set_xticks(x, labels)
        ax.set(ylabel="Eligible candidate windows (not replay)", title=title)
        for i, row in enumerate(local.itertuples()):
            ax.annotate(f"{int(row.eligible_windows):,}\n{int(row.available_sessions)}/{int(row.sessions)} sessions", (i, row.eligible_windows), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
        ax.margins(y=.25)
    ax = axes[1, 0]
    for j, (dataset, color) in enumerate([("pfeiffer_foster", "#176b8e"), ("tanni2022", "#b94748")]):
        for i, variant in enumerate(["detected_core", "peak_centered_200ms"]):
            local = overlap[overlap.dataset.eq(dataset) & overlap.window_variant.eq(variant)].groupby("animal").fraction_also_ripple_detected.mean().dropna()
            x = i + (j - .5) * .22
            offsets = np.linspace(-.055, .055, len(local))
            ax.scatter(x + offsets, local, color=color, label=dataset if i == 0 else None, alpha=.8)
    ax.set_xticks([0, 1], ["Detected core", "Fixed 200 ms"])
    ax.set(ylabel="MUA fraction also ripple-detected", ylim=(0, 1), title="Each point: one animal; available sessions only")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 1]
    mua = windows[windows.dataset.eq("tanni2022") & windows.detector.eq("source_high_mua") & windows.window_variant.eq("detected_core") & windows.eligible & windows.ripple_status.eq("available")]
    common_bins = np.histogram_bin_edges(mua.mean_ripple_envelope_z.dropna(), bins=35) if len(mua) else np.linspace(-1, 1, 36)
    for name, local in mua.groupby("animal", sort=True):
        values = local.mean_ripple_envelope_z.dropna()
        if len(values):
            ax.hist(values, bins=common_bins, histtype="step", density=True, label=f"{name} (n={len(values)})")
    ax.axvline(0, color="black", lw=.8)
    ax.set(xlabel="Within-MUA mean envelope z (amplitude, not power)", ylabel="Density", title="Tanni: event-average ripple expression")
    ax.legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Frozen event-definition inputs: detector and window sensitivity\nNo decoding, trajectory selection or biological comparison", fontsize=13)
    fig.savefig(out / "coverage_event_definition_overview.png", dpi=180)
    plt.close(fig)
    unavailable = [r for r in meta["results"] if r["ripple_status"] != "available"]
    lines = ["# Frozen Event-Definition Readout", "", f"{meta['sessions']} sessions; {meta['source_mua_events']:,} source MUA events retained before common eligibility.",
        f"Ripple definition unavailable in {len(unavailable)} session(s); these are missing, not zero-ripple sessions.", "",
        "| Dataset | Detector | Window | Threshold | Available sessions | Eligible windows |", "|---|---|---|---:|---:|---:|"]
    for row in total.itertuples(index=False):
        lines.append(f"| {row.dataset} | {row.detector} | {row.window_variant} | {row.peak_threshold_z:g} | {row.available_sessions}/{row.sessions} | {row.eligible_windows:g} |")
    lines.extend(["", "Threshold 0 means not rethresholded: original MUA or supplied native PF ripple rows.",
        "Only Tanni has new LFP thresholds. Every amplitude-qualified Tanni episode remains in the source table; duration failures are explicitly excluded.",
        "", "## Limits", "", "Counts are candidates, not replay or precision/recall. Fixed windows change both overlap and whole-window eligibility.",
        "PF native detector and newly applied Tanni detector are not interchangeable assays.",
        "LFP envelope amplitude z must not be described as power z. Technical channel QC is not an artifact or sharp-wave validation.",
        "Unknown ripple status is excluded from overlap denominators. Zero eligible windows are retained for available detectors.",
        "No decoder threshold, posterior, continuity or speed outcome entered preparation.",
        "The next experiment is paired full/half-population decoding within these frozen detector/window strata, followed by established shuffle-significant replay baselines.", "",
        "## Unavailable Sessions", ""])
    lines.extend(f"- {r['animal']}/{r['session']}: {r['raw_ripple_source']['baseline_duration_s']:.3f} s baseline, below frozen 60 s minimum." for r in unavailable)
    (out / "coverage_event_definition_summary.md").write_text("\n".join(lines) + "\n")
    provenance = build_script_provenance(input_paths={"preparation_manifest": root / "coverage_event_definition_manifest.json", "audit": root / "coverage_event_definition_reconstruction_audit.json", "reporter": Path(__file__)}, cwd=ROOT)
    provenance.update(status="complete", decode_performed=False, output_sha256={p.name: file_sha256(p) for p in out.iterdir() if p.is_file()})
    (out / "coverage_event_definition_report_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.input_dir.resolve(), args.output_dir.resolve())
