#!/usr/bin/env python3
"""Non-rescoring report of the independently audited MAZE transfer control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import build_script_provenance, file_sha256

CONTRASTS = ("behavior_minus_global", "behavior_real_minus_wrong", "iid_minus_global", "imm_minus_iid")
LABELS = ("Measured position - nonspatial", "Measured position: real - permuted", "Independent decoder - nonspatial", "IMM - independent decoder")


def validate(run, audit):
    manifest_path = run / "maze_transfer_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    checked = json.loads((audit / "audit_manifest.json").read_text())
    if manifest["status"] != "complete" or checked["status"] != "passed":
        raise ValueError("incomplete run or failed independent audit")
    if checked["input_file_sha256"]["run_manifest"] != file_sha256(manifest_path):
        raise ValueError("audit refers to a different run")
    names = ("summary", "by_animal", "decoder_qc", "gate_summary", "windows")
    tables = {}
    for name in names:
        path = run / f"maze_transfer_{name}.csv"
        if file_sha256(path) != manifest["output_sha256"][path.name]:
            raise ValueError(f"changed result: {name}")
        tables[name] = pd.read_csv(path)
    if tables["gate_summary"].empty or not tables["gate_summary"].passed.eq(True).all():
        raise ValueError("technical gates failed or empty")
    return manifest, checked, tables


def compact_table(summary):
    selected = summary[summary.encoding_variant.eq("direction_mixture") & summary.contrast.isin(CONTRASTS)].copy()
    keys = ["unit_regime", "count_regime", "contrast"]
    if selected.duplicated(keys).any() or len(selected) != 16:
        raise ValueError("missing or duplicate primary/sensitivity summary")
    if not np.isfinite(selected[["equal_animal_mean", "ci_low", "ci_high"]]).all().all():
        raise ValueError("nonfinite contrasts")
    return selected


def write_figure(table, animal, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Achilles": "#287c8e", "Buddy": "#9b4d85", "Cicero": "#cc6a30", "Gatsby": "#588542"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, count, units in zip(axes.ravel(), ("native", "sleep_total_cap") * 2, ("train_only_qc",) * 2 + ("frozen_parent_units",) * 2, strict=True):
        ss = table[table.count_regime.eq(count) & table.unit_regime.eq(units)].set_index("contrast").loc[list(CONTRASTS)]
        aa = animal[animal.count_regime.eq(count) & animal.unit_regime.eq(units) & animal.encoding_variant.eq("direction_mixture")]
        x = np.arange(4)
        ax.errorbar(
            x,
            ss.equal_animal_mean,
            yerr=np.vstack([ss.equal_animal_mean - ss.ci_low, ss.ci_high - ss.equal_animal_mean]),
            fmt="s",
            color="black",
            capsize=4,
            label="Equal-animal mean and 95% CI",
        )
        for offset, (rat, color) in zip(np.linspace(-0.15, 0.15, 4), colors.items(), strict=True):
            values = aa[aa.rat.eq(rat)].set_index("contrast").loc[list(CONTRASTS)].delta
            ax.scatter(x + offset, values, s=25, color=color, label=rat, zorder=3)
        ax.axhline(0, color="gray", linewidth=1)
        ax.set_xticks(x, LABELS, rotation=24, ha="right", fontsize=8)
        ax.set_ylabel("Predictive difference (nats / 200 ms)")
        ax.set_title(
            ("Training-only unit QC" if units == "train_only_qc" else "Frozen-parent units: sensitivity")
            + "\n"
            + ("Native spike counts" if count == "native" else "Sleep-total cap (not exact information matching)"),
            fontsize=10,
        )
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("hc-11 MAZE cross-time prediction: actual locomotion, not replay", fontsize=14)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(args):
    source, audit, out = (Path(p).resolve() for p in (args.run_dir, args.audit_dir, args.output_dir))
    manifest, checked, tables = validate(source, audit)
    table = compact_table(tables["summary"])
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing to overwrite a report")
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "maze_transfer_report_table.csv", index=False)
    write_figure(table, tables["by_animal"], out / "maze_transfer_report.png")
    lines = [
        "# Native MAZE transfer control",
        "",
        "Technical validation passed. This is an actual-running encoding control, not sleep replay, a theta-sweep result, or a new biological mechanism.",
        "",
        f"{len(tables['windows'])} nonoverlapping 200 ms windows, eight sessions, four animals; five neural splits. Encoding and primary unit QC use the other MAZE half only, with a five-second guard. Held-out neurons do not update latent inference.",
        "",
        "## Primary and sensitivity results",
        "",
        "Differences are event/split medians aggregated equally across halves, sessions and animals. Intervals resample four animal clusters; they condition on fitted maps and observed within-animal data. They are exploratory and do not establish a RUN-by-sleep interaction.",
        "",
        "| Units | Counts | Contrast | Mean [95% CI] | Positive animals |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"| {row.unit_regime} | {row.count_regime} | {row.contrast} | {row.equal_animal_mean:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}] | {row.positive_animals}/{row.animals} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Primary measured-position and neural-decoder controls are positive in all four animals. The total-count-capped condition remains positive, but does not match sleep per-bin spike allocation, active-cell support, or held-out information. Some running windows have fewer spikes than their target and are retained, not upsampled.",
            "",
            "The older frozen-parent unit list is not wholly independent unit-selection validation. In that sensitivity Buddy has a negative measured-position-minus-nonspatial score, and the aggregate interval includes zero. The primary result does not erase this caveat.",
            "",
            "Measured-position prediction is supervised: actual behavior selects the map location. It must not be called neural decoding. The temporal-model comparison uses replay-scale priors on running windows and is a pipeline control, not a physiological timescale conclusion.",
            "",
            "The primary control narrows a grossly broken RUN encoder explanation for the existing sleep result. It does not establish why transfer to sleep is weaker. A positive RUN result and an inconclusive sleep result do not establish a significant state difference.",
            "",
            "## Independent verification",
            "",
            checked["verification_scope"],
            "",
            f"{checked['count_matrices']} count matrices; {checked['independent_scores']} independently recomputed scores; maximum error {checked['maximum_score_error']:.3g}. Mean-position error and credible coverage are producer summaries, not independently recomputed audit endpoints.",
            "",
            f"Producer commit: `{manifest['code_commit']}`. Independent verifier: `{checked['code_commit']}`.",
        ]
    )
    (out / "maze_transfer_report.md").write_text("\n".join(lines) + "\n")
    provenance = build_script_provenance(input_paths={"run_manifest": source / "maze_transfer_manifest.json", "audit_manifest": audit / "audit_manifest.json"})
    provenance.update(status="complete", non_rescoring=True, output_sha256={p.name: file_sha256(p) for p in out.iterdir() if p.is_file()})
    (out / "report_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())
