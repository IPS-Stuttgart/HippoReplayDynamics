#!/usr/bin/env python3
"""Non-rescoring, audit-gated readout of the frozen hc-11 recovery benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from _provenance import build_script_provenance, file_sha256

GENERATORS = ("iid_position", "static_location", "diffusion", "first_order_imm")
LABELS = ("Independent", "Static", "Diffusion", "IMM")
COLORS = ("#777777", "#b34b55", "#16868a", "#4477aa")
PRIMARY = ("imm_minus_iid_position", "imm_minus_static_location")


def validate_audit(root, audit_path):
    manifest_path = root / "recovery_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    if manifest["status"] != "complete" or audit["status"] != "passed":
        raise ValueError("complete audited simulation required")
    if audit["input_file_sha256"]["recovery_manifest"] != file_sha256(manifest_path):
        raise ValueError("audit refers to another recovery run")
    scope = {"draws_checked": 64000, "regenerated_draws": 192, "independent_predictions": 1536, "independent_bootstrap_panels": 4800}
    if any(audit.get(key) != value for key, value in scope.items()):
        raise ValueError("incomplete independent audit")
    for name in (
        "recovery_simulation_panels.csv",
        "recovery_positive_pattern_summary.csv",
        "recovery_predictive_rank_confusion.csv",
        "recovery_information_summary.csv",
        "recovery_event_contrasts.csv",
    ):
        if file_sha256(root / name) != manifest["output_sha256"][name]:
            raise ValueError(f"changed summary: {name}")
    return manifest, audit


def build_readout(panels, patterns, real):
    if panels.empty or patterns.empty or real.empty:
        raise ValueError("empty readout inputs")
    rows = []
    for (phase, variant, generator), part in panels.groupby(["phase", "encoding_variant", "generator"]):
        if generator not in GENERATORS:
            raise ValueError("unexpected generator")
        pat = patterns[(patterns.phase == phase) & (patterns.encoding_variant == variant) & (patterns.generator == generator)]
        if len(pat) != 1 or pat.replicates.iloc[0] != 50:
            raise ValueError("missing or duplicate pattern summary")
        for contrast in PRIMARY:
            selected = part[part.contrast == contrast]
            observed = real[(real.phase == phase) & (real.encoding_variant == variant) & (real.inference_temperature == 1.0) & (real.contrast == contrast)]
            if len(selected) != 50 or set(selected.replicate) != set(range(50)) or len(observed) != 1:
                raise ValueError("missing or duplicate simulation/real contrast")
            values = selected.equal_animal_mean.to_numpy()
            row = observed.iloc[0]
            if not np.isfinite(values).all() or not np.isfinite([row.equal_animal_mean_event_median_delta, row.ci_low, row.ci_high]).all():
                raise ValueError("nonfinite effect")
            rows.append(
                {
                    "phase": phase,
                    "encoding_variant": variant,
                    "generator": generator,
                    "contrast": contrast,
                    "simulated_median": np.median(values),
                    "simulated_p05": np.quantile(values, 0.05),
                    "simulated_p95": np.quantile(values, 0.95),
                    "real_effect": row.equal_animal_mean_event_median_delta,
                    "real_ci_low": row.ci_low,
                    "real_ci_high": row.ci_high,
                    "positive_pattern_count": int(pat.positive_pattern_count.iloc[0]),
                    "replicates": 50,
                    "positive_pattern_fraction": pat.positive_pattern_fraction.iloc[0],
                    "pattern_ci_low": pat.binomial_ci_low.iloc[0],
                    "pattern_ci_high": pat.binomial_ci_high.iloc[0],
                }
            )
    result = pd.DataFrame(rows)
    expected = pd.MultiIndex.from_product([("POST", "PRE"), ("direction_mixture", "pooled"), GENERATORS, PRIMARY])
    actual = pd.MultiIndex.from_frame(result[["phase", "encoding_variant", "generator", "contrast"]])
    if len(result) != len(expected) or set(actual) != set(expected):
        raise ValueError("incomplete readout factors")
    return result


def verify_rank_table(events, confusion):
    keys = ["generator", "phase", "encoding_variant", "raw_predictive_winner"]
    counts = events.groupby(keys).size().rename("counted").reset_index()
    merged = counts.merge(confusion, on=keys, how="outer", validate="one_to_one")
    if merged.empty or not merged.counted.eq(merged["size"]).all():
        raise ValueError("rank confusion counts do not match audited event ranks")
    denominator = merged.groupby(keys[:-1])["counted"].transform("sum")
    if not np.allclose(merged.fraction, merged.counted / denominator):
        raise ValueError("rank confusion fractions do not match event ranks")


def real_information(source, parent):
    selection_path = source / "frozen_selection.csv"
    if file_sha256(selection_path) != parent["output_sha256"][selection_path.name]:
        raise ValueError("changed source selection")
    selection = pd.read_csv(selection_path)
    rows = []
    for session, selected in selection.groupby("session"):
        path = source / f"{session}_cache.npz"
        if file_sha256(path) != parent["output_sha256"][path.name]:
            raise ValueError("changed source population cache")
        with np.load(path) as data:
            for event in selected.itertuples(index=False):
                counts = data[f"counts_{event.phase}_{event.event_id}"]
                for split in range(5):
                    rows.append(
                        {
                            "session": session,
                            "rat": event.animal,
                            "phase": event.phase,
                            "event_index": event.event_id,
                            "split": split,
                            "n_spikes": int(counts.sum()),
                            "n_train_spikes": int(counts[:, data[f"train_{split}"]].sum()),
                            "n_heldout_spikes": int(counts[:, data[f"held_{split}"]].sum()),
                            "n_active_units": int((counts.sum(axis=0) > 0).sum()),
                        }
                    )
    return pd.DataFrame(rows)


def information_comparison(simulated, observed):
    columns = ["n_spikes", "n_train_spikes", "n_heldout_spikes", "n_active_units"]
    rows = []
    for (phase, generator), group in simulated.groupby(["phase", "generator"]):
        if len(group) != 50 or group.replicate.nunique() != 50:
            raise ValueError("incomplete information summary")
        for col in columns:
            rows.append(
                {
                    "phase": phase,
                    "generator": generator,
                    "metric": col,
                    "real_split_median": observed.loc[observed.phase == phase, col].median(),
                    "median_simulated_dataset_median": group[col].median(),
                    "p05_simulated_dataset_median": group[col].quantile(0.05),
                    "p95_simulated_dataset_median": group[col].quantile(0.95),
                }
            )
    return pd.DataFrame(rows)


def make_figure(panels, readout, confusion, output):
    primary = readout[(readout.phase == "POST") & (readout.encoding_variant == "direction_mixture") & (readout.contrast == PRIMARY[0])].set_index("generator").loc[list(GENERATORS)]
    distributions = [
        panels[
            (panels.phase == "POST") & (panels.encoding_variant == "direction_mixture") & (panels.contrast == PRIMARY[0]) & (panels.generator == gen)
        ].equal_animal_mean.to_numpy()
        for gen in GENERATORS
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3), gridspec_kw={"width_ratios": [1.15, 1, 1.35]})
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.22, top=0.79, wspace=0.4)
    positions = np.arange(4)
    ax = axes[0]
    boxes = ax.boxplot(distributions, positions=positions, widths=0.5, whis=(5, 95), patch_artist=True, showfliers=False, medianprops={"color": "black"})
    for box, color in zip(boxes["boxes"], COLORS, strict=True):
        box.set_facecolor(color)
        box.set_alpha(0.65)
    observed = primary.iloc[0]
    ax.axhspan(observed.real_ci_low, observed.real_ci_high, color="#e8b535", alpha=0.15, label="Real-data 95% bootstrap CI")
    ax.axhline(observed.real_effect, color="#8a6400", linestyle="--", label=f"Real POST: {observed.real_effect:+.3f}")
    ax.axhline(0, color="black", lw=0.65)
    ax.set_xticks(positions, LABELS)
    ax.set_ylabel("IMM - independent predictive score\n(equal-animal mean; nats/event)")
    ax.set_title("A  Cohort effects", loc="left", fontsize=12)
    ax.set_ylim(-0.25, 1.55)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.text(0, -0.21, "Boxes: 50 simulated cohorts; whiskers: p05-p95", transform=ax.transAxes, fontsize=8)
    ax = axes[1]
    frac = primary.positive_pattern_fraction.to_numpy()
    ax.bar(positions, frac, color=COLORS, alpha=0.8)
    ax.errorbar(positions, frac, yerr=np.vstack([frac - primary.pattern_ci_low, primary.pattern_ci_high - frac]), fmt="none", color="black", capsize=4)
    for x, upper, count in zip(positions, primary.pattern_ci_high, primary.positive_pattern_count, strict=True):
        ax.text(x, upper + 0.035, f"{count}/50", ha="center", fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "25", "50", "75", "100"])
    ax.set_ylabel("Simulated cohorts passing both contrasts (%)")
    ax.set_xticks(positions, LABELS)
    ax.set_title("B  Positive-pattern frequency", loc="left", fontsize=12)
    ax.text(0, -0.21, "Both IMM - independent and IMM - static;\npositive lower CI and all four animals positive", transform=ax.transAxes, fontsize=8)
    rank = confusion[(confusion.phase == "POST") & (confusion.encoding_variant == "direction_mixture")]
    winners = [*GENERATORS, "ambiguous"]
    matrix = rank.pivot(index="generator", columns="raw_predictive_winner", values="fraction").reindex(index=GENERATORS, columns=winners).fillna(0).to_numpy()
    if not np.allclose(matrix.sum(axis=1), 1):
        raise ValueError("incomplete rank confusion")
    ax = axes[2]
    ax.imshow(matrix, vmin=0, vmax=0.65, cmap="Blues", aspect="auto")
    for (y, x), value in np.ndenumerate(matrix):
        ax.text(x, y, f"{100 * value:.1f}%", ha="center", va="center", color="white" if value > 0.4 else "black", fontsize=9)
    ax.set_yticks(positions, LABELS)
    ax.set_xticks(np.arange(5), ["Indep.", "Static", "Diff.", "IMM", "Tie"], rotation=25)
    ax.set_ylabel("True generator")
    ax.set_xlabel("Raw best predictive model per simulated event")
    ax.set_title("C  Event labels are not reliably recovered", loc="left", fontsize=11)
    ax.text(0, -0.30, "8,000 event draws per row; repeated four-animal\ntemplates, not 8,000 independent biological events", transform=ax.transAxes, fontsize=8)
    for ax in axes[:2]:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("hc-11: count-matched sensitivity is not single-event model identification", fontsize=15, y=0.96)
    fig.text(0.5, 0.875, "Known RUN maps, native count profiles, frozen 70/30 cell splits; POST direction-mixture primary analysis", ha="center", fontsize=10)
    fig.savefig(output / "hc11_conditional_recovery.png", dpi=180)
    fig.savefig(output / "hc11_conditional_recovery.pdf")
    plt.close(fig)


def run(args):
    root, audit_path, output = Path(args.run_dir).resolve(), Path(args.audit_dir).resolve() / "audit_manifest.json", Path(args.output_dir).resolve()
    manifest, audit = validate_audit(root, audit_path)
    source = Path(manifest["source_dir"])
    parent_path = source / "hc11_conditional_manifest.json"
    if file_sha256(parent_path) != manifest["input_file_sha256"]["parent_manifest"]:
        raise ValueError("changed parent manifest")
    parent = json.loads(parent_path.read_text())
    real_path = source / "hc11_conditional_summary.csv"
    if file_sha256(real_path) != parent["output_sha256"][real_path.name]:
        raise ValueError("changed real summary")
    panels = pd.read_csv(root / "recovery_simulation_panels.csv")
    patterns = pd.read_csv(root / "recovery_positive_pattern_summary.csv")
    confusion = pd.read_csv(root / "recovery_predictive_rank_confusion.csv")
    verify_rank_table(pd.read_csv(root / "recovery_event_contrasts.csv"), confusion)
    readout = build_readout(panels, patterns, pd.read_csv(real_path))
    observed = real_information(source, parent)
    information = information_comparison(pd.read_csv(root / "recovery_information_summary.csv"), observed)
    if output.exists() and any(output.iterdir()):
        raise ValueError("refusing to overwrite a report")
    output.mkdir(parents=True, exist_ok=True)
    readout.to_csv(output / "recovery_readout.csv", index=False)
    information.to_csv(output / "recovery_information_comparison.csv", index=False)
    observed.to_csv(output / "recovery_observed_information.csv", index=False)
    make_figure(panels, readout, confusion, output)
    provenance = build_script_provenance(input_paths={"recovery_manifest": root / "recovery_manifest.json", "audit_manifest": audit_path, "parent_manifest": parent_path})
    provenance.update(status="complete", rescored=False, audit_scope=audit["scope"], output_sha256={p.name: file_sha256(p) for p in output.iterdir() if p.is_file()})
    (output / "report_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"output": str(output), "rows": len(readout), "status": "complete", "rescored": False}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())
