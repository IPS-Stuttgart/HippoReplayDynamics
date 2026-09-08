#!/usr/bin/env python3
"""Non-rescoring display of the audited hc-11 conditional prediction result."""

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _provenance import file_sha256


def render(run_dir, audit_dir, output_dir):
    run_dir, audit_dir, output_dir = map(Path, (run_dir, audit_dir, output_dir))
    audit = json.loads((audit_dir / "audit_manifest.json").read_text())
    if audit["status"] != "passed" or audit["input_file_sha256"]["run_manifest"] != file_sha256(run_dir / "hc11_conditional_manifest.json"):
        raise ValueError("audit does not validate this run")
    manifest = json.loads((run_dir / "hc11_conditional_manifest.json").read_text())
    for name in ("hc11_conditional_summary.csv", "hc11_conditional_by_animal.csv"):
        if file_sha256(run_dir / name) != manifest["output_sha256"][name]:
            raise ValueError("report input changed after verification")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(run_dir / "hc11_conditional_summary.csv")
    animals = pd.read_csv(run_dir / "hc11_conditional_by_animal.csv")
    primary = summary[(summary.phase == "POST") & (summary.encoding_variant == "direction_mixture") & (summary.inference_temperature == 1)].copy()
    primary.to_csv(output_dir / "hc11_conditional_primary_readout.csv", index=False)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1.1, 1, 1.2]})
    names = ["imm_minus_static_location", "imm_minus_diffusion", "imm_minus_iid_position", "real_minus_wrong_imm", "identity_minus_rate_imm"]
    labels = ["IMM - static", "IMM - diffusion", "IMM - independent", "IMM: real - wrong map", "IMM: identity - total rate"]
    values = primary.set_index("contrast").loc[names]
    estimate = values.equal_animal_mean_event_median_delta.to_numpy()
    axs[0].errorbar(estimate, np.arange(len(names)), xerr=np.vstack([estimate - values.ci_low, values.ci_high - estimate]), fmt="o", capsize=3, color="#286b68")
    axs[0].set(yticks=np.arange(len(names)), yticklabels=labels, xlabel="Held-out score difference (nats/event)", title="A. Primary POST contrasts")
    axs[0].invert_yaxis()
    selected = animals[
        (animals.encoding_variant == "direction_mixture")
        & (animals.inference_temperature == 1)
        & (animals.contrast == "imm_minus_iid_position")
        & animals.phase.isin(["PRE", "POST"])
    ]
    rat_names = sorted(selected.rat.unique())
    for i, rat in enumerate(rat_names):
        part = selected[selected.rat == rat].set_index("phase")
        pre, post = part.loc["PRE", "mean_event_median_delta"], part.loc["POST", "mean_event_median_delta"]
        axs[1].plot([pre, post], [i, i], color="#aaa", zorder=1)
        axs[1].scatter(pre, i, color="#888", marker="s", label="PRE" if i == 0 else None)
        axs[1].scatter(post, i, color="#1677a1", label="POST" if i == 0 else None)
    axs[1].set(yticks=range(len(rat_names)), yticklabels=rat_names, xlabel="IMM - independent (nats/event)", title="B. Each animal, T=1")
    axs[1].invert_yaxis()
    axs[1].legend(frameon=False, loc="best")
    sensitivity = summary[(summary.phase == "POST") & (summary.contrast == "imm_minus_iid_position")].sort_values(
        ["encoding_variant", "inference_temperature"], ascending=[True, False]
    )
    estimate = sensitivity.equal_animal_mean_event_median_delta.to_numpy()
    axs[2].errorbar(estimate, np.arange(len(sensitivity)), xerr=np.vstack([estimate - sensitivity.ci_low, sensitivity.ci_high - estimate]), fmt="o", capsize=3, color="#81538a")
    sensitivity_labels = [
        f"{'Direction mixture' if row.encoding_variant == 'direction_mixture' else 'Pooled'}; T={row.inference_temperature:g}" for row in sensitivity.itertuples()
    ]
    axs[2].set(yticks=np.arange(len(sensitivity)), yticklabels=sensitivity_labels, xlabel="IMM - independent (nats/event)", title="C. Declared POST sensitivities")
    axs[2].invert_yaxis()
    for ax in axs:
        ax.axvline(0, color="#333", linewidth=0.8, linestyle="--", zorder=0)
        ax.grid(axis="x", alpha=0.15)
    fig.suptitle("hc-11: conditional cross-cell prediction", fontsize=15)
    fig.text(
        0.5,
        0.02,
        "160 POST + 160 PRE candidates; 8 sessions, 4 animals. Event medians across 5 splits, then equal-animal means.\nBars: exploratory 95% hierarchical bootstrap intervals. Scoring held-out probabilities at T=1 in every condition.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.93), w_pad=2)
    path = output_dir / "hc11_conditional_external_prediction.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(render(args.run_dir, args.audit_dir, args.output_dir))
