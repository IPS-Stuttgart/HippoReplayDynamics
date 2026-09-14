#!/usr/bin/env python3
"""Plot frozen external outcomes, including failed informativeness checks."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    folder = args.result_dir/"validation"
    total = pd.read_csv(folder/"policy_summary.csv")
    animals = pd.read_csv(folder/"policy_by_animal.csv")
    primary = total.loc[total.split.eq(0) & total.retention.eq(.5)]
    actual = primary.loc[primary.source.eq("real")].set_index("policy")
    palette = ["#337A9E", "#B64D34"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    by_rat = animals.loc[animals.source.eq("real") & animals.split.eq(0) & animals.retention.eq(.5)]
    paired = by_rat.pivot(index="animal", columns="policy", values="endpoint_separation_cm")
    for rat, row in paired.iterrows():
        axes[0, 0].plot([0, 1], [row.random_expectation, row.diagnostic_full], "o-", label=rat)
    axes[0, 0].set(xticks=[0, 1], xticklabels=["Random expectation", "Diagnostic-selected"],
                   ylabel="A/B endpoint separation (cm)", title="A. Agreement improves in all five animals")
    axes[0, 0].legend(fontsize=8, frameon=False)
    x = np.arange(3)
    sources = ["run_test", "sim_matched", "sim_drift"]
    for offset, policy, color in zip([-.18, .18], ["random_expectation", "diagnostic_full"], palette, strict=True):
        values = [primary.loc[primary.source.eq(s) & primary.policy.eq(policy), "a_truth_error_cm"].iloc[0] for s in sources]
        axes[0, 1].bar(x+offset, values, width=.36, color=color, label=policy.replace("_", " "))
    axes[0, 1].set(xticks=x, xticklabels=["Observed RUN", "Matched simulation", "Drift simulation"],
                   ylabel="A's true-position error (cm)", title="B. But A's localization becomes worse")
    axes[0, 1].legend(fontsize=8, frameon=False)
    for offset, policy, color in zip([-.18, .18], ["random_expectation", "diagnostic_full"], palette, strict=True):
        axes[1, 0].bar(np.arange(2)+offset, actual.loc[policy, ["a_entropy", "b_entropy"]].to_numpy(float),
                       width=.36, color=color)
    axes[1, 0].set(xticks=[0, 1], xticklabels=["Population A", "Population B"], ylim=(0, 1),
                   ylabel="Normalized posterior entropy", title="C. Retained posteriors are more diffuse")
    policies = ["random_expectation", "diagnostic_full", "highest_spikes", "lowest_entropy"]
    axes[1, 1].bar(np.arange(4), actual.loc[policies, "regional_tv"], color=[palette[0], palette[1], "#779448", "#8D74A5"])
    axes[1, 1].set(xticks=np.arange(4), xticklabels=["Random", "Diagnostic", "Most spikes", "Lowest entropy"],
                   ylabel="Regional posterior total variation", title="D. Regional agreement is not correctness")
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.15)
        axis.set_axisbelow(True)
    fig.suptitle("PF-trained diagnostic tested on Tanni: full validation FAIL\nFixed 50% retention; equal animal weighting; no posterior smoothing", fontsize=13)
    fig.savefig(folder/"external_validation.png", dpi=160)
    fig.savefig(folder/"external_validation.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
