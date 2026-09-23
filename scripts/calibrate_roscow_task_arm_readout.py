#!/usr/bin/env python3
"""Nested task-only probability calibration; never score replay or outcome effects."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import build_script_provenance, file_sha256
from validate_roscow_task_arm_readout import PARAMETERS, READOUTS, fit_rates, scores, stable_seed

TEMPERATURES = np.array([1, 2, 4, 8, 16, 32, 64, 128, np.inf])
METHODS = ("uncalibrated", "nested_temperature")


def logits(counts, rates, exposure, kind):
    # Same fixed likelihood as the earlier posterior helper, kept in log space.
    total = rates.sum(axis=1)
    if kind == "composition":
        return counts @ np.log(rates / total[:, None]).T
    if kind == "count_only":
        return counts.sum(axis=1)[:, None] * np.log(total) - exposure * total
    return counts @ np.log(rates).T - exposure * total


def log_probabilities(values, temperature):
    if not np.isfinite(values).all() or temperature <= 0:
        raise ValueError("finite logits and positive temperature required")
    scaled = values / temperature
    return scaled - logsumexp(scaled, axis=-1, keepdims=True)


def fold_logits(counts, labels, exposure):
    """Cache symmetric leave-two-out fits; an outer trial is absent from its row."""
    n = len(labels)
    outer = {k: np.empty((n, 3)) for k in READOUTS}
    inner = {k: np.full((n, n, 3), np.nan) for k in READOUTS}
    for i in range(n):
        train = np.arange(n) != i
        rates = fit_rates(counts[train], labels[train], exposure)
        for kind in READOUTS:
            outer[kind][i] = logits(counts[i:i + 1], rates, exposure, kind)[0]
        for j in range(i + 1, n):
            train = (np.arange(n) != i) & (np.arange(n) != j)
            rates = fit_rates(counts[train], labels[train], exposure)
            for kind in READOUTS:
                values = logits(counts[[i, j]], rates, exposure, kind)
                inner[kind][i, j] = values[1]
                inner[kind][j, i] = values[0]
    return outer, inner


def choose_temperature(values, labels):
    quality = np.array([log_probabilities(values, t)[np.arange(len(labels)), labels].mean()
                        for t in TEMPERATURES])
    choice = np.flatnonzero(quality >= quality.max() - 1e-12)[-1]
    return TEMPERATURES[choice], float(quality[choice])


def calibrated_session(counts, labels, exposure):
    outer, inner = fold_logits(counts, labels, exposure)
    predictions, temperatures = {}, {}
    for kind in READOUTS:
        predictions[kind, "uncalibrated"] = log_probabilities(outer[kind], 1)
        calibrated, chosen = [], []
        for i in range(len(labels)):
            keep = np.arange(len(labels)) != i
            temperature, _ = choose_temperature(inner[kind][i, keep], labels[keep])
            calibrated.append(log_probabilities(outer[kind][i], temperature))
            chosen.append(temperature)
        predictions[kind, "nested_temperature"] = np.asarray(calibrated)
        temperatures[kind] = np.asarray(chosen)
    return predictions, temperatures


def evaluate(logp, labels):
    probability = np.exp(logp)
    result = scores(probability, labels)
    result["mean_log_score_above_chance"] = float(logp[np.arange(len(labels)), labels].mean() + np.log(3))
    result["brier_score"] = float(np.sum((probability - np.eye(3)[labels]) ** 2, axis=1).mean())
    result["mean_max_probability"] = float(probability.max(axis=1).mean())
    return result


def aggregate(table):
    output = []
    for (animal, kind), group in table.groupby(["animal", "readout"], sort=True):
        for method in METHODS:
            actual, null, uniform = [], [], []
            for session, rows in group[group.method == method].groupby("session"):
                rows = rows.sort_values("shift")
                real = rows[rows["shift"] == 0].iloc[0]
                actual.append(real.mean_log_score_above_chance)
                uniform.append(real.uniform_fraction)
                rng = np.random.default_rng(stable_seed(PARAMETERS["seed"], animal, int(session), kind, "calibration_null"))
                null.append(rng.choice(rows.mean_log_score_above_chance.to_numpy(), PARAMETERS["aggregate_null_draws"]))
            dist = np.mean(null, axis=0)
            observed, p95 = float(np.mean(actual)), float(np.quantile(dist, 0.95))
            output.append({
                "animal": animal, "readout": kind, "method": method, "n_sessions": len(actual),
                "mean_session_log_score_above_chance": observed,
                "null_median": float(np.median(dist)), "null_p95": p95,
                "empirical_p_value": float((1 + (dist >= observed).sum()) / (1 + len(dist))),
                "mean_session_uniform_fraction": float(np.mean(uniform)),
                "useful_probability_screen": bool(observed > max(0.0, p95)),
            })
    return pd.DataFrame(output)


def plot_summary(table, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), sharey=True)
    for ax, kind in zip(axes, READOUTS, strict=True):
        sub = table[(table.readout == kind) & (table.method == "nested_temperature")]
        for i, row in enumerate(sub.itertuples()):
            ax.plot([i, i], [row.null_median, row.null_p95], color="0.6", linewidth=3)
            ax.scatter(i, row.mean_session_log_score_above_chance, color="#166d8f", zorder=3)
        ax.axhline(0, linestyle=":", color="0.4")
        ax.set_xticks(range(len(sub)), [f"{r.animal}\nn={r.n_sessions}" for r in sub.itertuples()], fontsize=8)
        ax.set_title(kind.replace("_", " "))
    axes[0].set_ylabel("Held-out log score above uniform (nats/trial)")
    fig.suptitle("Nested task-only calibration: dots = real; gray = shift-null median to p95", fontsize=10)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def run(args):
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    previous = json.loads((source / "manifest.json").read_text())
    for name, digest in previous["outputs"].items():
        if Path(name).name != name or file_sha256(source / name) != digest:
            raise ValueError("Original output changed")
    provenance = build_script_provenance(cwd=ROOT, input_paths={
        "source_manifest": source / "manifest.json", "counts": source / "task_arm_count_matrices.npz",
        "source_sessions": source / "roscow_task_arm_sessions.csv", "producer": Path(__file__),
        "base_readout": ROOT / "scripts/validate_roscow_task_arm_readout.py",
        "protocol": ROOT / "docs/roscow_nested_calibration_protocol.md",
    })
    if provenance["git_dirty"] or provenance["code_commit"] == "unavailable":
        raise ValueError("Committed clean producer required")
    output.mkdir(parents=True, exist_ok=False)
    arrays = np.load(source / "task_arm_count_matrices.npz", allow_pickle=False)
    sessions = pd.read_csv(source / "roscow_task_arm_sessions.csv")
    exposure = PARAMETERS["window_stop_s"] - PARAMETERS["window_start_s"]
    results, prediction_rows = [], []
    for row in sessions[sessions.status == "scored"].itertuples():
        key = {"animal": row.animal, "session": int(row.session)}
        prefix = f"{row.animal}_session{row.session}"
        counts, labels = arrays[prefix + "_counts"], arrays[prefix + "_labels"]
        trials = arrays[prefix + "_trials"]
        for shift in range(len(labels)):
            shifted = np.roll(labels, shift)
            predictions, chosen = calibrated_session(counts, shifted, exposure)
            for (kind, method), logp in predictions.items():
                temperature = chosen[kind] if method == "nested_temperature" else np.ones(len(labels))
                results.append({**key, "shift": shift, "readout": kind, "method": method,
                                "n_trials": len(labels), "uniform_fraction": float(np.isinf(temperature).mean()),
                                **evaluate(logp, shifted)})
                if shift == 0:
                    for i in range(len(labels)):
                        prediction_rows.append({
                            **key, "readout": kind, "method": method, "trial_index_1based": int(trials[i]),
                            "chosen_arm_1based": int(labels[i] + 1),
                            "temperature": "uniform" if np.isinf(temperature[i]) else str(int(temperature[i])),
                            "true_arm_log_probability": float(logp[i, labels[i]]),
                            **{f"p_arm{j + 1}": float(np.exp(logp[i, j])) for j in range(3)},
                        })
        print(json.dumps({**key, "trials": len(labels), "label_shifts": len(labels), "status": "complete"}), flush=True)
    if not results:
        raise ValueError("No eligible sessions")
    result = pd.DataFrame(results)
    summary = aggregate(result)
    result.to_csv(output / "nested_calibration_shift_scores.csv", index=False)
    result[result["shift"] == 0].to_csv(output / "nested_calibration_session_scores.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output / "nested_calibration_trial_predictions.csv", index=False)
    summary.to_csv(output / "nested_calibration_by_animal.csv", index=False)
    plot_summary(summary, output / "nested_calibration_log_scores.png")
    good = summary[(summary.method == "nested_temperature") & summary.readout.isin(["poisson", "composition"])]
    all_pass = bool(len(good) == 6 and good.useful_probability_screen.all())
    manifest = {**provenance, "created_at_utc": datetime.now(UTC).isoformat(), "host": socket.gethostname(),
                "parameters": PARAMETERS, "temperature_grid": [1, 2, 4, 8, 16, 32, 64, 128, "uniform"],
                "nested_probability_screen_passed": all_pass,
                "original_task_arm_screen_passed": previous["task_arm_feasibility_passed"],
                "rest_analysis_authorized": False, "biological_hypothesis_scored": False,
                "exploratory_same_cohort_reanalysis": True,
                "outputs": {p.name: file_sha256(p) for p in output.iterdir() if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
