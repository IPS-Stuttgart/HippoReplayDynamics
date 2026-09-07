#!/usr/bin/env python3
"""Independently recount frozen score rows and figures; never rescore neurons."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ("iid_position", "static_location", "stationary", "diffusion", "first_order_imm")
OBS = ("full_poisson", "count_conditioned", "total_rate_only")
PRIMARY = {
    "count_conditioned:imm_minus_iid_position": "IMM - independent position",
    "count_conditioned:imm_minus_diffusion": "IMM - fixed diffusion",
    "count_conditioned:imm_minus_static_location": "IMM - static location",
    "count_conditioned:real_minus_wrong_imm": "Real map - permuted map (IMM)",
    "first_order_imm:identity_minus_rate_inference": "Identity-only - total-rate-only (IMM)",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("only completed runs can be interpreted")
    for name, expected in manifest["outputs_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"output digest mismatch: {name}")
    for name, expected in manifest["source_mat_sha256"].items():
        if digest(name) != expected:
            raise ValueError(f"source digest mismatch: {name}")
    scores = pd.read_csv(root / "split_scores.csv")
    events = pd.read_csv(root / "frozen_event_set.csv")
    expected_rows = len(events) * manifest["arguments"]["n_splits"] * len(manifest["arguments"]["temperatures"]) * 30
    if len(scores) != expected_rows or expected_rows == 0:
        raise ValueError("empty/incomplete row count")
    if not np.isfinite(scores[["conditional_heldout_log_score", "poisson_heldout_log_score"]]).all().all():
        raise ValueError("nonfinite scores")
    if (scores[["conditional_heldout_log_score", "poisson_heldout_log_score"]] > 1e-8).any().any():
        raise ValueError("discrete predictive probability greater than one")
    if not scores.heldout_temperature.eq(1).all():
        raise ValueError("powered held-out probabilities")
    if not scores.status.eq("success").all():
        raise ValueError("unsuccessful scoring rows")
    if not scores.cells_disjoint.eq(True).all() or not scores.heldout_used_for_inference.eq(False).all():
        raise ValueError("declared leakage failure")
    if not scores.posterior_unchanged.eq(True).all():
        raise ValueError("held-out scoring changed a training posterior")
    for _, row in scores[["train_cell_ids", "heldout_cell_ids", "n_train_cells", "n_heldout_cells"]].drop_duplicates().iterrows():
        train, held = set(str(row.train_cell_ids).split(",")), set(str(row.heldout_cell_ids).split(","))
        if not train.isdisjoint(held) or len(train) != row.n_train_cells or len(held) != row.n_heldout_cells:
            raise ValueError("invalid actual cell IDs/counts")
    group = ["session", "event_index", "split"]
    for column in ("training_counts_sha256", "n_train_spikes", "n_heldout_spikes", "n_time_bins", "n_heldout_nonzero_bins"):
        if (scores.groupby(group)[column].nunique() != 1).any():
            raise ValueError(f"observations changed between conditions: {column}")
    zero = scores.n_heldout_spikes.eq(0)
    if zero.any() and not np.allclose(scores.loc[zero, "conditional_heldout_log_score"], 0, atol=1e-9):
        raise ValueError("empty held-out bins contribute identity information")
    key = ["session", "event_index", "split", "inference_temperature", "map", "observation", "model"]
    if scores.duplicated(key).any():
        raise ValueError("duplicate condition key")
    frozen = set(events.itertuples(index=False, name=None))
    actual = set(scores[["session", "event_index"]].drop_duplicates().itertuples(index=False, name=None))
    if actual != frozen:
        raise ValueError("event selection changed")
    expected_blocks = {
        (sid, eid, split, temperature) for sid, eid in frozen for split in range(manifest["arguments"]["n_splits"]) for temperature in manifest["arguments"]["temperatures"]
    }
    actual_blocks = set(scores[key[:4]].drop_duplicates().itertuples(index=False, name=None))
    if actual_blocks != expected_blocks:
        raise ValueError("split/temperature coverage differs from the frozen design")
    rebuilt = []
    invariance_errors = []
    for (sid, eid, split, temperature), frame in scores.groupby(key[:4], sort=True):
        value = {(r.map, r.observation, r.model): r.conditional_heldout_log_score for r in frame.itertuples()}
        if len(value) != 30:
            raise ValueError("incomplete condition block")
        expected = {(m, o, d) for m in ("real", "population_code_permuted") for o in OBS for d in MODELS}
        if set(value) != expected:
            raise ValueError("unknown condition block")
        differences = {}
        for obs in OBS:
            for model in MODELS[:4]:
                differences[f"{obs}:imm_minus_{model}"] = value["real", obs, "first_order_imm"] - value["real", obs, model]
            differences[f"{obs}:real_minus_wrong_imm"] = value["real", obs, "first_order_imm"] - value["population_code_permuted", obs, "first_order_imm"]
            for model in MODELS[:2]:
                invariance_errors.append(abs(value["real", obs, model] - value["population_code_permuted", obs, model]))
        for model in MODELS:
            differences[f"{model}:identity_minus_rate_inference"] = value["real", "count_conditioned", model] - value["real", "total_rate_only", model]
            differences[f"{model}:identity_minus_full_inference"] = value["real", "count_conditioned", model] - value["real", "full_poisson", model]
        for label, delta in differences.items():
            rebuilt.append(
                {
                    "session": sid,
                    "event_index": eid,
                    "split": split,
                    "rat": sid.split("/")[0],
                    "inference_temperature": temperature,
                    "contrast": label,
                    "delta": delta,
                    "heldout_spikes": int(frame.n_heldout_spikes.iloc[0]),
                }
            )
    if max(invariance_errors) > 1e-7:
        raise ValueError("permutation changed analytical nonspatial-geometry baselines")
    rebuilt = pd.DataFrame(rebuilt)
    split_keys = ["session", "rat", "event_index", "split", "inference_temperature", "contrast"]
    exported_split = pd.read_csv(root / "split_contrasts.csv").set_index(split_keys).delta.sort_index()
    recounted_split = rebuilt.set_index(split_keys).delta.sort_index()
    pd.testing.assert_index_equal(exported_split.index, recounted_split.index)
    np.testing.assert_allclose(exported_split, recounted_split, atol=1e-9)
    event_keys = [k for k in split_keys if k != "split"]
    event = rebuilt.groupby(event_keys).delta.median().sort_index()
    exported = pd.read_csv(root / "event_medians.csv").set_index(event_keys).delta.sort_index()
    pd.testing.assert_index_equal(event.index, exported.index)
    np.testing.assert_allclose(event, exported, atol=1e-9)
    rats = event.reset_index().groupby(["contrast", "inference_temperature", "rat"]).delta.mean()
    estimate = rats.groupby(["contrast", "inference_temperature"]).mean().sort_index()
    reported = pd.read_csv(root / "contrast_summary.csv").set_index(["contrast", "inference_temperature"])
    np.testing.assert_allclose(estimate, reported.loc[estimate.index, "equal_animal_mean_event_median_delta"], atol=1e-9)
    # Descriptive normalization, not a substitute primary selected for significance.
    rebuilt["delta_per_heldout_spike"] = rebuilt.delta / rebuilt.heldout_spikes.replace(0, np.nan)
    normalized = rebuilt.groupby(event_keys).delta_per_heldout_spike.median().reset_index()
    normalized["status"] = np.where(normalized.delta_per_heldout_spike.notna(), "available", "zero_heldout_spikes")
    return {
        "status": "pass",
        "source_mat_files_verified": len(manifest["source_mat_sha256"]),
        "score_rows": len(scores),
        "split_contrasts_reconstructed": len(rebuilt),
        "event_contrasts_reconstructed": len(event),
        "aggregate_contrasts_reconstructed": len(estimate),
        "max_analytic_permutation_error": max(invariance_errors),
        "bootstrap_ci_independently_reconstructed": False,
        "raw_posteriors_independently_recomputed": False,
    }, normalized


def figures(root, output):
    manifest = json.loads((root / "manifest.json").read_text())
    splits = manifest["arguments"]["n_splits"]
    summary = pd.read_csv(root / "contrast_summary.csv")
    animals = pd.read_csv(root / "by_animal.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True, sharex=True, constrained_layout=True)
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for ax, temperature in zip(axes, (1.0, 0.3), strict=True):
        for y, key in enumerate(PRIMARY):
            row = summary[(summary.contrast == key) & summary.inference_temperature.eq(temperature)].iloc[0]
            values = animals[(animals.contrast == key) & animals.inference_temperature.eq(temperature)].sort_values("rat")
            ax.plot([row.ci_low, row.ci_high], [y, y], color="black", lw=1.8)
            ax.scatter(row.equal_animal_mean_event_median_delta, y, marker="D", c="black", s=35, zorder=3)
            for j, point in enumerate(values.itertuples()):
                ax.scatter(point.mean_event_median_delta, y + 0.11, color=palette[j], s=25, label=point.rat if y == 0 else None)
        ax.axvline(0, color="0.6", lw=1, ls="--")
        ax.set_title("Primary: inference temperature 1" if temperature == 1 else "Sensitivity: inference temperature 0.3")
        ax.set_xlabel("Conditional held-out log-score difference (nats/event)")
        ax.set_yticks(range(len(PRIMARY)), list(PRIMARY.values()))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    axes[0].legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle(
        f"Which cells fire: five held-out prediction contrasts\nEvent medians across {splits} split(s); black: equal-animal mean and hierarchical 95% interval", fontsize=12
    )
    fig.savefig(output / "conditional_prediction_contrasts.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("verification output directory must be empty")
    report, normalized = verify(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(args.output_dir / "descriptive_event_normalized_contrasts.csv", index=False)
    figures(args.input_dir, args.output_dir)
    report["input_manifest_sha256"] = digest(args.input_dir / "manifest.json")
    report["verifier_script_sha256"] = digest(__file__)
    report["created_at_utc"] = datetime.now(UTC).isoformat()
    report["input_directory"] = str(args.input_dir.resolve())
    report["outputs_sha256"] = {path.name: digest(path) for path in args.output_dir.iterdir() if path.is_file()}
    (args.output_dir / "independent_output_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
