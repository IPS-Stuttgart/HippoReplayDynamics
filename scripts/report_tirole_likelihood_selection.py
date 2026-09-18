"""Paired original-anchor comparison of Poisson and count-conditioned selection."""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from hipporeplayimm.tirole_two_track import file_sha256
from hipporeplayimm.two_track_content import stable_seed
from scripts.report_tirole_selection_transport import GROUPS, anchor_statistics

STATISTICS = ["group_mass", "label2_mass", "correct_label_mass", "poisson_z_numerator", "poisson_z_denominator", "conditional_count_z_numerator", "conditional_count_z_denominator"]


def metrics(anchors, weights):
    ids = sorted(anchors.anchor_event.unique())
    weights = np.atleast_2d(np.asarray(weights, float))
    if weights.shape[1] != len(ids) or not np.isfinite(weights).all() or (weights < 0).any() or not (weights.sum(axis=1) > 0).all():
        raise ValueError("valid original-anchor weights required")
    ix = pd.MultiIndex.from_product([ids, [1, 2], GROUPS], names=["anchor_event", "truth_track", "group"])
    if len(anchors) != len(ix) or anchors.duplicated(ix.names).any():
        raise ValueError("incomplete anchor truth/group cube")
    cube = anchors.set_index(ix.names).reindex(ix)[STATISTICS].to_numpy().reshape(len(ids), 2, 5, 7)
    if not np.isfinite(cube).all():
        raise ValueError("nonfinite anchor contributions")
    total = np.einsum("wa,atgs->wtgs", weights, cube)

    def ratio(a, b):
        return np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=b > 0)

    result = {}
    for k, g in enumerate(GROUPS):
        mass = total[:, :, k, 0].sum(axis=1)
        result[g + "_true_track2_fraction"] = ratio(total[:, 1, k, 0], mass)
        for t in [1, 2]:
            result[g + f"_acceptance_track{t}"] = ratio(total[:, t - 1, k, 0], weights.sum(axis=1))
        for col, name in [(1, "reported_track2_fraction"), (2, "correct_label_fraction")]:
            result[g + "_" + name] = ratio(total[:, :, k, col].sum(axis=1), mass)
        for col, name in [(3, "poisson"), (5, "conditional_count")]:
            result[g + "_" + name + "_true_signed_z"] = ratio(total[:, :, k, col].sum(axis=1), total[:, :, k, col + 1].sum(axis=1))
    for g in GROUPS[1:]:
        result[g + "_minus_full_true_track2_fraction"] = result[g + "_true_track2_fraction"] - result["full_true_track2_fraction"]
    for t in [1, 2]:
        result[f"loss_given_full_track{t}"] = ratio(result[f"lost_acceptance_track{t}"], result[f"full_acceptance_track{t}"])
    for g in ["full", "half"]:
        result[g + "_signed_distortion"] = result[g + "_true_track2_fraction"] - 0.5
        result[g + "_absolute_distortion"] = np.abs(result[g + "_signed_distortion"])
    return result


def paired_metrics(old, new):
    if set(old) != set(new):
        raise ValueError("metric mismatch")
    out = {name + "_conditional_minus_poisson": new[name] - old[name] for name in old}
    for g in ["full", "half"]:
        name = g + "_absolute_distortion"
        out[g + "_absolute_distortion_reduction"] = old[name] - new[name]
    return out


def check_scores(scores, ids):
    ix = pd.MultiIndex.from_product(
        [sorted(ids), range(5), range(20), [1, 2], ["ordered", "whole_bin_shuffled"], range(-1, 5)], names=["anchor_event", "split", "draw", "truth_track", "generator", "repeat"]
    )
    actual = pd.MultiIndex.from_frame(scores[ix.names])
    if len(scores) != len(ix) or actual.duplicated().any() or len(ix.difference(actual)) or not ((scores.repeat == -1) == scores.fraction.eq(1)).all():
        raise ValueError("incomplete frozen transport scores")


def run(source, output):
    if output.exists():
        raise ValueError("new immutable paired likelihood report required")
    m = json.loads((source / "manifest.json").read_text())
    old = Path(m["source_transport_dir"])
    om = json.loads((old / "manifest.json").read_text())
    if (
        m["status"] != "complete"
        or m["git_dirty"]
        or m["real_data_rescored"]
        or m["thresholds_changed"]
        or file_sha256(old / "manifest.json") != m["source_transport_manifest_sha256"]
    ):
        raise ValueError("invalid sensitivity provenance")
    for folder, manifest in [(source, m), (old, om)]:
        for name, h in manifest["output_sha256"].items():
            if file_sha256(folder / name) != h:
                raise ValueError("changed hashed source")
    anchors = pd.read_csv(source / "count_anchors.csv")
    content = pd.concat([pd.read_csv(p) for p in sorted((old / "content_shards").glob("*.csv"))], ignore_index=True)
    tables = []
    for name, folder in [("poisson", old), ("conditional_count", source)]:
        scores = pd.concat([pd.read_csv(p) for p in sorted((folder / "sequence_shards").glob("*.csv"))], ignore_index=True)
        check_scores(scores, anchors.event_id)
        a = anchor_statistics(scores, content)
        a["likelihood"] = name
        tables.append(a)
    table = pd.concat(tables, ignore_index=True)
    records = []
    intervals = []
    comparisons = []
    grouping = ["session", "animal", "epoch", "ripple_positive", "generator", "draw_partition"]
    for key, g in table.groupby(grouping, dropna=False):
        meta = dict(zip(grouping, key, strict=True))
        n = g.anchor_event.nunique()
        by = {k: a for k, a in g.groupby("likelihood")}
        point = {k: metrics(a, np.ones(n)) for k, a in by.items()}
        for name, values in point.items():
            records.append({**meta, "likelihood": name, "n_original_anchors": n, **{k: v[0] for k, v in values.items()}})
        if not (meta["epoch"] == "POST" and meta["ripple_positive"] and meta["draw_partition"] == "all"):
            continue
        weights = np.random.default_rng(stable_seed(20260918, meta["session"], meta["generator"], "selection-transport-anchor-bootstrap")).multinomial(
            n, np.full(n, 1 / n), size=2000
        )
        bootstrap = {k: metrics(a, weights) for k, a in by.items()}

        def row(metric, point, draws, n=n, meta=meta):
            good = draws[np.isfinite(draws)]
            lo, hi = np.quantile(good, [0.025, 0.975]) if len(good) >= 1900 and n >= 2 else [np.nan, np.nan]
            return {**meta, "n_original_anchors": n, "metric": metric, "estimate": float(point), "ci025": lo, "ci975": hi, "finite_bootstrap_fraction": len(good) / 2000}

        for name, values in point.items():
            for metric, value in values.items():
                intervals.append({**row(metric, value[0], bootstrap[name][metric]), "likelihood": name})
        paired = paired_metrics(point["poisson"], point["conditional_count"])
        draws = paired_metrics(bootstrap["poisson"], bootstrap["conditional_count"])
        for metric, value in paired.items():
            comparisons.append(row(metric, value[0], draws[metric]))
    output.mkdir(parents=True)
    table.to_csv(output / "anchor_statistics.csv", index=False)
    pd.DataFrame(records).to_csv(output / "likelihood_condition_summary.csv", index=False)
    uncertainty = pd.DataFrame(intervals)
    uncertainty.to_csv(output / "primary_anchor_uncertainty.csv", index=False)
    paired = pd.DataFrame(comparisons)
    paired.to_csv(output / "paired_likelihood_contrasts.csv", index=False)
    text = [
        "# Count-conditioned selection diagnostic",
        "",
        "Same known-label simulated observations; only the sequence likelihood changes.",
        "No real data rescored, no revised thresholds, no corrected biological experience fractions.",
        "Intervals resample original count anchors, not individual spike realizations.",
        "",
    ]
    for name in ["poisson", "conditional_count"]:
        for metric in ["full_true_track2_fraction", "half_true_track2_fraction", "half_minus_full_true_track2_fraction"]:
            a = uncertainty[uncertainty.likelihood.eq(name) & uncertainty.generator.eq("ordered") & uncertainty.metric.eq(metric)].iloc[0]
            text.append(f"{name} {metric}: {a.estimate:.6f} [{a.ci025:.6f}, {a.ci975:.6f}].")
    text += [
        "",
        "Conditional likelihood discards count information; lower distortion with no useful detections would not establish recovery.",
        "These simulation diagnostics cannot establish a replicated biological experience bias.",
    ]
    (output / "likelihood_selection.md").write_text("\n".join(text) + "\n")
    git = lambda *a: subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()
    out = {
        "status": "complete",
        "session": m["session"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "code_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "source_manifest_sha256": file_sha256(source / "manifest.json"),
        "source_dir": str(source.resolve()),
        "original_manifest_sha256": file_sha256(old / "manifest.json"),
        "non_rescoring": True,
        "n_anchor_rows": len(table),
        "n_summary_rows": len(records),
        "n_interval_rows": len(intervals),
        "n_paired_contrasts": len(comparisons),
        "biological_bias_established": False,
        "real_fractions_corrected": False,
        "output_sha256": {p.name: file_sha256(p) for p in output.iterdir()},
    }
    (output / "manifest.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    run(a.experiment_dir, a.output_dir)
