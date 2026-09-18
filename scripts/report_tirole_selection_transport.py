"""Known-label selection effects, aggregated at the original count-anchor level."""

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

GROUPS = ("full", "half", "retained", "lost", "gained")


def anchor_statistics(scores, content):
    keys = ["anchor_event", "split", "draw", "truth_track"]
    full = scores[scores.fraction == 1].set_index([*keys, "generator"])
    half = scores[scores.fraction == 0.5].copy()
    if full.index.duplicated().any() or half.duplicated([*keys, "generator", "repeat"]).any() or content.duplicated(keys).any():
        raise ValueError("duplicate input copies")
    lookup = full.reindex(pd.MultiIndex.from_frame(half[[*keys, "generator"]]))
    if lookup.sequence_accepted.isna().any():
        raise ValueError("missing paired full score")
    half["full_accepted"] = lookup.sequence_accepted.to_numpy()
    half["full_label"] = lookup.inferred_track.to_numpy()
    joined = half.merge(content[[*keys, "poisson_true_signed_z", "conditional_count_true_signed_z"]], on=keys, how="left", validate="many_to_one", indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("missing independent evaluation record")
    output = []
    grouping = ["session", "animal", "anchor_event", "epoch", "ripple_positive", "truth_track", "generator"]
    for part, frame in [("all", joined), ("even", joined[joined.draw % 2 == 0]), ("odd", joined[joined.draw % 2 == 1])]:
        f, h = frame.full_accepted.to_numpy(bool), frame.sequence_accepted.to_numpy(bool)
        for group, use in {"full": f, "half": h, "retained": f & h, "lost": f & ~h, "gained": ~f & h}.items():
            x = frame[grouping].copy()
            x["group_mass"] = use.astype(float)
            label = frame.full_label if group in ("full", "retained", "lost") else frame.inferred_track
            x["label2_mass"] = use & label.eq(2)
            x["correct_label_mass"] = use & label.eq(frame.truth_track)
            for readout in ["poisson", "conditional_count"]:
                z = frame[readout + "_true_signed_z"].to_numpy()
                x[readout + "_z_numerator"] = np.where(use & np.isfinite(z), z, 0.0)
                x[readout + "_z_denominator"] = use & np.isfinite(z)
            averaged = x.groupby(grouping, dropna=False).mean(numeric_only=True).reset_index()
            averaged["draw_partition"] = part
            averaged["group"] = group
            output.append(averaged)
    return pd.concat(output, ignore_index=True)


def ratios(anchor, weights=None):
    ids = sorted(anchor.anchor_event.unique())
    w = np.ones(len(ids)) if weights is None else np.asarray(weights, float)
    if w.shape != (len(ids),) or (w < 0).any() or w.sum() <= 0:
        raise ValueError("nonempty original-anchor weights required")
    result = {}
    for group in GROUPS:
        a = anchor[anchor.group == group]
        if len(a) != len(ids) * 2 or a.duplicated(["anchor_event", "truth_track"]).any():
            raise ValueError("incomplete paired-truth anchor summary")
        ix = pd.MultiIndex.from_product([ids, [1, 2]], names=["anchor_event", "truth_track"])
        a = a.set_index(ix.names).reindex(ix)
        mass = a.group_mass.to_numpy().reshape(-1, 2)
        total = (w[:, None] * mass).sum()
        result[group + "_true_track2_fraction"] = (w * mass[:, 1]).sum() / total if total else np.nan
        for t in [1, 2]:
            result[group + f"_acceptance_track{t}"] = np.dot(w, mass[:, t - 1]) / w.sum()
        for col, label in [("label2_mass", "reported_track2_fraction"), ("correct_label_mass", "correct_label_fraction")]:
            value = a[col].to_numpy().reshape(-1, 2)
            result[group + "_" + label] = (w[:, None] * value).sum() / total if total else np.nan
        for readout in ["poisson", "conditional_count"]:
            numerator = (w[:, None] * a[readout + "_z_numerator"].to_numpy().reshape(-1, 2)).sum()
            denominator = (w[:, None] * a[readout + "_z_denominator"].to_numpy().reshape(-1, 2)).sum()
            result[group + "_" + readout + "_true_signed_z"] = numerator / denominator if denominator else np.nan
    for group in ["half", "retained", "lost", "gained"]:
        result[group + "_minus_full_true_track2_fraction"] = result[group + "_true_track2_fraction"] - result["full_true_track2_fraction"]
    for t in [1, 2]:
        denominator = result[f"full_acceptance_track{t}"]
        result[f"loss_given_full_track{t}"] = result[f"lost_acceptance_track{t}"] / denominator if denominator else np.nan
    return result


def run(source, output):
    if output.exists():
        raise ValueError("new immutable selection transport report required")
    m = json.loads((source / "manifest.json").read_text())
    if m["status"] != "complete" or m["git_dirty"] or m["real_data_rescored"]:
        raise ValueError("invalid transport experiment")
    for name, digest in m["output_sha256"].items():
        if file_sha256(source / name) != digest:
            raise ValueError("changed input")
    scores = pd.concat([pd.read_csv(p) for p in sorted((source / "sequence_shards").glob("*.csv"))], ignore_index=True)
    content = pd.concat([pd.read_csv(p) for p in sorted((source / "content_shards").glob("*.csv"))], ignore_index=True)
    if len(scores) != m["n_anchors"] * 5 * 20 * 2 * 2 * 6 or len(content) != m["n_anchors"] * 5 * 20 * 2:
        raise ValueError("incomplete experiment")
    expected = pd.MultiIndex.from_product(
        [sorted(scores.anchor_event.unique()), range(5), range(20), [1, 2], ["ordered", "whole_bin_shuffled"], range(-1, 5)],
        names=["anchor_event", "split", "draw", "truth_track", "generator", "repeat"],
    )
    actual = pd.MultiIndex.from_frame(scores[expected.names])
    if len(actual) != len(expected) or actual.duplicated().any() or len(expected.difference(actual)):
        raise ValueError("missing or duplicate score keys")
    if not ((scores.repeat == -1) == scores.fraction.eq(1)).all():
        raise ValueError("full/half score identity mismatch")
    anchor = anchor_statistics(scores, content)
    records, uncertainty = [], []
    grouping = ["session", "animal", "epoch", "ripple_positive", "generator", "draw_partition"]
    for key, group in anchor.groupby(grouping, dropna=False):
        meta = dict(zip(grouping, key, strict=True))
        point = ratios(group)
        records.append({**meta, "n_original_anchors": group.anchor_event.nunique(), **point})
        if meta["epoch"] == "POST" and meta["ripple_positive"] and meta["draw_partition"] == "all":
            n = group.anchor_event.nunique()
            rng = np.random.default_rng(stable_seed(20260918, meta["session"], meta["generator"], "selection-transport-anchor-bootstrap"))
            draws = pd.DataFrame([ratios(group, w) for w in rng.multinomial(n, np.full(n, 1 / n), size=2000)])
            for metric, value in point.items():
                finite = draws[metric].dropna().to_numpy()
                lo, hi = np.quantile(finite, [0.025, 0.975]) if len(finite) >= 1900 and n >= 2 else [np.nan, np.nan]
                uncertainty.append(
                    {**meta, "metric": metric, "estimate": value, "ci025": lo, "ci975": hi, "n_original_anchors": n, "finite_bootstrap_fraction": len(finite) / 2000}
                )
    output.mkdir(parents=True)
    anchor.to_csv(output / "anchor_statistics.csv", index=False)
    pd.DataFrame(records).to_csv(output / "selection_summary.csv", index=False)
    pd.DataFrame(uncertainty).to_csv(output / "primary_anchor_uncertainty.csv", index=False)
    text = [
        "# Known-label selection transport",
        "",
        "Simulation only; no corrected real replay fractions.",
        "All Monte Carlo copies, cell splits and coverage repeats were averaged before original-anchor aggregation.",
        "Conditional uncertainty resamples original anchors, not simulated events or neurons.",
        "",
    ]
    for row in records:
        if row["epoch"] == "POST" and row["ripple_positive"] and row["draw_partition"] == "all":
            text.append(
                f"{row['session']} {row['generator']}: full true track-2 fraction {row['full_true_track2_fraction']:.6f}; half {row['half_true_track2_fraction']:.6f}; retained {row['retained_true_track2_fraction']:.6f}."
            )
    text += ["", "A selection effect in known-label simulations establishes a possible measurement mechanism, not replicated real experience bias."]
    (output / "selection_transport.md").write_text("\n".join(text) + "\n")
    manifest = {
        "status": "complete",
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "non_rescoring": True,
        "session": m["session"],
        "source_manifest": str((source / "manifest.json").resolve()),
        "source_manifest_sha256": file_sha256(source / "manifest.json"),
        "simulation_only": True,
        "real_fractions_corrected": False,
        "n_original_anchors": m["n_anchors"],
        "output_sha256": {p.name: file_sha256(p) for p in output.iterdir()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    run(a.experiment_dir, a.output_dir)
