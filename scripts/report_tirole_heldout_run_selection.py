"""Non-rescoring selection calibration against behavioral labels in real RUN."""

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

GROUPS = ("all", "full", "half", "retained", "lost", "gained", "lost_support", "lost_sequence")
READOUTS = ("evaluation_true_probability", "evaluation_true_z", "evaluation_correct")
META = ["window_id", "session", "animal", "truth_track", "fold", "time_block", "order", "likelihood"]


def check_cube(scores, content, windows):
    if windows.empty or windows.window_id.duplicated().any():
        raise ValueError("nonempty unique behavior windows required")
    keys = ["window_id", "split", "order", "likelihood", "repeat"]
    expected = pd.MultiIndex.from_product([sorted(windows.window_id), range(5), ["original", "whole_bin_shuffled"], ["poisson", "conditional_count"], range(-1, 5)], names=keys)
    actual = pd.MultiIndex.from_frame(scores[keys])
    if len(scores) != len(expected) or actual.duplicated().any() or len(expected.difference(actual)):
        raise ValueError("incomplete score cube")
    expected_b = pd.MultiIndex.from_product([sorted(windows.window_id), range(5)], names=["window_id", "split"])
    actual_b = pd.MultiIndex.from_frame(content[["window_id", "split"]])
    if len(content) != len(expected_b) or actual_b.duplicated().any() or len(expected_b.difference(actual_b)):
        raise ValueError("incomplete independent readouts")
    if not scores.fraction.eq(np.where(scores.repeat.eq(-1), 1.0, 0.5)).all():
        raise ValueError("coverage identity mismatch")
    for frame in (scores, content):
        ref = windows.set_index("window_id").reindex(frame.window_id)
        for col in ["truth_track", "fold", "time_block"]:
            if not np.array_equal(ref[col], frame[col]):
                raise ValueError("behavior metadata mismatch")
    for col in ["sequence_accepted", "sequence_eligible"]:
        if scores[col].isna().any() or not scores[col].isin([True, False]).all():
            raise ValueError("invalid selection flags")
    if (scores.sequence_accepted & ~scores.sequence_eligible).any():
        raise ValueError("unsupported selection")


def window_statistics(scores, content):
    keys = ["window_id", "split", "order", "likelihood"]
    full = scores[scores.repeat.eq(-1)].set_index(keys)
    half = scores[scores.repeat.ge(0)].copy()
    if full.index.duplicated().any() or half.duplicated([*keys, "repeat"]).any():
        raise ValueError("duplicate observations")
    ref = full.reindex(pd.MultiIndex.from_frame(half[keys]))
    if ref.sequence_accepted.isna().any():
        raise ValueError("missing full pairing")
    half["full_accepted"] = ref.sequence_accepted.to_numpy()
    half["full_label"] = ref.inferred_track.to_numpy()
    half = half.merge(content[["window_id", "split", *READOUTS]], on=["window_id", "split"], how="left", validate="many_to_one", indicator=True)
    if not half._merge.eq("both").all():
        raise ValueError("missing fixed evaluation cells")
    f, h = half.full_accepted.to_numpy(bool), half.sequence_accepted.to_numpy(bool)
    eligible = half.sequence_eligible.to_numpy(bool)
    groups = dict(zip(GROUPS, [np.ones(len(half), bool), f, h, f & h, f & ~h, ~f & h, f & ~h & ~eligible, f & ~h & eligible], strict=True))
    out = []
    for name, use in groups.items():
        a = half[META].copy()
        a["group_mass"] = use.astype(float)
        labels = half.inferred_track if name in ("half", "gained") else half.full_label
        a["decoded_label_mass"] = use & labels.isin([1, 2])
        a["decoded_correct_mass"] = use & labels.eq(half.truth_track)
        a["decoded_track2_mass"] = use & labels.eq(2)
        for col in READOUTS:
            valid = use & np.isfinite(half[col])
            a[col + "_numerator"] = np.where(valid, half[col], 0.0)
            a[col + "_denominator"] = valid.astype(float)
        a = a.groupby(META, dropna=False).mean(numeric_only=True).reset_index()
        a["group"] = name
        out.append(a)
    return pd.concat(out, ignore_index=True)


def metrics(table, weights):
    ids = sorted(table.window_id.unique())
    w = np.atleast_2d(np.asarray(weights, float))
    if w.shape[1] != len(ids) or not np.isfinite(w).all() or (w < 0).any() or (w.sum(axis=1) <= 0).any():
        raise ValueError("valid original-window weights required")
    out = {}

    def ratio(a, b):
        return np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=b > 0)

    for name in GROUPS:
        g = table[table.group.eq(name)].set_index("window_id").reindex(ids)
        if len(g) != len(ids) or g.group_mass.isna().any():
            raise ValueError("incomplete window groups")
        mass = g.group_mass.to_numpy()
        t2 = g.truth_track.eq(2).to_numpy()
        total = w @ mass
        out[name + "_mass"] = ratio(total, w.sum(axis=1))
        out[name + "_true_track2_fraction"] = ratio(w @ (mass * t2), total)
        for track in (1, 2):
            use = g.truth_track.eq(track).to_numpy()
            out[name + f"_acceptance_track{track}"] = ratio(w @ (mass * use), w @ use)
        for col in READOUTS:
            den = w @ g[col + "_denominator"].to_numpy()
            out[name + "_" + col] = ratio(w @ g[col + "_numerator"].to_numpy(), den)
            out[name + "_" + col + "_finite_fraction"] = ratio(den, total)
        den = w @ g.decoded_label_mass.to_numpy()
        out[name + "_decoded_correct_fraction"] = ratio(w @ g.decoded_correct_mass.to_numpy(), den)
        out[name + "_decoded_track2_fraction"] = ratio(w @ g.decoded_track2_mass.to_numpy(), den)
    out["half_minus_full_true_track2_fraction"] = out["half_true_track2_fraction"] - out["full_true_track2_fraction"]
    out["half_minus_full_acceptance"] = out["half_mass"] - out["full_mass"]
    for t in (1, 2):
        out[f"loss_given_full_track{t}"] = ratio(out[f"lost_acceptance_track{t}"], out[f"full_acceptance_track{t}"])
    return out


def block_weights(windows, seed, n_boot=2000):
    a = windows.sort_values("window_id").reset_index(drop=True)
    if a.empty or a.window_id.duplicated().any():
        raise ValueError("unique nonempty bootstrap windows required")
    weights = np.zeros((n_boot, len(a)))
    rng = np.random.default_rng(seed)
    strata = []
    for (track, fold), g in a.groupby(["truth_track", "fold"]):
        blocks, inverse = np.unique(g.time_block, return_inverse=True)
        multiplicity = rng.multinomial(len(blocks), np.full(len(blocks), 1 / len(blocks)), size=n_boot)
        values = multiplicity[:, inverse].astype(float)
        values *= len(g) / values.sum(axis=1, keepdims=True)
        weights[:, g.index] = values
        strata.append({"truth_track": track, "fold": fold, "n_windows": len(g), "n_blocks": len(blocks)})
    strata = pd.DataFrame(strata)
    adequate = len(strata) == 10 and strata.n_blocks.ge(2).all()
    return weights, strata, bool(adequate)


def run(source, output):
    if output.exists():
        raise ValueError("new immutable RUN report required")
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["git_dirty"] or not manifest["all_cells_and_maps_training_fold_only"]:
        raise ValueError("incomplete or unfrozen RUN control")
    for name, digest in manifest["output_sha256"].items():
        p = (source / name).resolve()
        if not p.is_relative_to(source.resolve()) or file_sha256(p) != digest:
            raise ValueError("changed or unsafe source artifact")
    windows = pd.read_csv(source / "RUN_windows.csv")
    scores = pd.read_csv(source / "RUN_sequence_scores.csv")
    content = pd.read_csv(source / "RUN_independent_content.csv")
    check_cube(scores, content, windows)
    stats = window_statistics(scores, content)
    weights, strata, adequate = block_weights(windows, stable_seed(20260918, manifest["session"], "RUN-stratified-block-bootstrap"))
    summaries = []
    intervals = []
    cache = {}

    def interval(name, value, draws, **meta):
        finite = draws[np.isfinite(draws)]
        ok = adequate and len(finite) >= 0.95 * len(draws)
        lo, hi = np.quantile(finite, [0.025, 0.975]) if ok else (np.nan, np.nan)
        return {**meta, "metric": name, "estimate": float(value), "ci025": lo, "ci975": hi, "finite_bootstrap_fraction": len(finite) / len(draws), "interval_supported": ok}

    for (order, likelihood), g in stats.groupby(["order", "likelihood"]):
        point = metrics(g, np.ones(len(windows)))
        draws = metrics(g, weights)
        cache[order, likelihood] = point, draws
        meta = {"session": manifest["session"], "order": order, "likelihood": likelihood, "n_windows": len(windows)}
        summaries.append({**meta, **{k: v[0] for k, v in point.items()}})
        intervals.extend(interval(k, v[0], draws[k], **meta) for k, v in point.items())
    for likelihood in ["poisson", "conditional_count"]:
        a, aa = cache["original", likelihood]
        b, bb = cache["whole_bin_shuffled", likelihood]
        for group in ["full", "half"]:
            k = group + "_mass"
            intervals.append(
                interval(
                    group + "_original_minus_shuffled_acceptance",
                    a[k][0] - b[k][0],
                    aa[k] - bb[k],
                    session=manifest["session"],
                    order="paired_order",
                    likelihood=likelihood,
                    n_windows=len(windows),
                )
            )
    ci = pd.DataFrame(intervals)
    balance = windows.groupby(["fold", "truth_track"]).size().unstack(fill_value=0)
    balanced = balance.shape == (5, 2) and balance[1].eq(balance[2]).all()
    gates = [
        {"gate": "complete_score_cube", "passed": True},
        {"gate": "balanced_contexts_all_folds", "passed": bool(balanced)},
        {"gate": "multiple_blocks_per_stratum", "passed": adequate},
        {"gate": "fold_only_maps_and_QC", "passed": True},
    ]
    gates.append({"gate": "technical_overall", "passed": all(g["passed"] for g in gates)})
    output.mkdir(parents=True)
    stats.to_csv(output / "window_statistics.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "RUN_selection_summary.csv", index=False)
    ci.to_csv(output / "RUN_block_uncertainty.csv", index=False)
    strata.to_csv(output / "bootstrap_strata.csv", index=False)
    pd.DataFrame(gates).to_csv(output / "gate_summary.csv", index=False)
    text = [
        "# Held-out RUN selection calibration",
        "",
        "Real RUN spikes and behavior-defined track labels; no generated spikes and no sleep replay rescoring.",
        "Maps and cell QC exclude the test block and one-second guards. Repeated splits and thinning draws are averaged within windows.",
        "Intervals resample ten-second blocks within track/fold strata, preserving the balanced behavioral context design.",
        "",
    ]
    for likelihood in ["poisson", "conditional_count"]:
        text.append("## " + likelihood)
        for name in [
            "full_mass",
            "half_mass",
            "full_true_track2_fraction",
            "half_true_track2_fraction",
            "half_minus_full_true_track2_fraction",
            "lost_evaluation_true_z",
            "lost_evaluation_correct",
        ]:
            row = ci[ci.order.eq("original") & ci.likelihood.eq(likelihood) & ci.metric.eq(name)].iloc[0]
            text.append(f"- {name}: {row.estimate:.6f} [{row.ci025:.6f}, {row.ci975:.6f}]")
    text += [
        "",
        "This control can identify a measurement effect in known behavioral contexts. It cannot establish selective loss of biological SleepPOST replay or correct its experience proportions.",
        "Two QC-selected sessions are not independent population replication. Unsuccessful or weak intervals must remain visible; no post-outcome threshold changes.",
    ]
    (output / "heldout_RUN_selection.md").write_text("\n".join(text) + "\n")
    git = lambda *a: subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()
    m = {
        "status": "complete",
        "session": manifest["session"],
        "code_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_dir": str(source.resolve()),
        "source_manifest_sha256": file_sha256(source / "manifest.json"),
        "non_rescoring": True,
        "biological_replay_bias_established": False,
        "n_windows": len(windows),
        "n_bootstrap": 2000,
        "output_sha256": {p.name: file_sha256(p) for p in output.iterdir()},
    }
    (output / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    run(args.experiment_dir, args.output_dir)
