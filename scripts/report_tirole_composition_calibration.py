"""Known-composition recovery using existing known-track simulations; no rescoring."""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from hipporeplayimm.tirole_two_track import file_sha256
from hipporeplayimm.two_track_content import stable_seed
from scripts.report_tirole_content_uncertainty import finite_row_mean, summarize_bootstrap
from scripts.report_tirole_measurement_calibration import check_rows

RETENTION = ((0.5, 0.5), (0.6, 0.4), (0.4, 0.6), (0.75, 0.25), (0.25, 0.75))
META = ["session", "animal", "cohort_stratum", "epoch", "anchor_ripple_positive"]


def probabilities(data):
    x = data.copy()
    sign = np.where(x.truth_track == 1, 1.0, -1.0)
    poisson = expit(-sign * x.evaluation_true_signed_log_odds)
    if not np.allclose(poisson, x.evaluation_track2_probability, atol=1e-12, equal_nan=True):
        raise ValueError("stored posterior mass and signed log odds differ")
    x["poisson"] = poisson
    x["conditional_count"] = expit(-sign * x.conditional_true_signed_log_odds)
    return x


def finite_mean(values):
    a = np.asarray(values, float)
    return float(a[np.isfinite(a)].mean()) if np.isfinite(a).any() else np.nan


def selected_composition(truth, q, full, half):
    truth, q, full, half = np.asarray(truth), np.asarray(q), np.asarray(full, bool), np.asarray(half, bool)
    if truth.shape != q.shape or full.shape != q.shape or half.shape != q.shape or q.ndim != 1 or not np.isin(truth, [0, 1]).all():
        raise ValueError("paired truth/readout/selection vectors required")
    support = np.isfinite(q)
    if ((q[support] < 0) | (q[support] > 1)).any():
        raise ValueError("posterior masses must be probabilities")
    masks = {"all": np.ones(len(q), bool), "full": full, "retained": full & half, "half": half, "lost": full & ~half, "gained": ~full & half}
    result = {}
    for group, mask in masks.items():
        good = mask & support
        result.update(
            {
                "n_" + group: int(mask.sum()),
                "n_" + group + "_supported": int(good.sum()),
                "true_" + group + "_all": finite_mean(truth[mask]),
                "true_" + group + "_supported": finite_mean(truth[good]),
                "readout_" + group: finite_mean(q[good]),
            }
        )
    for contrast, (a, b) in {"loss": ("retained", "full"), "gain": ("half", "retained"), "total": ("half", "full")}.items():
        result["true_" + contrast + "_shift_all"] = result["true_" + a + "_all"] - result["true_" + b + "_all"]
        result["true_" + contrast + "_shift_supported"] = result["true_" + a + "_supported"] - result["true_" + b + "_supported"]
        result["readout_" + contrast + "_shift"] = result["readout_" + a] - result["readout_" + b]
        result["error_" + contrast + "_shift"] = result["readout_" + contrast + "_shift"] - result["true_" + contrast + "_shift_supported"]
    return result


def known_weight_shift(q1, q2, retention, weights=None):
    """Anchor-by-split arrays, equal splits after within-split weighted means."""
    q1, q2 = np.asarray(q1, float), np.asarray(q2, float)
    if q1.ndim != 2 or q1.shape != q2.shape or not len(q1):
        raise ValueError("paired nonempty anchor-by-split probabilities required")
    p = np.asarray(retention, float)
    if p.shape != (2,) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or p.sum() <= 0:
        raise ValueError("two valid positive-total retention probabilities required")
    support = np.isfinite(q1) & np.isfinite(q2)
    if ((q1[support] < 0) | (q1[support] > 1) | (q2[support] < 0) | (q2[support] > 1)).any():
        raise ValueError("invalid posterior probabilities")
    w = np.ones((1, len(q1))) if weights is None else np.asarray(weights, float)
    if w.ndim != 2 or w.shape[1] != len(q1) or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("valid joint anchor weights required")
    den = w @ support.astype(float)
    means = []
    for q in [q1, q2]:
        num = w @ np.where(support, q, 0.0)
        means.append(np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0))
    slope = finite_row_mean(means[1] - means[0])
    before = finite_row_mean((means[0] + means[1]) / 2)
    after = finite_row_mean((p[0] * means[0] + p[1] * means[1]) / p.sum())
    return {
        "readout_before": before,
        "readout_after": after,
        "observed_shift": after - before,
        "recovery_slope": slope,
        "true_shift": float(p[1] / p.sum() - 0.5),
        "n_estimable_splits": (den > 0).sum(axis=1),
    }


def summarize_selected(rows):
    keys = [*META, "generator", "readout"]
    out = []
    for key, g in rows.groupby(keys):
        r = dict(zip(keys, key, strict=True), n_splits=len(g))
        for contrast in ["loss", "gain", "total"]:
            t = "true_" + contrast + "_shift_supported"
            e = "readout_" + contrast + "_shift"
            mask = np.isfinite(g[t]) & np.isfinite(g[e])
            r[contrast + "_n_estimable_splits"] = int(mask.sum())
            r[contrast + "_true_shift"] = finite_mean(g.loc[mask, t])
            r[contrast + "_readout_shift"] = finite_mean(g.loc[mask, e])
            r[contrast + "_mean_error"] = finite_mean(g.loc[mask, e] - g.loc[mask, t])
            r[contrast + "_mean_absolute_error"] = finite_mean(np.abs(g.loc[mask, e] - g.loc[mask, t]))
        out.append(r)
    return pd.DataFrame(out)


def analyze(data, n_boot=2000):
    x = probabilities(data)
    selected, interventions, pairs = [], [], []
    for key, g in x.groupby([*META, "generator", "split"]):
        full = g[g.fraction == 1].set_index(["anchor_event", "truth_track"])
        half = g[g.fraction == 0.5].set_index(["anchor_event", "truth_track"])
        if full.index.duplicated().any() or half.index.duplicated().any() or set(full.index) != set(half.index):
            raise ValueError("duplicate or unpaired full/half simulations")
        half = half.reindex(full.index)
        truth = (full.index.get_level_values("truth_track") == 2).astype(int)
        for column in ["poisson", "conditional_count"]:
            if not np.allclose(full[column], half[column], equal_nan=True):
                raise ValueError("evaluation mass changed with coverage")
            selected.append(
                dict(
                    zip([*META, "generator", "split"], key, strict=True),
                    readout=column,
                    **selected_composition(truth, full[column], full.sequence_accepted, half.sequence_accepted),
                )
            )
    ordered = x[(x.generator == "ordered") & (x.fraction == 1)]
    for key, g in ordered.groupby(META):
        anchor_ids = sorted(g.anchor_event.unique())
        for column in ["poisson", "conditional_count"]:
            arrays = []
            for truth in [1, 2]:
                arr = g[g.truth_track == truth].pivot(index="anchor_event", columns="split", values=column).reindex(index=anchor_ids, columns=range(5))
                arrays.append(arr.to_numpy())
            q1, q2 = arrays
            support = np.isfinite(q1) & np.isfinite(q2)
            n_supported = int(support.any(axis=1).sum())
            seed = stable_seed(20260918, *key, "composition-calibration-anchor-bootstrap")
            weights = np.random.default_rng(seed).multinomial(len(anchor_ids), np.full(len(anchor_ids), 1 / len(anchor_ids)), size=n_boot)
            base = dict(zip(META, key, strict=True), readout=column, n_anchors=len(anchor_ids), n_informative_anchors=n_supported, n_paired_readouts=int(support.sum()))
            for i, eid in enumerate(anchor_ids):
                for split in range(5):
                    pairs.append(
                        {
                            **dict(zip(META, key, strict=True)),
                            "readout": column,
                            "anchor_event": eid,
                            "split": split,
                            "q_truth1": q1[i, split],
                            "q_truth2": q2[i, split],
                            "finite_pair": bool(support[i, split]),
                        }
                    )
            for retention in RETENTION:
                point = known_weight_shift(q1, q2, retention)
                draws = known_weight_shift(q1, q2, retention, weights)
                ci = summarize_bootstrap(point["observed_shift"][0], draws["observed_shift"], n_supported, n_supported)
                interventions.append(
                    {
                        **base,
                        "retain_track1": retention[0],
                        "retain_track2": retention[1],
                        "true_shift": point["true_shift"],
                        "readout_before": point["readout_before"][0],
                        "readout_after": point["readout_after"][0],
                        "observed_shift": ci["point"],
                        "shift_error": ci["point"] - point["true_shift"],
                        "observed_shift_ci025": ci["ci025"],
                        "observed_shift_ci975": ci["ci975"],
                        "interval_status": ci["interval_status"],
                        "finite_bootstrap_fraction": ci["finite_bootstrap_fraction"],
                        "recovery_slope": point["recovery_slope"][0],
                        "n_estimable_splits": int(point["n_estimable_splits"][0]),
                    }
                )
    s = pd.DataFrame(selected)
    return {
        "actual_selection_by_split": s,
        "actual_selection_summary": summarize_selected(s),
        "known_selection_interventions": pd.DataFrame(interventions),
        "paired_track_readouts": pd.DataFrame(pairs),
    }


def run(source, output):
    if output.exists():
        raise ValueError("new immutable report directory required")
    git = lambda *a: subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()
    commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain"):
        raise ValueError("freeze audit before outcomes")
    frames, inputs = [], {}
    for p in sorted(source.glob("*/manifest.json")):
        m = json.loads(p.read_text())
        if m["status"] != "complete" or m["git_dirty"] or m["latent_paths_or_track_labels_given_to_decoder"]:
            raise ValueError("invalid calibration source")
        for name, digest in m["output_sha256"].items():
            if file_sha256(p.parent / name) != digest:
                raise ValueError("changed calibration output")
        c = pd.read_csv(p.parent / "known_track_calibration.csv")
        check_rows(c, pd.read_csv(p.parent / "known_temporal_calibration.csv"), pd.read_csv(p.parent / "count_anchors.csv"))
        frames.append(c)
        inputs[str(p.resolve())] = file_sha256(p)
    if not frames:
        raise ValueError("no calibration input")
    data = pd.concat(frames, ignore_index=True)
    if data.duplicated(["session", "anchor_event", "split", "truth_track", "generator", "fraction"]).any():
        raise ValueError("duplicate calibration sessions")
    tables = analyze(data)
    if git("status", "--porcelain") or git("rev-parse", "HEAD") != commit:
        raise ValueError("audit changed during execution")
    output.mkdir(parents=True)
    for name, frame in tables.items():
        frame.to_csv(output / (name + ".csv"), index=False)
    lines = [
        "# Known-composition recovery",
        "",
        "Non-rescoring measurement audit; no biological thresholds changed.",
        "",
        (
            "Mean evaluation posterior mass is compared with true simulated track proportions. Actual sequence selection "
            "is reported separately from deterministic selection-probability interventions. Missing evaluation spikes and empty "
            "selection sets remain explicit. Primary is POST ripple-positive anchors in the two strict RUN-pass sessions."
        ),
        "",
        "## Primary +25 percentage-point expected selection shift",
        "",
    ]
    d = tables["known_selection_interventions"]
    primary = d[(d.cohort_stratum == "strict_RUN_pass") & (d.epoch == "POST") & d.anchor_ripple_positive & (d.retain_track2 == 0.75)]
    for r in primary.to_dict("records"):
        lines.append(
            f"- {r['session']} / {r['readout']}: observed {100 * r['observed_shift']:.2f} pp "
            f"[{100 * r['observed_shift_ci025']:.2f}, {100 * r['observed_shift_ci975']:.2f}], recovery slope {r['recovery_slope']:.3f}; "
            f"{r['n_informative_anchors']}/{r['n_anchors']} informative anchors."
        )
    lines += [
        "",
        "## Boundaries",
        "",
        (
            "Anchor bootstraps share both tracks and all cell splits. They condition on the existing maps and simulated "
            "spikes and are not population intervals or additional simulations. The no-selection intervention is algebraic, not an "
            "empirical false-positive control. Conditional-count generation is not an exact Poisson generative model. "
            "Known maps are supplied, not known tracks/paths. Simulated PRE describes count anchors, not observed pre-experience replay."
        ),
        "",
        (
            "The actual-thinning half subsets vary with anchor rank in this calibration. Do not equate them with a single fixed "
            "real-data half-cell realization. A gain/loss shift with no selected events is undefined, not absent. "
            "Do not divide the real-data effect by this recovery slope: generator-to-real calibration transport is not established."
        ),
    ]
    (output / "composition_recovery.md").write_text("\n".join(lines) + "\n")
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "code_commit": commit,
        "git_dirty": False,
        "non_rescoring": True,
        "source_manifests": inputs,
        "n_bootstraps": 2000,
        "selection_retention_probabilities": RETENTION,
        "known_labels_used_for_audit_not_decoding": True,
        "changes_real_data_selection": False,
        "biological_confirmation": False,
        "command_line": sys.argv,
        "output_sha256": {p.name: file_sha256(p) for p in output.iterdir()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibration-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    run(a.calibration_dir, a.output_dir)
