"""Separate content changes caused by event loss from events newly accepted under thinning."""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_tirole_content_coverage import load_campaigns

from hipporeplayimm.tirole_two_track import file_sha256
from hipporeplayimm.two_track_content import stable_seed


def decompose(probability, full, thin, seed, n_null=499):
    q = np.asarray(probability, float)
    full, thin = np.asarray(full, bool), np.asarray(thin, bool)
    if q.ndim != 1 or q.shape != full.shape or q.shape != thin.shape or ((q[np.isfinite(q)] < 0) | (q[np.isfinite(q)] > 1)).any():
        raise ValueError("paired probability and acceptance vectors required")
    valid = np.isfinite(q)
    masks = {"full": full, "thin": thin, "retained": full & thin, "lost": full & ~thin, "gained": thin & ~full}
    vectors = {name: q[mask & valid] for name, mask in masks.items()}
    means = {name: float(x.mean()) if len(x) else np.nan for name, x in vectors.items()}
    out = {f"n_{k}": int(mask.sum()) for k, mask in masks.items()}
    out.update({f"n_{k}_with_content": len(v) for k, v in vectors.items()})
    out.update({f"mean_track2_{k}": v for k, v in means.items()})
    out.update(total_shift=means["thin"] - means["full"], loss_shift=means["retained"] - means["full"], gain_shift=means["thin"] - means["retained"])
    out["retained_fraction"] = len(vectors["retained"]) / len(vectors["full"]) if len(vectors["full"]) else np.nan
    out["loss_null_status"] = "insufficient_retained_or_full_content"
    for name in ["p_two_sided", "p025", "p975", "mean"]:
        out["loss_deletion_null_" + name] = np.nan
    if len(vectors["retained"]) and len(vectors["full"]):
        if n_null < 39:
            raise ValueError("at least 39 deletion draws required")
        rng = np.random.default_rng(seed)
        null = np.array([rng.choice(vectors["full"], len(vectors["retained"]), replace=False).mean() - means["full"] for _ in range(n_null)])
        out["loss_null_status"] = "estimable"
        out.update(
            loss_deletion_null_p_two_sided=(1 + (np.abs(null) >= abs(out["loss_shift"]) - 1e-12).sum()) / (n_null + 1),
            loss_deletion_null_p025=float(np.quantile(null, 0.025)),
            loss_deletion_null_p975=float(np.quantile(null, 0.975)),
            loss_deletion_null_mean=float(null.mean()),
        )
    if np.isfinite(out["gain_shift"]) and not np.isclose(out["total_shift"], out["loss_shift"] + out["gain_shift"], atol=1e-12):
        raise AssertionError("content decomposition does not close")
    return out


def analyze(data):
    x = data.loc[data.arm == "real"].copy()
    keys = ["animal", "session", "cohort_stratum", "candidate_stratum", "epoch", "split", "repeat"]
    records = []
    for key, g in x.groupby(keys):
        full = g[g.fraction == 1].set_index("event_id")
        if full.index.duplicated().any():
            raise ValueError("duplicate reference events")
        for fraction, h in g[g.fraction < 1].groupby("fraction"):
            half = h.set_index("event_id")
            if half.index.duplicated().any() or set(half.index) != set(full.index):
                raise ValueError("dose event identities differ")
            half = half.reindex(full.index)
            for readout, column in [("poisson", "evaluation_track2_probability"), ("conditional_count", "conditional_evaluation_track2_probability")]:
                if not np.allclose(full[column], half[column], equal_nan=True):
                    raise ValueError("evaluation population readout changed under thinning")
                result = decompose(full[column], full.sequence_accepted, half.sequence_accepted, stable_seed(20260918, *key, fraction, readout, "retained-only-null"))
                records.append(dict(zip(keys, key, strict=True), fraction=fraction, readout=readout, **result))
    rows = pd.DataFrame(records)
    group = ["animal", "session", "cohort_stratum", "candidate_stratum", "epoch", "fraction", "readout"]
    summary = rows.groupby(group, as_index=False).agg(
        n_split_repeats=("total_shift", "size"),
        estimable_total=("total_shift", "count"),
        estimable_loss=("loss_shift", "count"),
        mean_total_shift=("total_shift", "mean"),
        median_total_shift=("total_shift", "median"),
        mean_loss_shift=("loss_shift", "mean"),
        median_loss_shift=("loss_shift", "median"),
        mean_gain_shift=("gain_shift", "mean"),
        median_gain_shift=("gain_shift", "median"),
        median_full=("n_full_with_content", "median"),
        median_retained=("n_retained_with_content", "median"),
        median_lost=("n_lost_with_content", "median"),
        median_gained=("n_gained_with_content", "median"),
        median_loss_deletion_p=("loss_deletion_null_p_two_sided", "median"),
    )
    return rows, summary


def run(campaigns, output):
    if output.exists():
        raise ValueError("new report output required")
    data, sources = load_campaigns(campaigns)
    rows, summary = analyze(data)
    output.mkdir(parents=True)
    rows.to_csv(output / "selection_content_decomposition_by_split.csv", index=False)
    summary.to_csv(output / "selection_content_decomposition_by_session.csv", index=False)
    note = """# Selection-induced content changes

Thinning can both remove and newly admit events. A random-deletion null from
full-coverage accepted events is therefore applied ONLY to the retained subset.
The total track-2 probability change is separated into retained-minus-full
(selective loss) and thinned-minus-retained (gains), when all three means exist.
Empty retained sets do not become zero effects or passing tests. Evaluation
spikes and readout remain fixed across doses. Conditional-count readout is a
rate sensitivity, not a replacement for the frozen Poisson primary statistic.

The earlier report's total-shift/deletion comparison also included gained
events; it was a hypothetical deletion comparison, not an isolated test of
selective loss. Use this decomposition for that narrower attribution. The
unchanged total shifts remain descriptive. Repeated cell splits are not
independent events or animals; split-wise null p-values are diagnostics, not
population-level significance. Two primary animals cannot establish replication
across the dataset. No neural data are rescored here.
"""
    (output / "selection_content_decomposition.md").write_text(note)
    repo = Path(__file__).resolve().parents[1]
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "non_rescoring": True,
        "input_score_manifests": sources,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "output_sha256": {p.name: file_sha256(p) for p in output.iterdir()},
        "biological_confirmation": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign-dirs", nargs="+", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    run(a.campaign_dirs, a.output_dir)
