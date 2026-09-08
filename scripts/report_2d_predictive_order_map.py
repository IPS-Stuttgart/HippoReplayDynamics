#!/usr/bin/env python3
"""Non-rescoring, audit-gated report of the held-out order by map factorial."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import build_script_provenance, file_sha256

DATASETS = {"pfeiffer_foster": ("Pfeiffer/Foster", 4001, 8, 4), "tanni2022": ("Tanni", 5224, 25, 5)}
FACTORS = {
    "real_original_minus_iid": "Real map, original order",
    "real_shuffle_minus_iid": "Real map, shuffled order",
    "wrong_original_minus_iid": "Permuted map, original order",
    "wrong_shuffle_minus_iid": "Permuted map, shuffled order",
}
CONTRASTS = {
    "real_order_advantage": "Original - shuffled: real map",
    "wrong_order_advantage": "Original - shuffled: permuted map",
    "order_map_interaction": "Order x map interaction",
}
SENSITIVITIES = {"real_order_median_sensitivity": "Real map, median shuffle", "wrong_order_median_sensitivity": "Permuted map, median shuffle"}
PRIMARY = {"first_order_imm__real_order_advantage", "first_order_imm__order_map_interaction"}


def validate(root, audit_path):
    path = root / "predictive_order_map_manifest.json"
    manifest = json.loads(path.read_text())
    audit = json.loads(audit_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("k") != 20 or audit.get("status") != "pass":
        raise ValueError("complete K=20 run and passing independent audit required")
    if audit.get("input_file_sha256", {}).get("run_manifest") != file_sha256(path):
        raise ValueError("audit belongs to a different scoring run")
    expected = {
        "sessions": 33,
        "events": 9225,
        "score_rows": 1845000,
        "invariant_scores": 3690000,
        "permutations": 184500,
        "independent_predictions": 1584,
        "split_contrasts": 830250,
        "event_contrasts": 166050,
        "bootstrap_panels": 36,
    }
    if any(audit.get(k) != v for k, v in expected.items()):
        raise ValueError("incomplete independent audit coverage")
    error = audit.get("max_prediction_error", np.nan)
    if not np.isfinite(error) or error < 0 or error > 1e-7:
        raise ValueError("independent score reconstruction failed")
    for record, folder in ((manifest, root), (audit, audit_path.parent)):
        if not record.get("output_sha256"):
            raise ValueError("missing output hashes")
        for name, digest in record["output_sha256"].items():
            if file_sha256(folder / name) != digest:
                raise ValueError("changed scoring/audit artifact: " + name)
    return manifest, audit


def readout(summary):
    labels = FACTORS | CONTRASTS | SENSITIVITIES
    expected = {(d, m + "__" + c) for d in DATASETS for m in ("diffusion", "first_order_imm") for c in labels}
    if summary.duplicated(["dataset", "contrast"]).any() or set(summary[["dataset", "contrast"]].itertuples(index=False, name=None)) != expected:
        raise ValueError("missing, duplicate or unexpected summary panel")
    for dataset, (_, events, sessions, animals) in DATASETS.items():
        p = summary[summary.dataset.eq(dataset)]
        if not (p.events.eq(events).all() and p.sessions.eq(sessions).all() and p.animals.eq(animals).all()):
            raise ValueError("frozen cohort denominator mismatch")
    numeric = ["mean", "ci_low", "ci_high", "mean_per_heldout_spike", "per_spike_ci_low", "per_spike_ci_high", "positive_animals"]
    if (
        not np.isfinite(summary[numeric]).all().all()
        or (summary.ci_low > summary.ci_high).any()
        or (summary.per_spike_ci_low > summary.per_spike_ci_high).any()
        or summary.positive_animals.lt(0).any()
        or (summary.positive_animals > summary.animals).any()
    ):
        raise ValueError("invalid factorial estimate")
    result = summary.copy()
    result["primary"] = result.contrast.isin(PRIMARY)
    result["model"] = result.contrast.str.split("__").str[0]
    result["label"] = result.contrast.str.split("__").str[1].map(labels)
    result["raw_gate_passed"] = result["mean"].gt(0) & result.ci_low.gt(0) & result.positive_animals.eq(result.animals)
    return result


def decision(summary):
    table = readout(summary)
    rows = []
    for dataset in DATASETS:
        p = table[table.dataset.eq(dataset) & table.primary]
        passed = bool(p.raw_gate_passed.all())
        rows.append(
            {
                "dataset": dataset,
                "technical_status": "pass",
                "order_and_adjacency_gate_passed": passed,
                "failed_primary_axes": ",".join(p.loc[~p.raw_gate_passed, "contrast"]),
                "parent_replication_gate_changed": False,
                "new_mechanism_established": False,
                "recommendation": "bounded_order_and_adjacency_predictive_effect" if passed else "report_partial_or_absent_order_adjacency_effect",
            }
        )
    return pd.DataFrame(rows)


def plot(table, animals, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for col, (dataset, (label, events, sessions, rats)) in enumerate(DATASETS.items()):
        color = "#16746b" if col == 0 else "#ab3f60"
        for row, items in enumerate((FACTORS, CONTRASTS)):
            ax = axes[row, col]
            for y, contrast in enumerate(items):
                key = "first_order_imm__" + contrast
                r = table[table.dataset.eq(dataset) & table.contrast.eq(key)].iloc[0]
                points = animals[animals.dataset.eq(dataset) & animals.contrast.eq(key)].sort_values("animal")
                ax.scatter(points.delta, y + np.linspace(-0.12, 0.12, len(points)), s=25, color=color, alpha=0.45)
                ax.plot([r.ci_low, r.ci_high], [y, y], color=color, lw=2)
                ax.scatter(r["mean"], y, marker="D", s=35, color=color)
            ax.axvline(0, color="#666666", ls="--", lw=1)
            ax.set_yticks(range(len(items)), items.values())
            ax.set_ylim(len(items) - 0.5, -0.5)
            ax.grid(axis="x", alpha=0.15)
            ax.set_xlabel("Predictive advantage over independent positions (nats)" if row == 0 else "Paired predictive log-score difference (nats)")
            ax.set_title(
                f"{label}: {events:,} events, {sessions} sessions, {rats} animals\n" + ("All four factorial conditions" if row == 0 else "Paired order effects and interaction"),
                fontsize=12,
                pad=12,
            )
    fig.suptitle("Does held-out-cell prediction depend on time order and spatial adjacency?", fontsize=15)
    fig.supxlabel(
        "IMM, proper count-conditioned scores. Diamonds: equal-animal means; lines: 95% hierarchical intervals; dots: animals.\n20 whole-bin shuffles; five split medians per event. Interactions are paired before aggregation; panels need not add.",
        fontsize=10,
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def report_text(table, verdicts, support, manifest, audit):
    lines = ["# Predictive order by map factorial", "", "Non-rescoring report. All events and parameters were frozen before this run.", "", "## Decision", ""]
    for r in verdicts.itertuples(index=False):
        lines.append(f"- {DATASETS[r.dataset][0]}: `{r.recommendation}`; failed primary axes: {r.failed_primary_axes or 'none'}.")
    lines += [
        "",
        "The parent Tanni comparison against other-event cell composition failed its raw-score replication rule. Neither a positive shuffle result nor per-spike normalization changes that outcome. This experiment does not identify a unique IMM mechanism.",
        "",
        "## Paired Results",
        "",
        "| Dataset | Model | Contrast | Mean nats | 95% CI | Positive animals | Per-spike mean [95% CI] |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for model in ("first_order_imm", "diffusion"):
            for c in CONTRASTS | FACTORS | SENSITIVITIES:
                r = table[table.dataset.eq(dataset) & table.contrast.eq(model + "__" + c)].iloc[0]
                lines.append(
                    f"| {DATASETS[dataset][0]} | {model} | {r.label} | {r['mean']:+.4f} | [{r.ci_low:+.4f}, {r.ci_high:+.4f}] | {r.positive_animals}/{r.animals} | {r.mean_per_heldout_spike:+.4f} [{r.per_spike_ci_low:+.4f}, {r.per_spike_ci_high:+.4f}] |"
                )
    lines += [
        "",
        "## Frozen Cohort and Prediction",
        "",
        "All 4,001 PF and 5,224 Tanni high-MUA candidate windows, 33 sessions/nine animals, and five 70/30 cell splits are retained. These are not all confirmed replay events. Use the parent RUN-only 8 cm maps, nonoverlapping 20 ms bins including final partial bins, exact untempered multinomial cell-identity likelihood conditional on each bin's spike total, and the same fixed physical transition parameters.",
        "",
        "Infer smoothed latent position/mode posteriors from training neurons only. Freeze them before evaluating held-out cell-identity scores. The score sums time-bin marginal log predictions; it is not joint event evidence, a future-time forecast, or known actual position during replay. Candidate detection originally used all cells, so prediction is conditional on that fixed ascertainment.",
        "",
        "For each event, draw 20 whole-bin orders, shared across all neurons, cell splits and both maps. Preserve every population spike vector and carry its bin width with it. Include repeated and identity draws, especially in short events. Independent-position and static-location predictions are invariant checks. The wrong map is the parent's ONE shared population-code-column permutation: spatial adjacency changes, code content and cell identities remain intact. It is not a genuine alternate-context map or all possible wrong maps.",
        "",
        "Within each split subtract the MEAN shuffled score from the original. Interaction = (real original - real shuffled) - (wrong original - wrong shuffled). Compute these paired quantities first, then medians across five splits per event, means across events per session, means across sessions per animal, and equal-animal dataset means. Separately aggregated quantities need not add because split medians are nonlinear. Median-shuffle contrasts and per-spike ratios are sensitivity analyses, not alternative primary gates.",
        "",
        "The 5,000-draw hierarchical bootstrap resamples animals/sessions/events with maps, cell splits, shuffled orders and calibration fixed. Four/five animals remain the biological sample size. Positive pointwise intervals plus positive means in every animal are a bounded decision rule, not universal replication or multiple-testing-corrected event labels. Zero-held-out-spike cases remain in raw scores; their per-spike ratios are undefined.",
        "",
        "## Permutation Support",
        "",
    ]
    for dataset, frame in support.groupby("dataset"):
        lines.append(
            f"- {DATASETS[dataset][0]}: {len(frame):,} events; {frame.unique_permutations.lt(20).sum():,} have fewer than 20 unique draws; {frame.identity_permutations.gt(0).sum():,} include an identity draw; minimum/median/maximum time bins {frame.n_bins.min()}/{frame.n_bins.median():g}/{frame.n_bins.max()}."
        )
    lines += [
        "",
        "## Verification",
        "",
        f"Producer commit `{manifest['code_commit']}`; independent auditor `{audit['code_commit']}`. All {audit['score_rows']:,} shuffled score rows, {audit['permutations']:,} permutations, {audit['invariant_scores']:,} invariant scores, {audit['split_contrasts']:,} paired split contrasts, {audit['event_contrasts']:,} event contrasts, and {audit['bootstrap_panels']} hierarchical intervals checked. A separate dynamic solver reconstructs {audit['independent_predictions']:,} predictions across all 33 sessions, both maps, two splits and two orders of three fixed events/session; maximum discrepancy {audit['max_prediction_error']:.3g}. Parent native-count/map audit is hash-pinned and reused; maps are not refitted here.",
        "",
        "## Publication Boundary",
        "",
        "A positive original-order benefit means time order helps this cross-neuron predictive endpoint. A positive interaction means correct adjacency strengthens that benefit relative to this one spatial permutation. Neither establishes direction relative to experience, constant physical speed, a unique switching circuit, or exclusion of all temporally structured co-firing alternatives. Train and held-out neurons may share nonspatial latent causes; held-out prediction alone does not eliminate those.",
        "",
        "Temporal models, cross-validation, and time-swap controls for co-firing already have direct precedent in [Maboudi et al. (2018)](https://elifesciences.org/articles/34467). The possible contribution is a properly separated measurement/validation study, not discovery of ordered hippocampal activity. Preserve the parent failed comparator alongside these results; do not retune settings or select favorable animals to promote a mechanistic claim.",
        "",
    ]
    return "\n".join(lines)


def run(args):
    root, audit_path, out = Path(args.run_dir).resolve(), Path(args.audit_manifest).resolve(), Path(args.output_dir).resolve()
    manifest, audit = validate(root, audit_path)
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing to overwrite report")
    table = readout(pd.read_csv(root / "predictive_order_map_summary.csv"))
    verdicts = decision(table)
    animals = pd.read_csv(root / "predictive_order_map_by_animal.csv")
    support = pd.concat([pd.read_csv(root / f"{r['tag']}_support.csv.gz") for r in manifest["completed"]], ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "predictive_order_map_readout.csv", index=False)
    verdicts.to_csv(out / "predictive_order_map_report_decisions.csv", index=False)
    animals.to_csv(out / "predictive_order_map_animal_readout.csv", index=False)
    pd.read_csv(root / "predictive_order_map_by_session.csv").to_csv(out / "predictive_order_map_session_readout.csv", index=False)
    support.to_csv(out / "predictive_order_map_permutation_support.csv", index=False)
    plot(table, animals, out / "predictive_order_map.png")
    (out / "predictive_order_map_report.md").write_text(report_text(table, verdicts, support, manifest, audit))
    provenance = build_script_provenance(input_paths={"run_manifest": root / "predictive_order_map_manifest.json", "audit_manifest": audit_path})
    provenance.update(status="complete", non_rescoring=True, decisions=verdicts.to_dict("records"))
    provenance["output_sha256"] = {p.name: file_sha256(p) for p in out.iterdir()}
    (out / "predictive_order_map_report_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(verdicts.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--audit-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())
