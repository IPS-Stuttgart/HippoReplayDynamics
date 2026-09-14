#!/usr/bin/env python3
"""Non-rescoring paired endpoint report; earlier-time accuracy is not recovery."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd

from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_edge_support_content import POLICIES, SOURCES, KEYS

METRICS = ("separation_cm", "regional_tv", "a_entropy", "b_entropy", "a_width_cm", "b_width_cm",
           "a_spikes", "b_spikes", "a_active", "b_active", "a_original_truth_error_cm",
           "b_original_truth_error_cm", "a_selected_truth_error_cm", "b_selected_truth_error_cm")
GROUP = ["dataset", "source", "split", "policy"]


def session_summaries(frame):
    if frame.empty or frame.duplicated(KEYS).any():
        raise ValueError("empty or duplicated readouts")
    results, flags = [], []
    for key, g in frame.groupby(GROUP+["animal", "session"], sort=True):
        metadata = dict(zip(GROUP+["animal", "session"], key, strict=True))
        accepted = g.loc[g.status.eq("available")]
        common = dict(metadata, events=len(g), retained=len(accepted), availability=len(accepted)/len(g))
        for metric in METRICS:
            raw = accepted[f"raw_{metric}"]
            value = accepted[metric]
            difference = value-raw
            finite = np.isfinite(raw) & np.isfinite(value)
            if metadata["source"] != "real" or "truth_error" not in metric:
                if int(finite.sum()) != len(accepted):
                    raise ValueError(f"missing retained readout metrics: {metadata}:{metric}")
            results.append(dict(common, metric=metric, finite_pairs=int(finite.sum()),
                raw_all_events=g[f"raw_{metric}"].mean(), raw_same_retained=raw.mean(), selected=value.mean(),
                selected_minus_raw=difference.mean(), mean_shift_earlier_ms=accepted.shift_earlier_ms.mean(),
                median_shift_earlier_ms=accepted.shift_earlier_ms.median(), mean_truth_shift_cm=accepted.truth_shift_cm.mean()))
        if metadata["policy"] != "raw_endpoint":
            continue
        for metric in ("b_original_truth_error_cm", "separation_cm", "regional_tv", "b_entropy"):
            good = g.loc[~g.raw_a_activity_inadequate, metric]
            bad = g.loc[g.raw_a_activity_inadequate, metric]
            flags.append(dict(metadata, metric=metric, n_flagged=len(bad), n_unflagged=len(good),
                flagged_fraction=len(bad)/len(g), finite_flagged=int(np.isfinite(bad).sum()),
                finite_unflagged=int(np.isfinite(good).sum()), flagged_mean=bad.mean(), unflagged_mean=good.mean(),
                flagged_minus_unflagged=bad.mean()-good.mean()))
    return pd.DataFrame(results), pd.DataFrame(flags)


def aggregate(session, seed=20260914):
    keys = GROUP+["metric"]
    values = ["availability", "raw_all_events", "raw_same_retained", "selected", "selected_minus_raw",
              "mean_shift_earlier_ms", "median_shift_earlier_ms", "mean_truth_shift_cm"]
    animal = session.groupby(keys+["animal"], as_index=False)[values].mean()
    counts = session.groupby(keys+["animal"], as_index=False).agg(events=("events", "sum"),
        retained=("retained", "sum"), finite_pairs=("finite_pairs", "sum"), sessions=("session", "nunique"))
    animal = animal.merge(counts, on=keys+["animal"], validate="one_to_one")
    summaries = []
    rng = np.random.default_rng(seed)
    for key, g in animal.groupby(keys, sort=True):
        row = dict(zip(keys, key, strict=True))
        row.update(g[values].mean().to_dict())
        delta = g.selected_minus_raw.dropna().to_numpy()
        ci = np.quantile(rng.choice(delta, (10000, len(delta)), replace=True).mean(axis=1), [.025, .975]) if len(delta) else [np.nan]*2
        row.update(animals=len(g), finite_animals=len(delta), events=int(g.events.sum()), retained=int(g.retained.sum()),
            sessions=int(g.sessions.sum()), paired_delta_ci_low=ci[0], paired_delta_ci_high=ci[1],
            animals_delta_negative=int((delta < 0).sum()), animals_delta_positive=int((delta > 0).sum()),
            ci_scope="descriptive_rat_bootstrap_four_rats_not_population_proof")
        summaries.append(row)
    return animal, pd.DataFrame(summaries)


def gates(summary, animal, audit_passed, cohort_complete):
    rows = []
    for dataset in summary.dataset.unique():
        for policy in POLICIES[1:]:
            chosen = summary.loc[summary.dataset.eq(dataset) & summary.policy.eq(policy) & summary.split.eq(0)]

            def number(source, metric, field="selected_minus_raw"):
                r = chosen.loc[chosen.source.eq(source) & chosen.metric.eq(metric)]
                return float(r.iloc[0][field]) if len(r) == 1 else np.nan

            required = chosen.loc[chosen.source.ne("real") & chosen.metric.isin(["a_original_truth_error_cm", "b_original_truth_error_cm"])]
            error_complete = len(required) == 10 and required.finite_animals.eq(4).all()
            per_rat = animal.loc[animal.dataset.eq(dataset) & animal.source.eq("real") & animal.split.eq(0)
                & animal.policy.eq(policy) & animal.metric.eq("separation_cm")]
            checks = dict(independent_audit_passed=bool(audit_passed), all_frozen_sessions_complete=bool(cohort_complete),
                real_mean_separation_reduced=bool(number("real", "separation_cm") < 0),
                real_regional_tv_reduced=bool(number("real", "regional_tv") < 0),
                real_a_entropy_not_increased=bool(number("real", "a_entropy") <= 1e-10),
                real_b_entropy_not_increased=bool(number("real", "b_entropy") <= 1e-10),
                all_known_sources_original_time_error_not_worse=bool(error_complete and (required.selected_minus_raw <= 1e-10).all()),
                at_least_half_available_in_three_of_four_rats=bool(len(per_rat) == 4 and (per_rat.availability >= .5).sum() >= 3))
            checks["original_endpoint_remedy"] = all(checks.values())
            for name, passed in checks.items():
                rows.append(dict(dataset=dataset, policy=policy, gate=name, passed=passed,
                    claim_scope="original_endpoint_only; pooled/joint timing is not A-only predictive"))
    return pd.DataFrame(rows)


def plot(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = summary.loc[summary.split.eq(0)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    colors = ["#386cb0", "#007f66", "#a44860"]
    policies = list(POLICIES[1:])
    labels = ["Pooled >=2 spikes", "A >=3 spikes, >=2 cells", "Both supported"]
    for row, dataset in enumerate(("pfeiffer_foster", "hc11")):
        local = primary.loc[primary.dataset.eq(dataset)]
        for column, metric, title in ((0, "separation_cm", "Population separation change (cm)"),
                                      (1, "regional_tv", "Regional posterior TV change")):
            data = local.loc[local.source.eq("real") & local.metric.eq(metric)].set_index("policy").loc[policies]
            ax = axes[row, column]
            ax.bar(np.arange(3), data.selected_minus_raw, color=colors)
            ax.vlines(np.arange(3), data.paired_delta_ci_low, data.paired_delta_ci_high, color="black", lw=1)
            ax.axhline(0, color="black", lw=.8)
            ax.set(xticks=np.arange(3), xticklabels=labels, title=f"{dataset}\n{title}")
            ax.tick_params(axis="x", labelrotation=25, labelsize=8)
        ax = axes[row, 2]
        kinds = ["run_q4", "sim_stationary", "sim_moving", "sim_moving_gain", "sim_late_jump"]
        for j, policy in enumerate(policies):
            value = local.loc[local.policy.eq(policy) & local.metric.eq("b_original_truth_error_cm")].set_index("source").loc[kinds]
            ax.plot(np.arange(5), value.selected_minus_raw, marker="o", color=colors[j], label=labels[j])
        ax.axhline(0, color="black", lw=.8)
        ax.set(xticks=np.arange(5), xticklabels=["RUN", "Static", "Moving", "Gain shift", "Late jump"],
               title="Independent B error vs ORIGINAL time (cm)")
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)
        ax.legend(fontsize=8)
    fig.suptitle("Edge-support benchmark: selected minus raw, same retained events\nNegative is lower disagreement/error; timing shifts and abstentions remain explicit", fontsize=13)
    fig.savefig(output/"edge_support_content.png", dpi=160)
    fig.savefig(output/"edge_support_content.pdf")
    plt.close(fig)


def write_report(summary, gate, flags, sessions, output):
    lines = ["# Edge-support content benchmark", "", "Non-rescoring, frozen 200-candidate/session pilot; split0 is primary.",
        "Events average within session, sessions within rat, rats within dataset. Bootstrap intervals are descriptive (four rats/dataset).",
        "", "## Paired real-candidate readouts", "", "| Dataset | Policy | Available | Shift earlier (ms) | Separation change (cm) | TV change | A entropy change | B entropy change |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    s = summary.loc[summary.split.eq(0)]
    for (dataset, policy), part in s.loc[s.source.eq("real") & s.policy.ne("raw_endpoint")].groupby(["dataset", "policy"]):
        x = part.set_index("metric")
        d = x.selected_minus_raw
        common = x.loc["separation_cm"]
        lines.append(f"| {dataset} | {policy} | {common.availability:.1%} | {common.mean_shift_earlier_ms:.1f} | {d['separation_cm']:+.3f} | {d['regional_tv']:+.4f} | {d['a_entropy']:+.4f} | {d['b_entropy']:+.4f} |")
    lines += ["", "## Original-endpoint remedy gates", ""]
    for (dataset, policy), part in gate.groupby(["dataset", "policy"]):
        failures = part.loc[~part.passed & part.gate.ne("original_endpoint_remedy"), "gate"].tolist()
        lines.append(f"- {dataset}, {policy}: {'FAIL' if failures else 'PASS'}; failed components: {', '.join(failures) or 'none'}.")
    lines += ["", "## Known-position controls", "", "| Dataset | Policy | Source | A selected-time error change | A original-time error change | B selected-time error change | B original-time error change | Truth shift (cm) |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for (dataset, policy, source), part in s.loc[s.source.ne("real") & s.policy.ne("raw_endpoint")].groupby(["dataset", "policy", "source"]):
        x = part.set_index("metric")
        values = [x.loc[f"{side}_{when}_truth_error_cm", "selected_minus_raw"] for side, when in
                  (("a", "selected"), ("a", "original"), ("b", "selected"), ("b", "original"))]
        lines.append(f"| {dataset} | {policy} | {source} | " + " | ".join(f"{v:+.3f}" for v in values) + f" | {part.mean_truth_shift_cm.iloc[0]:.2f} |")
    lines += ["", "## A-only inadequacy flag", "", "Flag: original A window has <3 spikes or <2 active cells. Positive differences below mean larger B truth error in flagged windows. No thresholds fitted on hc11.",
              "This is association, not destination certification; absent comparison strata are undefined, never zero error.", ""]
    for (dataset, source, split, metric), g in flags.groupby(["dataset", "source", "split", "metric"]):
        if split != 0 or metric != "b_original_truth_error_cm" or source == "real":
            continue
        rat = g.groupby("animal").flagged_minus_unflagged.mean()
        lines.append(f"- {dataset}, {source}: flagged-minus-unflagged B error {rat.mean():+.3f} cm; positive {int((rat > 0).sum())}/{int(rat.notna().sum())} animals; {int(g.finite_unflagged.sum())} finite unflagged endpoints.")
    lines += ["", "## Boundaries", "", "- Earlier selected-time improvement does not recover original terminal content; both targets are reported.",
        "- Pooled/joint timing uses B spikes and cannot be called held-out A-only prediction.",
        "- Raw candidate endpoints are not the accepted trajectory endpoints in the matched-population Home analysis. This experiment does not explain away that result.",
        "- PF awake high-MUA and hc11 POST high-MUA candidate pools are not equivalent biological event classes. Native hc11 POST bursts are not automatically NREM/ripple/replay.",
        "- Graph simulations use occupied 2D bins, not verified track topology, preserve total count timecourses but not native noise correlations, and are not replay ground truth.",
        "- Native unbinned-count and RUN truth reconstruction is independent. Simulation totals/path invariants are audited; latent-path generation is covered by deterministic tests, not a second independent generator.",
        "- Training-only maps/unit QC exclude held-out RUN spikes. Geometry can use full RUN tracking support.",
        "- Negative results do not validate a remedy. Simple activity support alone is not a novel method.", "", "## Cohort", ""]
    for dataset, g in sessions.groupby("dataset"):
        lines.append(f"- {dataset}: {len(g)}/8 complete sessions, {g.animal.nunique()} rats, {int(g.selected_candidates.sum())} sampled candidates of {int(g.source_candidates.sum())} catalogued candidates.")
    (output/"edge_support_content_report.md").write_text("\n".join(lines)+"\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurement-dir", type=Path, action="append", required=True)
    p.add_argument("--audit-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    inputs = {f"sessions{i}": d/"measurement_sessions.csv" for i, d in enumerate(args.measurement_dir)}
    inputs.update(audit=args.audit_dir/"independent_audit.json", audit_sessions=args.audit_dir/"independent_audit_sessions.csv", reporter=Path(__file__))
    audit = json.loads(inputs["audit"].read_text())
    if audit["status"] != "passed":
        raise ValueError("independent audit required before report")
    audited = pd.read_csv(inputs["audit_sessions"])
    if not audited.status.eq("passed").all():
        raise ValueError("independent session audit failed")
    manifest = build_script_provenance(input_paths=inputs, cwd=ROOT)
    sessions = pd.concat([pd.read_csv(d/"measurement_sessions.csv") for d in args.measurement_dir], ignore_index=True)
    complete = all(len(g) == 8 and g.animal.nunique() == 4 and g.status.eq("complete").all() for _, g in sessions.groupby("dataset"))
    if not complete or sessions.duplicated(["dataset", "session"]).any():
        raise ValueError("incomplete frozen cohort")
    frames = []
    for row in sessions.itertuples(index=False):
        validated = audited.loc[audited.dataset.eq(row.dataset) & audited.session.eq(row.session)]
        path = Path(row.artifact_dir)/"edge_readouts.csv.gz"
        if len(validated) != 1 or file_sha256(path) != validated.iloc[0].readouts_sha256:
            raise ValueError("unaudited or altered readouts")
        frames.append(pd.read_csv(path))
    frame = pd.concat(frames, ignore_index=True)
    if set(frame.source) != set(SOURCES):
        raise ValueError("missing control source")
    by_session, flags = session_summaries(frame)
    animal, summary = aggregate(by_session)
    gate = gates(summary, animal, True, complete)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, table in (("edge_support_by_session", by_session), ("edge_support_by_animal", animal),
                        ("edge_support_summary", summary), ("edge_support_gate_summary", gate),
                        ("edge_support_a_only_flag_by_session", flags), ("measurement_sessions", sessions)):
        table.to_csv(args.output_dir/f"{name}.csv", index=False)
    write_report(summary, gate, flags, sessions, args.output_dir)
    plot(summary, args.output_dir)
    manifest.update(status="complete", non_rescoring=True, independent_audit_passed=True,
                    outputs_sha256={p.name: file_sha256(p) for p in args.output_dir.iterdir() if p.is_file()})
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
