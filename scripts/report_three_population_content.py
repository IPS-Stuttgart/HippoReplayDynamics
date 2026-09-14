#!/usr/bin/env python3
"""Non-rescoring report with independent aggregation and required-gate checks."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts._provenance import build_script_provenance, file_sha256


def verify_aggregates(predictions, directory):
    saved_session = pd.read_csv(directory/"external_by_session.csv")
    saved_animal = pd.read_csv(directory/"external_by_animal.csv")
    saved_summary = pd.read_csv(directory/"external_summary.csv")
    keys = ["dataset", "animal", "session", "split", "source", "draw", "policy"]
    metrics = [col for col in saved_session if col not in keys+["events", "retained"]]
    rows = []
    for identity, local in predictions.groupby(keys[:-1]):
        for policy in ("random_expectation", "full", "pooled", "a_entropy", "pooled_entropy"):
            chosen = local if policy == "random_expectation" else local.loc[local[f"retained_{policy}"]]
            row = dict(zip(keys[:-1], identity, strict=True), policy=policy, events=len(local), retained=len(chosen))
            row.update({metric: chosen[metric].mean() for metric in metrics})
            rows.append(row)
    session = pd.DataFrame(rows)
    groups = ["dataset", "animal", "split", "source", "policy"]
    # Explicit session-level draw averaging, then equal session and animal means.
    draws = session.groupby(groups+["session"])[metrics].mean().reset_index()
    animal = draws.groupby(groups)[metrics].mean().reset_index()
    summary = animal.groupby(["dataset", "split", "source", "policy"])[metrics].mean().reset_index()
    for actual, expected, sorting in ((saved_session, session, keys), (saved_animal, animal, groups),
                                     (saved_summary, summary, ["dataset", "split", "source", "policy"])):
        pd.testing.assert_frame_equal(actual.sort_values(sorting).reset_index(drop=True)[expected.columns],
            expected.sort_values(sorting).reset_index(drop=True), check_dtype=False, rtol=1e-10, atol=1e-10)
    return animal, summary


def verify_gates(directory, animal, summary, source, scored, frozen_before):
    gates = pd.read_csv(directory/"external_gate_summary.csv").set_index("gate")
    expected = {}
    def add(name, condition, value):
        expected[name] = (bool(condition), value)
    def lower(values):
        values = np.asarray(values, float)
        if len(values) < 2 or not np.isfinite(values).all():
            return np.nan
        rng = np.random.default_rng(20260914)
        return float(np.quantile(values[rng.integers(0, len(values), (5000, len(values)))].mean(axis=1), .025))
    real = animal.loc[(animal.split == 0) & animal.source.eq("real")]
    primary = summary.loc[(summary.split == 0) & summary.source.eq("real")].set_index("policy")
    base, keep = primary.loc["random_expectation"], primary.loc["full"]
    paired = real.pivot(index="animal", columns="policy")
    source_count = pd.to_numeric(source.candidates, errors="coerce")
    available = pd.to_numeric(scored.candidates, errors="coerce").fillna(0).sum()
    coverage = available/source_count.sum() if source_count.notna().all() and source_count.sum() > 0 else np.nan
    add("external_animals", real.animal.nunique() >= 4, real.animal.nunique())
    add("source_endpoint_coverage", np.isfinite(coverage) and coverage >= .8, coverage)
    add("model_frozen_before_external", frozen_before, frozen_before)
    for baseline in ("constant", "pooled"):
        value = 1-base.logloss_full/base[f"logloss_{baseline}"]
        add(f"logloss_improves_{baseline}", value >= .05, value)
        value = base[f"brier_{baseline}"]-base.brier_full
        add(f"brier_improves_{baseline}", value > 0, value)
    differences = paired[("logloss_pooled", "random_expectation")]-paired[("logloss_full", "random_expectation")]
    add("prediction_animal_consistency", (differences > 0).mean() >= .75, (differences > 0).mean())
    lo = lower(differences)
    add("prediction_bootstrap_lower_positive", lo > 0, lo)
    for metric in ("ac_regional_tv", "ac_separation_cm"):
        reduction = 1-keep[metric]/base[metric]
        diff = paired[(metric, "random_expectation")]-paired[(metric, "full")]
        add(f"{metric}_reduction", reduction >= .1, reduction)
        add(f"{metric}_animal_consistency", (diff > 0).mean() >= .75, (diff > 0).mean())
        lo = lower(diff)
        add(f"{metric}_bootstrap_lower_positive", lo > 0, lo)
    for side in ("a", "c"):
        change = keep[f"{side}_entropy"]-base[f"{side}_entropy"]
        add(f"{side}_entropy_not_increased", change <= 0, change)
        for name in ("run_test", "sim_matched", "sim_drift"):
            local = summary.loc[(summary.split == 0) & summary.source.eq(name)].set_index("policy")
            change = local.loc["full", f"{side}_truth_error_cm"]-local.loc["random_expectation", f"{side}_truth_error_cm"]
            add(f"{name}_{side}_truth_error_not_worse", np.isfinite(change) and change <= 0, change)
    assert set(gates.index) == set(expected) | {"independent_reconstruction_audit", "overall"}
    for name, (passed, value) in expected.items():
        assert bool(gates.loc[name, "passed"]) == passed, name
        if name == "model_frozen_before_external":
            assert str(gates.loc[name, "observed"]) == str(value)
        else:
            np.testing.assert_allclose(float(gates.loc[name, "observed"]), value, rtol=1e-10, atol=1e-10, equal_nan=True, err_msg=name)
    assert not gates.loc["independent_reconstruction_audit", "passed"] and not gates.loc["overall", "passed"]
    gates.loc["independent_reconstruction_audit", ["passed", "observed"]] = [True, "passed_dense_and_aggregation_audits"]
    gates.loc["overall", ["passed", "observed"]] = [all(x[0] for x in expected.values()), "all_frozen_gates_evaluated"]
    return gates.reset_index()


def make_figure(summary, animal, pf, output):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    colors = {"mn1173": "#16806a", "mn9686": "#ba4975"}
    real = animal.loc[(animal.split == 0) & animal.source.eq("real")]
    for ax, metric, title in ((axes[0, 0], "ac_regional_tv", "Regional disagreement"),
                             (axes[0, 1], "ac_separation_cm", "Endpoint-mean separation (cm)")):
        for rat, local in real.groupby("animal"):
            local = local.set_index("policy")
            ax.plot([0, 1], [local.loc["random_expectation", metric], local.loc["full", metric]],
                    "o-", color=colors.get(rat), label=rat)
        ax.set_xticks([0, 1], ["All candidates", "Selected half"])
        ax.set_xlim(-.25, 1.25)
        ax.set_title(title, fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    names = ["RUN truth", "Matched simulation", "Drift simulation"]
    primary = summary.loc[summary.split.eq(0)]
    ax = axes[1, 0]
    for side, color in (("a", "#246599"), ("c", "#ad6028")):
        changes = []
        for source in ("run_test", "sim_matched", "sim_drift"):
            table = primary.loc[primary.source.eq(source)].set_index("policy")
            changes.append(table.loc["full", f"{side}_truth_error_cm"]-table.loc["random_expectation", f"{side}_truth_error_cm"])
        ax.plot(range(3), changes, "o-", label=f"Population {side.upper()}", color=color)
    ax.axhline(0, color="0.5", lw=.8)
    ax.set_xticks(range(3), names, rotation=12)
    ax.set_title("Known-position error change (cm; lower is better)", fontsize=11)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1, 1]
    pf = pf.set_index("policy")
    external = primary.loc[primary.source.eq("real")].set_index("policy")
    for x, table, label, color in ((0, pf, "PF development", "#246599"), (1, external, "AutoPI external", "#ba4975")):
        ax.bar(x-.16, table.loc["random_expectation", "c_support"], .3, color=color, alpha=.45)
        ax.bar(x+.16, table.loc["full", "c_support"], .3, color=color)
        ax.text(x, -.04, label, ha="center", va="top", transform=ax.get_xaxis_transform())
    ax.set_xticks([])
    ax.set_ylim(0, .3)
    ax.set_ylabel("C-support fraction")
    ax.set_title("Light: all; dark: selected half\nAutoPI: no positive support examples", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Three-population diagnostic: independent validation failed", fontsize=15)
    fig.savefig(output/"three_population_validation.png", dpi=180)
    fig.savefig(output/"three_population_validation.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, directory = args.run_root, args.run_root/"external-validation"
    paths = dict(audit=root/"audit/independent_audit.json", predictions=directory/"external_predictions.csv.gz",
        source=root/"autopi-native-transfer/native-v2/native_sessions.csv", sessions=root/"autopi/sessions.csv",
        frozen=root/"pf-model/frozen_pf_model.json", measurement=root/"autopi/manifest.json",
        pf=root/"pf-model/pf_development_summary.csv", producer=Path(__file__))
    for name in ("by_session", "by_animal", "summary", "gate_summary"):
        paths[name] = directory/f"external_{name}.csv"
    provenance = build_script_provenance(input_paths=paths, cwd=ROOT)
    audit = json.loads(paths["audit"].read_text())
    if audit["status"] != "passed" or audit["prediction_rows"] == 0:
        raise ValueError("independent readout and prediction reconstruction not passed")
    frozen, measured = json.loads(paths["frozen"].read_text()), json.loads(paths["measurement"].read_text())
    frozen_before = (frozen["training_dataset"] == "pfeiffer_foster" and frozen["status"] == "frozen"
        and frozen["created_at_utc"] < measured["created_at_utc"]
        and measured["input_file_sha256"]["frozen_model"] == file_sha256(paths["frozen"])
        and audit["provenance"]["input_file_sha256"]["frozen_model"] == file_sha256(paths["frozen"]))
    for stage in ("pf", "autopi"):
        catalog = pd.read_csv(root/stage/"sessions.csv")
        for row in catalog.loc[catalog.status.eq("complete")].itertuples(index=False):
            digest = file_sha256(Path(row.artifact_dir)/"event_readouts.csv.gz")
            if digest != audit["verified_readouts"][row.dataset+":"+row.session]:
                raise ValueError("audit is stale relative to measured outcomes")
    predictions = pd.read_csv(paths["predictions"])
    # Rebind this exact prediction file to the independent audit by rechecking
    # its numeric content, rather than trusting an unrelated 'passed' JSON.
    from scripts.audit_three_population_content import verify_external_predictions
    catalog = pd.read_csv(paths["sessions"])
    rebuilt = pd.concat([pd.read_csv(Path(row.artifact_dir)/"event_readouts.csv.gz")
        for row in catalog.loc[catalog.status.eq("complete")].itertuples(index=False)])
    verify_external_predictions(rebuilt, frozen, predictions)
    animal, summary = verify_aggregates(predictions, directory)
    source = pd.read_csv(paths["source"])
    gates = verify_gates(directory, animal, summary, source, catalog, frozen_before)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    gates.to_csv(args.output_dir/"audited_gate_summary.csv", index=False)
    primary = predictions.loc[predictions.source.eq("real") & predictions.split.eq(0)]
    quality = primary.groupby(["animal", "session"], as_index=False).agg(
        events=("event_index", "size"), positive_c_support=("c_support", "sum"),
        mean_a_spikes=("a_spikes", "mean"), mean_b_spikes=("b_spikes", "mean"), mean_c_spikes=("c_spikes", "mean"),
        mean_a_entropy=("a_entropy", "mean"), mean_c_entropy=("c_entropy", "mean"))
    quality.to_csv(args.output_dir/"external_observability_by_session.csv", index=False)
    main_table = summary.loc[summary.source.eq("real") & summary.split.eq(0)].set_index("policy")
    baseline, selected = main_table.loc["random_expectation"], main_table.loc["full"]
    known = float(source.candidates.sum())
    lines = ["# Three-population independent validation", "", "**Outcome: NOT VALIDATED.**", "",
        "A/B-only features were trained on PF and frozen before external AutoPI decoding. C cells never supplied predictor features.",
        "This measures population support and instability, not biological replay ground truth.", "", "## Coverage", "",
        f"- Native recordings attempted: {len(source)}; catalogs extracted: {source.status.eq('extracted').sum()}.",
        f"- Evaluated primary endpoints: {len(primary)} in {primary.session.nunique()} recordings/{primary.animal.nunique()} animals.",
        f"- Known cataloged source candidates: {int(known)}; evaluated fraction among these: {len(primary)/known:.2%}.",
        "- Some failed source recordings have unknown candidate counts. The above fraction does not establish complete source coverage.",
        "- Frozen minimum: four animals and 80% source endpoints. Neither gate passes.", "", "## Predictive and retention results", "",
        "| Metric | All endpoints / random expectation | Selected half |", "|---|---:|---:|"]
    for metric in ("c_support", "ac_regional_tv", "ac_separation_cm", "a_entropy", "c_entropy"):
        lines.append(f"| {metric} | {baseline[metric]:.6f} | {selected[metric]:.6f} |")
    lines += ["", f"Full-model log loss: {baseline.logloss_full:.6f}; pooled A+B baseline: {baseline.logloss_pooled:.6f}; "
        f"training prevalence: {baseline.logloss_constant:.6f}.",
        f"C-support positives: {int(primary.c_support.sum())}/{len(primary)}. An all-negative target cannot demonstrate sensitivity to supported events.",
        "The lower loss is better prediction of absence in this low-observability subset, not successful positive content certification.",
        "Retaining predicted-best events increased regional disagreement and endpoint separation; this fails the requested instability remedy.",
        "Known-position errors decreased on the scored subset, but that partial improvement does not override coverage or disagreement failures.",
        "", "## Interpretation", "",
        "Low disagreement can be produced by diffuse posteriors concentrated around similar means. Conversely, a sharper estimate from one "
        "population can disagree more with an uninformative held-out population. Agreement, support and truth error are distinct validation targets.",
        "The current experiment does not establish a remedy or an externally validated positive-support diagnostic. No model, threshold, "
        "population seed or event endpoint was changed after external outcomes were inspected.", "", "## Verification", "",
        f"Independent dense reconstruction: {audit['reconstructed_rows']} readout rows, {audit['reconstructed_endpoints']} endpoints, "
        f"{audit['recordings']} recordings. All external probabilities, losses and fixed-coverage decisions were checked.",
        "Session/draw/animal aggregates and every frozen numeric gate were separately reconstructed for this report.",
        "Encoding maps are shared source caches; raw spike sorting and true biological replay content are not independently established.",
        "Bootstrap intervals from only two external animals are descriptive and cannot establish population generalization.",
        "", "## Next constraint", "",
        "Any further independent validation needs enough well-covered RUN-qualified cells and independently supported endpoints. "
        "Do not salvage this result by silently excluding failures or lowering the frozen thresholds."]
    (args.output_dir/"three_population_validation_outcome.md").write_text("\n".join(lines)+"\n")
    make_figure(summary, animal, pd.read_csv(paths["pf"]), args.output_dir)
    unchanged = all(file_sha256(path) == provenance["input_file_sha256"][key] for key, path in paths.items())
    provenance.update(status="complete" if unchanged else "failed", inputs_unchanged=unchanged,
        interpretation="not_validated", aggregate_and_gate_reconstruction="passed", no_rescoring=True,
        created_at_utc=datetime.now(UTC).isoformat())
    (args.output_dir/"report_manifest.json").write_text(json.dumps(provenance, indent=2)+"\n")
    if not unchanged:
        raise RuntimeError("report inputs changed")


if __name__ == "__main__":
    main()
