"""Non-rescoring, equal-animal report of detector/decoder population effects."""

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

CONDITION = ["dataset", "generator", "peak_gain", "likelihood", "region", "family", "side"]
METRICS = ["detection", "decoding", "interaction", "total", "true_selection_shift", "readout_bias_shift", "q_full_full", "q_subset_full", "q_full_subset", "q_subset_subset"]


def aggregate(contrasts, bootstrap=2000):
    if contrasts.empty or contrasts.duplicated(["dataset", "animal", "session", "source", "population", "likelihood", "region"]).any():
        raise ValueError("empty or duplicate factorial input")
    valid = contrasts.status.eq("complete")
    session = contrasts.loc[valid].groupby(CONDITION + ["animal", "session"], dropna=False)[METRICS].mean().reset_index()
    if session.empty:
        raise ValueError("no estimable contrasts")
    animal = session.groupby(CONDITION + ["animal"], dropna=False)[METRICS].mean().reset_index()
    result = []
    rng = np.random.default_rng(2026091702)
    for keys, group in animal.groupby(CONDITION, dropna=False, sort=True):
        row = dict(zip(CONDITION, keys, strict=True))
        row["animals"] = group.animal.nunique()
        for metric in METRICS:
            values = group[metric].dropna().to_numpy(float)
            row[metric] = float(values.mean()) if len(values) else np.nan
            row[metric + "_positive_animals"] = int((values > 0).sum())
            if len(values) > 1:
                draws = rng.choice(values, size=(bootstrap, len(values))).mean(axis=1)
                row[metric + "_ci_low"], row[metric + "_ci_high"] = np.quantile(draws, [0.025, 0.975])
            else:
                row[metric + "_ci_low"], row[metric + "_ci_high"] = np.nan, np.nan
        result.append(row)
    return session, animal, pd.DataFrame(result)


def detection_truth(epochs):
    rows = []
    keys = ["dataset", "animal", "session", "generator", "peak_gain", "population", "family", "side", "repeat"]
    for key, group in epochs.groupby(keys):
        inside = group.true_central_occupancy >= 1 - 1e-8
        outside = group.true_central_occupancy <= 1e-8
        rows.append(
            dict(zip(keys, key, strict=True))
            | {
                "epochs": len(group),
                "inside_epochs": int(inside.sum()),
                "outside_epochs": int(outside.sum()),
                "mixed_epochs": int((~inside & ~outside).sum()),
                "hit_inside": float(group.loc[inside, "n_gain_peak_detections"].gt(0).mean()) if inside.any() else np.nan,
                "hit_outside": float(group.loc[outside, "n_gain_peak_detections"].gt(0).mean()) if outside.any() else np.nan,
                "any_detection_fraction": float(group.n_detections.gt(0).mean()),
                "multiple_detection_fraction": float(group.n_detections.gt(1).mean()),
            }
        )
    return pd.DataFrame(rows)


def matched_timing(path):
    """Only uniquely matched pairs; no claim that this selected cohort is unbiased."""
    matches = pd.read_csv(path / "event_matches.csv")
    readouts = pd.read_csv(path / "window_readouts.csv.gz")
    events = pd.read_csv(path / "detector_events.csv")
    reference = events.loc[events.detector.eq("full"), ["event_start_s", "event_end_s"]].to_numpy()
    output = []
    for population, group in matches.loc[matches.population.ne("full") & matches.full_event_id.ge(0)].groupby("population"):
        alternate = events.loc[events.detector.eq(population)].set_index("event_id")
        windows = alternate.loc[group.event_id, ["event_start_s", "event_end_s"]].to_numpy()
        overlap = np.maximum(0, np.minimum(windows[:, 1, None], reference[None, :, 1]) - np.maximum(windows[:, 0, None], reference[None, :, 0]))
        tied = np.isclose(overlap, overlap.max(axis=1)[:, None], atol=1e-8, rtol=0).sum(axis=1) > 1
        group = group.loc[~group.full_event_id.duplicated(keep=False) & ~tied]
        for likelihood in ("poisson", "conditional_multinomial"):
            values = readouts.loc[readouts.likelihood.eq(likelihood)]
            full_full = values.loc[values.detector.eq("full") & values.decoder.eq("full")].set_index("event_id")
            full_sub = values.loc[values.detector.eq("full") & values.decoder.eq(population)].set_index("event_id")
            sub_full = values.loc[values.detector.eq(population) & values.decoder.eq("full")].set_index("event_id")
            sub_sub = values.loc[values.detector.eq(population) & values.decoder.eq(population)].set_index("event_id")
            if group.empty:
                continue
            f, a = group.full_event_id.to_numpy(int), group.event_id.to_numpy(int)
            meta = {k: group.iloc[0][k] for k in ("dataset", "animal", "session", "source", "generator", "peak_gain", "replicate")}
            output.append(
                dict(
                    **meta,
                    population=population,
                    likelihood=likelihood,
                    matched_unique_pairs=len(group),
                    same_full_window_population_delta=float((full_sub.loc[f, "mass_4"].to_numpy() - full_full.loc[f, "mass_4"].to_numpy()).mean()),
                    full_decoder_boundary_delta=float((sub_full.loc[a, "mass_4"].to_numpy() - full_full.loc[f, "mass_4"].to_numpy()).mean()),
                    subset_decoder_boundary_delta=float((sub_sub.loc[a, "mass_4"].to_numpy() - full_sub.loc[f, "mass_4"].to_numpy()).mean()),
                    median_endpoint_shift_ms=float(group.endpoint_shift_ms.median()),
                )
            )
    return pd.DataFrame(output)


def make_figure(animal, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for column, dataset in enumerate(("pfeiffer_foster", "tanni2022")):
        base = animal.loc[animal.dataset.eq(dataset) & animal.family.eq("targeted") & animal.side.eq("low") & animal.region.eq(4) & animal.likelihood.eq("poisson")]
        for line, (generator, gain, metrics, labels) in enumerate(
            [
                ("real", 0, ["detection", "decoding", "interaction", "total"], ["Detector", "Decoder", "Interaction", "Combined"]),
                ("stationary", 6, ["true_selection_shift", "readout_bias_shift", "total"], ["True selection", "Readout bias", "Combined"]),
            ]
        ):
            ax = axes[line, column]
            group = base.loc[base.generator.eq(generator) & base.peak_gain.eq(gain)]
            for i, metric in enumerate(metrics):
                values = group[metric].dropna().to_numpy() * 100
                if len(values):
                    ax.bar(i, values.mean(), width=0.6, alpha=0.6, color=("#7C6888" if metric == "total" else ["#277E8B", "#BC5153", "#739253"][i]))
                    ax.scatter(i + np.linspace(-0.16, 0.16, len(values)), values, color="black", s=22, zorder=3)
            ax.axhline(0, color="black", linewidth=0.7)
            ax.set_xticks(range(len(metrics)), labels)
            ax.set_ylabel("Change in mean central posterior mass (pp)" if line == 0 else "Change in central mass (pp)")
            ax.set_title({"pfeiffer_foster": "Pfeiffer-Foster", "tanni2022": "Tanni"}[dataset] + " / " + ("real sensitivity" if line == 0 else "known stationary content, gain 6"))
    fig.suptitle("RUN-coverage-poor half-population: detector versus decoder\nDots are animal means; real differences are not verified biological errors")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def table(frame, columns):
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for col in columns:
            value = row[col]
            values.append(f"{value:.2f}" if isinstance(value, (float, np.floating)) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.experiment / "manifest.json").read_text())
    audit = json.loads((args.audit / "manifest.json").read_text())
    if audit["status"] != "technical_pass" or audit["input_file_sha256"]["experiment"] != file_sha256(args.experiment / "manifest.json"):
        raise ValueError("valid linked audit required")
    sessions = pd.read_csv(args.experiment / "sessions.csv")
    frozen = pd.read_csv(args.experiment / "frozen_sessions.csv")
    if len(sessions) != len(frozen) or not sessions.status.eq("complete").all():
        raise ValueError("cohort incomplete")
    cs, summaries, epochs, matches, populations, sources, timing = [], [], [], [], [], [], []
    for row in sessions.itertuples():
        folder = Path(row.folder)
        pp = pd.read_csv(folder / "population_summary.csv")
        for k in ("dataset", "animal", "session"):
            pp[k] = getattr(row, k)
        populations.append(pp)
        ss = pd.read_csv(folder / "sources.csv")
        sources.append(ss)
        for s in ss.itertuples():
            path = Path(s.folder)
            cs.append(pd.read_csv(path / "factorial_contrasts.csv"))
            summaries.append(pd.read_csv(path / "readout_summary.csv"))
            matches.append(pd.read_csv(path / "event_matches.csv"))
            timing.append(matched_timing(path))
            if s.source != "real":
                epochs.append(pd.read_csv(path / "epoch_detection.csv"))
    contrast, readouts = pd.concat(cs, ignore_index=True), pd.concat(summaries, ignore_index=True)
    matches, pops, source = pd.concat(matches, ignore_index=True), pd.concat(populations, ignore_index=True), pd.concat(sources, ignore_index=True)
    by_session, by_animal, dataset = aggregate(contrast)
    primary = dataset.loc[dataset.region.eq(4)].copy()
    ep = detection_truth(pd.concat(epochs, ignore_index=True))
    met = ["hit_inside", "hit_outside", "any_detection_fraction", "multiple_detection_fraction"]
    ek = ["dataset", "generator", "peak_gain", "family", "side"]
    ep_session = ep.groupby(ek + ["animal", "session"])[met].mean().reset_index()
    ep_animal = ep_session.groupby(ek + ["animal"])[met].mean().reset_index()
    ep_dataset = ep_animal.groupby(ek)[met].mean().reset_index()
    match_summary = (
        matches.assign(matched=matches.full_event_id.ge(0))
        .groupby(["dataset", "animal", "session", "generator", "peak_gain", "population"])
        .agg(events=("matched", "size"), matched_fraction=("matched", "mean"), median_endpoint_shift_ms=("endpoint_shift_ms", "median"))
        .reset_index()
    )
    outputs = {
        "factorial_by_session.csv": by_session,
        "factorial_by_animal.csv": by_animal,
        "factorial_by_dataset.csv": dataset,
        "central_region_primary.csv": primary,
        "source_coverage.csv": source,
        "population_coverage.csv": pops,
        "detector_truth_by_population.csv": ep,
        "detector_truth_by_animal.csv": ep_animal,
        "detector_truth_by_dataset.csv": ep_dataset,
        "matching_summary.csv": match_summary,
        "matched_timing_by_source.csv": pd.concat(timing, ignore_index=True),
        "readout_summary.csv": readouts,
        "nonestimable_contrasts.csv": contrast.loc[~contrast.status.eq("complete")],
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / name, index=False)
    gates = [
        ("all_requested_sessions", len(sessions) == len(frozen), len(sessions)),
        ("all_session_jobs_complete", sessions.status.eq("complete").all(), int(sessions.status.eq("complete").sum())),
        ("both_datasets", set(sessions.dataset) == {"pfeiffer_foster", "tanni2022"}, sessions.dataset.nunique()),
        ("native_audit", audit["status"] == "technical_pass", audit["sources"]),
        ("all_primary_contrasts_nonempty", contrast.loc[contrast.region.eq(4)].status.eq("complete").all(), int(contrast.loc[contrast.region.eq(4)].status.eq("complete").sum())),
        (
            "all_population_families",
            pops.groupby(["dataset", "animal", "session"]).family.nunique().eq(4).all(),
            int(pops.groupby(["dataset", "animal", "session"]).family.nunique().eq(4).sum()),
        ),
    ]
    pd.DataFrame([{"gate": g, "passed": bool(ok), "observed": int(n)} for g, ok, n in gates]).to_csv(args.output_dir / "technical_gates.csv", index=False)
    make_figure(by_animal, args.output_dir / "detector_decoder_cross.png")
    shown = primary.loc[
        primary.family.eq("targeted") & primary.likelihood.eq("poisson") & (primary.generator.eq("real") | (primary.generator.eq("stationary") & primary.peak_gain.eq(6)))
    ].copy()
    for name in ["detection", "decoding", "interaction", "total", "true_selection_shift", "readout_bias_shift"]:
        shown[name] *= 100
    text = [
        "# Detector by decoder: frozen population-sampling experiment",
        "",
        "Non-rescoring report. Central 3x3 region, not a reward/Home region; full means all RUN-qualified cells.",
        f"Cohort: {len(sessions)} recordings, {sessions.groupby('dataset').animal.nunique().to_dict()} animals by dataset.",
        f"Source panels: {len(source)}. Nonestimable factorial rows: {int((contrast.status != 'complete').sum())}.",
        "",
        "## Primary contrasts",
        "",
        "Values are percentage points; replicas/subsets average within recording, then recordings within animal, then animals equally.",
        table(shown, ["dataset", "generator", "side", "detection", "decoding", "interaction", "total", "true_selection_shift", "readout_bias_shift"]),
        "",
        "Detection = subset-detector/full-decoder minus full/full. Decoding = full-detector/subset-decoder minus full/full.",
        "Interaction completes the crossed difference. Total is subset/subset minus full/full; no term is silently dropped.",
        "",
        "For simulations, true selection shift measures changed geometric content of selected windows; readout bias shift is the change in decoded-minus-true content.",
        "Their sum equals the combined decoded-content difference. In moving panels the true selection shift includes changes in event boundaries, not only membership.",
        "",
        "## Boundaries",
        "",
        "- No real biological content labels exist; full-population decoding is a reference, not truth.",
        "- All cells, positions, candidate rules and priors are frozen from RUN/geometry before outcomes. Targeted halves intentionally impose a coverage contrast and are not typical recordings.",
        "- Whole-tetrode controls preserve electrodes but may differ in cell count. Random and whole-tetrode repetitions are not independent animals.",
        "- These are MUA candidates, not verified continuous replay. Native detector settings were transferred to both datasets; historical event counts need not match.",
        "- Poisson decoding uses a fixed 20-ms exposure; simulated shared activity gain is not fitted. The conditional-multinomial sensitivity is separate, and moving windows can mix positions.",
        "- Synthetic path distribution and burst gain are specified model assumptions, not the true replay generator. No universal prevalence correction is validated.",
        "- Bootstrap intervals resample the four/five animal estimates and are descriptive with small animal counts; no multiplicity-adjusted biological claim follows.",
        "- The regional-content correction stopgate remains in force. This experiment diagnoses recording sensitivity, not a new correction method.",
        "",
        "The complete generator/gain/likelihood/region results and individual animal estimates accompany this report. Inspect them before interpreting the selected display above.",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(text))
    provenance = build_script_provenance(input_paths={"experiment": args.experiment / "manifest.json", "audit": args.audit / "manifest.json", "reporter": Path(__file__)}, cwd=ROOT)
    provenance.update(
        status="complete",
        scientific_result="descriptive_measurement_diagnostic",
        experiment_parameters=manifest["parameters"],
        outputs={p.name: file_sha256(p) for p in args.output_dir.iterdir() if p.is_file()},
    )
    (args.output_dir / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
