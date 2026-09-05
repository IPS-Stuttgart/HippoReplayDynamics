#!/usr/bin/env python3
"""Non-rescoring report of independent-map recovery and calibration sensitivity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts._provenance import build_script_provenance, file_sha256
from scripts.simulate_replay_coverage_map_mismatch import CONDITION, IDENTITY

MEASURES = ["acceptance_fraction", "eligible_recovery_fraction", "median_position_error_cm", "hpd95_coverage",
            "valid_bins", "all_steps", "selected_steps", "large_jump_fraction", "all_median_speed_cm_s",
            "selected_median_speed_cm_s", "truth_training_supported_fraction", "truth_generator_supported_fraction"]


def aggregate(table, groups, values, seed=20260913):
    sessions = table.groupby(groups + ["animal", "session"], as_index=False, dropna=False)[values].mean()
    animals = sessions.groupby(groups + ["animal"], as_index=False, dropna=False)[values].mean()
    rows, rng = [], np.random.default_rng(seed)
    for key, group in animals.groupby(groups, sort=True, dropna=False):
        for metric in values:
            data = group[metric].dropna().to_numpy(float)
            ci = [np.nan, np.nan]
            if len(data) > 1:
                ci = np.quantile(rng.choice(data, (2000, len(data))).mean(axis=1), [.025, .975])
            rows.append({**dict(zip(groups, key, strict=True)), "metric": metric, "animals_expected": len(group),
                         "animals_available": len(data), "mean": float(data.mean()) if len(data) else np.nan,
                         "ci95_low": ci[0], "ci95_high": ci[1]})
    return sessions, animals, pd.DataFrame(rows)


def gradient_responses(gradients):
    keys = IDENTITY + CONDITION + ["selection", "coordinate", "readout"]
    pivot = gradients[gradients.gradient.isin([-.5, .5])].pivot(index=keys, columns="gradient", values="normalized_slope").reindex(columns=[-.5, .5])
    frame = pivot.index.to_frame(index=False)
    frame["gradient_response"] = (pivot[.5] - pivot[-.5]).to_numpy()
    frame["both_gradients_available"] = pivot.notna().all(axis=1).to_numpy()
    return frame


def paired_map_effect(table, keys, metrics):
    index = [name for name in keys if name != "decoder_map"]
    oracle = table[table.decoder_map.eq("generator_known")][index+metrics]
    changed = table[table.decoder_map.eq("independent_RUN_half")][index+metrics]
    pair = changed.merge(oracle, on=index, how="outer", validate="one_to_one", suffixes=("_estimated", "_known"), indicator=True)
    if not pair._merge.eq("both").all():
        raise ValueError("unpaired independent/known map conditions")
    for metric in metrics:
        pair[metric] = pair[f"{metric}_estimated"] - pair[f"{metric}_known"]
    return pair[index+metrics]


def plot(summary, response, output):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for col, (dataset, title) in enumerate([("pfeiffer_foster", "Pfeiffer/Foster maps"), ("tanni2022", "Tanni maps: all sizes")]):
        base = summary[summary.dataset.eq(dataset) & summary.observation.eq("poisson") & summary.likelihood.eq("poisson")
            & summary.bin_filter.eq("unfiltered") & summary.estimator.eq("map") & summary.truth_kind.eq("continuous")
            & summary.gradient.eq(0) & summary.metric.eq("eligible_recovery_fraction")]
        for map_name, label, color in [("generator_known", "Generator-known map", "#1d7b83"), ("independent_RUN_half", "Independent RUN-half map", "#a94f62")]:
            part = base[base.decoder_map.eq(map_name)].sort_values("cell_fraction")
            axes[0, col].plot(part.cell_fraction, 100*part["mean"], "o-", label=label, color=color)
            axes[0, col].fill_between(part.cell_fraction, 100*part.ci95_low, 100*part.ci95_high, color=color, alpha=.15)
            sub = response[response.dataset.eq(dataset) & response.observation.eq("poisson") & response.likelihood.eq("poisson")
                & response.bin_filter.eq("unfiltered") & response.estimator.eq("posterior_mean") & response.selection.eq("all")
                & response.coordinate.eq("true_coordinate") & response.readout.eq("decoded") & response.decoder_map.eq(map_name)].sort_values("cell_fraction")
            axes[1, col].plot(sub.cell_fraction, sub["mean"], "o-", label=label, color=color)
            axes[1, col].fill_between(sub.cell_fraction, sub.ci95_low, sub.ci95_high, color=color, alpha=.15)
        axes[0, col].set(title=title, ylabel="Eligible true paths recovered (%)", ylim=(0, 100))
        axes[1, col].set(ylabel="Decoded speed-gradient response")
        axes[1, col].axhline(1, color=".4", ls=":", label="Injected contrast")
        axes[1, col].axhline(0, color=".6", lw=.6)
        for row in [0, 1]:
            axes[row, col].set(xticks=[.5, 1.], xticklabels=["Half cells", "Full cells"], xlim=(.4, 1.1))
            axes[row, col].legend(fontsize=8)
    fig.suptitle("Separate RUN-half maps; identical simulated spikes for each map comparison\nKnown-path surrogates, not measured replay truth; animal-bootstrap intervals")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run(args):
    root, out = args.input_dir.resolve(), args.output_dir.resolve()
    files = {name: root / f"coverage_map_mismatch_{name}.csv" for name in ["session_summary", "session_gradients", "batches", "observations", "trials", "gate_summary"]}
    files["manifest"] = root / "coverage_map_mismatch_manifest.json"
    files["reporter"] = Path(__file__)
    manifest = json.loads(files["manifest"].read_text())
    if manifest["status"] != "complete":
        raise ValueError("scoring did not complete technically")
    for path in files.values():
        if path.name in manifest["output_sha256"] and file_sha256(path) != manifest["output_sha256"][path.name]:
            raise ValueError("report input hash mismatch")
    out.mkdir(parents=True, exist_ok=False)
    table, gradients = pd.read_csv(files["session_summary"]), pd.read_csv(files["session_gradients"])
    groups = ["dataset"] + CONDITION + ["truth_kind", "gradient"]
    sessions, animals, summary = aggregate(table, groups, MEASURES)
    response = gradient_responses(gradients)
    response_groups = ["dataset"] + CONDITION + ["selection", "coordinate", "readout"]
    response_sessions, response_animals, response_summary = aggregate(response, response_groups, ["gradient_response"])
    map_pairs = paired_map_effect(table, IDENTITY + CONDITION + ["truth_kind", "gradient"], MEASURES)
    _, paired_animals, paired_summary = aggregate(map_pairs, [name for name in groups if name != "decoder_map"], MEASURES)
    response_pairs = paired_map_effect(response, IDENTITY + CONDITION + ["selection", "coordinate", "readout"], ["gradient_response"])
    _, response_pair_animals, response_pair_summary = aggregate(response_pairs, [name for name in response_groups if name != "decoder_map"], ["gradient_response"])
    availability = response.groupby(["dataset"]+CONDITION+["selection", "coordinate", "readout"], as_index=False).agg(
        expected_directions=("both_gradients_available", "size"), available_directions=("both_gradients_available", "sum"))
    trials, observations = pd.read_csv(files["trials"]), pd.read_csv(files["observations"])
    budget = observations.merge(trials[IDENTITY+["source_event_index", "truth_kind", "gradient", "duration_s", "source_spikes_training_units"]],
                               on=IDENTITY+["source_event_index", "truth_kind", "gradient"], validate="many_to_one")
    budget["source_rate_hz"] = budget.source_spikes_retained_units / budget.duration_s
    budget["simulated_rate_hz"] = budget.spikes / budget.duration_s
    _, _, budget_summary = aggregate(budget, ["dataset", "observation", "cell_fraction", "truth_kind", "gradient"], ["source_rate_hz", "simulated_rate_hz"])
    outputs = {"session_endpoints": sessions, "animal_endpoints": animals, "endpoint_summary": summary,
        "gradient_by_direction": response, "gradient_by_session": response_sessions, "gradient_by_animal": response_animals,
        "gradient_response_summary": response_summary, "gradient_availability": availability,
        "paired_map_by_animal": paired_animals, "paired_map_summary": paired_summary,
        "paired_gradient_by_animal": response_pair_animals, "paired_gradient_summary": response_pair_summary,
        "spike_budget_summary": budget_summary}
    for name, frame in outputs.items():
        frame.to_csv(out / f"coverage_map_mismatch_{name}.csv", index=False)
    plot(summary, response_summary, out / "coverage_map_mismatch_recovery.png")
    batches = pd.read_csv(files["batches"])
    audit_path = root / "coverage_map_mismatch_reconstruction_audit.json"
    audited = False
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        audited = audit["status"] == "pass" and audit["input_file_sha256"]["scoring_manifest"] == file_sha256(files["manifest"])
        if not audited:
            raise ValueError("audit is failed or refers to a different scoring manifest")
        files["reconstruction_audit"] = audit_path
    report = ("# Independent RUN-Map Mismatch\n\nNon-rescoring report.\n\n"
        f"{len(batches)} planned half directions; {batches.status.eq('scored').sum()} scored; independent reconstruction audit present/pass: {audited}.\n\n"
        "Compare independent-minus-generator-known on identical spikes, states, cells and priors. Directions average within session, sessions within animal, animals equally. Bootstrap intervals condition on the empirical maps/synthetic draws and use only four PF or five Tanni animals. Missing gradient directions remain explicit.\n\n"
        "Half-map differences may include neural drift and sampling, not only estimation noise. Generator-known is an oracle-rate comparison with training-restricted state support. Both extents constrain the synthetic domain; the decoder's fitting and support remain training-only.\n\n"
        "Shared gain is a declared stress model, not fitted replay covariance. Conditional decoding may reduce gain sensitivity but also discards population-rate information. Moving-within-window approximation remains in both families.\n\n"
        "These are simulated continuity/null acceptances, not measured biological replay prevalence or false-positive rates. No speed-uniformity or calibrated replay-uncertainty claim follows. Calibration/abstention on independent populations and real event-definition sensitivity remain required.\n")
    (out / "coverage_map_mismatch_report.md").write_text(report)
    provenance = build_script_provenance(input_paths=files, cwd=ROOT)
    provenance["output_sha256"] = {path.name: file_sha256(path) for path in out.iterdir() if path.is_file()}
    (out / "coverage_map_mismatch_report_manifest.json").write_text(json.dumps(provenance, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
