#!/usr/bin/env python3
"""Non-rescoring animal-deletion and exact sign sensitivity of frozen results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import build_script_provenance, file_sha256
from report_replay_measurement_paper_evidence import DATASETS, checked, load_sources, one

CONTRASTS = (
    "imm_minus_iid",
    "imm_minus_static",
    "imm_minus_composition",
    "imm_order_advantage",
    "imm_order_map_interaction",
)


def animal_rows(frame, filters, expected):
    mask = pd.Series(True, index=frame.index)
    for key, value in filters.items():
        mask &= frame[key].eq(value)
    selected = frame.loc[mask].copy()
    if selected.animal.isna().any() or selected.animal.duplicated().any() or len(selected) != expected:
        raise ValueError(f"incomplete or duplicate animal rows: {filters}")
    return selected.set_index("animal").sort_index()


def sign_sensitivity(values, direction):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or direction not in (-1, 1):
        raise ValueError("finite animal vector and declared direction required")
    directed = direction * values
    nonzero = int(np.count_nonzero(directed))
    supporting = int(np.count_nonzero(directed > 0))
    return {
        "animals": len(values),
        "animals_supporting_direction": supporting,
        "zero_ties": len(values) - nonzero,
        "mean": float(values.mean()),
        "sign_p_one_sided": float(binomtest(supporting, nonzero, 0.5, alternative="greater").pvalue) if nonzero else 1.0,
        "sign_p_two_sided": float(binomtest(supporting, nonzero, 0.5).pvalue) if nonzero else 1.0,
        "minimum_attainable_one_sided_p": 0.5**nonzero if nonzero else 1.0,
        "minimum_attainable_two_sided_p": min(1.0, 2 * 0.5**nonzero) if nonzero else 1.0,
    }


def collect_values(coverage, prediction, gradients, reference):
    rows = []
    identities = {}

    def add(dataset, family, endpoint, units, selected, value, direction, source, filters, reference_mean=None):
        array = np.asarray(value, dtype=float)
        if array.shape != (DATASETS[dataset][1],) or not np.isfinite(array).all():
            raise ValueError("nonfinite or incomplete animal endpoint")
        current = set(selected.index)
        if current != identities.setdefault(dataset, current):
            raise ValueError("animal identities differ between endpoints")
        if reference_mean is not None and not np.isclose(array.mean(), reference_mean, atol=1e-10, rtol=0):
            raise ValueError(f"animal mean does not reconstruct published endpoint: {endpoint}")
        for animal, estimate in zip(selected.index, array, strict=True):
            rows.append(
                {
                    "dataset": dataset,
                    "family": family,
                    "endpoint": endpoint,
                    "units": units,
                    "animal": animal,
                    "value": float(estimate),
                    "direction": direction,
                    "source_table": source,
                    "source_filter": json.dumps(filters, sort_keys=True),
                }
            )

    for dataset, (_, count) in DATASETS.items():
        ripple = "native_ripple_table" if dataset == "pfeiffer_foster" else "lfp_ripple_detected"
        for detector in ("source_high_mua", ripple):
            f = {"dataset": dataset, "detector": detector, "observation": "original_order", "bin_filter": "edge_only", "min_frames": 10, "alpha": 0.02}
            selected = animal_rows(coverage, f, count)
            target = one(reference["shuffle_baseline"], {"dataset": dataset, "detector": detector, "observation": "original_order"})
            add(dataset, "continuity", detector, "percentage_points", selected, 100 * selected.accepted_fraction_delta, -1, "coverage", f, target.half_minus_full_pp)

        for group in ("rejected_with_opportunity", "lost_with_thinning"):
            for contrast in CONTRASTS:
                f = {"dataset": dataset, "support": "parent", "bin_filter": "edge_only", "min_frames": 10, "group": group, "contrast": contrast}
                selected = animal_rows(prediction, f, count)
                target = one(reference["prediction"], f)
                for field, units, target_field in [("delta", "nats", "mean"), ("delta_per_heldout_spike", "nats_per_heldout_spike", "mean_per_heldout_spike")]:
                    add(dataset, "prediction", f"{group}:{contrast}", units, selected, selected[field], 1, "prediction", f, target[target_field])

        f = {
            "dataset": dataset,
            "observation": "poisson",
            "cell_fraction": 1.0,
            "likelihood": "poisson",
            "estimator": "posterior_mean",
            "bin_filter": "unfiltered",
            "selection": "all",
            "coordinate": "true_coordinate",
        }
        estimated_filter = f | {"decoder_map": "independent_RUN_half", "readout": "decoded"}
        estimated = animal_rows(gradients, estimated_filter, count)
        estimated_ref = one(reference["map_mismatch"], estimated_filter | {"metric": "gradient_response"})["mean"]
        for name, readout in [("arclength_truth", "true_arclength"), ("window_truth", "true_chord"), ("known_map", "decoded")]:
            comparator_filter = f | {"decoder_map": "generator_known", "readout": readout}
            comparator = animal_rows(gradients, comparator_filter, count)
            if not estimated.index.equals(comparator.index):
                raise ValueError("gradient animal identities do not match")
            comparator_ref = one(reference["map_mismatch"], comparator_filter | {"metric": "gradient_response"})["mean"]
            add(
                dataset,
                "gradient",
                f"estimated_minus_{name}",
                "gradient_contrast",
                estimated,
                estimated.gradient_response - comparator.gradient_response,
                -1,
                "gradients",
                {"estimated": estimated_filter, "comparator": comparator_filter},
                estimated_ref - comparator_ref,
            )
    return pd.DataFrame(rows)


def summarize(values):
    keys = ["dataset", "family", "endpoint", "units"]
    summaries, deletions = [], []
    if values.duplicated(keys + ["animal"]).any():
        raise ValueError("duplicate animal endpoint")
    for key, frame in values.groupby(keys, sort=True):
        if len(frame) != DATASETS[key[0]][1] or frame.direction.nunique() != 1:
            raise ValueError("incomplete endpoint or conflicting direction")
        meta = dict(zip(keys, key, strict=True))
        direction = int(frame.direction.iloc[0])
        result = sign_sensitivity(frame.value, direction)
        loo = []
        for row in frame.itertuples():
            mean = float(frame.loc[frame.animal.ne(row.animal), "value"].mean())
            loo.append(mean)
            deletions.append(
                meta | {"excluded_animal": row.animal, "remaining_animals": len(frame) - 1, "mean": mean, "direction": direction, "supports_direction": bool(direction * mean > 0)}
            )
        summaries.append(
            meta
            | result
            | {"direction": direction, "loo_mean_min": min(loo), "loo_mean_max": max(loo), "all_loo_means_support_direction": bool(np.all(direction * np.asarray(loo) > 0))}
        )
    return pd.DataFrame(summaries), pd.DataFrame(deletions)


def run(args):
    reference, paths, _ = load_sources(args.coverage_index.resolve(), args.prediction_report.resolve(), args.prediction_audit.resolve())
    paths["script"] = Path(__file__)
    paths["protocol"] = ROOT / "docs/replay_measurement_animal_sensitivity_protocol.md"
    manifest = json.loads(paths["shuffle_baseline_report"].read_text())
    paths["animal_coverage"] = checked(manifest["input_file_paths"]["animal_summary"], manifest["input_file_sha256"]["animal_summary"])
    for key, report_key, filename in [
        ("animal_prediction", "prediction_report_manifest", "training_continuity_by_animal.csv"),
        ("animal_gradients", "map_mismatch_report", "coverage_map_mismatch_gradient_by_animal.csv"),
    ]:
        manifest = json.loads(paths[report_key].read_text())
        paths[key] = checked(paths[report_key].parent / filename, manifest["output_sha256"][filename])
    before = {key: file_sha256(path) for key, path in paths.items()}
    values = collect_values(*(pd.read_csv(paths[key]) for key in ["animal_coverage", "animal_prediction", "animal_gradients"]), reference)
    summary, deletions = summarize(values)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name, frame in [("animal_values", values), ("leave_one_animal_out", deletions), ("animal_sensitivity_summary", summary)]:
        frame.to_csv(out / f"replay_measurement_{name}.csv", index=False)
    report = [
        "# Replay Measurement: Animal-Level Sensitivity",
        "",
        "Retrospective, non-rescoring check. Frozen source analyses and decisions are unchanged.",
        "",
        "Each source already averages within animals. Deletion means remove one animal from those fixed estimates; they do not refit maps or decoders.",
        "",
        "Exact sign p-values concern independent animal signs with null probability 0.5, not the hierarchical-bootstrap magnitude estimand. These unadjusted diagnostics are not new confirmatory tests. A non-small p-value does not establish no effect or equivalence.",
        "",
        "Four nonzero animals permit a minimum one-sided sign p of 0.0625 (two-sided 0.125). Five permit 0.03125 (two-sided 0.0625). Thousands of events cannot increase that biological replicate count. No pooling of datasets, detectors or contrasts is performed.",
        "",
        "| Dataset | Endpoint | Units | Mean | Leave-one-animal-out range | Directional animals | Sign p, one / two sided |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for row in summary.itertuples():
        report.append(
            f"| {row.dataset} | {row.endpoint} | {row.units} | {row.mean:.4f} | [{row.loo_mean_min:.4f}, {row.loo_mean_max:.4f}] | {row.animals_supporting_direction}/{row.animals} | {row.sign_p_one_sided:.5f} / {row.sign_p_two_sided:.5f} |"
        )
    report += [
        "",
        "## Boundaries",
        "",
        "Continuity endpoints use the full transferred two-shuffle criterion. Prediction endpoints use training-only geometry and held-out population identity scores, not true replay labels. Gradient endpoints use prescribed-path simulations with empirical maps, not observed biological gradients. The five predictive contrasts and both normalizations remain visible even when adverse.",
        "",
        "Leave-one-animal-out sign stability is a descriptive robustness result, not prospective external replication. Conditional bootstrap intervals and exact sign sensitivities make different assumptions; neither should silently replace the other. A methods-paper contribution can concern the measured recording populations without claiming a population-wide biological mechanism.",
        "",
        "Exact binomial reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html",
        "",
    ]
    (out / "replay_measurement_animal_sensitivity.md").write_text("\n".join(report))
    for key, path in paths.items():
        checked(path, before[key])
    manifest = build_script_provenance(input_paths=paths, cwd=ROOT)
    manifest.update(
        status="complete",
        non_rescoring=True,
        endpoint_count=len(summary),
        animal_rows=len(values),
        verification_scope="Source manifest/hash linkage; animal means reconstructed against published summaries; no new event rescore",
        output_sha256={p.name: file_sha256(p) for p in sorted(out.iterdir()) if p.is_file()},
    )
    (out / "replay_measurement_animal_sensitivity_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "complete", "endpoints": len(summary), "animal_rows": len(values)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-index", required=True, type=Path)
    parser.add_argument("--prediction-report", required=True, type=Path)
    parser.add_argument("--prediction-audit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    run(parser.parse_args())
