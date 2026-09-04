#!/usr/bin/env python3
"""Reproduce a pinned RatInABox coverage run and audit count conditioning.

The legacy source is an explicit, hashed input, not an installed dependency.
It supplies identical populations, paths, and counts to the prior experiment.
Both likelihoods and both point estimators see exactly the same spike arrays.
This diagnosis does not claim to complete the broader coverage calibration.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hipporeplayimm.replay_coverage import decode_independent, path_metrics
from scripts._provenance import build_script_provenance, file_sha256

KEYS = ["variant", "condition", "field_sigma_cm", "population_replicate", "event_index"]
STRATA = ["variant", "field_sigma_cm", "likelihood", "estimator", "bin_filter"]


@contextmanager
def load_legacy(directory: Path):
    """Load exactly two supplied modules without mixing in another worktree."""
    names = ["scripts.simulate_2d_place_field_coverage_continuity", "coverage_legacy_ratinabox"]
    filenames = ["simulate_2d_place_field_coverage_continuity.py", "simulate_ratinabox_place_field_coverage_continuity.py"]
    prior = {name: sys.modules.get(name) for name in names}
    try:
        for name, filename in zip(names, filenames, strict=True):
            spec = importlib.util.spec_from_file_location(name, directory / filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        yield sys.modules[names[-1]]
    finally:
        for name, old in prior.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def compare_reproduction(reproduced: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Fail missing keys, duplicates, or numeric/Boolean discrepancies."""
    if reproduced.duplicated(KEYS).any() or reference.duplicated(KEYS).any():
        raise ValueError("duplicate legacy event identities")
    merged = reproduced.merge(reference, on=KEYS, suffixes=("_new", "_old"), how="left", validate="one_to_one", indicator=True)
    rows = [{"gate": "reference_keys_present", "passed": bool(len(merged) and merged["_merge"].eq("both").all()), "max_abs_difference": np.nan}]
    metrics = ["event_spikes", "decoded_large_jump_fraction", "median_map_error_cm", "foster_continuous_trajectory"]
    for metric in metrics:
        left = pd.to_numeric(merged[f"{metric}_new"], errors="coerce").to_numpy(float)
        right = pd.to_numeric(merged[f"{metric}_old"], errors="coerce").to_numpy(float)
        finite = np.isfinite(left) & np.isfinite(right)
        error = np.abs(left - right)
        rows.append({"gate": f"legacy_{metric}_reproduced", "passed": bool(len(left) and finite.all() and np.allclose(left, right, atol=1e-9, rtol=1e-9)), "max_abs_difference": float(error[finite].max()) if finite.any() else np.nan})
    return pd.DataFrame(rows)


def population_tables(events: pd.DataFrame, bootstrap: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = STRATA + ["condition", "population_replicate"]
    populations = events.groupby(keys, as_index=False).agg(
        events=("event_index", "size"),
        pass_fraction=("continuity_pass", "mean"),
        large_jump_fraction=("large_jump_fraction", "mean"),
        median_event_speed_cm_s=("median_speed_cm_s", "median"),
        median_event_position_error_cm=("median_position_error_cm", "median"),
        median_selected_event_speed_cm_s=("selected_median_speed_cm_s", "median"),
        mean_valid_adjacent_steps=("valid_adjacent_steps", "mean"),
    )
    definitions = [
        ("large_minus_small_arena", "pf_area_same_cells", "tanni_area_same_cells"),
        ("density_rescue", "tanni_area_same_cells", "tanni_area_density_matched"),
    ]
    rows = []
    rng = np.random.default_rng(seed)
    for key, group in populations.groupby(STRATA, sort=True):
        for name, left, right in definitions:
            ref = group[group.condition.eq(left)].set_index("population_replicate")
            comp = group[group.condition.eq(right)].set_index("population_replicate")
            if not ref.index.equals(comp.index):
                raise ValueError("paired population coverage mismatch")
            for metric in ["pass_fraction", "large_jump_fraction", "median_event_speed_cm_s", "median_event_position_error_cm"]:
                differences = comp[metric].to_numpy(float) - ref[metric].to_numpy(float)
                finite = differences[np.isfinite(differences)]
                if len(finite) >= 2:
                    draws = rng.choice(finite, (bootstrap, len(finite)), replace=True).mean(axis=1)
                    lo, hi = np.quantile(draws, [0.025, 0.975])
                else:
                    lo, hi = np.nan, np.nan
                rows.append({**dict(zip(STRATA, key, strict=True)), "contrast": name, "metric": metric, "paired_populations": len(differences), "finite_populations": len(finite), "comparison_minus_reference": float(np.mean(finite)) if finite.size else np.nan, "ci95_low": lo, "ci95_high": hi})
    return populations, pd.DataFrame(rows)


def likelihood_interactions(populations: pd.DataFrame, bootstrap: int, seed: int) -> pd.DataFrame:
    """Paired difference-in-differences; do not subtract unpaired CI endpoints."""
    strata = ["variant", "field_sigma_cm", "estimator", "bin_filter"]
    metrics = ["pass_fraction", "large_jump_fraction", "median_event_speed_cm_s", "median_event_position_error_cm"]
    rows = []
    rng = np.random.default_rng(seed)
    for key, local in populations.groupby(strata, sort=True):
        for metric in metrics:
            pivot = local.pivot(index="population_replicate", columns=["likelihood", "condition"], values=metric)
            difference = (pivot["conditional_multinomial", "tanni_area_same_cells"] - pivot["conditional_multinomial", "pf_area_same_cells"]) - (pivot["poisson", "tanni_area_same_cells"] - pivot["poisson", "pf_area_same_cells"])
            finite = difference[np.isfinite(difference)].to_numpy(float)
            lo, hi = np.nan, np.nan
            if len(finite) >= 2:
                lo, hi = np.quantile(rng.choice(finite, (bootstrap, len(finite)), replace=True).mean(axis=1), [0.025, 0.975])
            rows.append({**dict(zip(strata, key, strict=True)), "metric": metric, "interaction_definition": "(large-small)_conditional-(large-small)_poisson", "paired_populations": len(difference), "finite_populations": len(finite), "mean_interaction": float(finite.mean()) if finite.size else np.nan, "ci95_low": lo, "ci95_high": hi})
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source_manifest = json.loads(args.reference_manifest.read_text())
    settings = source_manifest["settings"]
    population_count = args.population_replicates or int(settings["population_replicates"])
    events_per_population = args.events_per_population or int(settings["events_per_population"])
    if not 1 <= population_count <= int(settings["population_replicates"]) or not 1 <= events_per_population <= int(settings["events_per_population"]):
        raise ValueError("audit subset must fit inside the reference experiment")
    input_paths = {
        "reference_manifest": args.reference_manifest,
        "reference_events": args.reference_events,
        "legacy_custom_source": args.legacy_scripts / "simulate_2d_place_field_coverage_continuity.py",
        "legacy_ratinabox_source": args.legacy_scripts / "simulate_ratinabox_place_field_coverage_continuity.py",
        "audit_source": Path(__file__),
        "decoder_source": ROOT / "src/hipporeplayimm/replay_coverage.py",
    }
    start_hashes = {key: file_sha256(path) for key, path in input_paths.items()}
    rows, originals = [], []
    with load_legacy(args.legacy_scripts) as legacy:
        framework_version = legacy.require_ratinabox()
        expected_version = source_manifest["ratinabox_version"]
        if framework_version != expected_version:
            raise ValueError(f"RatInABox version mismatch: {framework_version} != {expected_version}")
        config = legacy.SimulationSettings(
            spatial_bin_size_cm=settings["spatial_bin_size_cm"],
            true_speed_cm_s=settings["true_speed_cm_s"],
            event_duration_s=settings["event_duration_ms"] / 1000.0,
            mean_spikes_per_base_bin=settings["mean_spikes_per_5ms"],
            peak_rate_hz=settings["peak_rate_hz"],
            baseline_rate_hz=settings["baseline_rate_hz"],
        )
        conditions = legacy.primary_conditions(settings["common_cell_count"])
        maximum_cells = max(condition.n_cells for condition in conditions)
        variants = [item[0] for description in sorted({item[1] for item in legacy.VARIANTS}) for item in legacy.VARIANTS if item[1] == description]
        for sigma in legacy.comma_floats(settings["field_sigmas_cm"]):
            for rep in range(population_count):
                for condition in conditions:
                    batches = []

                    def capture(counts, rates, grid, duration, _batches=batches):
                        decoded = {name: decode_independent(counts, rates, grid, duration, likelihood=name) for name in ["poisson", "conditional_multinomial"]}
                        _batches.append((counts.copy(), decoded))
                        return decoded["poisson"]["map"]

                    old_decoder = legacy.decode_map_positions
                    legacy.decode_map_positions = capture
                    try:
                        original = legacy.simulate_population(condition, config, field_sigma_cm=sigma, population_replicate=rep, events_per_population=events_per_population, maximum_cells=maximum_cells, seed=settings["seed"])
                    finally:
                        legacy.decode_map_positions = old_decoder
                    originals.append(original)
                    if len(batches) != len(variants):
                        raise ValueError("legacy decoder call contract changed")
                    n_base = round(config.event_duration_s / config.base_step_s)
                    window = round(config.decode_window_s / config.base_step_s)
                    n_frames = n_base - window + 1
                    for variant, (counts, decoded) in zip(variants, batches, strict=True):
                        old_rows = original[original.variant.eq(variant)].set_index("event_index")
                        for event in range(events_per_population):
                            angle, midpoint, _ = legacy._event_random_values(settings["seed"], rep, event, n_base, config.mean_spikes_per_base_bin)
                            path = legacy.straight_constant_speed_path(condition.width_cm, condition.height_cm, n_base, config.true_speed_cm_s * config.base_step_s, angle, midpoint)
                            truth = legacy.window_mean_positions(path, window)
                            sl = slice(event * n_frames, (event + 1) * n_frames)
                            local_counts = counts[sl]
                            masks = {"unfiltered": np.ones(n_frames, dtype=bool), "two_cells_three_spikes": (local_counts.sum(axis=1) >= 3) & ((local_counts > 0).sum(axis=1) >= 2)}
                            for likelihood, decoded_batch in decoded.items():
                                for estimator in ["map", "posterior_mean"]:
                                    for bin_filter, mask in masks.items():
                                        metrics = path_metrics(decoded_batch[estimator][sl], truth, step_s=config.base_step_s, valid_bins=mask)
                                        rows.append({"variant": variant, "condition": condition.name, "field_sigma_cm": sigma, "population_replicate": rep, "event_index": event, "likelihood": likelihood, "estimator": estimator, "bin_filter": bin_filter, "event_spikes": int(old_rows.loc[event, "event_spikes"]), "median_entropy_nats": float(np.median(decoded_batch["posterior_entropy_nats"][sl])), "median_rms_cm": float(np.median(decoded_batch["posterior_rms_cm"][sl])), **metrics})
                    print(f"sigma={sigma:g} population={rep + 1}/{population_count} condition={condition.name}", flush=True)
    events = pd.DataFrame(rows)
    reproduction = compare_reproduction(pd.concat(originals, ignore_index=True), pd.read_csv(args.reference_events))
    populations, contrasts = population_tables(events, args.bootstrap_replicates, settings["seed"])
    interactions = likelihood_interactions(populations, args.bootstrap_replicates, settings["seed"])
    inputs_unchanged = start_hashes == {key: file_sha256(path) for key, path in input_paths.items()}
    gates = pd.concat([reproduction, pd.DataFrame([
        {"gate": "inputs_unchanged", "passed": inputs_unchanged},
        {"gate": "all_metrics_rows_present", "passed": len(events) == population_count * events_per_population * len(conditions) * len(legacy.comma_floats(settings["field_sigmas_cm"])) * len(variants) * 8},
        {"gate": "finite_unfiltered_metrics", "passed": np.isfinite(events.loc[events.bin_filter.eq("unfiltered"), ["median_speed_cm_s", "median_position_error_cm", "large_jump_fraction"]].to_numpy(float)).all()},
    ])], ignore_index=True)
    gates = pd.concat([gates, pd.DataFrame([{"gate": "overall_technical", "passed": bool(gates.passed.all())}])], ignore_index=True)
    for name, frame in [("event_metrics", events), ("population_summary", populations), ("paired_contrasts", contrasts), ("interaction_summary", interactions), ("gate_summary", gates)]:
        frame.to_csv(out / f"coverage_likelihood_{name}.csv", index=False)
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        **build_script_provenance(input_paths=input_paths, cwd=ROOT),
        "initial_input_sha256": start_hashes,
        "ratinabox_version": framework_version,
        "package_versions": {name: version(name) for name in ["numpy", "scipy", "pandas", "matplotlib", "ratinabox"]},
        "python_version": sys.version,
        "reference_settings": settings,
        "population_replicates": population_count,
        "events_per_population": events_per_population,
        "bootstrap_replicates": args.bootstrap_replicates,
        "uncertainty_unit": "paired synthetic population, not animal",
        "scope": "likelihood diagnosis of legacy simulation; not real replay prevalence or biological speed uniformity",
        "technical_pass": bool(gates.passed.all()),
    }
    (out / "coverage_likelihood_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    figure(contrasts, out / "coverage_likelihood_audit.png")
    table = contrasts[(contrasts.estimator == "map") & (contrasts.bin_filter == "unfiltered") & (contrasts.contrast == "large_minus_small_arena") & (contrasts.metric == "pass_fraction")]
    lines = ["# Coverage likelihood audit", "", f"Technical reproduction pass: {manifest['technical_pass']}", "", "Identical simulated spikes; independent flat-prior decoding. The fixed-total generator is matched by the conditional multinomial likelihood. The Poisson generator uses an unconditional likelihood; its conditional decoder is a count-information sensitivity.", "", "## Large-minus-small arena continuity pass fraction", "", "| Generator | Field SD (cm) | Likelihood | Difference | 95% population CI |", "|---|---:|---|---:|---|"]
    for row in table.itertuples():
        lines.append(f"| {row.variant} | {row.field_sigma_cm:g} | {row.likelihood} | {row.comparison_minus_reference:+.4f} | [{row.ci95_low:+.4f}, {row.ci95_high:+.4f}] |")
    lines += ["", "## Limits", "", "These are geometric continuity passes, not shuffle-significant replay. Finite 20 ms integration remains a shared approximation. The legacy arena contrast changes aspect ratio as well as area. Events are synthetic 80 ms straight paths at 1000 cm/s, not empirically matched candidates. The conditional likelihood is a diagnosis, not a recommendation to replace the real-data observation model without testing it. No fraction of the observed dataset gap is attributed to coverage here.", "", "The broader study remains open: actual population calibration, native event support, cell subsampling, known speed gradients, negative controls, and held-out recovery are required."]
    (out / "coverage_likelihood_report.md").write_text("\n".join(lines) + "\n")
    if not manifest["technical_pass"]:
        raise RuntimeError("technical audit failed; do not interpret contrasts")


def figure(contrasts: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for column, variant in enumerate(["gaussian_exact_count", "gaussian_mean_rate_poisson"]):
        for row, metric in enumerate(["pass_fraction", "median_event_position_error_cm"]):
            ax = axes[row, column]
            local = contrasts[(contrasts.variant == variant) & (contrasts.estimator == "map") & (contrasts.bin_filter == "unfiltered") & (contrasts.contrast == "large_minus_small_arena") & (contrasts.metric == metric)]
            for likelihood, color in [("poisson", "#3465a4"), ("conditional_multinomial", "#c45528")]:
                line = local[local.likelihood.eq(likelihood)].sort_values("field_sigma_cm")
                ax.plot(line.field_sigma_cm, line.comparison_minus_reference, "o-", color=color, label=likelihood)
                ax.fill_between(line.field_sigma_cm, line.ci95_low, line.ci95_high, alpha=0.18, color=color)
            ax.axhline(0, color="black", linewidth=0.7)
            ax.set(xlabel="Field SD (cm)", ylabel="Large minus small arena", title=f"{variant}\n{metric}")
            ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-scripts", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--reference-events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--population-replicates", type=int)
    parser.add_argument("--events-per-population", type=int)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        parser.error("at least 100 bootstrap replicates required")
    run(args)


if __name__ == "__main__":
    main()
