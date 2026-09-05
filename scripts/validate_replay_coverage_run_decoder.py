#!/usr/bin/env python3
"""Blocked, training-only RUN validation of recording-coverage encoding maps."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.replay_coverage import decode_independent
from hipporeplayimm.replay_coverage_data import CoverageInputConfig, index_spike_times
from hipporeplayimm.replay_coverage_validation import (
    count_windows,
    fit_training_fold,
    heldout_centers,
    posterior_coverage,
    truth_state_indices,
    window_truth,
)
from scripts._provenance import build_script_provenance, file_sha256

KEYS = ["dataset", "animal", "session", "likelihood", "window_s"]


def session_from_cache(record, source_units):
    """Do not load cached rate maps, support masks, or all-RUN unit-QC masks."""
    with np.load(record.artifact_path, allow_pickle=False) as data:
        position = data["position"]
        spikes = data["spikes"]
        intervals = data["supported_run_intervals"]
        bounds = data["arena_bounds_cm"]
    allowed = source_units.loc[source_units.source_cell_type_allowed, "cell_id"].to_numpy(int)
    empty = np.empty((0, 2))
    return ReplaySession(
        rat=record.animal, name=record.session.split("/")[-1], path=Path(record.artifact_path).parent,
        position=position, spikes=spikes, tetrode_cell_ids=empty,
        excitatory_neurons=allowed, inhibitory_neurons=np.empty(0, dtype=int),
        ripple_events=np.empty((0, 6)), run_times=intervals,
        sleep_box_immobile_times=empty, sleep_times=empty, rem_times=empty,
        well_sequence=None,
        metadata={"source_dataset": record.dataset, "arena_bounds_cm": bounds.tolist() if np.isfinite(bounds).all() else None},
    )


def score_fold(session, model, centers, widths, null_permutations, seed):
    """Held-out neural data are read only after the fold model has been frozen."""
    indexed = index_spike_times(session.spikes, model["cell_ids"])
    rng = np.random.default_rng(seed)
    permutations = [rng.permutation(len(model["cell_ids"])) for _ in range(null_permutations)]
    frames = []
    for width in widths:
        truth = window_truth(session.position, centers, width)
        true_index = truth_state_indices(model, truth["center_xy"])
        mean_index = truth_state_indices(model, truth["window_mean_xy"])
        counts = count_windows(indexed, model["cell_ids"], centers, width)
        for likelihood in ["poisson", "conditional_multinomial"]:
            decoded = decode_independent(counts, model["rates_hz"], model["grid_cm"], width, likelihood=likelihood)
            null_mean, null_map = [], []
            for permutation in permutations:
                null = decode_independent(counts, model["rates_hz"][permutation], model["grid_cm"], width, likelihood=likelihood)
                null_mean.append(np.linalg.norm(null["posterior_mean"] - truth["center_xy"], axis=1))
                null_map.append(np.linalg.norm(null["map"] - truth["center_xy"], axis=1))
            errors = np.linalg.norm(decoded["posterior_mean"] - truth["center_xy"], axis=1)
            null_errors = np.median(null_mean, axis=0)
            uniform_error = np.linalg.norm(model["grid_cm"].mean(axis=0) - truth["center_xy"], axis=1)
            frame = pd.DataFrame({
                "center_s": centers, "window_s": width, "likelihood": likelihood,
                "truth_x_cm": truth["center_xy"][:, 0], "truth_y_cm": truth["center_xy"][:, 1],
                "truth_mean_x_cm": truth["window_mean_xy"][:, 0], "truth_mean_y_cm": truth["window_mean_xy"][:, 1],
                "mean_behavior_speed_cm_s": truth["mean_behavior_speed_cm_s"],
                "tracking_speed_over_200_cm_s": truth["mean_behavior_speed_cm_s"] > 200,
                "n_spikes": counts.sum(axis=1), "n_active_units": np.count_nonzero(counts, axis=1),
                "spike_support_pass": (counts.sum(axis=1) >= 3) & (np.count_nonzero(counts, axis=1) >= 2),
                "posterior_mean_error_cm": errors,
                "map_error_cm": np.linalg.norm(decoded["map"] - truth["center_xy"], axis=1),
                "posterior_mean_error_to_window_mean_cm": np.linalg.norm(decoded["posterior_mean"] - truth["window_mean_xy"], axis=1),
                "posterior_rms_cm": decoded["posterior_rms_cm"],
                "posterior_entropy_nats": decoded["posterior_entropy_nats"],
                "true_bin_in_training_support": true_index >= 0,
                "window_mean_bin_in_training_support": mean_index >= 0,
                **posterior_coverage(decoded["posterior"], true_index),
                "hpd95_window_mean": posterior_coverage(decoded["posterior"], mean_index)["hpd95"],
                "null_posterior_mean_error_cm": null_errors,
                "null_map_error_cm": np.median(null_map, axis=0),
                "uniform_mean_error_cm": uniform_error,
                "paired_improvement_over_cell_ID_null_cm": null_errors - errors,
                "paired_improvement_over_uniform_mean_cm": uniform_error - errors,
                "null_permutations": null_permutations,
            })
            frames.append(frame)
    return pd.concat(frames, ignore_index=True), [order.tolist() for order in permutations]


def summarize_predictions(predictions):
    rows = []
    for key, group in predictions.groupby(KEYS, sort=True):
        for label, keep in [
            ("all_behavior_selected_windows", np.ones(len(group), dtype=bool)),
            ("at_least_2cells_3spikes", group.spike_support_pass.to_numpy(bool)),
        ]:
            local = group.loc[keep]
            supported = local[local.true_bin_in_training_support]
            row = {
                **dict(zip(KEYS, key, strict=True)), "test_bin_filter": label,
                "planned_selected_windows": len(group), "measured_windows": len(local),
                "folds_represented": int(local.fold.nunique()),
                "minimum_training_units": int(group.n_training_units.min()),
                "true_bin_support_fraction": float(local.true_bin_in_training_support.mean()) if len(local) else np.nan,
                "median_test_spikes": float(local.n_spikes.median()) if len(local) else np.nan,
                "median_active_units": float(local.n_active_units.median()) if len(local) else np.nan,
                "spike_support_fraction": float(group.spike_support_pass.mean()),
            }
            for source, target in [
                ("posterior_mean_error_cm", "posterior_mean_error_cm"), ("map_error_cm", "map_error_cm"),
                ("posterior_mean_error_to_window_mean_cm", "posterior_mean_error_to_window_mean_cm"),
                ("null_posterior_mean_error_cm", "null_posterior_mean_error_cm"),
                ("uniform_mean_error_cm", "uniform_mean_error_cm"),
                ("posterior_rms_cm", "posterior_rms_cm"),
                ("paired_improvement_over_cell_ID_null_cm", "paired_improvement_over_cell_ID_null_cm"),
                ("paired_improvement_over_uniform_mean_cm", "paired_improvement_over_uniform_mean_cm"),
            ]:
                row[f"median_{target}"] = float(local[source].median()) if len(local) else np.nan
            for quantile in [.75, .90]:
                row[f"posterior_mean_error_cm_p{int(quantile * 100)}"] = float(local.posterior_mean_error_cm.quantile(quantile)) if len(local) else np.nan
            for mass in [50, 80, 95]:
                row[f"hpd{mass}_coverage"] = float(local[f"hpd{mass}"].mean()) if len(local) else np.nan
                row[f"hpd{mass}_coverage_given_training_support"] = float(supported[f"hpd{mass}"].mean()) if len(supported) else np.nan
            row["hpd95_window_mean_coverage"] = float(local.hpd95_window_mean.mean()) if len(local) else np.nan
            row["beats_cell_ID_null_descriptively"] = row["median_paired_improvement_over_cell_ID_null_cm"] > 0
            row["beats_uniform_mean_descriptively"] = row["median_paired_improvement_over_uniform_mean_cm"] > 0
            for threshold in [8, 16, 20, 40]:
                row[f"median_error_at_most_{threshold}_cm"] = row["median_posterior_mean_error_cm"] <= threshold
            rows.append(row)
    return pd.DataFrame(rows)


def plot_validation(summary, output):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    base = summary[summary.likelihood.eq("poisson") & summary.test_bin_filter.eq("all_behavior_selected_windows")]
    for column, (dataset, title) in enumerate([("pfeiffer_foster", "Pfeiffer/Foster"), ("tanni2022", "Tanni: all arena sizes")]):
        for width, color, label in [(.25, "#087e8b", "250 ms"), (.02, "#b04b6d", "20 ms")]:
            group = base[base.dataset.eq(dataset) & base.window_s.eq(width)]
            axes[0, column].scatter(group.median_null_posterior_mean_error_cm, group.median_posterior_mean_error_cm, color=color, s=25, alpha=.8, label=label)
            axes[1, column].plot([50, 80, 95], [group[f"hpd{m}_coverage"].mean() * 100 for m in [50, 80, 95]], "o-", color=color, label=label)
        ax = axes[0, column]
        limit = max(ax.get_xlim()[1], ax.get_ylim()[1], 1)
        ax.plot([0, limit], [0, limit], color="0.6", linestyle="--", linewidth=1)
        ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="Cell-identity null error (cm)", ylabel="Held-out posterior-mean error (cm)", title=title)
        ax.legend(frameon=False)
        axes[1, column].plot([50, 95], [50, 95], color="0.6", linestyle="--", linewidth=1)
        axes[1, column].set(xlabel="Nominal posterior mass (%)", ylabel="Empirical coverage (%)", ylim=(0, 100), title="Session-mean coverage, including unsupported truth")
        for row in range(2):
            axes[row, column].spines[["top", "right"]].set_visible(False)
    fig.suptitle("Training-only RUN decoder validation\nNo test spikes or positions used to select units or fit maps", fontsize=12)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def validation_gates(folds, predictions, expected_folds, unchanged):
    complete = bool(len(folds) == expected_folds and expected_folds > 0 and folds.status.eq("scored").all())
    required = {("poisson", .25), ("poisson", .02), ("conditional_multinomial", .25), ("conditional_multinomial", .02)}
    rows_complete = False
    if complete and len(predictions):
        keys = ["dataset", "animal", "session", "fold", "center_s"]
        groups = predictions.groupby(keys, sort=False)
        combinations = set(zip(predictions.likelihood, predictions.window_s, strict=True))
        rows_complete = bool(
            combinations == required and not predictions.duplicated([*keys, "likelihood", "window_s"]).any()
            and groups.size().eq(4).all() and groups.ngroups == int(folds.n_test_windows.sum())
        )
    gates = [
        ("planned_fold_rows_present", len(folds) == expected_folds and expected_folds > 0),
        ("all_folds_fitted_and_scored", complete),
        ("all_window_likelihood_rows_present", rows_complete),
        ("all_prediction_errors_finite", len(predictions) > 0 and np.isfinite(predictions[["posterior_mean_error_cm", "map_error_cm", "null_posterior_mean_error_cm"]]).all().all()),
        ("inputs_and_code_unchanged", unchanged),
    ]
    return [*gates, ("overall_technical", all(bool(value) for _, value in gates))]


def run(args):
    if args.folds < 2 or args.guard_s < 0 or args.null_permutations < 1 or args.max_samples_per_fold < 20:
        raise ValueError("invalid fold, guard, null, or sample settings")
    root = args.input_dir.resolve()
    out = args.output_dir.resolve()
    (out / "fold_models").mkdir(parents=True, exist_ok=False)
    inputs = {
        "cache_manifest": root / "coverage_input_manifest.json", "cache_sessions": root / "coverage_input_sessions.csv",
        "source_unit_types": root / "coverage_input_units.csv", "script": Path(__file__),
        "validation_library": ROOT / "src/hipporeplayimm/replay_coverage_validation.py",
        "input_library": ROOT / "src/hipporeplayimm/replay_coverage_data.py",
        "encoding_library": ROOT / "src/hipporeplayimm/encoding.py",
        "decoder_library": ROOT / "src/hipporeplayimm/replay_coverage.py",
    }
    hashes = {key: file_sha256(value) for key, value in inputs.items()}
    provenance = build_script_provenance(input_paths=inputs, cwd=ROOT)
    provenance["created_at_utc"] = datetime.now(UTC).isoformat()
    settings = CoverageInputConfig(**json.loads(inputs["cache_manifest"].read_text())["settings"])
    sessions = pd.read_csv(inputs["cache_sessions"])
    if args.sessions:
        sessions = sessions[sessions.apply(lambda row: f"{row.dataset}:{row.animal}:{row.session}" in args.sessions, axis=1)]
        if len(sessions) != len(set(args.sessions)):
            raise ValueError("requested session identity mismatch")
    if sessions.empty or not sessions.status.eq("cached").all():
        raise ValueError("no complete cache sessions")
    units = pd.read_csv(inputs["source_unit_types"], usecols=["dataset", "animal", "session", "cell_id", "source_cell_type_allowed"], dtype={"source_cell_type_allowed": "boolean"})
    if units.source_cell_type_allowed.isna().any():
        raise ValueError("missing source unit classification")
    all_predictions, all_units, folds = [], [], []
    cache_hashes = {}
    for record in sessions.itertuples(index=False):
        if file_sha256(record.artifact_path) != record.artifact_sha256:
            raise ValueError("cache digest mismatch")
        cache_hashes[record.artifact_path] = record.artifact_sha256
        key = f"{record.dataset}:{record.animal}:{record.session}"
        source_units = units[units.dataset.eq(record.dataset) & units.animal.eq(record.animal) & units.session.eq(record.session)]
        session = session_from_cache(record, source_units)
        limits = np.linspace(session.run_times[:, 0].min(), session.run_times[:, 1].max(), args.folds + 1)
        token = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
        for fold in range(args.folds):
            start = time.monotonic()
            row = {"dataset": record.dataset, "animal": record.animal, "session": record.session, "fold": fold}
            print(f"validate {key} fold {fold + 1}/{args.folds}", flush=True)
            try:
                model, unit_qc, meta = fit_training_fold(session, limits[fold], limits[fold + 1], args.guard_s, settings)
                centers, eligible = heldout_centers(session, limits[fold], limits[fold + 1], .25, .25, args.max_samples_per_fold, [args.seed, token, fold], min_speed_cm_s=settings.min_speed_cm_s)
                if len(centers) < 20:
                    raise ValueError("fewer than 20 behavior-selected test windows")
                prediction, permutations = score_fold(session, model, centers, [.25, .02], args.null_permutations, [args.seed + 1, token, fold])
                for name, value in row.items():
                    prediction[name] = value
                    unit_qc[name] = value
                prediction["n_training_units"] = len(model["cell_ids"])
                prediction["training_states"] = len(model["grid_cm"])
                all_predictions.append(prediction)
                all_units.append(unit_qc)
                model_path = out / "fold_models" / f"{record.dataset}__{record.animal}__{session.name}__fold{fold}.npz"
                np.savez_compressed(model_path, **model, test_centers_s=centers)
                (model_path.with_suffix(".json")).write_text(json.dumps({**row, **meta, "eligible_test_windows": eligible, "selected_test_windows": len(centers), "cell_ID_null_permutations": permutations, "model_sha256": file_sha256(model_path)}, indent=2, allow_nan=False) + "\n")
                row.update(status="scored", failure_reason="", n_test_windows=len(centers), n_eligible_test_windows=eligible, n_training_units=len(model["cell_ids"]), training_states=len(model["grid_cm"]), training_running_duration_s=meta["running_duration_s"], model_path=str(model_path))
            except (ValueError, OSError, RuntimeError) as exc:
                row.update(status="failed", failure_reason=f"{type(exc).__name__}: {exc}")
                print(row["failure_reason"], flush=True)
            row["runtime_s"] = time.monotonic() - start
            folds.append(row)
    folds = pd.DataFrame(folds)
    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    unit_qc = pd.concat(all_units, ignore_index=True) if all_units else pd.DataFrame()
    summary = summarize_predictions(predictions) if len(predictions) else pd.DataFrame()
    for name, frame in [("predictions", predictions), ("folds", folds), ("training_units", unit_qc), ("session_summary", summary)]:
        frame.to_csv(out / f"coverage_RUN_validation_{name}.csv", index=False)
    if len(summary):
        numeric = summary.select_dtypes(include="number").columns.difference(["window_s"])
        summary.groupby(["dataset", "animal", "likelihood", "window_s", "test_bin_filter"], as_index=False)[numeric].mean().to_csv(out / "coverage_RUN_validation_animal_summary.csv", index=False)
        plot_validation(summary, out / "coverage_RUN_validation.png")
    unchanged = all(file_sha256(value) == hashes[key] for key, value in inputs.items()) and all(file_sha256(path) == value for path, value in cache_hashes.items())
    gates = validation_gates(folds, predictions, len(sessions) * args.folds, unchanged)
    pd.DataFrame([{"gate": name, "passed": bool(value)} for name, value in gates]).to_csv(out / "coverage_RUN_validation_gate_summary.csv", index=False)
    manifest = {
        **provenance, "input_sha256_at_start": hashes, "cache_sha256": cache_hashes,
        "encoding_settings": asdict(settings), "folds": args.folds, "guard_s": args.guard_s,
        "maximum_test_windows_per_fold": args.max_samples_per_fold, "seed": args.seed,
        "null_permutations": args.null_permutations, "window_widths_s": [.25, .02],
        "test_window_stride_s": .25, "sessions": len(sessions), "prediction_rows": len(predictions),
        "uses_cached_full_RUN_QC_mask_or_maps": False,
        "claim_boundary": "behavioral encoder validation, not replay ground truth or uniformity",
    }
    (out / "coverage_RUN_validation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "coverage_RUN_validation_report.md").write_text(
        "# Training-Only RUN Decoder Validation\n\n"
        f"{len(sessions)} sessions; {args.folds} chronological folds each. {int(folds.status.eq('scored').sum())} folds scored.\n\n"
        "Grid, unit inclusion, rate maps, and spatial occupancy support are estimated after removing held-out/guard positions and spikes. Test windows are selected from behavior only. No all-RUN map or stability mask is reused.\n\n"
        "250 ms and 20 ms windows use the same behavior-selected centers. The primary table includes all windows, including zero-spike windows and positions outside training support; the spike-support-filtered table is a conditional sensitivity.\n\n"
        "Cell-identity permutations are descriptive controls, not formal per-session significance tests. Error thresholds of 8/16/20/40 cm are reported without choosing a favorable one. Technical gates do not assert calibrated uncertainty.\n\n"
        "HPD coverage includes unsupported true locations as misses and reports a support-conditional sensitivity. Discrete ties are included. Center-position and window-mean errors are both retained to expose temporal integration.\n\n"
        "RUN decoding does not establish replay speed. Very short RUN bins have different spike support and potentially theta-related represented positions; undercoverage cannot automatically be attributed to a software error or carried quantitatively into replay.\n"
    )
    if not all(bool(value) for _, value in gates):
        raise RuntimeError("RUN validation technical gates failed; written failures remain in the denominator")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sessions", nargs="*")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--guard-s", type=float, default=1.0)
    parser.add_argument("--max-samples-per-fold", type=int, default=250)
    parser.add_argument("--null-permutations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260906)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
