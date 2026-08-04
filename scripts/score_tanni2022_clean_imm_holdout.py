#!/usr/bin/env python3
"""Score a frozen Tanni 2022 large-2D holdout with the exact model core."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
SRC_DIR = ROOT / "src"
for path in (SCRIPT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from _provenance import build_script_provenance  # noqa: E402
from analyze_tanni2022_wall_distance_replay import (  # noqa: E402
    fit_decoder_encoding,
    make_replay_session,
)
from hipporeplayimm.data import RippleEvent  # noqa: E402
from hipporeplayimm.encoding import EmissionConfig, build_emissions  # noqa: E402
from hipporeplayimm.evidence_reporting import ensure_evidence_support_columns  # noqa: E402
from hipporeplayimm.sorted_spike_state_space import SortedSpikeStateSpaceReplayModel  # noqa: E402
from hipporeplayimm.state_space import StateSpaceDecoderConfig  # noqa: E402
from hipporeplayimm.tanni2022 import read_tanni_position  # noqa: E402


EVIDENCE_OUTPUT = "tanni2022_clean_imm_holdout_event_model_evidence.csv"
SELECTION_OUTPUT = "tanni2022_clean_imm_holdout_scored_selection.csv"
UNIT_OUTPUT = "tanni2022_clean_imm_holdout_encoding_unit_qc.csv"
GATE_OUTPUT = "tanni2022_clean_imm_holdout_scoring_gate_summary.csv"
MANIFEST_OUTPUT = "tanni2022_clean_imm_holdout_scoring_manifest.json"

MODEL_MODES = {
    "stationary": "stationary",
    "diffusion": "diffusion",
    "fragmented": "fragmented",
    "first_order_imm": "first-order-imm",
    "exact_sparse_momentum": "momentum-exact-sparse",
}
MODELS = tuple(MODEL_MODES)
KEYS = ["animal", "session", "event_index"]


def holdout_decisions(evidence: pd.DataFrame, *, margin_threshold: float) -> pd.DataFrame:
    successful = evidence[evidence["status"].eq("success") & evidence["evidence_comparable"].astype(bool)]
    pivot = successful.pivot_table(index=KEYS, columns="model", values="log_evidence", aggfunc="first").reset_index()
    missing = [model for model in MODELS if model not in pivot.columns]
    if missing:
        raise ValueError(f"missing exact-core model columns: {missing}")
    values = pivot[list(MODELS)].to_numpy(dtype=float)
    order = np.argsort(values, axis=1)
    pivot["best_model"] = np.asarray(MODELS, dtype=object)[order[:, -1]]
    pivot["runner_up_model"] = np.asarray(MODELS, dtype=object)[order[:, -2]]
    pivot["best_minus_runner_up_log_evidence"] = (
        values[np.arange(len(values)), order[:, -1]]
        - values[np.arange(len(values)), order[:, -2]]
    )
    trajectory = pivot[["diffusion", "fragmented", "first_order_imm", "exact_sparse_momentum"]].max(axis=1)
    ordered = pivot[["diffusion", "first_order_imm", "exact_sparse_momentum"]].max(axis=1)
    pivot["delta_trajectory_minus_stationary"] = trajectory - pivot["stationary"]
    pivot["delta_ordered_minus_static_or_fragmented"] = ordered - pivot[["stationary", "fragmented"]].max(axis=1)
    pivot["delta_imm_minus_fragmented"] = pivot["first_order_imm"] - pivot["fragmented"]
    pivot["trajectory_confident"] = pivot["delta_trajectory_minus_stationary"] >= margin_threshold
    pivot["ordered_trajectory_confident"] = pivot["delta_ordered_minus_static_or_fragmented"] >= margin_threshold
    pivot["imm_confident_over_fragmented"] = pivot["delta_imm_minus_fragmented"] >= margin_threshold
    pivot["joint_family_and_imm_margin_positive"] = (
        pivot["trajectory_confident"]
        & pivot["imm_confident_over_fragmented"]
    )
    pivot["strict_clean_imm"] = (
        pivot["joint_family_and_imm_margin_positive"]
        & pivot["best_model"].eq("first_order_imm")
    )
    return pivot


def scoring_gates(
    selection: pd.DataFrame,
    evidence: pd.DataFrame,
    *,
    expected_events: int,
) -> pd.DataFrame:
    successful = evidence[evidence["status"].eq("success")] if not evidence.empty else pd.DataFrame()
    coverage = (
        successful.groupby(KEYS)["model"].agg(lambda values: set(values))
        if not successful.empty
        else pd.Series(dtype=object)
    )
    checks = [
        ("selected_events_present", len(selection) == expected_events and expected_events > 0, f"{len(selection)}/{expected_events}"),
        ("selection_keys_unique", bool(not selection.empty and ~selection.duplicated(KEYS).any()), f"duplicates={int(selection.duplicated(KEYS).sum()) if not selection.empty else 0}"),
        ("prior_model_overlap_zero", bool(not selection.empty and ~selection["excluded_prior_model_event"].astype(bool).any()), f"overlap={int(selection['excluded_prior_model_event'].astype(bool).sum()) if not selection.empty else 0}"),
        ("required_models_complete", bool(len(coverage) == expected_events and coverage.map(lambda value: value == set(MODELS)).all()), f"complete={int(coverage.map(lambda value: value == set(MODELS)).sum()) if len(coverage) else 0}/{expected_events}"),
        ("no_model_scoring_failures", bool(not evidence.empty and evidence["status"].eq("success").all()), f"failures={int((~evidence['status'].eq('success')).sum()) if not evidence.empty else 0}"),
        ("all_rows_evidence_comparable", bool(not evidence.empty and evidence["evidence_comparable"].astype(bool).all()), f"comparable={int(evidence['evidence_comparable'].astype(bool).sum()) if not evidence.empty else 0}/{len(evidence)}"),
    ]
    overall = all(passed for _, passed, _ in checks)
    checks.append(("overall_technical", overall, "biological outcomes are not technical gates"))
    return pd.DataFrame(
        {"gate": gate, "passed": bool(passed), "detail": detail}
        for gate, passed, detail in checks
    )


def run(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    evidence_dir = Path(args.evidence_dir).resolve()
    selection_path = Path(args.selection_csv).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(evidence_dir / "tanni2022_session_manifest.csv")
    selection = pd.read_csv(selection_path)
    requested_animals = set(args.animal)
    if requested_animals:
        missing = sorted(requested_animals.difference(manifest["animal"].astype(str)))
        if missing:
            raise ValueError(f"requested animals are absent from session manifest: {missing}")
        manifest = manifest[manifest["animal"].astype(str).isin(requested_animals)].copy()
        selection = selection[selection["animal"].astype(str).isin(requested_animals)].copy()
    if args.max_events_per_animal > 0:
        selection = (
            selection.sort_values(["animal", "selection_rank_within_animal"], kind="mergesort")
            .groupby("animal", sort=True, group_keys=False)
            .head(args.max_events_per_animal)
            .copy()
        )
    evidence_rows: list[dict[str, object]] = []
    unit_frames: list[pd.DataFrame] = []

    for manifest_row in manifest.itertuples(index=False):
        animal_events = selection[
            selection["animal"].astype(str).eq(str(manifest_row.animal))
            & selection["session"].astype(str).eq(str(manifest_row.session))
        ]
        if animal_events.empty:
            continue
        nwb_path = Path(manifest_row.nwb_path)
        position = read_tanni_position(nwb_path)
        replay_session = make_replay_session(nwb_path, position)
        encoding, unit_qc = fit_decoder_encoding(
            replay_session,
            position,
            bin_size_cm=args.model_bin_size_cm,
            smoothing_sigma_bins=args.smoothing_sigma_bins,
            running_speed_cm_s=args.running_speed_cm_s,
            min_running_spikes=args.min_running_spikes,
            max_mean_rate_hz=args.max_mean_rate_hz,
            min_peak_rate_hz=args.min_peak_rate_hz,
            min_split_half_stability=args.min_split_half_stability,
        )
        unit_qc.insert(0, "session", str(manifest_row.session))
        unit_qc.insert(0, "animal", str(manifest_row.animal))
        unit_frames.append(unit_qc)
        replay_session.excitatory_neurons = encoding.cell_ids
        scorers = {
            label: SortedSpikeStateSpaceReplayModel(
                mode=mode,
                config=StateSpaceDecoderConfig(
                    mode=mode,
                    valid_occupancy_threshold_s=args.valid_occupancy_threshold_s,
                ),
                name=label,
            )
            for label, mode in MODEL_MODES.items()
        }
        for event in animal_events.itertuples(index=False):
            ripple = RippleEvent(
                start=float(event.window_start_time_s),
                end=float(event.window_end_time_s),
                peak=float(event.peak_time_s),
                raw_power=float(event.peak_ripple_z),
                z_power_session=float(event.peak_ripple_z),
                z_power_epoch=float(event.peak_ripple_z),
            )
            emissions = build_emissions(
                replay_session,
                encoding,
                ripple,
                EmissionConfig(time_bin_s=args.decode_bin_s),
            )
            for model, scorer in scorers.items():
                started = time.perf_counter()
                try:
                    score = scorer.score(
                        emissions,
                        encoding.bin_centers,
                        occupancy_s=encoding.occupancy_s,
                        return_trajectory=False,
                    )
                    diagnostics = dict(score.diagnostics)
                    row = {
                        "animal": str(event.animal),
                        "session": str(event.session),
                        "event_index": int(event.event_index),
                        "selection_rank_within_animal": int(event.selection_rank_within_animal),
                        "model": model,
                        "model_mode": MODEL_MODES[model],
                        "model_family": "nontrajectory" if model == "stationary" else "trajectory",
                        "log_evidence": float(score.log_likelihood),
                        "n_spikes": int(emissions.n_spikes),
                        "n_active_cells": int(event.n_active_cells),
                        "n_time_bins": int(emissions.n_time),
                        "n_encoding_cells": int(len(encoding.cell_ids)),
                        "runtime_s": float(time.perf_counter() - started),
                        "status": "success",
                        "failure_reason": "",
                        **{f"diagnostic_{key}": value for key, value in diagnostics.items()},
                    }
                except Exception as exc:  # pragma: no cover - real-data failure path.
                    row = {
                        "animal": str(event.animal),
                        "session": str(event.session),
                        "event_index": int(event.event_index),
                        "selection_rank_within_animal": int(event.selection_rank_within_animal),
                        "model": model,
                        "model_mode": MODEL_MODES[model],
                        "model_family": "nontrajectory" if model == "stationary" else "trajectory",
                        "log_evidence": np.nan,
                        "n_spikes": int(emissions.n_spikes),
                        "n_active_cells": int(event.n_active_cells),
                        "n_time_bins": int(emissions.n_time),
                        "n_encoding_cells": int(len(encoding.cell_ids)),
                        "runtime_s": float(time.perf_counter() - started),
                        "status": "failure",
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                    }
                evidence_rows.append(row)
        print(f"{manifest_row.animal}: scored {len(animal_events)} holdout events", flush=True)

    evidence = ensure_evidence_support_columns(pd.DataFrame(evidence_rows))
    units = pd.concat(unit_frames, ignore_index=True) if unit_frames else pd.DataFrame()
    expected_events = int(len(selection))
    gates = scoring_gates(selection, evidence, expected_events=expected_events)
    selection.to_csv(output_dir / SELECTION_OUTPUT, index=False)
    evidence.to_csv(output_dir / EVIDENCE_OUTPUT, index=False)
    units.to_csv(output_dir / UNIT_OUTPUT, index=False)
    gates.to_csv(output_dir / GATE_OUTPUT, index=False)
    manifest_output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "Tanni_et_al_2022_large_2d",
        "analysis": "exact_core_clean_imm_holdout_scoring",
        "models": list(MODELS),
        "claim_boundary": "diagnostic external 2D replication; no Gate 2/3/4 promotion without distributed strict clean IMM",
        "parameters": {key: value for key, value in vars(args).items() if key not in {"evidence_dir", "selection_csv", "output_dir"}},
        "selected_events": int(len(selection)),
        "evidence_rows": int(len(evidence)),
        **build_script_provenance(
            input_paths={
                "session_manifest": evidence_dir / "tanni2022_session_manifest.csv",
                "selection_csv": selection_path,
            }
        ),
    }
    (output_dir / MANIFEST_OUTPUT).write_text(
        json.dumps(manifest_output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"selection": selection, "evidence": evidence, "units": units, "gates": gates}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--selection-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--animal", action="append", default=[])
    parser.add_argument("--max-events-per-animal", type=int, default=0)
    parser.add_argument("--model-bin-size-cm", type=float, default=16.0)
    parser.add_argument("--decode-bin-s", type=float, default=0.020)
    parser.add_argument("--smoothing-sigma-bins", type=float, default=1.5)
    parser.add_argument("--running-speed-cm-s", type=float, default=10.0)
    parser.add_argument("--min-running-spikes", type=int, default=30)
    parser.add_argument("--max-mean-rate-hz", type=float, default=4.0)
    parser.add_argument("--min-peak-rate-hz", type=float, default=2.0)
    parser.add_argument("--min-split-half-stability", type=float, default=0.25)
    parser.add_argument("--valid-occupancy-threshold-s", type=float, default=0.05)
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
