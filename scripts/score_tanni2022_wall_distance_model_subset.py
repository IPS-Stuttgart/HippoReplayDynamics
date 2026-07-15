#!/usr/bin/env python3
"""Score a pre-model, wall-balanced Tanni event subset with the exact 2D core."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from hipporeplayimm.data import RippleEvent
from hipporeplayimm.encoding import EmissionConfig, build_emissions
from hipporeplayimm.sorted_spike_state_space import SortedSpikeStateSpaceReplayModel
from hipporeplayimm.state_space import StateSpaceDecoderConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_tanni2022_wall_distance_replay import (  # noqa: E402
    association_summary,
    fit_decoder_encoding,
    make_replay_session,
)
from hipporeplayimm.tanni2022 import read_tanni_position  # noqa: E402


MODELS = ("stationary", "diffusion", "fragmented", "first-order-imm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--events-per-animal", type=int, default=40)
    parser.add_argument("--model-bin-size-cm", type=float, default=16.0)
    parser.add_argument("--decode-bin-s", type=float, default=0.020)
    parser.add_argument("--claim-margin", type=float, default=5.5)
    parser.add_argument("--seed", type=int, default=20220714)
    return parser.parse_args()


def select_balanced_model_subset(events: pd.DataFrame, events_per_animal: int, seed: int) -> pd.DataFrame:
    """Select equal numbers per represented-wall quartile without model scores."""

    rng = np.random.default_rng(seed)
    selected_rows = []
    per_quartile = max(int(events_per_animal) // 4, 1)
    candidates = events.copy()
    candidates["wall_distance_normalized"] = candidates["median_wall_distance_cm"] / 125.0
    candidates["wall_quartile"] = np.minimum(np.floor(np.clip(candidates["wall_distance_normalized"], 0.0, 1.0) * 4.0).astype(int), 3) + 1
    for animal, animal_frame in candidates.groupby("animal", sort=True):
        for quartile in range(1, 5):
            frame = animal_frame.loc[animal_frame["wall_quartile"] == quartile]
            if frame.empty:
                continue
            count = min(per_quartile, frame.shape[0])
            indices = np.sort(rng.choice(frame.index.to_numpy(), size=count, replace=False))
            chosen = frame.loc[indices].copy()
            chosen["selection_rank_within_animal_quartile"] = np.arange(1, chosen.shape[0] + 1)
            selected_rows.append(chosen)
    selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    if not selected.empty:
        selected["selection_rule"] = "uniform_random_within_animal_and_represented_wall_quartile_pre_model"
        selected["selection_seed"] = int(seed)
    return selected


def evidence_decisions(evidence: pd.DataFrame, claim_margin: float) -> pd.DataFrame:
    """Return ordered-vs-static/fragmented decisions for complete model rows."""

    pivot = evidence.pivot_table(index=["animal", "session", "event_index"], columns="model", values="log_evidence", aggfunc="first").reset_index()
    missing = [model for model in MODELS if model not in pivot]
    if missing:
        raise ValueError(f"Missing model evidence columns: {missing}")
    model_values = pivot[list(MODELS)].to_numpy(dtype=float)
    order = np.argsort(model_values, axis=1)
    pivot["best_model"] = np.asarray(MODELS, dtype=object)[order[:, -1]]
    pivot["runner_up_model"] = np.asarray(MODELS, dtype=object)[order[:, -2]]
    pivot["best_minus_runner_up_log_evidence"] = model_values[np.arange(model_values.shape[0]), order[:, -1]] - model_values[
        np.arange(model_values.shape[0]), order[:, -2]
    ]
    ordered = np.maximum(pivot["diffusion"], pivot["first-order-imm"])
    nonordered = np.maximum(pivot["stationary"], pivot["fragmented"])
    pivot["delta_ordered_minus_static_or_fragmented"] = ordered - nonordered
    pivot["delta_imm_minus_fragmented"] = pivot["first-order-imm"] - pivot["fragmented"]
    pivot["ordered_trajectory_confident"] = pivot["delta_ordered_minus_static_or_fragmented"] >= float(claim_margin)
    pivot["stationary_confident"] = pivot["stationary"] - pivot[["diffusion", "fragmented", "first-order-imm"]].max(axis=1) >= float(claim_margin)
    pivot["fragmented_confident"] = pivot["fragmented"] - pivot[["stationary", "diffusion", "first-order-imm"]].max(axis=1) >= float(claim_margin)
    pivot["imm_confident_over_fragmented"] = pivot["delta_imm_minus_fragmented"] >= float(claim_margin)
    pivot["ambiguous"] = ~(pivot["ordered_trajectory_confident"] | pivot["stationary_confident"] | pivot["fragmented_confident"])
    return pivot


def main() -> int:
    args = parse_args()
    evidence_dir = args.evidence_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(evidence_dir / "tanni2022_session_manifest.csv")
    event_speed = pd.read_csv(evidence_dir / "tanni2022_replay_speed_events.csv")
    ripple_events = pd.read_csv(evidence_dir / "tanni2022_ripple_candidates.csv")
    speed_segments = pd.read_csv(evidence_dir / "tanni2022_replay_speed_segments.csv")
    selected = select_balanced_model_subset(event_speed, args.events_per_animal, args.seed)
    selected = selected.merge(
        ripple_events[
            [
                "animal",
                "session",
                "event_index",
                "window_start_time_s",
                "window_end_time_s",
                "peak_time_s",
                "peak_ripple_z",
            ]
        ],
        on=["animal", "session", "event_index", "peak_time_s", "peak_ripple_z"],
        how="left",
        validate="one_to_one",
    )
    selected.to_csv(output_dir / "tanni2022_wall_balanced_model_subset.csv", index=False)
    evidence_rows: list[dict[str, object]] = []
    for manifest_row in manifest.itertuples(index=False):
        animal_events = selected.loc[(selected["animal"] == manifest_row.animal) & (selected["session"] == manifest_row.session)]
        if animal_events.empty:
            continue
        nwb_path = Path(manifest_row.nwb_path)
        position = read_tanni_position(nwb_path)
        session = make_replay_session(nwb_path, position)
        encoding, _unit_qc = fit_decoder_encoding(
            session,
            position,
            bin_size_cm=args.model_bin_size_cm,
            smoothing_sigma_bins=1.5,
            running_speed_cm_s=10.0,
            min_running_spikes=30,
            max_mean_rate_hz=4.0,
            min_peak_rate_hz=2.0,
            min_split_half_stability=0.25,
        )
        session.excitatory_neurons = encoding.cell_ids
        config_by_model = {
            model: StateSpaceDecoderConfig(mode=model, valid_occupancy_threshold_s=0.05)
            for model in MODELS
        }
        scorers = {
            model: SortedSpikeStateSpaceReplayModel(mode=model, config=config_by_model[model], name=model)
            for model in MODELS
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
            emissions = build_emissions(session, encoding, ripple, EmissionConfig(time_bin_s=args.decode_bin_s))
            for model, scorer in scorers.items():
                score = scorer.score(
                    emissions,
                    encoding.bin_centers,
                    occupancy_s=encoding.occupancy_s,
                    return_trajectory=False,
                )
                evidence_rows.append(
                    {
                        "animal": event.animal,
                        "session": event.session,
                        "event_index": int(event.event_index),
                        "wall_quartile": int(event.wall_quartile),
                        "model": model,
                        "log_evidence": float(score.log_likelihood),
                        "n_spikes": int(emissions.n_spikes),
                        "n_time_bins": int(emissions.n_time),
                        "status": "ok",
                    }
                )
        print(f"{manifest_row.animal}: scored {animal_events.shape[0]} events", flush=True)
    evidence = pd.DataFrame(evidence_rows)
    decisions = evidence_decisions(evidence, args.claim_margin)
    decisions = decisions.merge(
        selected[["animal", "session", "event_index", "wall_quartile", "median_wall_distance_cm"]],
        on=["animal", "session", "event_index"],
        how="left",
        validate="one_to_one",
    )
    evidence.to_csv(output_dir / "tanni2022_wall_balanced_model_evidence.csv", index=False)
    decisions.to_csv(output_dir / "tanni2022_wall_balanced_model_decisions.csv", index=False)
    summary = (
        decisions.groupby("best_model").size().rename("best_model_count").reset_index().sort_values("best_model_count", ascending=False)
    )
    summary["events"] = int(decisions.shape[0])
    summary["ordered_trajectory_confident_count"] = int(decisions["ordered_trajectory_confident"].sum())
    summary["imm_confident_over_fragmented_count"] = int(decisions["imm_confident_over_fragmented"].sum())
    summary.to_csv(output_dir / "tanni2022_wall_balanced_model_summary.csv", index=False)
    association_frames = []
    for subset_name, keys in (
        ("all_model_subset", decisions),
        ("ordered_trajectory_confident", decisions.loc[decisions["ordered_trajectory_confident"]]),
        ("imm_confident_over_fragmented", decisions.loc[decisions["imm_confident_over_fragmented"]]),
    ):
        subset_segments = speed_segments.merge(
            keys[["animal", "session", "event_index"]],
            on=["animal", "session", "event_index"],
            how="inner",
        )
        if subset_segments.empty:
            continue
        association = association_summary(subset_segments, pd.DataFrame(), bootstrap_replicates=5000, seed=args.seed)
        association.insert(0, "model_subset", subset_name)
        association_frames.append(association)
    associations = pd.concat(association_frames, ignore_index=True) if association_frames else pd.DataFrame()
    associations.to_csv(output_dir / "tanni2022_wall_distance_model_subset_associations.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
