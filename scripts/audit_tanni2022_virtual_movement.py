#!/usr/bin/env python3
"""Test whether Tanni ripple candidates contain ordered virtual movement.

Every event in the pre-model exact-core subset is screened with complete saved
exact-core evidence. Expensive null rescoring is then short-circuited to the
events that pass the predeclared 5.5 family margin. A strict candidate must:

* favor diffusion/first-order IMM over stationary/fragmented by 5.5 log units;
* exceed its within-event whole-bin time-shuffle p95;
* exceed its within-event independently shifted cell-map p95; and
* displace by at least two model grid bins under the winning ordered model.

The result remains an exploratory Tanni-specific audit because the broad ripple
candidate definition was not curated as a replay event set.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from hipporeplayimm.data import RippleEvent
from hipporeplayimm.encoding import EmissionConfig, LogEmissionTensor, build_emissions
from hipporeplayimm.sorted_spike_state_space import SortedSpikeStateSpaceReplayModel
from hipporeplayimm.state_space import StateSpaceDecoderConfig
from hipporeplayimm.tanni2022 import read_tanni_position

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _provenance import build_script_provenance  # noqa: E402
from analyze_tanni2022_wall_distance_replay import fit_decoder_encoding, make_replay_session  # noqa: E402
from clean_imm_time_order_shuffle_control import permute_emission_time_bins  # noqa: E402


MODELS = ("stationary", "diffusion", "fragmented", "first-order-imm")
ORDERED_MODELS = ("diffusion", "first-order-imm")
NONORDERED_MODELS = ("stationary", "fragmented")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-shuffles", type=int, default=100)
    parser.add_argument("--model-bin-size-cm", type=float, default=16.0)
    parser.add_argument("--decode-bin-s", type=float, default=0.020)
    parser.add_argument("--claim-margin", type=float, default=5.5)
    parser.add_argument("--min-displacement-bins", type=float, default=2.0)
    parser.add_argument("--source-overlap-gap-s", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20220715)
    return parser.parse_args()


def score_models(emissions: LogEmissionTensor, encoding, *, return_trajectory: bool) -> dict[str, object]:
    scores = {}
    for model in MODELS:
        scorer = SortedSpikeStateSpaceReplayModel(
            mode=model,
            config=StateSpaceDecoderConfig(mode=model, valid_occupancy_threshold_s=0.05),
            name=model,
        )
        scores[model] = scorer.score(
            emissions,
            encoding.bin_centers,
            occupancy_s=encoding.occupancy_s,
            return_trajectory=return_trajectory,
        )
    return scores


def ordered_margin(scores: dict[str, object]) -> float:
    ordered = max(float(scores[model].log_likelihood) for model in ORDERED_MODELS)
    nonordered = max(float(scores[model].log_likelihood) for model in NONORDERED_MODELS)
    return ordered - nonordered


def independently_shift_cell_maps(encoding, rng: np.random.Generator):
    """Shift each cell's 2D rate map independently with toroidal wrapping."""

    rates = np.asarray(encoding.rates_hz, dtype=float).reshape((encoding.n_cells, *encoding.grid_shape))
    shifted = np.empty_like(rates)
    for cell_index in range(encoding.n_cells):
        x_shift = int(rng.integers(0, encoding.grid_shape[0]))
        y_shift = int(rng.integers(0, encoding.grid_shape[1]))
        if x_shift == 0 and y_shift == 0:
            x_shift = 1 % encoding.grid_shape[0]
        shifted[cell_index] = np.roll(rates[cell_index], shift=(x_shift, y_shift), axis=(0, 1))
    return replace(encoding, rates_hz=shifted.reshape(encoding.rates_hz.shape))


def posterior_path_metrics(score, bin_centers: np.ndarray, model_bin_size_cm: float) -> dict[str, float | bool]:
    log_posterior = np.asarray(score.trajectory_log_posterior, dtype=float)
    posterior = np.exp(log_posterior - logsumexp(log_posterior, axis=1, keepdims=True))
    path = posterior @ np.asarray(bin_centers, dtype=float)
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    path_length = float(steps.sum())
    displacement = float(np.linalg.norm(path[-1] - path[0]))
    return {
        "posterior_path_length_cm": path_length,
        "posterior_net_displacement_cm": displacement,
        "posterior_path_efficiency": displacement / path_length if path_length > 0.0 else 0.0,
        "posterior_mean_step_cm": float(np.mean(steps)) if steps.size else 0.0,
        "posterior_max_step_cm": float(np.max(steps)) if steps.size else 0.0,
        "posterior_large_jump_fraction_4bins": float(np.mean(steps > 4.0 * model_bin_size_cm)) if steps.size else 0.0,
    }


def empirical_upper_p(original: float, null_values: np.ndarray) -> float:
    values = np.asarray(null_values, dtype=float)
    return float((1 + np.count_nonzero(values >= float(original))) / (1 + values.size))


def source_event_groups(events: pd.DataFrame, overlap_gap_s: float) -> pd.Series:
    """Assign groups to overlapping fixed event windows within each session."""

    output = pd.Series(index=events.index, dtype="Int64")
    next_group = 0
    for _, frame in events.groupby(["animal", "session"], sort=True):
        current_end = -np.inf
        current_group = -1
        for index, row in frame.sort_values("window_start_time_s").iterrows():
            if float(row["window_start_time_s"]) > current_end + float(overlap_gap_s):
                current_group = next_group
                next_group += 1
                current_end = float(row["window_end_time_s"])
            else:
                current_end = max(current_end, float(row["window_end_time_s"]))
            output.loc[index] = current_group
    return output.astype(int)


def summarize(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, frame in [("all_events", events), ("one_per_source_group", events.loc[events["source_group_representative"]])]:
        rows.append(
            {
                "scope": scope,
                "events": int(frame.shape[0]),
                "animals": int(frame["animal"].nunique()),
                "ordered_model_confident": int(frame["ordered_model_confident"].sum()),
                "time_order_sensitive": int(frame["time_order_sensitive"].sum()),
                "map_content_sensitive": int(frame["map_content_sensitive"].sum()),
                "displacing": int(frame["displacing"].sum()),
                "strict_virtual_movement": int(frame["strict_virtual_movement"].sum()),
                "strict_virtual_movement_animals": int(frame.loc[frame["strict_virtual_movement"], "animal"].nunique()),
                "median_original_ordered_margin": float(frame["original_ordered_margin"].median()) if len(frame) else np.nan,
                "median_time_order_advantage": float(frame["time_order_advantage"].median()) if len(frame) else np.nan,
                "median_map_specific_excess": float(frame["map_specific_excess"].median()) if len(frame) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def gate_summary(events: pd.DataFrame, n_shuffles: int) -> pd.DataFrame:
    strict = events.loc[events["source_group_representative"] & events["strict_virtual_movement"]]
    candidates = events.loc[events["ordered_model_confident"]]
    complete_candidates = candidates["n_time_shuffles"].eq(n_shuffles) & candidates["n_map_shuffles"].eq(n_shuffles)
    gates = [
        ("events_present", len(events) > 0, f"{len(events)} events"),
        ("all_exact_core_rows_complete", events["all_model_scores_finite"].all() and len(events) > 0, f"{int(events['all_model_scores_finite'].sum())}/{len(events)}"),
        (
            "shuffle_counts_complete_for_margin_candidates",
            len(candidates) > 0 and complete_candidates.all(),
            f"{int(complete_candidates.sum())}/{len(candidates)} at K={n_shuffles}",
        ),
        ("strict_virtual_movement_exists", len(strict) > 0, f"{len(strict)} de-duplicated events"),
        ("strict_virtual_movement_multi_animal", strict["animal"].nunique() >= 2, f"{strict['animal'].nunique()} animals"),
    ]
    technical = all(
        passed
        for name, passed, _ in gates
        if name
        in {
            "events_present",
            "all_exact_core_rows_complete",
            "shuffle_counts_complete_for_margin_candidates",
        }
    )
    biological = all(passed for name, passed, _ in gates if name in {"strict_virtual_movement_exists", "strict_virtual_movement_multi_animal"})
    gates.extend(
        [
            ("technical_overall", technical, "technical readiness only"),
            ("biological_virtual_movement_supported", biological, "requires de-duplicated strict events in at least two animals"),
        ]
    )
    return pd.DataFrame([{"gate": name, "passed": bool(passed), "value": value} for name, passed, value in gates])


def make_figure(events: pd.DataFrame, output_path: Path, claim_margin: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    colors = np.where(events["strict_virtual_movement"], "#b6323b", "#9aa3ad")
    axes[0].scatter(events["median_time_shuffle_margin"], events["original_ordered_margin"], c=colors, alpha=0.8)
    limits = [
        float(np.nanmin([events["median_time_shuffle_margin"].min(), events["original_ordered_margin"].min()])),
        float(np.nanmax([events["median_time_shuffle_margin"].max(), events["original_ordered_margin"].max()])),
    ]
    axes[0].plot(limits, limits, color="#555555", linewidth=1)
    axes[0].axhline(claim_margin, color="#777777", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Median time-shuffle ordered margin")
    axes[0].set_ylabel("Original ordered margin")
    axes[0].set_title("Temporal-order control")
    axes[1].scatter(events["median_map_shuffle_margin"], events["original_ordered_margin"], c=colors, alpha=0.8)
    limits = [
        float(np.nanmin([events["median_map_shuffle_margin"].min(), events["original_ordered_margin"].min()])),
        float(np.nanmax([events["median_map_shuffle_margin"].max(), events["original_ordered_margin"].max()])),
    ]
    axes[1].plot(limits, limits, color="#555555", linewidth=1)
    axes[1].axhline(claim_margin, color="#777777", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Median cell-map-shuffle ordered margin")
    axes[1].set_ylabel("Original ordered margin")
    axes[1].set_title("Spatial-map control")
    axes[2].scatter(events["posterior_net_displacement_cm"], events["original_ordered_margin"], c=colors, alpha=0.8)
    axes[2].axhline(claim_margin, color="#777777", linestyle="--", linewidth=1)
    axes[2].set_xlabel("Ordered-winner posterior displacement (cm)")
    axes[2].set_ylabel("Original ordered margin")
    axes[2].set_title("Posterior movement content")
    for row in events.loc[events["strict_virtual_movement"]].itertuples(index=False):
        axes[2].annotate(f"{row.animal}:{row.event_index}", (row.posterior_net_displacement_cm, row.original_ordered_margin), fontsize=7)
    fig.suptitle("Tanni large-arena virtual-movement audit (red: passes all exploratory gates)")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(summary: pd.DataFrame, gates: pd.DataFrame, output_path: Path) -> None:
    all_row = summary.loc[summary["scope"] == "all_events"].iloc[0]
    dedup = summary.loc[summary["scope"] == "one_per_source_group"].iloc[0]
    biological = bool(gates.loc[gates["gate"] == "biological_virtual_movement_supported", "passed"].iloc[0])
    verdict = "exploratory virtual movement candidates exist across animals" if biological else "robust virtual movement not established"
    lines = [
        "# Tanni 2022 virtual-movement audit",
        "",
        f"**Verdict:** {verdict}.",
        "",
        "Diffusion is treated as ordered virtual movement: it is a local continuous random walk, not a stationary or fragmented model.",
        "",
        f"- Exact-core events tested without positive-event preselection: {int(all_row['events'])}",
        f"- Original ordered-model confident events: {int(all_row['ordered_model_confident'])}",
        f"- Strict events before source-window de-duplication: {int(all_row['strict_virtual_movement'])}",
        f"- Strict one-per-source events: {int(dedup['strict_virtual_movement'])}",
        f"- Animals with strict one-per-source events: {int(dedup['strict_virtual_movement_animals'])}",
        "",
        "A strict event clears the 5.5 family margin, survives Bonferroni correction across all margin-positive events for both shuffle tests, exceeds both event-specific shuffle p95 values, and displaces by at least two 16 cm grid bins.",
        "This is an exploratory audit of broad awake ripple candidates, not a prevalence estimate for curated replay.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    evidence_dir = args.evidence_dir.resolve()
    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(evidence_dir / "tanni2022_session_manifest.csv")
    selected = pd.read_csv(model_dir / "tanni2022_wall_balanced_model_subset.csv")
    decisions = pd.read_csv(model_dir / "tanni2022_wall_balanced_model_decisions.csv")
    ripple_events = pd.read_csv(evidence_dir / "tanni2022_ripple_candidates.csv")
    event_table = selected.merge(
        ripple_events[
            [
                "animal",
                "session",
                "event_index",
                "window_start_time_s",
                "window_end_time_s",
                "peak_time_s",
                "peak_ripple_z",
                "n_spikes",
                "n_active_cells",
            ]
        ],
        on=["animal", "session", "event_index", "peak_time_s", "peak_ripple_z"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_detector"),
    )
    event_table = event_table.merge(
        decisions[
            [
                "animal",
                "session",
                "event_index",
                *MODELS,
                "delta_ordered_minus_static_or_fragmented",
            ]
        ],
        on=["animal", "session", "event_index"],
        how="left",
        validate="one_to_one",
    )
    candidate_mask = event_table["delta_ordered_minus_static_or_fragmented"].ge(args.claim_margin)
    candidate_count = int(candidate_mask.sum())
    audit_by_key: dict[tuple[str, str, int], dict[str, object]] = {}
    for _, event in event_table.iterrows():
        key = (str(event["animal"]), str(event["session"]), int(event["event_index"]))
        ordered_winner = max(ORDERED_MODELS, key=lambda model: float(event[model]))
        model_values = np.asarray([float(event[model]) for model in MODELS], dtype=float)
        audit_by_key[key] = {
            "animal": key[0],
            "session": key[1],
            "event_index": key[2],
            "window_start_time_s": float(event["window_start_time_s"]),
            "window_end_time_s": float(event["window_end_time_s"]),
            "peak_time_s": float(event["peak_time_s"]),
            "peak_ripple_z": float(event["peak_ripple_z"]),
            "n_spikes": int(event["n_spikes"]),
            "n_active_cells": int(event["n_active_cells"]),
            "ordered_winner": ordered_winner,
            "original_ordered_margin": float(event["delta_ordered_minus_static_or_fragmented"]),
            "recomputed_ordered_margin": np.nan,
            "max_abs_logZ_reproduction_error": np.nan,
            "median_time_shuffle_margin": np.nan,
            "p95_time_shuffle_margin": np.nan,
            "time_order_advantage": np.nan,
            "time_order_empirical_p": np.nan,
            "time_order_bonferroni_p": np.nan,
            "median_map_shuffle_margin": np.nan,
            "p95_map_shuffle_margin": np.nan,
            "map_specific_excess": np.nan,
            "map_shuffle_empirical_p": np.nan,
            "map_shuffle_bonferroni_p": np.nan,
            "n_time_shuffles": 0,
            "n_map_shuffles": 0,
            "posterior_path_length_cm": np.nan,
            "posterior_net_displacement_cm": np.nan,
            "posterior_path_efficiency": np.nan,
            "posterior_mean_step_cm": np.nan,
            "posterior_max_step_cm": np.nan,
            "posterior_large_jump_fraction_4bins": np.nan,
            "all_model_scores_finite": bool(np.isfinite(model_values).all()),
            "ordered_model_confident": bool(event["delta_ordered_minus_static_or_fragmented"] >= args.claim_margin),
            "time_order_sensitive": False,
            "map_content_sensitive": False,
            "displacing": False,
            "strict_virtual_movement": False,
            **{f"logZ_{model}": float(event[model]) for model in MODELS},
        }
    rng = np.random.default_rng(args.seed)
    null_rows = []
    for manifest_row in manifest.itertuples(index=False):
        events = event_table.loc[candidate_mask & (event_table["animal"] == manifest_row.animal) & (event_table["session"] == manifest_row.session)]
        if events.empty:
            continue
        position = read_tanni_position(Path(manifest_row.nwb_path))
        session = make_replay_session(Path(manifest_row.nwb_path), position)
        encoding, _ = fit_decoder_encoding(
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
        selected_session = replace(session, excitatory_neurons=encoding.cell_ids)
        for event in events.itertuples(index=False):
            key = (str(event.animal), str(event.session), int(event.event_index))
            ripple = RippleEvent(
                start=float(event.window_start_time_s),
                end=float(event.window_end_time_s),
                peak=float(event.peak_time_s),
                raw_power=float(event.peak_ripple_z),
                z_power_session=float(event.peak_ripple_z),
                z_power_epoch=float(event.peak_ripple_z),
            )
            emissions = build_emissions(selected_session, encoding, ripple, EmissionConfig(time_bin_s=args.decode_bin_s))
            original_scores = score_models(emissions, encoding, return_trajectory=True)
            original_margin = ordered_margin(original_scores)
            ordered_winner = max(ORDERED_MODELS, key=lambda model: float(original_scores[model].log_likelihood))
            path = posterior_path_metrics(original_scores[ordered_winner], encoding.bin_centers, args.model_bin_size_cm)
            time_margins = []
            map_margins = []
            for shuffle_index in range(args.n_shuffles):
                permutation = rng.permutation(emissions.n_time)
                shuffled_emissions = permute_emission_time_bins(emissions, permutation)
                time_margin = ordered_margin(score_models(shuffled_emissions, encoding, return_trajectory=False))
                shifted_encoding = independently_shift_cell_maps(encoding, rng)
                wrong_map_emissions = build_emissions(
                    selected_session,
                    shifted_encoding,
                    ripple,
                    EmissionConfig(time_bin_s=args.decode_bin_s),
                )
                map_margin = ordered_margin(score_models(wrong_map_emissions, shifted_encoding, return_trajectory=False))
                time_margins.append(time_margin)
                map_margins.append(map_margin)
                null_rows.extend(
                    [
                        {
                            "animal": event.animal,
                            "session": event.session,
                            "event_index": int(event.event_index),
                            "null_type": "whole_bin_time_shuffle",
                            "shuffle_index": shuffle_index,
                            "ordered_margin": time_margin,
                        },
                        {
                            "animal": event.animal,
                            "session": event.session,
                            "event_index": int(event.event_index),
                            "null_type": "independent_cell_map_shift",
                            "shuffle_index": shuffle_index,
                            "ordered_margin": map_margin,
                        },
                    ]
                )
            time_values = np.asarray(time_margins, dtype=float)
            map_values = np.asarray(map_margins, dtype=float)
            raw_time_p = empirical_upper_p(original_margin, time_values)
            raw_map_p = empirical_upper_p(original_margin, map_values)
            row = audit_by_key[key]
            row.update(
                {
                    "ordered_winner": ordered_winner,
                    "recomputed_ordered_margin": original_margin,
                    "max_abs_logZ_reproduction_error": float(max(abs(float(original_scores[model].log_likelihood) - float(row[f"logZ_{model}"])) for model in MODELS)),
                    "median_time_shuffle_margin": float(np.median(time_values)),
                    "p95_time_shuffle_margin": float(np.quantile(time_values, 0.95)),
                    "time_order_advantage": original_margin - float(np.median(time_values)),
                    "time_order_empirical_p": raw_time_p,
                    "time_order_bonferroni_p": min(raw_time_p * candidate_count, 1.0),
                    "median_map_shuffle_margin": float(np.median(map_values)),
                    "p95_map_shuffle_margin": float(np.quantile(map_values, 0.95)),
                    "map_specific_excess": original_margin - float(np.median(map_values)),
                    "map_shuffle_empirical_p": raw_map_p,
                    "map_shuffle_bonferroni_p": min(raw_map_p * candidate_count, 1.0),
                    "n_time_shuffles": int(time_values.size),
                    "n_map_shuffles": int(map_values.size),
                }
            )
            row.update(path)
            row["time_order_sensitive"] = bool(original_margin > row["p95_time_shuffle_margin"] and row["time_order_bonferroni_p"] <= 0.05)
            row["map_content_sensitive"] = bool(original_margin > row["p95_map_shuffle_margin"] and row["map_shuffle_bonferroni_p"] <= 0.05)
            row["displacing"] = bool(row["posterior_net_displacement_cm"] >= args.min_displacement_bins * args.model_bin_size_cm)
            row["strict_virtual_movement"] = bool(row["ordered_model_confident"] and row["time_order_sensitive"] and row["map_content_sensitive"] and row["displacing"])
        print(f"{manifest_row.animal}: null-tested {len(events)} margin-positive events", flush=True)
    audit = pd.DataFrame(audit_by_key.values())
    audit["source_event_group"] = source_event_groups(audit, args.source_overlap_gap_s)
    audit["source_group_representative"] = False
    representatives = audit.sort_values(["strict_virtual_movement", "original_ordered_margin"], ascending=[False, False]).groupby("source_event_group", sort=False).head(1).index
    audit.loc[representatives, "source_group_representative"] = True
    summary = summarize(audit)
    gates = gate_summary(audit, args.n_shuffles)
    audit.to_csv(output_dir / "tanni2022_virtual_movement_event_audit.csv", index=False)
    pd.DataFrame(null_rows).to_csv(output_dir / "tanni2022_virtual_movement_null_scores.csv", index=False)
    summary.to_csv(output_dir / "tanni2022_virtual_movement_summary.csv", index=False)
    gates.to_csv(output_dir / "tanni2022_virtual_movement_gate_summary.csv", index=False)
    make_figure(audit, output_dir / "tanni2022_virtual_movement_audit_figure.png", args.claim_margin)
    write_report(summary, gates, output_dir / "tanni2022_virtual_movement_report.md")
    provenance = build_script_provenance(
        input_paths={
            "session_manifest": evidence_dir / "tanni2022_session_manifest.csv",
            "ripple_candidates": evidence_dir / "tanni2022_ripple_candidates.csv",
            "model_subset": model_dir / "tanni2022_wall_balanced_model_subset.csv",
            "model_decisions": model_dir / "tanni2022_wall_balanced_model_decisions.csv",
        }
    )
    payload = {
        "analysis": "tanni2022_virtual_movement_audit",
        "parameters": vars(args) | {"evidence_dir": str(evidence_dir), "model_dir": str(model_dir), "output_dir": str(output_dir)},
        "provenance": provenance,
    }
    (output_dir / "tanni2022_virtual_movement_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
