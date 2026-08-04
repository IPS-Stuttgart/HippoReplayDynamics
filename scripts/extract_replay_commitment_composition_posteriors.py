#!/usr/bin/env python3
"""Export common replay paths and exact IMM mode histories for frozen events."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import hipporeplayimm  # noqa: E402

from _provenance import build_script_provenance, git_metadata  # noqa: E402
from hipporeplayimm.data import load_replay_session  # noqa: E402
from hipporeplayimm.encoding import (  # noqa: E402
    EmissionConfig,
    EncodingConfig,
    build_emissions,
    fit_place_field_encoding,
)
from hipporeplayimm.sorted_spike_state_space import (  # noqa: E402
    SortedSpikeStateSpaceReplayModel,
)
from hipporeplayimm.state_space import StateSpaceDecoderConfig  # noqa: E402


IMM = "sorted-spike-state-space-first-order-imm"
MOMENTUM = "sorted-spike-state-space-momentum-exact-sparse"
MODE_NAMES = ("stationary", "continuous_diffusion", "fragmented_jump")

BIN_OUTPUT = "replay_commitment_composition_posterior_bins.csv"
TRANSITION_OUTPUT = "replay_commitment_composition_posterior_transitions.csv"
EVENT_OUTPUT = "replay_commitment_composition_posterior_event_summary.csv"
GATE_OUTPUT = "replay_commitment_composition_posterior_gate_summary.csv"
MANIFEST_OUTPUT = "replay_commitment_composition_posterior_manifest.json"
SUMMARY_OUTPUT = "replay_commitment_composition_posterior_summary.md"


def _successful_rows(evidence: pd.DataFrame) -> pd.DataFrame:
    out = evidence.copy()
    if "status" in out:
        status = out["status"].astype("string").str.lower()
        out = out[status.isna() | status.eq("success")]
    if "evidence_comparable" in out:
        comparable = out["evidence_comparable"].astype(str).str.lower().isin(
            {"true", "1", "1.0", "yes"}
        )
        out = out[comparable]
    out["event_index"] = pd.to_numeric(out["event_index"], errors="raise").astype(int)
    out["log_evidence"] = pd.to_numeric(out["log_evidence"], errors="coerce")
    return out.dropna(subset=["log_evidence"])


def _unique_number(rows: pd.DataFrame, column: str, default: float) -> float:
    if column not in rows:
        return float(default)
    values = pd.to_numeric(rows[column], errors="coerce").dropna().unique()
    if not len(values):
        return float(default)
    if len(values) != 1:
        raise ValueError(f"{column} is not constant: {values.tolist()}")
    return float(values[0])


def decoder_configs(
    evidence: pd.DataFrame,
) -> tuple[EncodingConfig, EmissionConfig, StateSpaceDecoderConfig]:
    imm_rows = evidence[evidence["model"].eq(IMM)]
    momentum_rows = evidence[evidence["model"].eq(MOMENTUM)]
    if imm_rows.empty or momentum_rows.empty:
        raise ValueError("both first-order IMM and exact-sparse momentum rows are required")
    encoding = EncodingConfig(
        bin_size_cm=_unique_number(imm_rows, "bin_size_cm", 6.0),
        smoothing_sigma_bins=_unique_number(imm_rows, "smoothing_sigma_bins", 2.0),
        min_speed_cm_s=_unique_number(imm_rows, "min_speed_cm_s", 5.0),
    )
    emission = EmissionConfig(
        time_bin_s=_unique_number(imm_rows, "time_bin_s", 0.004),
        spike_rate_scale=_unique_number(imm_rows, "spike_rate_scale", 2.0),
        likelihood_temperature=_unique_number(
            imm_rows,
            "emission_likelihood_temperature",
            0.3,
        ),
        negative_binomial_overdispersion=_unique_number(
            imm_rows,
            "emission_negative_binomial_overdispersion",
            0.0,
        ),
    )
    state = StateSpaceDecoderConfig(
        mode="first-order-imm",
        stationary_sigma_cm=_unique_number(
            imm_rows,
            "diagnostic_state_space_stationary_sigma_cm",
            2.0,
        ),
        diffusion_sigma_cm_sqrt_s=_unique_number(
            imm_rows,
            "diagnostic_state_space_diffusion_sigma_cm_sqrt_s",
            60.0,
        ),
        max_step_sigma=_unique_number(
            imm_rows,
            "diagnostic_state_space_max_step_sigma",
            3.0,
        ),
        imm_mode_stickiness=_unique_number(
            imm_rows,
            "state_space_imm_mode_stickiness",
            0.95,
        ),
        imm_switch_tau_s=_unique_number(
            imm_rows,
            "state_space_imm_switch_tau_s",
            0.06,
        ),
        momentum_sigma_cm_sqrt_s=_unique_number(
            momentum_rows,
            "diagnostic_state_space_momentum_sigma_cm_sqrt_s",
            85.0,
        ),
        momentum_initial_sigma_cm_sqrt_s=_unique_number(
            momentum_rows,
            "diagnostic_state_space_momentum_initial_sigma_cm_sqrt_s",
            85.0,
        ),
        momentum_velocity_decay=_unique_number(
            momentum_rows,
            "diagnostic_state_space_momentum_velocity_decay",
            0.95,
        ),
        momentum_velocity_decay_tau_s=_unique_number(
            momentum_rows,
            "state_space_momentum_velocity_decay_tau_s",
            0.0,
        ),
        momentum_candidate_top_k=int(
            _unique_number(momentum_rows, "state_space_common_support_top_k", 128)
        ),
        momentum_predicted_candidate_top_k=int(
            _unique_number(
                momentum_rows,
                "state_space_momentum_predicted_candidate_top_k",
                8,
            )
        ),
        valid_occupancy_threshold_s=_unique_number(
            imm_rows,
            "state_space_valid_occupancy_threshold_s",
            0.0,
        ),
    )
    return encoding, emission, state


def normalized_posterior(log_values: np.ndarray) -> np.ndarray:
    values = np.asarray(log_values, dtype=float)
    normalizer = logsumexp(values, axis=1, keepdims=True)
    probability = np.exp(values - normalizer)
    probability[~np.isfinite(probability)] = 0.0
    row_sum = probability.sum(axis=1, keepdims=True)
    return np.divide(probability, row_sum, out=np.zeros_like(probability), where=row_sum > 0.0)


def posterior_path_summary(
    log_posterior: np.ndarray,
    centers: np.ndarray,
) -> dict[str, np.ndarray]:
    probability = normalized_posterior(log_posterior)
    center_array = np.asarray(centers, dtype=float)
    mean_xy = probability @ center_array
    map_xy = center_array[np.argmax(probability, axis=1)]
    entropy = -np.sum(
        probability * np.log(np.maximum(probability, np.finfo(float).tiny)),
        axis=1,
    )
    return {
        "probability": probability,
        "mean_xy": mean_xy,
        "map_xy": map_xy,
        "entropy": entropy,
    }


def continuous_bout_ids(map_mode: np.ndarray) -> np.ndarray:
    mode = np.asarray(map_mode, dtype=int)
    continuous = mode == 1
    starts = continuous & np.concatenate(([True], ~continuous[:-1]))
    cumulative = np.cumsum(starts) - 1
    return np.where(continuous, cumulative, -1).astype(int)


def _path_length(xy: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) >= 2 else 0.0


def _displacement(xy: np.ndarray) -> float:
    return float(np.linalg.norm(xy[-1] - xy[0])) if len(xy) >= 2 else 0.0


def _reference_lookup(evidence: pd.DataFrame, model: str) -> pd.Series:
    return (
        evidence[evidence["model"].eq(model)]
        .drop_duplicates(["session", "event_index"], keep="last")
        .set_index(["session", "event_index"])["log_evidence"]
    )


def extract_posteriors(
    *,
    dataset_root: str | Path,
    frozen_events: pd.DataFrame,
    evidence: pd.DataFrame,
    include_momentum_posterior: bool = True,
    max_events: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    encoding_config, emission_config, state_config = decoder_configs(evidence)
    imm_model = SortedSpikeStateSpaceReplayModel(
        mode="first-order-imm",
        config=state_config,
        name=IMM,
    )
    momentum_model = SortedSpikeStateSpaceReplayModel(
        mode="momentum-exact-sparse",
        config=state_config,
        name=MOMENTUM,
    )
    imm_reference = _reference_lookup(evidence, IMM)
    momentum_reference = _reference_lookup(evidence, MOMENTUM)
    selected = frozen_events.sort_values(["session", "event_index"]).copy()
    if int(max_events) > 0:
        selected = selected.head(int(max_events))
    bin_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    dataset = Path(dataset_root)
    for session_id, session_events in selected.groupby("session", sort=True):
        session = load_replay_session(dataset / Path(str(session_id)))
        encoding = fit_place_field_encoding(session, encoding_config)
        centers = np.asarray(encoding.bin_centers, dtype=float)
        for event in session_events.itertuples(index=False):
            event_index = int(event.event_index)
            emissions = build_emissions(session, encoding, event_index, emission_config)
            imm_score = imm_model.score(emissions, centers)
            if imm_score.trajectory_log_posterior is None:
                raise ValueError(f"{session_id} event {event_index}: IMM trajectory missing")
            imm_path = posterior_path_summary(imm_score.trajectory_log_posterior, centers)
            emission_path = posterior_path_summary(emissions.log_likelihood, centers)
            mode = np.asarray(
                json.loads(imm_score.diagnostics["state_space_imm_mode_posterior_over_time"]),
                dtype=float,
            )
            transition = np.asarray(
                json.loads(
                    imm_score.diagnostics[
                        "state_space_imm_mode_transition_posterior_over_time"
                    ]
                ),
                dtype=float,
            )
            switch = np.asarray(
                json.loads(
                    imm_score.diagnostics["state_space_imm_switch_probability_over_time"]
                ),
                dtype=float,
            )
            map_mode = np.argmax(mode, axis=1)
            bout_id = continuous_bout_ids(map_mode)
            momentum_score = None
            momentum_path = None
            if include_momentum_posterior:
                momentum_score = momentum_model.score(emissions, centers)
                if momentum_score.trajectory_log_posterior is None:
                    raise ValueError(
                        f"{session_id} event {event_index}: momentum trajectory missing"
                    )
                momentum_path = posterior_path_summary(
                    momentum_score.trajectory_log_posterior,
                    centers,
                )
            for time_index, time_s in enumerate(emissions.times):
                row: dict[str, object] = {
                    "session": str(session_id),
                    "rat": str(session_id).split("/", 1)[0],
                    "event_index": event_index,
                    "time_bin": int(time_index),
                    "time_s": float(time_s),
                    "bin_duration_s": float(emissions.bin_durations[time_index]),
                    "imm_posterior_mean_x_cm": float(imm_path["mean_xy"][time_index, 0]),
                    "imm_posterior_mean_y_cm": float(imm_path["mean_xy"][time_index, 1]),
                    "imm_map_x_cm": float(imm_path["map_xy"][time_index, 0]),
                    "imm_map_y_cm": float(imm_path["map_xy"][time_index, 1]),
                    "imm_posterior_spatial_entropy": float(imm_path["entropy"][time_index]),
                    "emission_only_mean_x_cm": float(emission_path["mean_xy"][time_index, 0]),
                    "emission_only_mean_y_cm": float(emission_path["mean_xy"][time_index, 1]),
                    "emission_only_map_x_cm": float(emission_path["map_xy"][time_index, 0]),
                    "emission_only_map_y_cm": float(emission_path["map_xy"][time_index, 1]),
                    "emission_only_spatial_entropy": float(emission_path["entropy"][time_index]),
                    "stationary_mode_probability": float(mode[time_index, 0]),
                    "continuous_diffusion_mode_probability": float(mode[time_index, 1]),
                    "fragmented_jump_mode_probability": float(mode[time_index, 2]),
                    "map_mode_index": int(map_mode[time_index]),
                    "map_mode_name": MODE_NAMES[int(map_mode[time_index])],
                    "continuous_diffusion_bin": bool(map_mode[time_index] == 1),
                    "continuous_bout_id": int(bout_id[time_index]),
                }
                if momentum_path is not None:
                    row.update(
                        {
                            "momentum_posterior_mean_x_cm": float(
                                momentum_path["mean_xy"][time_index, 0]
                            ),
                            "momentum_posterior_mean_y_cm": float(
                                momentum_path["mean_xy"][time_index, 1]
                            ),
                            "momentum_map_x_cm": float(
                                momentum_path["map_xy"][time_index, 0]
                            ),
                            "momentum_map_y_cm": float(
                                momentum_path["map_xy"][time_index, 1]
                            ),
                            "momentum_posterior_spatial_entropy": float(
                                momentum_path["entropy"][time_index]
                            ),
                        }
                    )
                bin_rows.append(row)
            for transition_index in range(len(switch)):
                dominant = np.unravel_index(
                    int(np.argmax(transition[transition_index])),
                    transition[transition_index].shape,
                )
                transition_rows.append(
                    {
                        "session": str(session_id),
                        "rat": str(session_id).split("/", 1)[0],
                        "event_index": event_index,
                        "transition_index": int(transition_index),
                        "transition_time_s": float(
                            0.5
                            * (
                                emissions.times[transition_index]
                                + emissions.times[transition_index + 1]
                            )
                        ),
                        "switch_probability": float(switch[transition_index]),
                        "map_mode_switch": bool(
                            map_mode[transition_index] != map_mode[transition_index + 1]
                        ),
                        "source_map_mode_index": int(map_mode[transition_index]),
                        "destination_map_mode_index": int(map_mode[transition_index + 1]),
                        "dominant_source_mode_index": int(dominant[0]),
                        "dominant_destination_mode_index": int(dominant[1]),
                    }
                )
            continuous_ids = np.unique(bout_id[bout_id >= 0])
            bout_lengths = [int(np.sum(bout_id == value)) for value in continuous_ids]
            key = (str(session_id), event_index)
            imm_reference_value = float(imm_reference.loc[key])
            momentum_reference_value = float(momentum_reference.loc[key])
            event_rows.append(
                {
                    "session": str(session_id),
                    "rat": str(session_id).split("/", 1)[0],
                    "event_index": event_index,
                    "n_time_bins": int(len(emissions.times)),
                    "n_spikes": int(emissions.n_spikes),
                    "n_active_cells": int(np.sum(emissions.spike_counts.sum(axis=0) > 0)),
                    "continuous_diffusion_bins": int(np.sum(map_mode == 1)),
                    "stationary_bins": int(np.sum(map_mode == 0)),
                    "fragmented_jump_bins": int(np.sum(map_mode == 2)),
                    "continuous_diffusion_fraction": float(np.mean(map_mode == 1)),
                    "continuous_bout_count": int(len(continuous_ids)),
                    "longest_continuous_bout_bins": max(bout_lengths, default=0),
                    "imm_reference_log_evidence": imm_reference_value,
                    "imm_rescored_log_evidence": float(imm_score.log_likelihood),
                    "imm_rescore_error": float(imm_score.log_likelihood - imm_reference_value),
                    "momentum_reference_log_evidence": momentum_reference_value,
                    "momentum_rescored_log_evidence": float(momentum_score.log_likelihood)
                    if momentum_score is not None else np.nan,
                    "momentum_rescore_error": float(
                        momentum_score.log_likelihood - momentum_reference_value
                    ) if momentum_score is not None else np.nan,
                    "imm_mean_path_length_cm": _path_length(imm_path["mean_xy"]),
                    "imm_mean_net_displacement_cm": _displacement(imm_path["mean_xy"]),
                    "emission_only_mean_path_length_cm": _path_length(
                        emission_path["mean_xy"]
                    ),
                    "emission_only_mean_net_displacement_cm": _displacement(
                        emission_path["mean_xy"]
                    ),
                    "momentum_mean_path_length_cm": _path_length(
                        momentum_path["mean_xy"]
                    ) if momentum_path is not None else np.nan,
                    "momentum_mean_net_displacement_cm": _displacement(
                        momentum_path["mean_xy"]
                    ) if momentum_path is not None else np.nan,
                }
            )
    return pd.DataFrame(bin_rows), pd.DataFrame(transition_rows), pd.DataFrame(event_rows)


def build_gate_summary(
    frozen_events: pd.DataFrame,
    bins: pd.DataFrame,
    transitions: pd.DataFrame,
    events: pd.DataFrame,
    *,
    rescore_tolerance: float,
    include_momentum_posterior: bool,
) -> pd.DataFrame:
    expected = int(len(frozen_events))
    observed = int(len(events))
    imm_error = float(events["imm_rescore_error"].abs().max()) if observed else np.inf
    momentum_error = (
        float(events["momentum_rescore_error"].abs().max())
        if observed and include_momentum_posterior
        else np.nan
    )
    gates = [
        ("frozen_events_present", expected > 0, expected, ">0"),
        ("all_events_scored", observed == expected and expected > 0, observed, expected),
        ("posterior_bins_present", len(bins) > 0, int(len(bins)), ">0"),
        ("mode_transitions_present", len(transitions) > 0, int(len(transitions)), ">0"),
        (
            "imm_rescore_matches_frozen_evidence",
            imm_error <= float(rescore_tolerance),
            imm_error,
            f"<={float(rescore_tolerance):g}",
        ),
        (
            "momentum_rescore_matches_frozen_evidence",
            (not include_momentum_posterior)
            or momentum_error <= float(rescore_tolerance),
            momentum_error,
            f"<={float(rescore_tolerance):g}",
        ),
        (
            "all_three_common_path_estimators_present",
            bool(
                not bins.empty
                and {"imm_posterior_mean_x_cm", "emission_only_mean_x_cm"}.issubset(bins)
                and (
                    not include_momentum_posterior
                    or "momentum_posterior_mean_x_cm" in bins
                )
            ),
            bool(not bins.empty),
            True,
        ),
        (
            "stationary_and_fragmented_bins_explicitly_labeled",
            bool(not bins.empty and {0, 2}.issubset(set(bins["map_mode_index"].astype(int)))),
            sorted(set(bins["map_mode_index"].astype(int))) if not bins.empty else [],
            "contains 0 and 2",
        ),
    ]
    rows = [
        {"gate": gate, "passed": bool(passed), "value": value, "required": required}
        for gate, passed, value, required in gates
    ]
    rows.append(
        {
            "gate": "overall",
            "passed": bool(all(row["passed"] for row in rows)),
            "value": int(sum(bool(row["passed"]) for row in rows)),
            "required": len(rows),
        }
    )
    return pd.DataFrame(rows)


def run_analysis(
    *,
    dataset_root: str | Path,
    frozen_events_csv: str | Path,
    event_evidence_csv: str | Path,
    output_dir: str | Path,
    include_momentum_posterior: bool = True,
    max_events: int = 0,
    rescore_tolerance: float = 0.025,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frozen_events = pd.read_csv(frozen_events_csv)
    if int(max_events) > 0:
        frozen_for_run = frozen_events.sort_values(["session", "event_index"]).head(int(max_events))
    else:
        frozen_for_run = frozen_events
    evidence = _successful_rows(pd.read_csv(event_evidence_csv))
    bins, transitions, events = extract_posteriors(
        dataset_root=dataset_root,
        frozen_events=frozen_for_run,
        evidence=evidence,
        include_momentum_posterior=bool(include_momentum_posterior),
        max_events=0,
    )
    gates = build_gate_summary(
        frozen_for_run,
        bins,
        transitions,
        events,
        rescore_tolerance=float(rescore_tolerance),
        include_momentum_posterior=bool(include_momentum_posterior),
    )
    frames = {
        BIN_OUTPUT: bins,
        TRANSITION_OUTPUT: transitions,
        EVENT_OUTPUT: events,
        GATE_OUTPUT: gates,
    }
    paths: dict[str, Path] = {}
    for name, frame in frames.items():
        path = output / name
        frame.to_csv(path, index=False)
        paths[name] = path
    provenance = build_script_provenance(
        input_paths={
            "dataset_root": dataset_root,
            "frozen_events_csv": frozen_events_csv,
            "event_evidence_csv": event_evidence_csv,
        },
        cwd=ROOT,
    )
    manifest = {
        "analysis": "replay_commitment_composition_posterior_export",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "events": int(len(events)),
        "include_momentum_posterior": bool(include_momentum_posterior),
        "max_events": int(max_events),
        "mode_semantics": {
            "0": "stationary_excluded",
            "1": "continuous_diffusion_included",
            "2": "fragmented_jump_excluded",
        },
        "primary_path_estimator": "first_order_imm_posterior_mean_for_all_events",
        "sensitivity_path_estimators": [
            "emission_only_posterior_mean",
            "exact_sparse_momentum_posterior_mean",
        ] if include_momentum_posterior else ["emission_only_posterior_mean"],
        "rescore_tolerance": float(rescore_tolerance),
        "scoring_package_file": str(Path(hipporeplayimm.__file__).resolve()),
        "scoring_source_git": git_metadata(Path(hipporeplayimm.__file__).resolve().parents[2]),
        "outputs": {name: str(path) for name, path in paths.items()},
        "provenance": provenance,
    }
    manifest_path = output / MANIFEST_OUTPUT
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths[MANIFEST_OUTPUT] = manifest_path
    overall = bool(gates.loc[gates["gate"].eq("overall"), "passed"].iloc[0])
    summary = [
        "# Replay commitment/composition posterior export",
        "",
        f"- Events: {len(events)}",
        f"- Posterior bins: {len(bins)}",
        f"- Posterior transitions: {len(transitions)}",
        f"- Exact-sparse momentum posterior included: {bool(include_momentum_posterior)}",
        f"- Overall technical gate: {'PASS' if overall else 'FAIL'}",
        "",
        "Primary comparisons use one common first-order-IMM posterior estimator for every event. Stationary and fragmented/jump MAP phases are explicitly excluded from composition bouts. Emission-only and exact-sparse momentum paths are sensitivity estimators.",
    ]
    summary_path = output / SUMMARY_OUTPUT
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    paths[SUMMARY_OUTPUT] = summary_path
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--frozen-events", required=True)
    parser.add_argument("--event-evidence", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skip-momentum-posterior", action="store_true")
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--rescore-tolerance", type=float, default=0.025)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_analysis(
        dataset_root=args.dataset_root,
        frozen_events_csv=args.frozen_events,
        event_evidence_csv=args.event_evidence,
        output_dir=args.output_dir,
        include_momentum_posterior=not args.skip_momentum_posterior,
        max_events=args.max_events,
        rescore_tolerance=args.rescore_tolerance,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
