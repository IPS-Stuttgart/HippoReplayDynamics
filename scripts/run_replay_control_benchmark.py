#!/usr/bin/env python3
"""Run opt-in replay evidence controls without changing the main benchmark CLI."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from hipporeplayimm.data import load_replay_session
from hipporeplayimm.empirical_transition import fit_empirical_transition_model
from hipporeplayimm.encoding import EmissionConfig, EncodingConfig, build_emissions, fit_place_field_encoding
from hipporeplayimm.ground_truth import infer_well_locations
from hipporeplayimm.models import CandidateKinematicModel, RandomModel, StationaryModel
from hipporeplayimm.replay_emission_calibration import apply_replay_cell_gains, fit_replay_cell_gains
from hipporeplayimm.reverse_models import BidirectionalReplayModel, ReverseTimeReplayModel
from hipporeplayimm.shuffle_controls import ShuffleControlConfig, score_shuffle_controls
from hipporeplayimm.sorted_spike_state_space import SortedSpikeStateSpaceReplayModel
from hipporeplayimm.state_space import StateSpaceDecoderConfig
from hipporeplayimm.well_route_state_space import WellRouteStateSpaceReplayModel, routes_from_wells


def _session_path(root: str | Path, session: str) -> Path:
    rat, open_field = session.replace("\\", "/").split("/", 1)
    return Path(root) / rat / open_field


def _parse_events(spec: str, session) -> list[int]:
    text = spec.strip().lower()
    if text == "run":
        return [int(x) for x in session.ripple_indices_in_run()]
    if text == "all":
        return list(range(session.ripple_count))
    if text.startswith("run:"):
        run_events = [int(x) for x in session.ripple_indices_in_run()]
        ordinals = _parse_int_ranges(text.split(":", 1)[1])
        return [run_events[index] for index in ordinals if 0 <= index < len(run_events)]
    return _parse_int_ranges(text)


def _parse_int_ranges(spec: str) -> list[int]:
    out = []
    for item in spec.replace(" ", ",").split(","):
        if not item:
            continue
        if "-" in item:
            lo, hi = [int(x) for x in item.split("-", 1)]
            if lo > hi:
                raise ValueError(f"event range must be ascending: {item!r}")
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(item))
    return sorted(dict.fromkeys(out))


def _build_model(name: str, session, encoding, args):
    if name.endswith("-reverse"):
        return ReverseTimeReplayModel(_build_model(name[: -len("-reverse")], session, encoding, args), name=name)
    if name.startswith("bidirectional-"):
        base_name = name[len("bidirectional-") :]
        return BidirectionalReplayModel(_build_model(base_name, session, encoding, args), name=name)
    if name == "random":
        return RandomModel()
    if name == "stationary":
        return StationaryModel()
    if name in {"diffusion", "momentum", "imm"}:
        return CandidateKinematicModel(mode=name, top_k=args.candidate_top_k, name=name)
    if name.startswith("sorted-spike-state-space-"):
        mode = name.removeprefix("sorted-spike-state-space-")
        if mode == "empirical-transition":
            return fit_empirical_transition_model(session, encoding)
        if mode == "route":
            wells = infer_well_locations(session)
            routes = None
            if not wells.empty:
                routes = routes_from_wells(wells[["well_x", "well_y"]].to_numpy(float))
            return WellRouteStateSpaceReplayModel(candidate_routes=routes)
        config = StateSpaceDecoderConfig(
            mode=mode,
            diffusion_sigma_cm_sqrt_s=args.state_space_sigma_cm_sqrt_s,
            momentum_sigma_cm_sqrt_s=args.state_space_sigma_cm_sqrt_s,
            momentum_initial_sigma_cm_sqrt_s=args.state_space_sigma_cm_sqrt_s,
            momentum_candidate_top_k=args.state_space_candidate_top_k,
            momentum_predicted_candidate_top_k=args.state_space_predicted_candidate_top_k,
        )
        return SortedSpikeStateSpaceReplayModel(mode=mode, config=config, name=name)
    raise ValueError(f"Unknown model {name!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--events", default="run:0-25")
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", default="random stationary sorted-spike-state-space-diffusion sorted-spike-state-space-empirical-transition sorted-spike-state-space-route")
    parser.add_argument("--candidate-top-k", type=int, default=64)
    parser.add_argument("--state-space-candidate-top-k", type=int, default=128)
    parser.add_argument("--state-space-predicted-candidate-top-k", type=int, default=8)
    parser.add_argument("--state-space-sigma-cm-sqrt-s", type=float, default=85.0)
    parser.add_argument("--time-bin-s", type=float, default=0.003)
    parser.add_argument("--bin-size-cm", type=float, default=6.0)
    parser.add_argument("--smoothing-sigma-bins", type=float, default=2.0)
    parser.add_argument("--min-speed-cm-s", type=float, default=5.0)
    parser.add_argument("--replay-calibrate-emissions", action="store_true")
    parser.add_argument("--calibration-events", default="run")
    parser.add_argument("--shuffle-controls", type=int, default=0)
    parser.add_argument("--shuffle-mode", default="spatial-roll")
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--hyperparameter-source", default="diagnostic-control-script")
    parser.add_argument("--selection-dataset", default="not-selected")
    parser.add_argument("--selection-metric", default="not-selected")
    args = parser.parse_args()

    session = load_replay_session(_session_path(args.dataset_root, args.session))
    event_ids = _parse_events(args.events, session)
    encoding = fit_place_field_encoding(
        session,
        EncodingConfig(
            bin_size_cm=args.bin_size_cm,
            smoothing_sigma_bins=args.smoothing_sigma_bins,
            min_speed_cm_s=args.min_speed_cm_s,
        ),
    )
    emission_config = EmissionConfig(time_bin_s=args.time_bin_s)
    calibration_metadata = {}
    if args.replay_calibrate_emissions:
        calibration = fit_replay_cell_gains(session, encoding, _parse_events(args.calibration_events, session), emission_config)
        encoding = apply_replay_cell_gains(encoding, calibration)
        calibration_metadata = calibration.as_metadata()
    model_names = [item.strip() for item in args.models.replace(",", " ").split() if item.strip()]
    models = {name: _build_model(name, session, encoding, args) for name in model_names}

    rows = []
    for event_index in event_ids:
        emissions = build_emissions(session, encoding, int(event_index), emission_config)
        for requested_name, model in models.items():
            start = time.perf_counter()
            score = model.score(emissions, encoding.bin_centers)
            row = {
                "status": "success",
                "session": session.session_id,
                "event_index": int(event_index),
                "requested_model": requested_name,
                "model": score.model_name,
                "log_evidence": float(score.log_likelihood),
                "n_time": int(score.n_time),
                "n_spikes": int(score.n_spikes),
                "runtime_s": float(time.perf_counter() - start),
                "hyperparameter_source": args.hyperparameter_source,
                "selection_dataset": args.selection_dataset,
                "selection_metric": args.selection_metric,
                **calibration_metadata,
            }
            row.update({f"diagnostic_{key}": value for key, value in score.diagnostics.items()})
            rows.append(row)
    scores = pd.DataFrame(rows)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / "event_model_evidence_controls.csv", index=False)
    if args.shuffle_controls > 0:
        controls = score_shuffle_controls(
            session,
            encoding,
            event_ids,
            models,
            emission_config,
            ShuffleControlConfig(mode=args.shuffle_mode, n_shuffles=args.shuffle_controls, random_seed=args.random_seed),
        )
        controls.to_csv(output / "shuffle_control_event_model_evidence.csv", index=False)
    print(scores.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
