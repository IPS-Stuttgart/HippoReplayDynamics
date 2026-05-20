#!/usr/bin/env python3
"""Estimate exact-vs-pruned evidence gaps on a selected event."""

from __future__ import annotations

import argparse
from pathlib import Path


from hipporeplayimm.candidate_pruning_calibration import score_pruning_gaps
from hipporeplayimm.data import load_replay_session
from hipporeplayimm.encoding import EmissionConfig, EncodingConfig, build_emissions, fit_place_field_encoding
from hipporeplayimm.sorted_spike_state_space import SortedSpikeStateSpaceReplayModel
from hipporeplayimm.state_space import StateSpaceDecoderConfig


def _session_path(root: str | Path, session: str) -> Path:
    rat, open_field = session.replace("\\", "/").split("/", 1)
    return Path(root) / rat / open_field


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--event-index", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", default="momentum,imm")
    parser.add_argument("--candidate-top-k", type=int, default=128)
    parser.add_argument("--predicted-candidate-top-k", type=int, default=8)
    parser.add_argument("--time-bin-s", type=float, default=0.003)
    parser.add_argument("--bin-size-cm", type=float, default=6.0)
    parser.add_argument("--smoothing-sigma-bins", type=float, default=2.0)
    parser.add_argument("--min-speed-cm-s", type=float, default=5.0)
    parser.add_argument("--diffusion-sigma-cm-sqrt-s", type=float, default=85.0)
    parser.add_argument("--momentum-sigma-cm-sqrt-s", type=float, default=85.0)
    parser.add_argument("--momentum-initial-sigma-cm-sqrt-s", type=float, default=85.0)
    parser.add_argument("--momentum-velocity-decay", type=float, default=0.95)
    args = parser.parse_args()

    session = load_replay_session(_session_path(args.dataset_root, args.session))
    encoding = fit_place_field_encoding(
        session,
        EncodingConfig(
            bin_size_cm=args.bin_size_cm,
            smoothing_sigma_bins=args.smoothing_sigma_bins,
            min_speed_cm_s=args.min_speed_cm_s,
        ),
    )
    emissions = build_emissions(session, encoding, args.event_index, EmissionConfig(time_bin_s=args.time_bin_s))
    models = []
    for mode in [item.strip() for item in args.models.replace(" ", ",").split(",") if item.strip()]:
        config = StateSpaceDecoderConfig(
            mode=mode,
            diffusion_sigma_cm_sqrt_s=args.diffusion_sigma_cm_sqrt_s,
            momentum_sigma_cm_sqrt_s=args.momentum_sigma_cm_sqrt_s,
            momentum_initial_sigma_cm_sqrt_s=args.momentum_initial_sigma_cm_sqrt_s,
            momentum_velocity_decay=args.momentum_velocity_decay,
            momentum_candidate_top_k=args.candidate_top_k,
            momentum_predicted_candidate_top_k=args.predicted_candidate_top_k,
        )
        models.append(SortedSpikeStateSpaceReplayModel(mode=mode, config=config))
    rows = score_pruning_gaps(models, emissions, encoding.bin_centers)
    rows.insert(0, "session", session.session_id)
    rows.insert(1, "event_index", int(args.event_index))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output, index=False)
    print(rows.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
