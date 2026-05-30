#!/usr/bin/env python3
"""Score replay events with a wrong-environment place-field map.

This script is a diagnostic control.  It uses spikes and ripple windows from the
``--event-session`` but fits the spatial map from ``--map-session``.  It is most
interpretable when cell IDs are stable across the two sessions, for example
Rat1/Open1 events scored under Rat1/Open2 place fields.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_model_evidence import _events, _models
from hipporeplayimm.data import load_replay_session
from hipporeplayimm.encoding import EmissionConfig, EncodingConfig, build_emissions, fit_place_field_encoding
from hipporeplayimm.models import CandidateKinematicModel


def _session_path(root: str | Path, session: str) -> Path:
    return Path(root) / session.replace("\\", "/")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--event-session", required=True, help="Session containing replay events")
    parser.add_argument("--map-session", required=True, help="Session used to fit the place-field map")
    parser.add_argument("--events", default="run")
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--models", default="random stationary sorted-spike-state-space-stationary sorted-spike-state-space-diffusion")
    parser.add_argument("--candidate-top-k", type=int, default=64)
    parser.add_argument("--stationary-sigma-cm", type=float, default=2.0)
    parser.add_argument("--diffusion-sigma-cm", type=float, default=12.0)
    parser.add_argument("--momentum-sigma-cm", type=float, default=12.0)
    parser.add_argument("--velocity-decay", type=float, default=0.95)
    parser.add_argument("--mode-stickiness", type=float, default=0.94)
    parser.add_argument("--state-space-stationary-sigma-cm", type=float, default=2.0)
    parser.add_argument("--state-space-diffusion-sigma-cm-sqrt-s", type=float, default=85.0)
    parser.add_argument("--state-space-max-step-sigma", type=float, default=4.0)
    parser.add_argument("--state-space-valid-occupancy-threshold-s", type=float, default=0.0)
    parser.add_argument("--state-space-imm-mode-stickiness", type=float, default=0.95)
    parser.add_argument("--state-space-momentum-sigma-cm-sqrt-s", type=float, default=85.0)
    parser.add_argument("--state-space-momentum-initial-sigma-cm-sqrt-s", type=float, default=85.0)
    parser.add_argument("--state-space-momentum-velocity-decay", type=float, default=0.95)
    parser.add_argument("--state-space-momentum-candidate-top-k", type=int, default=128)
    parser.add_argument("--state-space-momentum-predicted-candidate-top-k", type=int, default=8)
    parser.add_argument("--state-space-momentum-candidate-source", default="emission")
    parser.add_argument("--clusterless-mark-smoothing-sigma-bins", type=float, default=1.0)
    parser.add_argument("--clusterless-mark-prior-count", type=float, default=1.0)
    parser.add_argument("--clusterless-mark-variance-floor", type=float, default=1.0)
    parser.add_argument("--clusterless-rate-floor-hz", type=float, default=1e-4)
    parser.add_argument("--clusterless-mark-likelihood", default="local-kde")
    parser.add_argument("--time-bin-s", type=float, default=0.003)
    parser.add_argument("--spike-rate-scale", type=float, default=1.0)
    parser.add_argument("--emission-likelihood-temperature", type=float, default=1.0)
    parser.add_argument("--emission-negative-binomial-overdispersion", type=float, default=0.0)
    parser.add_argument("--bin-size-cm", type=float, default=6.0)
    parser.add_argument("--smoothing-sigma-bins", type=float, default=2.0)
    parser.add_argument("--min-speed-cm-s", type=float, default=5.0)
    parser.add_argument("--output", default="results/wrong-map-controls")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    event_session = load_replay_session(_session_path(args.dataset_root, args.event_session))
    map_session = load_replay_session(_session_path(args.dataset_root, args.map_session))
    event_ids = _events(args.events, event_session)
    if args.max_events is not None:
        event_ids = event_ids[: args.max_events]

    encoding = fit_place_field_encoding(
        map_session,
        EncodingConfig(
            bin_size_cm=args.bin_size_cm,
            smoothing_sigma_bins=args.smoothing_sigma_bins,
            min_speed_cm_s=args.min_speed_cm_s,
        ),
    )
    models = _models(args)
    emissions_cfg = EmissionConfig(
        time_bin_s=args.time_bin_s,
        spike_rate_scale=args.spike_rate_scale,
        likelihood_temperature=args.emission_likelihood_temperature,
        negative_binomial_overdispersion=args.emission_negative_binomial_overdispersion,
    )
    rows = []
    for event_id in event_ids:
        emissions = build_emissions(event_session, encoding, int(event_id), emissions_cfg)
        if emissions.n_time == 0:
            continue
        for requested_model, model in models.items():
            start = time.perf_counter()
            try:
                if isinstance(model, CandidateKinematicModel):
                    candidates = model.candidate_indices(emissions)
                    score = model.score(emissions, encoding.bin_centers, candidate_indices=candidates)
                else:
                    score = model.score(emissions, encoding.bin_centers)
                rows.append(
                    {
                        "status": "success",
                        "session": event_session.session_id,
                        "event_index": int(event_id),
                        "map_session": map_session.session_id,
                        "model": score.model_name,
                        "requested_model": requested_model,
                        "log_evidence": float(score.log_likelihood),
                        "n_time": int(score.n_time),
                        "n_spikes": int(score.n_spikes),
                        "runtime_s": float(time.perf_counter() - start),
                        "wrong_map_control": True,
                        "error": "",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "status": "failure",
                        "session": event_session.session_id,
                        "event_index": int(event_id),
                        "map_session": map_session.session_id,
                        "model": requested_model,
                        "requested_model": requested_model,
                        "log_evidence": np.nan,
                        "n_time": int(emissions.n_time),
                        "n_spikes": int(emissions.n_spikes),
                        "runtime_s": float(time.perf_counter() - start),
                        "wrong_map_control": True,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if not args.continue_on_error:
                    raise
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "event_model_evidence.csv", index=False)
    print(frame.head(20).to_string(index=False))
    print(f"Rows: {len(frame)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
