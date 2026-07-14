#!/usr/bin/env python3
"""Generate opt-in accuracy-upgrade diagnostics for one replay session."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hipporeplayimm.accuracy_upgrades import (
    ContinuousTimeEmissionConfig,
    behavioral_well_context,
    build_continuous_time_emissions,
    fit_empirical_transition_matrix,
    leave_one_rat_splits,
    robust_position_filter,
    summarize_tetrode_mark_partitions,
    ValidStateConfig,
    valid_grid_graph_transition,
    valid_state_mask_from_encoding,
)
from hipporeplayimm.data import load_replay_session
from hipporeplayimm.encoding import EncodingConfig, fit_place_field_encoding


def _session_path(root: str | Path, session: str) -> Path:
    parts = session.replace("\\", "/").split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("session must have the form 'RatN/OpenM', e.g. 'Rat1/Open1'")
    return Path(root) / parts[0] / parts[1]


def _event_ids(spec: str, session) -> list[int]:
    text = spec.strip().lower()
    if text == "run":
        return [int(x) for x in session.ripple_indices_in_run()]
    if text == "all":
        return list(range(session.ripple_count))
    tokens = _event_tokens(text)
    out: list[int] = []
    for token in tokens:
        out.extend(_event_id_token_values(token))
    return sorted(dict.fromkeys(out))


def _event_tokens(text: str) -> list[str]:
    if not text:
        raise ValueError("--events must contain 'run', 'all', or at least one event id")
    comma_parts = text.split(",")
    if any(not part.strip() for part in comma_parts):
        raise ValueError("--events must not contain empty comma-separated entries")
    tokens: list[str] = []
    for part in comma_parts:
        tokens.extend(part.split())
    if not tokens:
        raise ValueError("--events must contain 'run', 'all', or at least one event id")
    return tokens


def _event_id_token_values(token: str) -> list[int]:
    if "-" not in token:
        return [_event_index(token)]
    if token.count("-") != 1:
        raise ValueError("--events ranges must have the form START-END")
    lo_text, hi_text = token.split("-", 1)
    if not lo_text or not hi_text:
        raise ValueError("--events ranges must have the form START-END")
    lo = _event_index(lo_text)
    hi = _event_index(hi_text)
    if hi < lo:
        raise ValueError("--events ranges must be increasing")
    return list(range(lo, hi + 1))


def _event_index(token: str) -> int:
    try:
        value = int(token)
    except ValueError as exc:
        raise ValueError("--events entries must be non-negative integers or ranges") from exc
    if value < 0:
        raise ValueError("--events entries must be non-negative")
    return value


def _nonnegative_int(value: str) -> int:
    """Return a nonnegative CLI integer instead of allowing negative slicing."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--events", default="run")
    parser.add_argument("--max-events", type=_nonnegative_int, default=25)
    parser.add_argument("--output", default="results/accuracy-upgrade-report")
    parser.add_argument("--bin-size-cm", type=float, default=6.0)
    parser.add_argument("--smoothing-sigma-bins", type=float, default=2.0)
    parser.add_argument("--min-speed-cm-s", type=float, default=5.0)
    parser.add_argument("--valid-min-occupancy-s", type=float, default=0.02)
    parser.add_argument("--continuous-spike-rate-scale", type=float, default=1.0)
    args = parser.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    session = load_replay_session(_session_path(args.dataset_root, args.session))
    encoding = fit_place_field_encoding(
        session,
        EncodingConfig(
            bin_size_cm=args.bin_size_cm,
            smoothing_sigma_bins=args.smoothing_sigma_bins,
            min_speed_cm_s=args.min_speed_cm_s,
        ),
    )
    try:
        event_ids = _event_ids(args.events, session)
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_events is not None:
        event_ids = event_ids[: args.max_events]

    valid_mask = valid_state_mask_from_encoding(
        encoding,
        ValidStateConfig(min_occupancy_s=args.valid_min_occupancy_s),
    )
    valid_summary = pd.DataFrame(
        [
            {
                "session": session.session_id,
                "spatial_bins": int(encoding.n_bins),
                "valid_state_bins": int(valid_mask.sum()),
                "valid_state_fraction": float(valid_mask.mean()),
                "min_occupancy_s": float(args.valid_min_occupancy_s),
                "median_valid_occupancy_s": float(np.median(encoding.occupancy_s[valid_mask])),
            }
        ]
    )
    valid_summary.to_csv(outdir / "valid_state_summary.csv", index=False)

    empirical_transition = fit_empirical_transition_matrix(
        session,
        encoding,
        min_speed_cm_s=args.min_speed_cm_s,
        valid_mask=valid_mask,
    )
    graph_transition = valid_grid_graph_transition(encoding.grid_shape, valid_mask)
    pd.DataFrame(
        [
            {
                "session": session.session_id,
                "empirical_transition_nonzeros": int(empirical_transition.nnz),
                "empirical_transition_density": float(empirical_transition.nnz / np.prod(empirical_transition.shape)),
                "valid_graph_transition_nonzeros": int(graph_transition.nnz),
                "valid_graph_transition_density": float(graph_transition.nnz / np.prod(graph_transition.shape)),
            }
        ]
    ).to_csv(outdir / "transition_prior_summary.csv", index=False)

    continuous_rows = []
    context_rows = []
    for event_index in event_ids:
        emissions = build_continuous_time_emissions(
            session,
            encoding,
            int(event_index),
            ContinuousTimeEmissionConfig(spike_rate_scale=args.continuous_spike_rate_scale),
        )
        continuous_rows.append(
            {
                "session": session.session_id,
                "event_index": int(event_index),
                "continuous_time_intervals": int(emissions.n_time),
                "spikes": int(emissions.n_spikes),
                "median_interval_s": float(emissions.dt),
            }
        )
        context_rows.append(behavioral_well_context(session, int(event_index)))
    pd.DataFrame(continuous_rows).to_csv(outdir / "continuous_time_emission_summary.csv", index=False)
    pd.DataFrame(context_rows).to_csv(outdir / "behavioral_context_labels.csv", index=False)

    _, position_quality = robust_position_filter(session.position)
    position_quality.to_csv(outdir / "position_quality_flags.csv", index=False)
    pd.DataFrame(
        [
            {
                "session": session.session_id,
                "position_frames": int(len(position_quality)),
                "high_speed_flag_fraction": float(position_quality["position_high_speed_flag"].mean()) if not position_quality.empty else np.nan,
            }
        ]
    ).to_csv(outdir / "position_quality_summary.csv", index=False)

    marks = getattr(session, "spike_marks", None)
    if marks is not None:
        summarize_tetrode_mark_partitions(marks).to_csv(outdir / "tetrode_mark_partitions.csv", index=False)
    else:
        pd.DataFrame(columns=["tetrode_id", "features", "n_features"]).to_csv(
            outdir / "tetrode_mark_partitions.csv",
            index=False,
        )

    pd.DataFrame(leave_one_rat_splits([session.session_id])).to_csv(outdir / "leave_one_rat_split_example.csv", index=False)

    print(f"Wrote accuracy-upgrade diagnostics to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
