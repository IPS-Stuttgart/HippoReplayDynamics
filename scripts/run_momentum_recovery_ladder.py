#!/usr/bin/env python3
"""Run the momentum-recovery ladder on one Pfeiffer/Foster session."""

from __future__ import annotations

import argparse

from hipporeplayimm.encoding import EncodingConfig
from hipporeplayimm.momentum_recovery_ladder import (
    default_ladder_tiers,
    run_momentum_recovery_ladder,
)
from hipporeplayimm.position_validation import (
    VALIDATED_POSITION_BIN_SIZE_CM,
    VALIDATED_POSITION_MIN_SPEED_CM_S,
    VALIDATED_POSITION_SMOOTHING_SIGMA_BINS,
)
from hipporeplayimm.simulation_recovery import SimulationRecoveryConfig
from hipporeplayimm.state_space import StateSpaceDecoderConfig


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a four-tier synthetic momentum-recovery ladder: full-grid pairwise, "
            "exact finite-velocity/displacement, oracle candidate support, and native support."
        )
    )
    parser.add_argument("--dataset-root", required=True, help="Root containing DataSetFromPfeifferFoster.")
    parser.add_argument("--session", required=True, help="Session such as Rat1/Open1.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--events", default="run:0-25", help="Template event selector.")
    parser.add_argument("--max-template-events", type=int, default=25)
    parser.add_argument("--events-per-model", type=int, default=25)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--time-bin-s", type=float, default=0.003)
    parser.add_argument("--spike-rate-scale", type=float, default=1.0)
    parser.add_argument("--likelihood-temperature", type=float, default=1.0)
    parser.add_argument("--negative-binomial-overdispersion", type=float, default=0.0)
    parser.add_argument("--continue-on-error", action="store_true")

    parser.add_argument("--bin-size-cm", type=float, default=VALIDATED_POSITION_BIN_SIZE_CM)
    parser.add_argument("--smoothing-sigma-bins", type=float, default=VALIDATED_POSITION_SMOOTHING_SIGMA_BINS)
    parser.add_argument("--min-speed-cm-s", type=float, default=VALIDATED_POSITION_MIN_SPEED_CM_S)
    parser.add_argument("--min-occupancy-s", type=float, default=0.01)
    parser.add_argument("--rate-floor-hz", type=float, default=1e-4)

    parser.add_argument("--state-space-diffusion-sigma-cm-sqrt-s", type=float, default=85.0)
    parser.add_argument("--state-space-momentum-sigma-cm-sqrt-s", type=float, default=85.0)
    parser.add_argument("--state-space-momentum-initial-sigma-cm-sqrt-s", type=float, default=85.0)
    parser.add_argument("--state-space-momentum-velocity-decay", type=float, default=0.95)
    parser.add_argument("--state-space-momentum-velocity-decay-tau-s", type=float, default=0.0)
    parser.add_argument("--state-space-max-step-sigma", type=float, default=4.0)
    parser.add_argument("--state-space-imm-mode-stickiness", type=float, default=0.95)

    parser.add_argument("--native-candidate-top-k", type=int, default=128)
    parser.add_argument("--native-predicted-candidate-top-k", type=int, default=8)
    parser.add_argument("--finite-displacement-radius-bins", type=int, default=2)
    parser.add_argument("--displacement-position-sigma-cm", type=float, default=0.0)
    parser.add_argument("--displacement-transition-sigma-cm-sqrt-s", type=float, default=0.0)
    parser.add_argument("--displacement-prior-sigma-cm", type=float, default=0.0)
    parser.add_argument("--score-with-occupancy", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    encoding = EncodingConfig(
        bin_size_cm=float(args.bin_size_cm),
        smoothing_sigma_bins=float(args.smoothing_sigma_bins),
        min_speed_cm_s=float(args.min_speed_cm_s),
        min_occupancy_s=float(args.min_occupancy_s),
        rate_floor_hz=float(args.rate_floor_hz),
    )
    state_space = StateSpaceDecoderConfig(
        diffusion_sigma_cm_sqrt_s=float(args.state_space_diffusion_sigma_cm_sqrt_s),
        momentum_sigma_cm_sqrt_s=float(args.state_space_momentum_sigma_cm_sqrt_s),
        momentum_initial_sigma_cm_sqrt_s=float(args.state_space_momentum_initial_sigma_cm_sqrt_s),
        momentum_velocity_decay=float(args.state_space_momentum_velocity_decay),
        momentum_velocity_decay_tau_s=float(args.state_space_momentum_velocity_decay_tau_s),
        max_step_sigma=float(args.state_space_max_step_sigma),
        imm_mode_stickiness=float(args.state_space_imm_mode_stickiness),
        displacement_radius_bins=int(args.finite_displacement_radius_bins),
        displacement_position_sigma_cm=float(args.displacement_position_sigma_cm),
        displacement_transition_sigma_cm_sqrt_s=float(args.displacement_transition_sigma_cm_sqrt_s),
        displacement_prior_sigma_cm=float(args.displacement_prior_sigma_cm),
    )
    base = SimulationRecoveryConfig(
        true_models=("momentum",),
        events=str(args.events),
        max_template_events=args.max_template_events,
        events_per_model=int(args.events_per_model),
        random_seed=int(args.random_seed),
        time_bin_s=float(args.time_bin_s),
        spike_rate_scale=float(args.spike_rate_scale),
        likelihood_temperature=float(args.likelihood_temperature),
        negative_binomial_overdispersion=float(args.negative_binomial_overdispersion),
        encoding=encoding,
        state_space=state_space,
        true_state_space=state_space,
        score_with_occupancy=bool(args.score_with_occupancy),
        continue_on_error=bool(args.continue_on_error),
    )
    tiers = default_ladder_tiers(
        state_space,
        native_candidate_top_k=int(args.native_candidate_top_k),
        native_predicted_candidate_top_k=int(args.native_predicted_candidate_top_k),
        finite_displacement_radius_bins=int(args.finite_displacement_radius_bins),
    )
    result = run_momentum_recovery_ladder(
        args.dataset_root,
        args.session,
        base,
        tiers=tiers,
        output=args.output,
    )
    result.write(args.output)
    if result.tier_summary.empty:
        print("No ladder event-recovery rows were produced.")
    else:
        print(result.tier_summary.to_string(index=False))
    if not result.interpretation.empty:
        print("\n" + result.interpretation.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
