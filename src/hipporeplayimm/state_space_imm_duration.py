"""Duration-aware four-mode state-space IMM patch."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .duration_dynamics import (
    _decays,
    _ps,
    _pss,
    _rep,
    _scales,
    attach_duration_metadata,
    transition_durations_s,
)


def apply_state_space_imm_duration_patch() -> None:
    """Patch four-mode state-space IMM to use per-transition durations."""

    import hipporeplayimm.state_space as ss

    if getattr(ss, "_state_space_imm_duration_patch_applied", False):
        return

    previous_score = ss.StateSpaceReplayModel.score

    def score(self, emissions, bin_centers, candidate_indices=None):
        centers = _as_xy_bin_centers(bin_centers)
        if self.mode != "imm":
            derived_candidates = None
            if self.mode == "momentum" and candidate_indices is None:
                derived_candidates = self.candidate_indices(emissions, centers)
                candidate_indices = derived_candidates
            event_score = previous_score(self, emissions, centers, candidate_indices)
            if self.mode == "momentum":
                candidates = derived_candidates if derived_candidates is not None else candidate_indices
                if candidates is not None:
                    event_score.diagnostics.setdefault(
                        "mean_candidate_count",
                        float(np.mean([len(curr) for curr in candidates])),
                    )
                event_score.diagnostics.setdefault(
                    "state_space_momentum_predicted_candidate_top_k",
                    int(self.config.momentum_predicted_candidate_top_k),
                )
                if derived_candidates is not None:
                    event_score.diagnostics["state_space_momentum_candidate_support"] = "derived"
            return event_score
        if emissions.n_time == 0:
            raise ValueError("emissions must contain at least one time bin")
        if emissions.n_bins != centers.shape[0]:
            raise ValueError("emissions.n_bins must match bin_centers rows")
        assert self.config is not None

        durations = transition_durations_s(emissions)
        attach_duration_metadata(emissions)
        candidates = self.candidate_indices(emissions, centers) if candidate_indices is None else candidate_indices
        ss._validate_candidate_indices(candidates, emissions.n_time, emissions.n_bins)

        diffusion_sigmas = _pss(self.config.diffusion_sigma_cm_sqrt_s, durations, float(emissions.dt))
        transition_sigma_cm = _rep(self.config.diffusion_sigma_cm_sqrt_s, durations, float(emissions.dt))
        momentum_sigmas = _pss(self.config.momentum_sigma_cm_sqrt_s, durations, float(emissions.dt))
        momentum_transition_sigma_cm = _rep(self.config.momentum_sigma_cm_sqrt_s, durations, float(emissions.dt))
        initial_sigma = _ps(
            self.config.momentum_initial_sigma_cm_sqrt_s,
            durations[0] if len(durations) else float(emissions.dt),
        )
        decays = _decays(self.config.momentum_velocity_decay, durations, float(emissions.dt))
        scales = _scales(durations)

        logp, trajectory, mode_post, masses = _score_imm_duration(
            ss,
            emissions,
            centers,
            candidates,
            stationary_sigma_cm=self.config.stationary_sigma_cm,
            diffusion_sigmas_cm=diffusion_sigmas,
            momentum_sigmas_cm=momentum_sigmas,
            initial_momentum_sigma_cm=initial_sigma,
            velocity_decays=decays,
            time_scales=scales,
            mode_stickiness=self.config.imm_mode_stickiness,
        )

        names = ("stationary", "diffusion", "momentum", "jump")
        extra = {
            f"state_space_mode_{name}_terminal_probability": float(mode_post[-1, idx])
            for idx, name in enumerate(names)
        }
        extra.update(
            {
                "mean_candidate_log_mass": float(np.mean(masses)),
                "mean_candidate_count": float(np.mean([len(curr) for curr in candidates])),
                "state_space_imm_modes": ",".join(names),
                "state_space_imm_candidate_top_k": int(self.config.momentum_candidate_top_k),
                "state_space_imm_predicted_candidate_top_k": int(self.config.momentum_predicted_candidate_top_k),
                "state_space_imm_candidate_support": "derived" if candidate_indices is None else "provided",
                "state_space_imm_trajectory_posterior": "smoothed_pair_marginal",
                "state_space_imm_evidence_support": "truncated_full_grid",
                "state_space_momentum_transition_sigma_cm": float(momentum_transition_sigma_cm),
                "state_space_momentum_initial_transition_sigma_cm": float(initial_sigma),
            }
        )

        terminal = trajectory[-1]
        diagnostics = {
            "state_space_mode": str(self.mode),
            "state_space_time_bin_s": float(emissions.dt),
            "state_space_transition_durations": ",".join(f"{duration:.12g}" for duration in durations),
            "state_space_trajectory_posterior": 1,
            "state_space_trajectory_time_bins": int(emissions.n_time),
            "state_space_stationary_sigma_cm": float(self.config.stationary_sigma_cm),
            "state_space_diffusion_sigma_cm_sqrt_s": float(self.config.diffusion_sigma_cm_sqrt_s),
            "state_space_max_step_sigma": float(self.config.max_step_sigma),
            "state_space_imm_mode_stickiness": float(self.config.imm_mode_stickiness),
            "state_space_momentum_sigma_cm_sqrt_s": float(self.config.momentum_sigma_cm_sqrt_s),
            "state_space_momentum_initial_sigma_cm_sqrt_s": float(self.config.momentum_initial_sigma_cm_sqrt_s),
            "state_space_momentum_velocity_decay": float(self.config.momentum_velocity_decay),
            "state_space_transition_sigma_cm": float(transition_sigma_cm),
            "mean_trajectory_posterior_entropy": ss._mean_entropy(trajectory),
            **extra,
        }
        diagnostics.update(ss._posterior_diagnostics(terminal, centers))
        return ss.EventScore(
            str(self.name),
            float(logp),
            emissions.n_time,
            emissions.n_spikes,
            diagnostics=diagnostics,
            terminal_log_posterior=terminal,
            trajectory_log_posterior=trajectory,
        )

    ss.StateSpaceReplayModel.__imm_duration_previous_score__ = previous_score
    ss.StateSpaceReplayModel.score = score
    ss._state_space_imm_duration_patch_applied = True


def _score_imm_duration(
    ss,
    emissions,
    bin_centers,
    candidates,
    *,
    stationary_sigma_cm,
    diffusion_sigmas_cm,
    momentum_sigmas_cm,
    initial_momentum_sigma_cm,
    velocity_decays,
    time_scales,
    mode_stickiness,
):
    """Candidate-pruned four-mode IMM with duration-dependent transitions."""

    modes = ("stationary", "diffusion", "momentum", "jump")
    if emissions.n_time == 1:
        logp, trajectory = ss._score_fragmented(emissions)
        mode_posterior = np.full((1, len(modes)), 1.0 / len(modes), dtype=float)
        return logp, trajectory, mode_posterior, [0.0]

    masses = ss._candidate_log_masses(emissions.log_likelihood, candidates)
    mode_transition = ss._mode_transition_matrix(len(modes), mode_stickiness)
    with np.errstate(divide="ignore"):
        log_mode_transition = np.log(mode_transition)

    by_mode = [
        ss._init_imm_pair_log_alpha(
            emissions.log_likelihood,
            candidates[0],
            candidates[1],
            bin_centers,
            mode=mode,
            stationary_sigma_cm=stationary_sigma_cm,
            diffusion_sigma_cm=float(diffusion_sigmas_cm[0]),
            momentum_initial_sigma_cm=float(initial_momentum_sigma_cm),
        )
        for mode in modes
    ]
    log_pair = np.stack(by_mode, axis=0) - np.log(len(modes))
    pair_alphas = [log_pair]

    for time_index in range(2, emissions.n_time):
        transition_index = time_index - 1
        next_alpha = []
        for dst_mode_index, dst_mode in enumerate(modes):
            mixed_prev = logsumexp(
                log_pair + log_mode_transition[:, dst_mode_index][:, None, None],
                axis=0,
            )
            next_alpha.append(
                ss._advance_imm_pair_log_alpha(
                    mixed_prev,
                    candidates[time_index - 2],
                    candidates[time_index - 1],
                    candidates[time_index],
                    emissions.log_likelihood[time_index, candidates[time_index]],
                    bin_centers,
                    mode=dst_mode,
                    stationary_sigma_cm=stationary_sigma_cm,
                    diffusion_sigma_cm=float(diffusion_sigmas_cm[transition_index]),
                    momentum_sigma_cm=float(momentum_sigmas_cm[transition_index]),
                    velocity_decay=float(velocity_decays[transition_index]) * float(time_scales[transition_index]),
                )
            )
        log_pair = np.stack(next_alpha, axis=0)
        pair_alphas.append(log_pair)

    logp = float(logsumexp(pair_alphas[-1]))
    pair_betas = [np.zeros_like(pair_alphas[-1]) for _ in pair_alphas]
    for pair_index in range(len(pair_alphas) - 2, -1, -1):
        transition_index = pair_index + 1
        curr_time = pair_index + 2
        pair_betas[pair_index] = ss._backward_imm_pair(
            pair_betas[pair_index + 1],
            candidates[pair_index],
            candidates[pair_index + 1],
            candidates[curr_time],
            emissions.log_likelihood[curr_time, candidates[curr_time]],
            bin_centers,
            modes=modes,
            mode_transition=mode_transition,
            stationary_sigma_cm=stationary_sigma_cm,
            diffusion_sigma_cm=float(diffusion_sigmas_cm[transition_index]),
            momentum_sigma_cm=float(momentum_sigmas_cm[transition_index]),
            velocity_decay=float(velocity_decays[transition_index]) * float(time_scales[transition_index]),
        )

    trajectory = np.full((emissions.n_time, emissions.n_bins), ss.LOG_ZERO, dtype=float)
    mode_log_posterior = np.full((emissions.n_time, len(modes)), ss.LOG_ZERO, dtype=float)
    for pair_index, (alpha, beta) in enumerate(zip(pair_alphas, pair_betas, strict=True)):
        pair_log_posterior = alpha + beta - logp
        if pair_index == 0:
            trajectory[0, candidates[0]] = logsumexp(pair_log_posterior, axis=(0, 2))
            mode_log_posterior[0] = logsumexp(pair_log_posterior, axis=(1, 2))
        trajectory[pair_index + 1, candidates[pair_index + 1]] = logsumexp(pair_log_posterior, axis=(0, 1))
        mode_log_posterior[pair_index + 1] = logsumexp(pair_log_posterior, axis=(1, 2))
    for time_index in range(emissions.n_time):
        trajectory[time_index] -= logsumexp(trajectory[time_index])
    mode_log_posterior -= logsumexp(mode_log_posterior, axis=1)[:, None]
    return logp, trajectory, np.exp(mode_log_posterior), masses


def _as_xy_bin_centers(bin_centers):
    centers = np.asarray(bin_centers, dtype=float)
    if centers.ndim != 2:
        raise ValueError("bin_centers must have shape (n_bins, position_dim)")
    if centers.shape[1] == 1:
        return np.column_stack([centers[:, 0], np.zeros(centers.shape[0], dtype=float)])
    return centers
