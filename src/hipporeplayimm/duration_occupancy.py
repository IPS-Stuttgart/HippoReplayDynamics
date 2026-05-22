"""Duration-aware state-space scoring with occupancy-mask support.

The historical duration patch replaced ``StateSpaceReplayModel.score`` but fell
back to the original scalar-dt implementation whenever callers supplied
``occupancy_s``.  Benchmark and decode paths routinely pass occupancy arrays,
even when the occupancy threshold is disabled, so duration metadata could be
silently ignored.  This module installs a final score implementation that keeps
per-transition durations and valid-occupancy masks active at the same time.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .duration_dynamics import attach_duration_metadata, transition_durations_s


def apply_duration_occupancy_patch() -> None:
    """Install the duration-aware state-space scorer that honors occupancy masks."""

    import hipporeplayimm.state_space as ss

    if getattr(ss, "_duration_occupancy_patch_applied", False):
        return

    if getattr(ss.StateSpaceReplayModel.score, "_native_duration_occupancy_aware", False):
        ss._duration_occupancy_patch_applied = True
        return

    ss.StateSpaceReplayModel.__duration_occupancy_previous_score__ = ss.StateSpaceReplayModel.score
    ss.StateSpaceReplayModel.score = _score_state_space_duration_with_occupancy
    ss._duration_occupancy_patch_applied = True


def _score_state_space_duration_with_occupancy(
    self,
    emissions,
    bin_centers,
    candidate_indices=None,
    *,
    occupancy_s=None,
):
    import hipporeplayimm.state_space as ss

    if emissions.n_time == 0:
        raise ValueError("emissions must contain at least one time bin")
    if emissions.n_bins != bin_centers.shape[0]:
        raise ValueError("emissions.n_bins must match bin_centers rows")
    assert self.config is not None

    valid_bin_mask = ss._valid_bin_mask_from_occupancy(
        occupancy_s,
        self.config.valid_occupancy_threshold_s,
        emissions.n_bins,
    )
    durations = transition_durations_s(emissions)
    attach_duration_metadata(emissions)

    extra: dict[str, float | int | str] = {}
    if self.mode == "stationary":
        logp, trajectory = ss._score_stationary(emissions, valid_bin_mask=valid_bin_mask)
        transition_sigma_cm = 0.0
    elif self.mode in {"fragmented", "jump"}:
        logp, trajectory = ss._score_fragmented(emissions, valid_bin_mask=valid_bin_mask)
        transition_sigma_cm = float("inf")
    elif self.mode == "diffusion":
        diffusion_sigmas = _per_transition_sigmas(
            self.config.diffusion_sigma_cm_sqrt_s,
            durations,
        )
        transition_sigma_cm = _representative_sigma(
            self.config.diffusion_sigma_cm_sqrt_s,
            durations,
            float(emissions.dt),
        )
        transitions = [
            ss._gaussian_transition_matrix(
                bin_centers,
                float(sigma),
                self.config.max_step_sigma,
                valid_bin_mask=valid_bin_mask,
            )
            for sigma in diffusion_sigmas
        ]
        logp, trajectory = _forward_backward_variable(
            ss,
            emissions.log_likelihood,
            transitions,
            valid_bin_mask=valid_bin_mask,
        )
    elif self.mode == "first-order-imm":
        diffusion_sigmas = _per_transition_sigmas(
            self.config.diffusion_sigma_cm_sqrt_s,
            durations,
        )
        transition_sigma_cm = _representative_sigma(
            self.config.diffusion_sigma_cm_sqrt_s,
            durations,
            float(emissions.dt),
        )
        diffusion_transitions = [
            ss._gaussian_transition_matrix(
                bin_centers,
                float(sigma),
                self.config.max_step_sigma,
                valid_bin_mask=valid_bin_mask,
            )
            for sigma in diffusion_sigmas
        ]
        logp, trajectory, mode_post = _score_first_order_imm_variable(
            ss,
            emissions.log_likelihood,
            bin_centers,
            stationary_sigma_cm=self.config.stationary_sigma_cm,
            diffusion_transitions=diffusion_transitions,
            max_step_sigma=self.config.max_step_sigma,
            mode_stickiness=self.config.imm_mode_stickiness,
            valid_bin_mask=valid_bin_mask,
        )
        names = ("stationary", "diffusion", "fragmented")
        extra = {
            f"state_space_mode_{name}_terminal_probability": float(mode_post[-1, idx])
            for idx, name in enumerate(names)
        }
        extra.update(
            {
                "state_space_imm_modes": ",".join(names),
                "state_space_imm_evidence_support": "exact_full_grid",
            }
        )
    elif self.mode == "momentum":
        candidates = _duration_candidates(ss, self, emissions, bin_centers, candidate_indices, valid_bin_mask)
        momentum_sigmas = _per_transition_sigmas(
            self.config.momentum_sigma_cm_sqrt_s,
            durations,
        )
        transition_sigma_cm = _representative_sigma(
            self.config.momentum_sigma_cm_sqrt_s,
            durations,
            float(emissions.dt),
        )
        initial_sigma = _per_bin_sigma(
            self.config.momentum_initial_sigma_cm_sqrt_s,
            durations[0] if len(durations) else float(emissions.dt),
        )
        decays = _duration_adjusted_decays(self.config, durations, float(emissions.dt))
        time_scales = _time_scales(durations)
        logp, trajectory, masses = _score_momentum_duration(
            ss,
            emissions,
            bin_centers,
            candidates,
            sigmas_cm=momentum_sigmas,
            initial_sigma_cm=initial_sigma,
            velocity_decays=decays,
            time_scales=time_scales,
            valid_bin_mask=valid_bin_mask,
        )
        evidence_support = ss._candidate_evidence_support_label(
            candidates,
            emissions.n_bins,
            valid_bin_mask,
        )
        candidate_support_label = "full_grid" if evidence_support == "exact_full_grid" else (
            "derived" if candidate_indices is None else "provided"
        )
        extra = {
            "mean_candidate_log_mass": float(np.mean(masses)),
            "min_candidate_log_mass": float(np.min(masses)),
            "mean_candidate_count": float(np.mean([len(curr) for curr in candidates])),
            "state_space_momentum_candidate_support": candidate_support_label,
            "state_space_momentum_trajectory_posterior": "smoothed_pair_marginal",
            "state_space_momentum_evidence_support": evidence_support,
            **ss._candidate_support_config_diagnostics("state_space_momentum", self.config),
            "state_space_momentum_candidate_selection": (
                "provided" if candidate_indices is not None else ss._candidate_selection_label(self.config)
            ),
        }
    elif self.mode == "imm":
        candidates = _duration_candidates(ss, self, emissions, bin_centers, candidate_indices, valid_bin_mask)
        diffusion_sigmas = _per_transition_sigmas(
            self.config.diffusion_sigma_cm_sqrt_s,
            durations,
        )
        transition_sigma_cm = _representative_sigma(
            self.config.diffusion_sigma_cm_sqrt_s,
            durations,
            float(emissions.dt),
        )
        momentum_sigmas = _per_transition_sigmas(
            self.config.momentum_sigma_cm_sqrt_s,
            durations,
        )
        momentum_transition_sigma_cm = _representative_sigma(
            self.config.momentum_sigma_cm_sqrt_s,
            durations,
            float(emissions.dt),
        )
        initial_sigma = _per_bin_sigma(
            self.config.momentum_initial_sigma_cm_sqrt_s,
            durations[0] if len(durations) else float(emissions.dt),
        )
        decays = _duration_adjusted_decays(self.config, durations, float(emissions.dt))
        time_scales = _time_scales(durations)
        logp, trajectory, mode_post, masses = _score_imm_duration(
            ss,
            emissions,
            bin_centers,
            candidates,
            stationary_sigma_cm=self.config.stationary_sigma_cm,
            diffusion_sigmas_cm=diffusion_sigmas,
            momentum_sigmas_cm=momentum_sigmas,
            initial_momentum_sigma_cm=initial_sigma,
            velocity_decays=decays,
            time_scales=time_scales,
            mode_stickiness=self.config.imm_mode_stickiness,
            valid_bin_mask=valid_bin_mask,
        )
        evidence_support = ss._candidate_evidence_support_label(
            candidates,
            emissions.n_bins,
            valid_bin_mask,
        )
        candidate_support_label = "full_grid" if evidence_support == "exact_full_grid" else (
            "derived" if candidate_indices is None else "provided"
        )
        names = ("stationary", "diffusion", "momentum", "jump")
        extra = {
            f"state_space_mode_{name}_terminal_probability": float(mode_post[-1, idx])
            for idx, name in enumerate(names)
        }
        extra.update(
            {
                "mean_candidate_log_mass": float(np.mean(masses)),
                "min_candidate_log_mass": float(np.min(masses)),
                "mean_candidate_count": float(np.mean([len(curr) for curr in candidates])),
                "state_space_imm_modes": ",".join(names),
                "state_space_imm_candidate_support": candidate_support_label,
                "state_space_imm_trajectory_posterior": "smoothed_pair_marginal",
                "state_space_imm_evidence_support": evidence_support,
                **ss._candidate_support_config_diagnostics("state_space_imm", self.config),
                "state_space_imm_candidate_selection": (
                    "provided" if candidate_indices is not None else ss._candidate_selection_label(self.config)
                ),
                "state_space_momentum_transition_sigma_cm": float(momentum_transition_sigma_cm),
                "state_space_momentum_initial_transition_sigma_cm": float(initial_sigma),
            }
        )
    else:  # pragma: no cover - StateSpaceReplayModel.__post_init__ validates this.
        raise ValueError(f"Unsupported state-space mode: {self.mode}")

    terminal = trajectory[-1]
    diagnostics: dict[str, float | int | str] = {
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
        "state_space_momentum_velocity_decay_tau_s": float(self.config.momentum_velocity_decay_tau_s),
        "state_space_valid_occupancy_threshold_s": float(self.config.valid_occupancy_threshold_s),
        "state_space_transition_sigma_cm": float(transition_sigma_cm),
        "mean_trajectory_posterior_entropy": ss._mean_entropy(trajectory),
        **extra,
    }
    if valid_bin_mask is not None:
        diagnostics.update(
            {
                "state_space_valid_bin_count": int(np.sum(valid_bin_mask)),
                "state_space_valid_bin_fraction": float(np.mean(valid_bin_mask)),
            }
        )
    diagnostics.update(ss._posterior_diagnostics(terminal, bin_centers))
    return ss.EventScore(
        str(self.name),
        float(logp),
        emissions.n_time,
        emissions.n_spikes,
        diagnostics=diagnostics,
        terminal_log_posterior=terminal,
        trajectory_log_posterior=trajectory,
    )


def _duration_candidates(ss, model, emissions, bin_centers, candidate_indices, valid_bin_mask):
    candidates = (
        model.candidate_indices(emissions, bin_centers)
        if candidate_indices is None
        else candidate_indices
    )
    candidates = ss._restrict_candidates_to_valid_bins(
        candidates,
        emissions.log_likelihood,
        valid_bin_mask,
    )
    return ss._validate_candidate_indices(candidates, emissions.n_time, emissions.n_bins)


def _per_bin_sigma(sigma_cm_sqrt_s: float, dt_s: float) -> float:
    sigma = float(sigma_cm_sqrt_s)
    dt = float(dt_s)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma_cm_sqrt_s must be finite and positive")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    return max(sigma * np.sqrt(dt), np.finfo(float).eps)


def _per_transition_sigmas(sigma_cm_sqrt_s: float, durations: np.ndarray) -> np.ndarray:
    return np.asarray([_per_bin_sigma(sigma_cm_sqrt_s, duration) for duration in durations], dtype=float)


def _representative_sigma(sigma_cm_sqrt_s: float, durations: np.ndarray, fallback_dt: float) -> float:
    dt = float(np.median(durations)) if len(durations) else float(fallback_dt)
    return _per_bin_sigma(sigma_cm_sqrt_s, dt)


def _duration_adjusted_decays(
    config_or_decay: object,
    durations: np.ndarray,
    reference_dt: float,
) -> np.ndarray:
    reference_dt = float(reference_dt)
    if not np.isfinite(reference_dt) or reference_dt <= 0.0:
        raise ValueError("reference dt must be finite and positive")

    durations = np.asarray(durations, dtype=float)
    if hasattr(config_or_decay, "momentum_velocity_decay"):
        tau_s = float(getattr(config_or_decay, "momentum_velocity_decay_tau_s", 0.0))
        if not np.isfinite(tau_s) or tau_s < 0.0:
            raise ValueError("momentum_velocity_decay_tau_s must be finite and nonnegative")
        if tau_s > 0.0:
            return np.asarray(np.exp(-durations / tau_s), dtype=float)
        decay = float(getattr(config_or_decay, "momentum_velocity_decay"))
    else:
        decay = float(config_or_decay)

    if not np.isfinite(decay) or decay < 0.0:
        raise ValueError("momentum_velocity_decay must be finite and nonnegative")
    return np.asarray([decay ** (float(duration) / reference_dt) for duration in durations], dtype=float)


def _duration_adjusted_decays_from_config(config, durations: np.ndarray, reference_dt: float) -> np.ndarray:
    """Return transition-specific momentum velocity decays.

    Historically the duration-aware scorer scaled a per-bin decay by
    ``duration / reference_dt``.  That is still the backwards-compatible path.
    When ``momentum_velocity_decay_tau_s`` is positive, use the physical-time
    decay ``exp(-duration / tau)`` instead so the same setting is meaningful for
    1, 2, 3, or 5 ms replay bins.
    """

    tau_s = float(getattr(config, "momentum_velocity_decay_tau_s", 0.0))
    if tau_s <= 0.0:
        return _duration_adjusted_decays(
            float(getattr(config, "momentum_velocity_decay", 0.95)),
            durations,
            reference_dt,
        )
    if not np.isfinite(tau_s):
        raise ValueError("momentum_velocity_decay_tau_s must be finite when positive")
    durations = np.asarray(durations, dtype=float)
    if np.any(durations <= 0.0) or not np.all(np.isfinite(durations)):
        raise ValueError("transition durations must be finite and positive")
    return np.exp(-durations / tau_s)


def _time_scales(durations: np.ndarray) -> np.ndarray:
    scales = np.ones_like(durations, dtype=float)
    if len(durations) > 1:
        scales[1:] = durations[1:] / durations[:-1]
    return scales


def _transition_at(transitions, transition_index: int):
    return transitions[transition_index] if isinstance(transitions, (list, tuple)) else transitions


def _forward_backward_variable(ss, log_likelihood, transitions, *, valid_bin_mask=None):
    n_time, n_bins = log_likelihood.shape
    scaled, offsets = ss._scaled_emissions(log_likelihood)
    filtered = np.zeros((n_time, n_bins), dtype=float)
    scales = np.zeros(n_time, dtype=float)

    alpha = scaled[0] * _uniform_probabilities(n_bins, valid_bin_mask)
    scales[0] = float(alpha.sum())
    if scales[0] <= 0.0:
        raise ValueError("first emission row has no finite likelihood mass")
    alpha /= scales[0]
    filtered[0] = alpha
    logp = float(np.log(scales[0]) + offsets[0])

    for time_index in range(1, n_time):
        alpha = np.asarray(_transition_at(transitions, time_index - 1) @ alpha, dtype=float) * scaled[time_index]
        scales[time_index] = float(alpha.sum())
        if scales[time_index] <= 0.0:
            raise ValueError(f"emission row {time_index} has no finite predicted mass")
        alpha /= scales[time_index]
        filtered[time_index] = alpha
        logp += float(np.log(scales[time_index]) + offsets[time_index])

    smoothed = np.zeros_like(filtered)
    beta = np.ones(n_bins, dtype=float)
    smoothed[-1] = filtered[-1]
    for time_index in range(n_time - 1, 0, -1):
        beta = np.asarray(
            _transition_at(transitions, time_index - 1).T @ (scaled[time_index] * beta),
            dtype=float,
        ) / scales[time_index]
        gamma = filtered[time_index - 1] * beta
        total = float(gamma.sum())
        smoothed[time_index - 1] = gamma / total if total > 0.0 else filtered[time_index - 1]
    return logp, ss._as_log_probs(smoothed)


def _score_first_order_imm_variable(
    ss,
    log_likelihood,
    bin_centers,
    *,
    stationary_sigma_cm,
    diffusion_transitions,
    max_step_sigma,
    mode_stickiness,
    valid_bin_mask=None,
):
    modes = ("stationary", "diffusion", "fragmented")
    n_modes = len(modes)
    n_time, n_bins = log_likelihood.shape
    transitions = {
        "stationary": ss._gaussian_transition_matrix(
            bin_centers,
            stationary_sigma_cm,
            max_step_sigma,
            valid_bin_mask=valid_bin_mask,
        ),
        "diffusion": diffusion_transitions,
        "fragmented": None,
    }
    mode_transition = ss._mode_transition_matrix(n_modes, mode_stickiness)
    scaled, offsets = ss._scaled_emissions(log_likelihood)
    filtered = np.zeros((n_time, n_modes, n_bins), dtype=float)
    scales = np.zeros(n_time, dtype=float)

    alpha = np.tile(scaled[0] * _uniform_probabilities(n_bins, valid_bin_mask) / n_modes, (n_modes, 1))
    scales[0] = float(alpha.sum())
    if scales[0] <= 0.0:
        raise ValueError("first emission row has no finite likelihood mass")
    alpha /= scales[0]
    filtered[0] = alpha
    logp = float(np.log(scales[0]) + offsets[0])

    for time_index in range(1, n_time):
        predicted = np.zeros_like(alpha)
        for dst_idx, dst_mode in enumerate(modes):
            dst = np.zeros(n_bins, dtype=float)
            for src_idx in range(n_modes):
                transition = transitions[dst_mode]
                if transition is None:
                    value = _uniform_probabilities(n_bins, valid_bin_mask) * float(alpha[src_idx].sum())
                else:
                    value = np.asarray(_transition_at(transition, time_index - 1) @ alpha[src_idx], dtype=float)
                dst += mode_transition[src_idx, dst_idx] * value
            predicted[dst_idx] = dst
        alpha = predicted * scaled[time_index][None, :]
        scales[time_index] = float(alpha.sum())
        if scales[time_index] <= 0.0:
            raise ValueError(f"emission row {time_index} has no finite predicted mass")
        alpha /= scales[time_index]
        filtered[time_index] = alpha
        logp += float(np.log(scales[time_index]) + offsets[time_index])

    smoothed = np.zeros_like(filtered)
    beta = np.ones((n_modes, n_bins), dtype=float)
    smoothed[-1] = filtered[-1]
    for time_index in range(n_time - 1, 0, -1):
        beta_prev = np.zeros_like(beta)
        for src_idx in range(n_modes):
            for dst_idx, dst_mode in enumerate(modes):
                transition = transitions[dst_mode]
                values = scaled[time_index] * beta[dst_idx]
                if transition is None:
                    value = _uniform_backward(values, valid_bin_mask)
                else:
                    value = np.asarray(_transition_at(transition, time_index - 1).T @ values, dtype=float)
                beta_prev[src_idx] += mode_transition[src_idx, dst_idx] * value
        beta = beta_prev / scales[time_index]
        gamma = filtered[time_index - 1] * beta
        total = float(gamma.sum())
        smoothed[time_index - 1] = gamma / total if total > 0.0 else filtered[time_index - 1]

    return logp, ss._as_log_probs(smoothed.sum(axis=1)), smoothed.sum(axis=2)


def _score_momentum_duration(
    ss,
    emissions,
    bin_centers,
    candidates,
    *,
    sigmas_cm,
    initial_sigma_cm,
    velocity_decays,
    time_scales,
    valid_bin_mask=None,
):
    if emissions.n_time == 1:
        logp, trajectory = ss._score_fragmented(emissions, valid_bin_mask=valid_bin_mask)
        return logp, trajectory, [0.0]

    masses = ss._candidate_log_masses(emissions.log_likelihood, candidates)
    pair = ss._init_pair_log_alpha(
        emissions.log_likelihood,
        candidates[0],
        candidates[1],
        bin_centers,
        sigma_cm=float(initial_sigma_cm),
        valid_bin_mask=valid_bin_mask,
    )
    pair_alphas = [pair]
    for time_index in range(2, emissions.n_time):
        transition_index = time_index - 1
        pair = ss._advance_momentum_pair(
            pair,
            candidates[time_index - 2],
            candidates[time_index - 1],
            candidates[time_index],
            emissions.log_likelihood[time_index, candidates[time_index]],
            bin_centers,
            sigma_cm=float(sigmas_cm[transition_index]),
            velocity_decay=float(velocity_decays[transition_index]) * float(time_scales[transition_index]),
            valid_bin_mask=valid_bin_mask,
        )
        pair_alphas.append(pair)

    logp = float(logsumexp(pair_alphas[-1]))
    pair_betas = [np.zeros_like(pair_alphas[-1]) for _ in pair_alphas]
    for pair_index in range(len(pair_alphas) - 2, -1, -1):
        transition_index = pair_index + 1
        curr_time = pair_index + 2
        pair_betas[pair_index] = ss._backward_momentum_pair(
            pair_betas[pair_index + 1],
            candidates[pair_index],
            candidates[pair_index + 1],
            candidates[curr_time],
            emissions.log_likelihood[curr_time, candidates[curr_time]],
            bin_centers,
            sigma_cm=float(sigmas_cm[transition_index]),
            velocity_decay=float(velocity_decays[transition_index]) * float(time_scales[transition_index]),
            valid_bin_mask=valid_bin_mask,
        )

    trajectory = np.full((emissions.n_time, emissions.n_bins), ss.LOG_ZERO, dtype=float)
    for pair_index, (alpha, beta) in enumerate(zip(pair_alphas, pair_betas, strict=True)):
        pair_log_posterior = alpha + beta - logp
        if pair_index == 0:
            trajectory[0, candidates[0]] = logsumexp(pair_log_posterior, axis=1)
        trajectory[pair_index + 1, candidates[pair_index + 1]] = logsumexp(pair_log_posterior, axis=0)
    for time_index in range(emissions.n_time):
        trajectory[time_index] -= logsumexp(trajectory[time_index])
    return logp, trajectory, masses


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
    valid_bin_mask=None,
):
    modes = ("stationary", "diffusion", "momentum", "jump")
    if emissions.n_time == 1:
        logp, trajectory = ss._score_fragmented(emissions, valid_bin_mask=valid_bin_mask)
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
            valid_bin_mask=valid_bin_mask,
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
                    valid_bin_mask=valid_bin_mask,
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
            valid_bin_mask=valid_bin_mask,
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


def _uniform_probabilities(n_bins: int, valid_bin_mask: np.ndarray | None = None) -> np.ndarray:
    if valid_bin_mask is None:
        return np.full(n_bins, 1.0 / n_bins, dtype=float)
    mask = np.asarray(valid_bin_mask, dtype=bool)
    if mask.shape != (n_bins,):
        raise ValueError("valid_bin_mask must contain one boolean value per spatial bin")
    if not np.any(mask):
        raise ValueError("valid_bin_mask must contain at least one valid spatial bin")
    probabilities = np.zeros(n_bins, dtype=float)
    probabilities[mask] = 1.0 / int(np.sum(mask))
    return probabilities


def _uniform_backward(values: np.ndarray, valid_bin_mask: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if valid_bin_mask is None:
        return np.full(values.shape, float(values.sum()) / values.shape[0], dtype=float)
    mask = np.asarray(valid_bin_mask, dtype=bool)
    if mask.shape != values.shape:
        raise ValueError("valid_bin_mask must contain one boolean value per spatial bin")
    if not np.any(mask):
        raise ValueError("valid_bin_mask must contain at least one valid spatial bin")
    return np.full(values.shape, float(values[mask].sum()) / int(np.sum(mask)), dtype=float)
