"""State-space replay decoder baselines with full trajectory posteriors."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.special import logsumexp

from .encoding import LogEmissionTensor
from .models import EventScore, _posterior_diagnostics
from .state_space_candidates import _score_imm_candidates
from .state_space_candidates_momentum import _score_momentum_candidates
from .state_space_first_order import (
    _forward_backward_first_order,
    _forward_backward_first_order_time_varying,
    _score_first_order_imm,
    _score_fragmented,
    _score_stationary,
)
from .state_space_utils import (
    _gaussian_transition_matrix,
    _mass_retaining_candidate_indices,
    _mean_entropy,
    _per_bin_sigma,
    _restrict_candidates_to_valid_bins,
    _top_candidate_indices,
    _valid_bin_mask_from_occupancy,
    _validate_candidate_indices,
)


@dataclass(frozen=True)
class StateSpaceDecoderConfig:
    """Configuration for state-space replay baselines.

    Diffusion and momentum noise are specified in cm/sqrt(s). They are converted
    to per-bin standard deviations using the emission tensor's ``dt`` so that the
    same model can be evaluated with 1--3 ms replay bins.
    """

    mode: str = "diffusion"
    stationary_sigma_cm: float = 2.0
    diffusion_sigma_cm_sqrt_s: float = 85.0
    max_step_sigma: float = 4.0
    imm_mode_stickiness: float = 0.95
    momentum_sigma_cm_sqrt_s: float = 85.0
    momentum_initial_sigma_cm_sqrt_s: float = 85.0
    momentum_velocity_decay: float = 0.95
    momentum_velocity_decay_tau_s: float = 0.0
    momentum_candidate_top_k: int = 128
    momentum_candidate_mass_threshold: float | None = None
    displacement_radius_bins: int = 2
    displacement_position_sigma_cm: float = 0.0
    displacement_transition_sigma_cm_sqrt_s: float = 0.0
    displacement_prior_sigma_cm: float = 0.0
    momentum_candidate_min_k: int = 1
    momentum_candidate_max_k: int = 0
    momentum_predicted_candidate_top_k: int = 8
    momentum_candidate_source: str = "emission"
    valid_occupancy_threshold_s: float = 0.0


@dataclass
class StateSpaceReplayModel:
    """Replay decoder baseline returning a posterior for every replay bin.

    Supported modes are ``stationary``, ``diffusion``, ``fragmented``/``jump``,
    ``first-order-imm``, ``trajectory-imm-exact-sparse``, ``momentum``, ``imm``,
    ``displacement-momentum``, and ``displacement-imm``. The first-order models use exact full-grid
    forward-backward recursions. ``first-order-imm`` is the legacy exact
    first-order switcher over stationary, diffusion, and fragmented/jump
    dynamics. ``momentum`` and ``imm`` use candidate-pruned second-order
    dynamics for scalability. ``imm`` switches among stationary, diffusion,
    momentum, and jump modes.

    Candidate recursions keep full-grid prior and transition normalizers and
    drop off-support paths, so their evidences are conservative truncated
    full-grid evidences. They return candidate-supported per-bin posterior
    marginals. ``momentum-exact-sparse`` is an exact pair-grid dynamic program
    over a finite-radius sparse transition model and is intended as the
    comparable paper-facing second-order momentum row. ``displacement-momentum``
    is an exact finite-state surrogate over ``(position, displacement)`` states.
    ``displacement-imm`` is an exact finite-state IMM surrogate over ``(mode,
    position, displacement)`` states. These finite-state surrogates are not the
    full pairwise momentum model, but their evidences are exact over the
    declared finite displacement grid and can be used as comparable diagnostic
    rows.
    """

    mode: str = "diffusion"
    config: StateSpaceDecoderConfig | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        allowed = {
            "stationary",
            "diffusion",
            "fragmented",
            "jump",
            "first-order-imm",
            "trajectory-imm-exact-sparse",
            "imm",
            "momentum",
            "momentum-exact-sparse",
            "displacement-momentum",
            "displacement-imm",
        }
        if self.mode not in allowed:
            raise ValueError(f"mode must be one of {sorted(allowed)}")
        if self.name is None:
            self.name = f"state-space-{self.mode}"
        if self.config is None:
            self.config = StateSpaceDecoderConfig(mode=self.mode)
        elif self.config.mode != self.mode:
            self.config = replace(self.config, mode=self.mode)

    def candidate_indices(self, emissions: LogEmissionTensor, bin_centers: np.ndarray | None = None) -> list[np.ndarray]:
        """Return the candidate support used by pruned momentum/IMM recursions.

        The base support is either the per-bin emission top-k set or, when
        ``momentum_candidate_mass_threshold`` is finite, the smallest emission
        support retaining that normalized mass subject to the configured
        min/max bounds. When ``momentum_predicted_candidate_top_k`` is positive
        and ``bin_centers`` is supplied, each time bin is enlarged by nearest
        grid states to forward- and backward-momentum predictions built from
        the top adjacent emission candidates. This keeps deterministic emission
        support while recovering dynamically plausible states whose local
        emission rank is too low for the fixed top-k beam. The augmentation is
        bounded by the prediction top-k and never changes externally supplied
        candidate supports.
        """

        assert self.config is not None
        support_log_values = _candidate_support_log_values(emissions, bin_centers, self.config)
        mass_threshold = self.config.momentum_candidate_mass_threshold
        if mass_threshold is None or not np.isfinite(float(mass_threshold)):
            base = [
                _top_candidate_indices(row, self.config.momentum_candidate_top_k)
                for row in support_log_values
            ]
        else:
            base = [
                _mass_retaining_candidate_indices(
                    row,
                    float(mass_threshold),
                    top_k=self.config.momentum_candidate_top_k,
                    min_k=self.config.momentum_candidate_min_k,
                    max_k=self.config.momentum_candidate_max_k,
                )
                for row in support_log_values
            ]
        predicted_top_k = int(self.config.momentum_predicted_candidate_top_k)
        if predicted_top_k <= 0 or bin_centers is None or emissions.n_time < 3:
            return base
        return _augment_candidates_with_momentum_predictions(
            base,
            bin_centers,
            predicted_top_k=predicted_top_k,
            velocity_decay=_representative_transition_value(
                _momentum_velocity_decays(
                    self.config,
                    _emission_transition_durations(emissions),
                ),
                fallback=float(self.config.momentum_velocity_decay),
            ),
            velocity_decays=_momentum_velocity_decays(
                self.config,
                _emission_transition_durations(emissions),
            ),
        )

    def score(
        self,
        emissions: LogEmissionTensor,
        bin_centers: np.ndarray,
        candidate_indices: list[np.ndarray] | None = None,
        *,
        occupancy_s: np.ndarray | None = None,
        return_trajectory: bool = True,
    ) -> EventScore:
        # Native duration-aware implementation. The historical runtime patch
        # modules detect the marker below and skip monkey-patching this method.
        # The legacy scalar-dt body is intentionally left as unreachable fallback
        # for source compatibility while the patch modules are phased out.
        from .duration_occupancy import _score_state_space_duration_with_occupancy

        return _score_state_space_duration_with_occupancy(
            self,
            emissions,
            bin_centers,
            candidate_indices=candidate_indices,
            occupancy_s=occupancy_s,
            return_trajectory=return_trajectory,
        )
        if emissions.n_time == 0:
            raise ValueError("emissions must contain at least one time bin")
        if emissions.n_bins != bin_centers.shape[0]:
            raise ValueError("emissions.n_bins must match bin_centers rows")
        assert self.config is not None
        valid_bin_mask = _valid_bin_mask_from_occupancy(
            occupancy_s,
            self.config.valid_occupancy_threshold_s,
            emissions.n_bins,
        )

        if self.mode == "stationary":
            logp, trajectory = _score_stationary(emissions, valid_bin_mask=valid_bin_mask)
            transition_sigma_cm = 0.0
            extra: dict[str, float | int | str] = {}
        elif self.mode in {"fragmented", "jump"}:
            logp, trajectory = _score_fragmented(emissions, valid_bin_mask=valid_bin_mask)
            transition_sigma_cm = float("inf")
            extra = {}
        elif self.mode == "diffusion":
            transition_sigma_cm = _per_bin_sigma(self.config.diffusion_sigma_cm_sqrt_s, emissions.dt)
            transition_durations = _emission_transition_durations(emissions)
            if transition_durations.size and not np.allclose(transition_durations, float(emissions.dt)):
                transition_sigmas_cm = [
                    _per_bin_sigma(self.config.diffusion_sigma_cm_sqrt_s, duration)
                    for duration in transition_durations
                ]
                transitions = [
                    _gaussian_transition_matrix(
                        bin_centers,
                        sigma_cm,
                        self.config.max_step_sigma,
                        valid_bin_mask=valid_bin_mask,
                    )
                    for sigma_cm in transition_sigmas_cm
                ]
                logp, trajectory = _forward_backward_first_order_time_varying(
                    emissions.log_likelihood,
                    transitions,
                    valid_bin_mask=valid_bin_mask,
                )
                transition_sigma_cm = float(np.median(transition_sigmas_cm))
            else:
                transition = _gaussian_transition_matrix(
                    bin_centers,
                    transition_sigma_cm,
                    self.config.max_step_sigma,
                    valid_bin_mask=valid_bin_mask,
                )
                logp, trajectory = _forward_backward_first_order(
                    emissions.log_likelihood,
                    transition,
                    valid_bin_mask=valid_bin_mask,
                )
            extra = {}
        elif self.mode == "first-order-imm":
            transition_sigma_cm = _per_bin_sigma(self.config.diffusion_sigma_cm_sqrt_s, emissions.dt)
            logp, trajectory, mode_post = _score_first_order_imm(
                emissions.log_likelihood,
                bin_centers,
                stationary_sigma_cm=self.config.stationary_sigma_cm,
                diffusion_sigma_cm=transition_sigma_cm,
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
        elif self.mode == "imm":
            transition_durations = _emission_transition_durations(emissions)
            diffusion_transition_sigmas_cm = _per_transition_sigmas_cm(
                self.config.diffusion_sigma_cm_sqrt_s,
                transition_durations,
            )
            momentum_transition_sigmas_cm = _per_transition_sigmas_cm(
                self.config.momentum_sigma_cm_sqrt_s,
                transition_durations,
            )
            momentum_initial_transition_sigmas_cm = _per_transition_sigmas_cm(
                self.config.momentum_initial_sigma_cm_sqrt_s,
                transition_durations,
            )
            velocity_decays = _momentum_velocity_decays(self.config, transition_durations)
            transition_sigma_cm = _representative_transition_value(
                diffusion_transition_sigmas_cm,
                fallback=_per_bin_sigma(self.config.diffusion_sigma_cm_sqrt_s, emissions.dt),
            )
            momentum_transition_sigma_cm = _representative_transition_value(
                momentum_transition_sigmas_cm,
                fallback=_per_bin_sigma(self.config.momentum_sigma_cm_sqrt_s, emissions.dt),
            )
            momentum_initial_sigma_cm = _first_transition_value(
                momentum_initial_transition_sigmas_cm,
                fallback=_per_bin_sigma(self.config.momentum_initial_sigma_cm_sqrt_s, emissions.dt),
            )
            momentum_velocity_decay = _representative_transition_value(
                velocity_decays,
                fallback=float(self.config.momentum_velocity_decay),
            )
            candidates = self.candidate_indices(emissions, bin_centers) if candidate_indices is None else candidate_indices
            candidates = _restrict_candidates_to_valid_bins(
                candidates,
                emissions.log_likelihood,
                valid_bin_mask,
            )
            candidates = _validate_candidate_indices(candidates, emissions.n_time, emissions.n_bins)
            logp, trajectory, mode_post, masses = _score_imm_candidates(
                emissions,
                bin_centers,
                stationary_sigma_cm=self.config.stationary_sigma_cm,
                diffusion_sigma_cm=transition_sigma_cm,
                momentum_sigma_cm=momentum_transition_sigma_cm,
                momentum_initial_sigma_cm=momentum_initial_sigma_cm,
                velocity_decay=momentum_velocity_decay,
                mode_stickiness=self.config.imm_mode_stickiness,
                diffusion_sigmas_cm=diffusion_transition_sigmas_cm,
                momentum_sigmas_cm=momentum_transition_sigmas_cm,
                velocity_decays=velocity_decays,
                candidate_indices=candidates,
                valid_bin_mask=valid_bin_mask,
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
                    "state_space_imm_candidate_support": "derived" if candidate_indices is None else "provided",
                    "state_space_imm_trajectory_posterior": "smoothed_pair_marginal",
                    "state_space_imm_evidence_support": "truncated_full_grid",
                    **_candidate_support_config_diagnostics("state_space_imm", self.config),
                    "state_space_imm_candidate_selection": (
                        "provided" if candidate_indices is not None else _candidate_selection_label(self.config)
                    ),
                    "state_space_momentum_transition_sigma_cm": float(momentum_transition_sigma_cm),
                    "state_space_momentum_initial_transition_sigma_cm": float(momentum_initial_sigma_cm),
                    "state_space_momentum_transition_sigma_cm_per_step": _format_float_series(momentum_transition_sigmas_cm),
                    "state_space_momentum_initial_transition_sigma_cm_per_step": _format_float_series(momentum_initial_transition_sigmas_cm),
                    "state_space_diffusion_transition_sigma_cm_per_step": _format_float_series(diffusion_transition_sigmas_cm),
                    "state_space_momentum_velocity_decay_effective": float(momentum_velocity_decay),
                    "state_space_momentum_velocity_decay_per_step": _format_float_series(velocity_decays),
                    "state_space_transition_durations_s": _format_float_series(transition_durations),
                }
            )
        elif self.mode == "momentum":
            transition_durations = _emission_transition_durations(emissions)
            momentum_transition_sigmas_cm = _per_transition_sigmas_cm(self.config.momentum_sigma_cm_sqrt_s, transition_durations)
            momentum_initial_transition_sigmas_cm = _per_transition_sigmas_cm(self.config.momentum_initial_sigma_cm_sqrt_s, transition_durations)
            velocity_decays = _momentum_velocity_decays(self.config, transition_durations)
            transition_sigma_cm = _representative_transition_value(momentum_transition_sigmas_cm, fallback=_per_bin_sigma(self.config.momentum_sigma_cm_sqrt_s, emissions.dt))
            momentum_initial_sigma_cm = _first_transition_value(momentum_initial_transition_sigmas_cm, fallback=_per_bin_sigma(self.config.momentum_initial_sigma_cm_sqrt_s, emissions.dt))
            momentum_velocity_decay = _representative_transition_value(velocity_decays, fallback=float(self.config.momentum_velocity_decay))
            candidates = self.candidate_indices(emissions, bin_centers) if candidate_indices is None else candidate_indices
            candidates = _restrict_candidates_to_valid_bins(
                candidates,
                emissions.log_likelihood,
                valid_bin_mask,
            )
            candidates = _validate_candidate_indices(candidates, emissions.n_time, emissions.n_bins)
            logp, trajectory, masses = _score_momentum_candidates(
                emissions,
                bin_centers,
                candidates,
                sigma_cm=transition_sigma_cm,
                initial_sigma_cm=momentum_initial_sigma_cm,
                velocity_decay=momentum_velocity_decay,
                transition_sigmas_cm=momentum_transition_sigmas_cm,
                velocity_decays=velocity_decays,
                valid_bin_mask=valid_bin_mask,
            )
            extra = {
                "mean_candidate_log_mass": float(np.mean(masses)),
                "min_candidate_log_mass": float(np.min(masses)),
                "mean_candidate_count": float(np.mean([len(curr) for curr in candidates])),
                "state_space_momentum_candidate_support": "derived" if candidate_indices is None else "provided",
                "state_space_momentum_trajectory_posterior": "smoothed_pair_marginal",
                "state_space_momentum_evidence_support": "truncated_full_grid",
                **_candidate_support_config_diagnostics("state_space_momentum", self.config),
                "state_space_momentum_candidate_selection": (
                    "provided" if candidate_indices is not None else _candidate_selection_label(self.config)
                ),
                "state_space_momentum_transition_sigma_cm": float(transition_sigma_cm),
                "state_space_momentum_initial_transition_sigma_cm": float(momentum_initial_sigma_cm),
                "state_space_momentum_transition_sigma_cm_per_step": _format_float_series(momentum_transition_sigmas_cm),
                "state_space_momentum_initial_transition_sigma_cm_per_step": _format_float_series(momentum_initial_transition_sigmas_cm),
                "state_space_momentum_velocity_decay_effective": float(momentum_velocity_decay),
                "state_space_momentum_velocity_decay_per_step": _format_float_series(velocity_decays),
                "state_space_transition_durations_s": _format_float_series(transition_durations),
            }
        else:  # pragma: no cover - __post_init__ validates this.
            raise ValueError(f"Unsupported state-space mode: {self.mode}")

        terminal = trajectory[-1]
        diagnostics: dict[str, float | int | str] = {
            "state_space_mode": str(self.mode),
            "state_space_time_bin_s": float(emissions.dt),
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
            "mean_trajectory_posterior_entropy": _mean_entropy(trajectory),
            **extra,
        }
        if valid_bin_mask is not None:
            diagnostics.update(
                {
                    "state_space_valid_bin_count": int(np.sum(valid_bin_mask)),
                    "state_space_valid_bin_fraction": float(np.mean(valid_bin_mask)),
                }
            )
        diagnostics.update(_posterior_diagnostics(terminal, bin_centers))
        return EventScore(
            str(self.name),
            float(logp),
            emissions.n_time,
            emissions.n_spikes,
            diagnostics=diagnostics,
            terminal_log_posterior=terminal,
            trajectory_log_posterior=trajectory,
        )


def _emission_transition_durations(emissions: LogEmissionTensor) -> np.ndarray:
    values = getattr(emissions, "transition_durations", None)
    if values is not None:
        out = np.asarray(values, dtype=float)
    else:
        out = np.diff(np.asarray(emissions.times, dtype=float)) if emissions.n_time > 1 else np.empty(0, dtype=float)
    if out.shape != (max(emissions.n_time - 1, 0),):
        return np.full(max(emissions.n_time - 1, 0), float(emissions.dt), dtype=float)
    if not np.all(np.isfinite(out)) or np.any(out <= 0.0):
        return np.full(max(emissions.n_time - 1, 0), float(emissions.dt), dtype=float)
    return out


def _candidate_support_config_diagnostics(
    prefix: str,
    config: StateSpaceDecoderConfig,
) -> dict[str, float | int | str]:
    """Return diagnostics describing how candidate support was generated."""

    mass_threshold = config.momentum_candidate_mass_threshold
    if mass_threshold is None or not np.isfinite(float(mass_threshold)):
        threshold_value = float("nan")
    else:
        threshold_value = float(mass_threshold)
    return {
        f"{prefix}_candidate_top_k": int(config.momentum_candidate_top_k),
        f"{prefix}_candidate_mass_threshold": threshold_value,
        f"{prefix}_candidate_min_k": int(config.momentum_candidate_min_k),
        f"{prefix}_candidate_max_k": int(config.momentum_candidate_max_k),
        f"{prefix}_candidate_selection": _candidate_selection_label(config),
        f"{prefix}_candidate_source": _candidate_source_label(config),
        f"{prefix}_velocity_decay_tau_s": float(config.momentum_velocity_decay_tau_s),
        f"{prefix}_predicted_candidate_top_k": int(
            config.momentum_predicted_candidate_top_k
        ),
    }


def _candidate_evidence_support_label(
    candidates: list[np.ndarray],
    n_bins: int,
    valid_bin_mask: np.ndarray | None = None,
) -> str:
    """Classify candidate-pruned evidence as exact only for full support.

    The candidate recursions use full-grid transition normalizers.  They are
    therefore exact full-grid evidences when every time bin's support contains
    every spatial bin allowed by the occupancy mask, and conservative truncated
    lower bounds otherwise.  This distinction matters for paper-level model
    comparison because exact rows can be normalized against diffusion/static
    baselines, while truncated rows cannot.
    """

    expected = _full_candidate_index_set(n_bins, valid_bin_mask)
    for current in candidates:
        observed = np.sort(np.asarray(current, dtype=int))
        if observed.shape != expected.shape or not np.array_equal(observed, expected):
            return "truncated_full_grid"
    return "exact_full_grid"


def _full_candidate_index_set(n_bins: int, valid_bin_mask: np.ndarray | None) -> np.ndarray:
    if valid_bin_mask is None:
        return np.arange(n_bins, dtype=int)
    mask = np.asarray(valid_bin_mask, dtype=bool)
    if mask.shape != (n_bins,):
        raise ValueError("valid_bin_mask must contain one boolean value per spatial bin")
    if not np.any(mask):
        raise ValueError("valid_bin_mask must contain at least one valid spatial bin")
    return np.flatnonzero(mask).astype(int)


def _candidate_selection_label(config: StateSpaceDecoderConfig) -> str:
    mass_threshold = config.momentum_candidate_mass_threshold
    if mass_threshold is not None and np.isfinite(float(mass_threshold)) and float(mass_threshold) > 0.0:
        return "adaptive_mass"
    if int(config.momentum_candidate_top_k) <= 0:
        return "full_grid"
    return "top_k"


def _candidate_source_label(config: StateSpaceDecoderConfig) -> str:
    source = str(config.momentum_candidate_source).strip().lower().replace("_", "-")
    aliases = {
        "likelihood": "emission",
        "log-likelihood": "emission",
        "train-posterior": "posterior",
        "diffusion-posterior": "posterior",
        "first-order-posterior": "posterior",
    }
    source = aliases.get(source, source)
    if source not in {"emission", "posterior"}:
        raise ValueError("momentum_candidate_source must be 'emission' or 'posterior'")
    return source


def _candidate_support_log_values(
    emissions: LogEmissionTensor,
    bin_centers: np.ndarray | None,
    config: StateSpaceDecoderConfig,
) -> np.ndarray:
    """Return train-only log support scores used for candidate selection."""

    source = _candidate_source_label(config)
    if source == "emission" or bin_centers is None:
        return np.asarray(emissions.log_likelihood, dtype=float)
    return _diffusion_candidate_log_posterior(emissions, bin_centers, config)


def _diffusion_candidate_log_posterior(
    emissions: LogEmissionTensor,
    bin_centers: np.ndarray,
    config: StateSpaceDecoderConfig,
) -> np.ndarray:
    """Use the exact first-order diffusion posterior as a leakage-free beam source."""

    if emissions.n_time <= 1:
        return _normalize_log_rows(emissions.log_likelihood)
    transition_durations = _emission_transition_durations(emissions)
    sigmas = _per_transition_sigmas_cm(config.diffusion_sigma_cm_sqrt_s, transition_durations)
    if sigmas.size and not np.allclose(sigmas, _per_bin_sigma(config.diffusion_sigma_cm_sqrt_s, emissions.dt)):
        transitions = [
            _gaussian_transition_matrix(
                bin_centers,
                float(sigma_cm),
                config.max_step_sigma,
            )
            for sigma_cm in sigmas
        ]
        _, trajectory = _forward_backward_first_order_time_varying(
            emissions.log_likelihood,
            transitions,
        )
        return trajectory
    transition = _gaussian_transition_matrix(
        bin_centers,
        _per_bin_sigma(config.diffusion_sigma_cm_sqrt_s, emissions.dt),
        config.max_step_sigma,
    )
    _, trajectory = _forward_backward_first_order(emissions.log_likelihood, transition)
    return trajectory


def _normalize_log_rows(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    out -= logsumexp(out, axis=1)[:, None]
    return out


def _per_transition_sigmas_cm(sigma_cm_sqrt_s: float, transition_durations: np.ndarray) -> np.ndarray:
    return np.asarray(
        [_per_bin_sigma(sigma_cm_sqrt_s, float(duration)) for duration in np.asarray(transition_durations, dtype=float)],
        dtype=float,
    )


def _momentum_velocity_decays(config: StateSpaceDecoderConfig, transition_durations: np.ndarray) -> np.ndarray:
    durations = np.asarray(transition_durations, dtype=float)
    if durations.size == 0:
        return np.empty(0, dtype=float)
    tau_s = float(config.momentum_velocity_decay_tau_s)
    if tau_s > 0.0:
        return np.exp(-durations / tau_s)
    return np.full(durations.shape, float(config.momentum_velocity_decay), dtype=float)


def _representative_transition_value(values: np.ndarray, *, fallback: float) -> float:
    arr = np.asarray(values, dtype=float)
    return float(fallback) if arr.size == 0 else float(np.median(arr))


def _first_transition_value(values: np.ndarray, *, fallback: float) -> float:
    arr = np.asarray(values, dtype=float)
    return float(fallback) if arr.size == 0 else float(arr[0])


def _format_float_series(values: np.ndarray) -> str:
    return ",".join(f"{float(value):.12g}" for value in np.asarray(values, dtype=float))


def _augment_candidates_with_momentum_predictions(
    candidates: list[np.ndarray],
    bin_centers: np.ndarray,
    *,
    predicted_top_k: int,
    velocity_decay: float,
    velocity_decays: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Union emission candidates with states nearest to bounded momentum predictions."""

    if predicted_top_k <= 0:
        return [np.asarray(curr, dtype=int).copy() for curr in candidates]
    top_k = max(1, int(predicted_top_k))
    augmented = [set(int(idx) for idx in np.asarray(curr, dtype=int)) for curr in candidates]

    for time_index in range(2, len(candidates)):
        prev_prev = np.asarray(candidates[time_index - 2], dtype=int)[:top_k]
        prev = np.asarray(candidates[time_index - 1], dtype=int)[:top_k]
        if prev_prev.size == 0 or prev.size == 0:
            continue
        decay = _transition_decay_at(velocity_decays, time_index - 1, velocity_decay)
        predictions = bin_centers[prev][None, :, :] + decay * (
            bin_centers[prev][None, :, :] - bin_centers[prev_prev][:, None, :]
        )
        _add_nearest_predictions(augmented[time_index], bin_centers, predictions)

    for time_index in range(len(candidates) - 2):
        decay = _transition_decay_at(velocity_decays, time_index + 1, velocity_decay)
        if abs(decay) <= np.finfo(float).eps:
            continue
        nxt = np.asarray(candidates[time_index + 1], dtype=int)[:top_k]
        nxt_nxt = np.asarray(candidates[time_index + 2], dtype=int)[:top_k]
        if nxt.size == 0 or nxt_nxt.size == 0:
            continue
        predictions = bin_centers[nxt][None, :, :] - (
            bin_centers[nxt_nxt][:, None, :] - bin_centers[nxt][None, :, :]
        ) / decay
        _add_nearest_predictions(augmented[time_index], bin_centers, predictions)

    return [np.fromiter(sorted(curr), dtype=int) for curr in augmented]


def _transition_decay_at(values: np.ndarray | None, transition_index: int, fallback: float) -> float:
    if values is None:
        return float(fallback)
    arr = np.asarray(values, dtype=float)
    if transition_index < 0 or transition_index >= arr.size:
        return float(fallback)
    return float(arr[transition_index])


def _add_nearest_predictions(
    target: set[int],
    bin_centers: np.ndarray,
    predictions: np.ndarray,
) -> None:
    flat = predictions.reshape(-1, bin_centers.shape[1])
    if flat.size == 0:
        return
    for predicted in flat:
        dist2 = np.sum((bin_centers - predicted[None, :]) ** 2, axis=1)
        target.add(int(np.argmin(dist2)))


StateSpaceReplayModel.score._native_duration_occupancy_aware = True  # type: ignore[attr-defined]
