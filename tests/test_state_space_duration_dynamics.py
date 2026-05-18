import itertools

import numpy as np
from scipy.special import logsumexp

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.encoding import EncodingConfig, EmissionConfig, EncodingModel, LogEmissionTensor
from hipporeplayimm.state_space import StateSpaceDecoderConfig, StateSpaceReplayModel


def test_state_space_four_mode_imm_uses_transition_durations():
    emissions = LogEmissionTensor(
        log_likelihood=np.log(
            np.array(
                [
                    [0.60, 0.30, 0.10],
                    [0.20, 0.70, 0.10],
                    [0.05, 0.30, 0.65],
                ]
            )
        ),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.0, 1.0, 5.0]),
        dt=1.0,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    centers = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    config = StateSpaceDecoderConfig(
        mode="imm",
        stationary_sigma_cm=1.0,
        diffusion_sigma_cm_sqrt_s=1.0,
        imm_mode_stickiness=0.8,
        momentum_sigma_cm_sqrt_s=1.0,
        momentum_initial_sigma_cm_sqrt_s=1.0,
        momentum_velocity_decay=1.0,
        momentum_candidate_top_k=3,
    )

    score = StateSpaceReplayModel(mode="imm", config=config).score(emissions, centers)

    modes = ("stationary", "diffusion", "momentum", "jump")
    mode_transition = np.full((4, 4), (1.0 - config.imm_mode_stickiness) / 3.0)
    np.fill_diagonal(mode_transition, config.imm_mode_stickiness)
    durations = np.array([1.0, 4.0])
    diffusion_sigmas = config.diffusion_sigma_cm_sqrt_s * np.sqrt(durations)
    momentum_sigmas = config.momentum_sigma_cm_sqrt_s * np.sqrt(durations)
    momentum_initial_sigma = config.momentum_initial_sigma_cm_sqrt_s * np.sqrt(durations[0])
    momentum_coefficients = np.ones_like(durations)
    momentum_coefficients[1:] = durations[1:] / durations[:-1]
    momentum_coefficients *= config.momentum_velocity_decay ** (durations / 1.0)

    def kernel_log(str_mode: str, src: int, prev: int, dst: int, *, transition_index: int, initial: bool = False) -> float:
        if str_mode == "jump":
            return -np.log(centers.shape[0])
        if initial:
            sigma = (
                config.stationary_sigma_cm
                if str_mode == "stationary"
                else diffusion_sigmas[0]
                if str_mode == "diffusion"
                else momentum_initial_sigma
            )
            predicted = centers[src]
        elif str_mode == "stationary":
            sigma = config.stationary_sigma_cm
            predicted = centers[prev]
        elif str_mode == "diffusion":
            sigma = diffusion_sigmas[transition_index]
            predicted = centers[prev]
        elif str_mode == "momentum":
            sigma = momentum_sigmas[transition_index]
            predicted = centers[prev] + momentum_coefficients[transition_index] * (centers[prev] - centers[src])
        else:
            raise AssertionError(str_mode)
        weights = np.exp(-0.5 * np.sum((centers - predicted) ** 2, axis=1) / (sigma * sigma))
        return float(np.log(weights[dst] / weights.sum()))

    brute_terms = []
    for x0, x1, x2 in itertools.product(range(centers.shape[0]), repeat=3):
        for mode1_idx, mode1 in enumerate(modes):
            for mode2_idx, mode2 in enumerate(modes):
                brute_terms.append(
                    -np.log(centers.shape[0])
                    + emissions.log_likelihood[0, x0]
                    - np.log(len(modes))
                    + kernel_log(mode1, x0, x0, x1, transition_index=0, initial=True)
                    + emissions.log_likelihood[1, x1]
                    + np.log(mode_transition[mode1_idx, mode2_idx])
                    + kernel_log(mode2, x0, x1, x2, transition_index=1)
                    + emissions.log_likelihood[2, x2]
                )

    assert np.allclose(score.log_likelihood, logsumexp(brute_terms))
    assert score.diagnostics["state_space_transition_durations"] == "1,4"


def test_duration_patch_updates_imported_build_emissions_aliases():
    import hipporeplayimm.benchmarks as benchmarks
    import hipporeplayimm.encoding as encoding
    import hipporeplayimm.ground_truth as ground_truth

    assert benchmarks.build_emissions is encoding.build_emissions
    assert ground_truth.build_emissions is encoding.build_emissions

    emissions = benchmarks.build_emissions(
        _single_partial_bin_ripple_session(),
        _single_cell_encoding(),
        0,
        EmissionConfig(time_bin_s=0.02),
    )

    assert hasattr(emissions, "transition_durations")
    np.testing.assert_allclose(emissions.transition_durations, np.array([0.02, 0.0165]))


def _single_partial_bin_ripple_session() -> ReplaySession:
    return ReplaySession(
        rat="RatX",
        name="OpenX",
        path=None,
        position=np.empty((0, 4)),
        spikes=np.array([[0.049, 1.0]]),
        tetrode_cell_ids=np.array([[1, 1]]),
        excitatory_neurons=np.array([1]),
        inhibitory_neurons=np.array([]),
        ripple_events=np.array([[0.0, 0.053, 0.0265, 0.0, 0.0, 0.0]]),
        run_times=np.empty((0, 2)),
        sleep_box_immobile_times=np.empty((0, 2)),
        sleep_times=np.empty((0, 2)),
        rem_times=np.empty((0, 2)),
        well_sequence=None,
        metadata={},
    )


def _single_cell_encoding() -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([0.0, 1.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.5, 0.5]]),
        rates_hz=np.array([[10.0]]),
        occupancy_s=np.array([1.0]),
        cell_ids=np.array([1]),
        config=EncodingConfig(),
    )
