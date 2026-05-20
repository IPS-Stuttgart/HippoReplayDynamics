import numpy as np
import pytest
from scipy.special import logsumexp

from hipporeplayimm.clusterless import (
    ClusterlessMarkConfig,
    ClusterlessStateSpaceReplayModel,
    build_clusterless_mark_emissions,
    fit_clusterless_mark_encoding,
)
from hipporeplayimm.data import ReplaySession, SpikeMarkData
from hipporeplayimm.encoding import EmissionConfig, EncodingConfig, LogEmissionTensor
from hipporeplayimm.state_space import StateSpaceDecoderConfig


def test_clusterless_emissions_use_mark_likelihood_to_localize_spikes():
    session = _clusterless_session()
    encoding = fit_clusterless_mark_encoding(
        session,
        ClusterlessMarkConfig(
            encoding=EncodingConfig(
                bin_size_cm=10.0,
                smoothing_sigma_bins=0.0,
                min_speed_cm_s=0.0,
                arena_padding_cm=5.0,
            ),
            mark_smoothing_sigma_bins=0.0,
            mark_prior_count=0.1,
            mark_variance_floor=0.05,
        ),
    )

    emissions = build_clusterless_mark_emissions(
        session,
        encoding,
        0,
        EmissionConfig(time_bin_s=1.0),
    )
    left_bin = int(np.argmin(np.linalg.norm(encoding.bin_centers - np.array([0.0, 0.0]), axis=1)))
    right_bin = int(np.argmin(np.linalg.norm(encoding.bin_centers - np.array([10.0, 0.0]), axis=1)))

    assert emissions.n_spikes == 1
    assert emissions.spike_counts.shape == (1, 1)
    assert emissions.log_likelihood[0, left_bin] > emissions.log_likelihood[0, right_bin]
    assert emissions.metadata["clusterless_mark_likelihood"] == "local-kde"


def test_clusterless_tetrode_grouping_uses_group_specific_likelihood_and_rate():
    session = _grouped_clusterless_session()
    encoding = fit_clusterless_mark_encoding(
        session,
        ClusterlessMarkConfig(
            encoding=EncodingConfig(
                bin_size_cm=10.0,
                smoothing_sigma_bins=0.0,
                min_speed_cm_s=0.0,
                arena_padding_cm=5.0,
            ),
            mark_likelihood="diagonal-gaussian",
            mark_group_by="tetrode",
            mark_smoothing_sigma_bins=0.0,
            mark_prior_count=0.1,
            mark_variance_floor=0.05,
            rate_floor_hz=1e-4,
        ),
    )

    emissions = build_clusterless_mark_emissions(
        session,
        encoding,
        0,
        EmissionConfig(time_bin_s=1.0),
    )
    left_bin = int(np.argmin(np.linalg.norm(encoding.bin_centers - np.array([0.0, 0.0]), axis=1)))
    right_bin = int(np.argmin(np.linalg.norm(encoding.bin_centers - np.array([10.0, 0.0]), axis=1)))

    assert encoding.n_mark_groups == 2
    assert emissions.metadata["clusterless_mark_group_by"] == "tetrode"
    assert emissions.metadata["clusterless_mark_groups"] == 2
    assert emissions.log_likelihood[0, right_bin] > emissions.log_likelihood[0, left_bin]


def test_clusterless_local_kde_preserves_multimodal_mark_structure():
    session = _multimodal_clusterless_session()
    common = dict(
        encoding=EncodingConfig(
            bin_size_cm=10.0,
            smoothing_sigma_bins=0.0,
            min_speed_cm_s=0.0,
            arena_padding_cm=5.0,
        ),
        mark_smoothing_sigma_bins=0.0,
        mark_prior_count=0.0,
        mark_variance_floor=0.01,
    )
    kde_encoding = fit_clusterless_mark_encoding(
        session,
        ClusterlessMarkConfig(
            **common,
            mark_likelihood="local-kde",
            mark_kde_spatial_sigma_bins=0.0,
            mark_kde_bandwidth=0.2,
            mark_kde_max_neighbors=4,
        ),
    )
    gaussian_encoding = fit_clusterless_mark_encoding(
        session,
        ClusterlessMarkConfig(**common, mark_likelihood="diagonal-gaussian"),
    )
    left_bin = int(np.argmin(np.linalg.norm(kde_encoding.bin_centers - np.array([0.0, 0.0]), axis=1)))

    kde_logp = kde_encoding.log_mark_likelihood(np.array([[0.05], [5.05]]))[:, left_bin]
    gaussian_logp = gaussian_encoding.log_mark_likelihood(np.array([[0.05], [5.05]]))[:, left_bin]

    assert kde_encoding.mark_likelihood == "local-kde"
    assert kde_logp[0] > kde_logp[1]
    assert gaussian_encoding.mark_likelihood == "diagonal-gaussian"
    assert gaussian_logp[1] > gaussian_logp[0]


def test_clusterless_diagonal_gaussian_mark_likelihood_remains_available():
    session = _clusterless_session()
    encoding = fit_clusterless_mark_encoding(
        session,
        ClusterlessMarkConfig(
            encoding=EncodingConfig(
                bin_size_cm=10.0,
                smoothing_sigma_bins=0.0,
                min_speed_cm_s=0.0,
                arena_padding_cm=5.0,
            ),
            mark_likelihood="diagonal-gaussian",
            mark_smoothing_sigma_bins=0.0,
            mark_prior_count=0.1,
            mark_variance_floor=0.05,
        ),
    )

    likelihood = encoding.log_mark_likelihood(np.array([[0.0]]))

    assert encoding.mark_likelihood == "diagonal-gaussian"
    assert likelihood.shape == (1, encoding.n_bins)
    assert encoding.mark_kde_neighbor_indices is None


def test_clusterless_emissions_clip_bins_to_ripple_end_and_ignore_post_ripple_marks():
    session = _clusterless_session()
    old_marks = session.spike_marks
    assert old_marks is not None
    session.spike_marks = SpikeMarkData(
        times=np.append(old_marks.times, 5.1),
        marks=np.vstack([old_marks.marks, [[0.0]]]),
        source_file=old_marks.source_file,
        source_variable=old_marks.source_variable,
        feature_names=old_marks.feature_names,
        cell_ids=np.append(old_marks.cell_ids, 1) if old_marks.cell_ids is not None else None,
    )
    encoding = fit_clusterless_mark_encoding(
        session,
        ClusterlessMarkConfig(
            encoding=EncodingConfig(
                bin_size_cm=10.0,
                smoothing_sigma_bins=0.0,
                min_speed_cm_s=0.0,
                arena_padding_cm=5.0,
            ),
            mark_smoothing_sigma_bins=0.0,
            mark_prior_count=0.1,
            mark_variance_floor=0.05,
            mark_likelihood="diagonal-gaussian",
        ),
    )

    emissions = build_clusterless_mark_emissions(
        session,
        encoding,
        0,
        EmissionConfig(time_bin_s=0.6),
    )

    assert emissions.times == pytest.approx(np.array([4.3, 4.8]))
    assert emissions.dt == pytest.approx(0.5)
    assert emissions.n_spikes == 1
    assert emissions.spike_counts[:, 0].tolist() == [1, 0]
    assert np.allclose(emissions.log_likelihood[-1], -encoding.rate_hz * 0.4)


def test_clusterless_emissions_apply_spike_rate_scale_to_intensity():
    session = _clusterless_session()
    encoding = fit_clusterless_mark_encoding(
        session,
        ClusterlessMarkConfig(
            encoding=EncodingConfig(
                bin_size_cm=10.0,
                smoothing_sigma_bins=0.0,
                min_speed_cm_s=0.0,
                arena_padding_cm=5.0,
            ),
            mark_smoothing_sigma_bins=0.0,
            mark_prior_count=0.1,
            mark_variance_floor=0.05,
            mark_likelihood="diagonal-gaussian",
        ),
    )

    base = build_clusterless_mark_emissions(
        session,
        encoding,
        0,
        EmissionConfig(time_bin_s=0.6, spike_rate_scale=1.0),
    )
    scaled = build_clusterless_mark_emissions(
        session,
        encoding,
        0,
        EmissionConfig(time_bin_s=0.6, spike_rate_scale=2.0),
    )

    expected_delta = np.empty_like(base.log_likelihood)
    expected_delta[0] = np.log(2.0) - encoding.rate_hz * 0.6
    expected_delta[1] = -encoding.rate_hz * 0.4
    assert np.allclose(scaled.log_likelihood - base.log_likelihood, expected_delta)


def test_clusterless_emissions_reject_nonpositive_spike_rate_scale():
    session = _clusterless_session()
    encoding = fit_clusterless_mark_encoding(
        session,
        ClusterlessMarkConfig(
            encoding=EncodingConfig(
                bin_size_cm=10.0,
                smoothing_sigma_bins=0.0,
                min_speed_cm_s=0.0,
                arena_padding_cm=5.0,
            ),
            mark_smoothing_sigma_bins=0.0,
            mark_prior_count=0.1,
            mark_variance_floor=0.05,
        ),
    )

    with pytest.raises(ValueError, match="spike_rate_scale must be positive"):
        build_clusterless_mark_emissions(
            session,
            encoding,
            0,
            EmissionConfig(time_bin_s=1.0, spike_rate_scale=0.0),
        )


def test_clusterless_encoding_excludes_ripple_intervals_by_default():
    session = _clusterless_session()
    base_kwargs = dict(
        bin_size_cm=10.0,
        smoothing_sigma_bins=0.0,
        min_speed_cm_s=0.0,
        arena_padding_cm=5.0,
    )
    clusterless_kwargs = dict(
        mark_likelihood="diagonal-gaussian",
        mark_smoothing_sigma_bins=0.0,
        mark_prior_count=0.0,
        mark_variance_floor=0.05,
    )

    excluded = fit_clusterless_mark_encoding(
        session,
        ClusterlessMarkConfig(
            encoding=EncodingConfig(**base_kwargs, exclude_ripple_intervals=True),
            **clusterless_kwargs,
        ),
    )
    included = fit_clusterless_mark_encoding(
        session,
        ClusterlessMarkConfig(
            encoding=EncodingConfig(**base_kwargs, exclude_ripple_intervals=False),
            **clusterless_kwargs,
        ),
    )
    left_bin = int(np.argmin(np.linalg.norm(excluded.bin_centers - np.array([0.0, 0.0]), axis=1)))
    right_bin = int(np.argmin(np.linalg.norm(excluded.bin_centers - np.array([10.0, 0.0]), axis=1)))

    assert excluded.effective_spike_count[left_bin] == pytest.approx(3.0)
    assert included.effective_spike_count[left_bin] == pytest.approx(3.0)
    assert excluded.mark_mean[left_bin, 0] == pytest.approx(0.0)
    assert excluded.effective_spike_count[right_bin] == pytest.approx(3.0)
    assert included.effective_spike_count[right_bin] == pytest.approx(4.0)
    assert excluded.mark_mean[right_bin, 0] == pytest.approx(10.0)
    assert included.mark_mean[right_bin, 0] == pytest.approx(7.5125)


def test_clusterless_encoding_requires_spike_marks():
    session = _clusterless_session()
    session.spike_marks = None

    with pytest.raises(ValueError, match="spike marks"):
        fit_clusterless_mark_encoding(session)


def test_clusterless_encoding_rejects_unknown_mark_likelihood():
    session = _clusterless_session()

    with pytest.raises(ValueError, match="Unknown clusterless mark likelihood"):
        fit_clusterless_mark_encoding(session, ClusterlessMarkConfig(mark_likelihood="not-a-model"))


def test_clusterless_state_space_model_reports_clusterless_diagnostics():
    emissions = LogEmissionTensor(
        log_likelihood=np.log(np.array([[0.7, 0.3], [0.2, 0.8]])),
        spike_counts=np.array([[1], [0]]),
        times=np.array([0.0, 0.003]),
        dt=0.003,
        cell_ids=np.array([0]),
        n_spikes=1,
    )
    centers = np.array([[0.0, 0.0], [1.0, 0.0]])
    model = ClusterlessStateSpaceReplayModel(
        mode="diffusion",
        config=StateSpaceDecoderConfig(mode="diffusion", diffusion_sigma_cm_sqrt_s=1.0, max_step_sigma=10.0),
    )

    score = model.score(emissions, centers)

    assert score.model_name == "clusterless-state-space-diffusion"
    assert np.isfinite(score.log_likelihood)
    assert score.trajectory_log_posterior is not None
    assert np.allclose(logsumexp(score.trajectory_log_posterior, axis=1), 0.0)
    assert score.diagnostics["state_space_observation_model"] == "clusterless-marked-point-process"
    assert score.diagnostics["clusterless_mark_likelihood"] == "local-kde"


def _clusterless_session() -> ReplaySession:
    position_times = np.linspace(0.0, 5.0, 51)
    x = np.where(position_times < 2.5, 0.0, 10.0)
    y = np.zeros_like(x)
    position = np.column_stack([position_times, x, y, np.zeros_like(x)])
    mark_times = np.array([0.2, 0.4, 0.6, 3.2, 3.4, 3.6, 4.2])
    cell_ids = np.ones(mark_times.shape[0], dtype=int)
    marks = np.array([[0.0], [0.1], [-0.1], [10.0], [10.2], [9.8], [0.05]])
    spikes = np.column_stack([mark_times, cell_ids])
    return ReplaySession(
        rat="RatX",
        name="OpenX",
        path=None,
        position=position,
        spikes=spikes,
        tetrode_cell_ids=np.array([[1, 1]]),
        excitatory_neurons=np.array([1]),
        inhibitory_neurons=np.array([]),
        ripple_events=np.array([[4.0, 5.0, 4.5, 0.0, 0.0, 0.0]]),
        run_times=np.array([[0.0, 5.0]]),
        sleep_box_immobile_times=np.empty((0, 2)),
        sleep_times=np.empty((0, 2)),
        rem_times=np.empty((0, 2)),
        well_sequence=None,
        metadata={},
        spike_marks=SpikeMarkData(
            times=mark_times,
            marks=marks,
            source_file="Spike_Data.mat",
            source_variable="Spike_Amplitude_Marks",
            feature_names=("amp",),
            cell_ids=cell_ids,
        ),
    )


def _grouped_clusterless_session() -> ReplaySession:
    position_times = np.linspace(0.0, 5.0, 51)
    x = np.where(position_times < 2.5, 0.0, 10.0)
    y = np.zeros_like(x)
    position = np.column_stack([position_times, x, y, np.zeros_like(x)])
    mark_times = np.array([0.2, 0.4, 0.6, 3.2, 3.4, 3.6, 4.2])
    cell_ids = np.array([1, 1, 1, 2, 2, 2, 2], dtype=int)
    group_ids = np.array([1, 1, 1, 2, 2, 2, 2], dtype=int)
    marks = np.array([[0.0], [0.1], [-0.1], [0.0], [0.2], [-0.2], [0.05]])
    spikes = np.column_stack([mark_times, cell_ids])
    return ReplaySession(
        rat="RatX",
        name="OpenGrouped",
        path=None,
        position=position,
        spikes=spikes,
        tetrode_cell_ids=np.array([[1, 1], [2, 2]]),
        excitatory_neurons=np.array([1, 2]),
        inhibitory_neurons=np.array([]),
        ripple_events=np.array([[4.0, 5.0, 4.5, 0.0, 0.0, 0.0]]),
        run_times=np.array([[0.0, 5.0]]),
        sleep_box_immobile_times=np.empty((0, 2)),
        sleep_times=np.empty((0, 2)),
        rem_times=np.empty((0, 2)),
        well_sequence=None,
        metadata={},
        spike_marks=SpikeMarkData(
            times=mark_times,
            marks=marks,
            source_file="Spike_Data.mat",
            source_variable="Spike_Amplitude_Marks",
            feature_names=("amp",),
            cell_ids=cell_ids,
            group_ids=group_ids,
        ),
    )


def _multimodal_clusterless_session() -> ReplaySession:
    session = _clusterless_session()
    mark_times = np.array([0.2, 0.4, 0.6, 0.8, 3.2, 3.4, 3.6, 3.8, 4.2])
    cell_ids = np.ones(mark_times.shape[0], dtype=int)
    marks = np.array([[0.0], [0.1], [10.0], [10.1], [4.9], [5.0], [5.1], [5.2], [0.05]])
    session.spikes = np.column_stack([mark_times, cell_ids])
    session.spike_marks = SpikeMarkData(
        times=mark_times,
        marks=marks,
        source_file="Spike_Data.mat",
        source_variable="Spike_Amplitude_Marks",
        feature_names=("amp",),
        cell_ids=cell_ids,
    )
    return session
