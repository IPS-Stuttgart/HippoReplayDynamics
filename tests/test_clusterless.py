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


def test_clusterless_encoding_excludes_ripple_intervals_by_default():
    session = _clusterless_session()
    base_kwargs = dict(
        bin_size_cm=10.0,
        smoothing_sigma_bins=0.0,
        min_speed_cm_s=0.0,
        arena_padding_cm=5.0,
    )
    clusterless_kwargs = dict(
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

    assert excluded.effective_spike_count[left_bin] == pytest.approx(3.0)
    assert included.effective_spike_count[left_bin] == pytest.approx(4.0)
    assert included.mark_mean[left_bin, 0] == pytest.approx(0.0125)
    assert excluded.mark_mean[left_bin, 0] == pytest.approx(0.0)


def test_clusterless_encoding_requires_spike_marks():
    session = _clusterless_session()
    session.spike_marks = None

    with pytest.raises(ValueError, match="spike marks"):
        fit_clusterless_mark_encoding(session)


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
    assert score.diagnostics["clusterless_mark_likelihood"] == "diagonal-gaussian"


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
