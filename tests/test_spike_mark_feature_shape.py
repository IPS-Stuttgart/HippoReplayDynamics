import numpy as np

from hipporeplayimm.clusterless import ClusterlessMarkConfig, fit_clusterless_mark_encoding
from hipporeplayimm.data import ReplaySession, SpikeMarkData
from hipporeplayimm.encoding import EncodingConfig


def test_one_dimensional_spike_marks_are_single_feature_clusterless_input() -> None:
    session = _one_dimensional_mark_session()
    assert session.spike_marks is not None

    assert session.spike_marks.n_spikes == 6
    assert session.spike_marks.n_features == 1
    assert session.has_spike_marks

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

    assert encoding.n_features == 1
    assert encoding.mark_mean.shape[1] == 1


def _one_dimensional_mark_session() -> ReplaySession:
    position_times = np.linspace(0.0, 3.0, 31)
    x = np.where(position_times < 1.5, 0.0, 10.0)
    y = np.zeros_like(x)
    position = np.column_stack([position_times, x, y, np.zeros_like(x)])
    mark_times = np.array([0.2, 0.4, 0.6, 2.2, 2.4, 2.6])
    cell_ids = np.ones(mark_times.shape[0], dtype=int)
    one_dimensional_marks = np.array([0.0, 0.1, -0.1, 10.0, 10.2, 9.8])

    return ReplaySession(
        rat="RatX",
        name="OpenOneDimensionalMarks",
        path=None,
        position=position,
        spikes=np.column_stack([mark_times, cell_ids]),
        tetrode_cell_ids=np.array([[1, 1]]),
        excitatory_neurons=np.array([1]),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.empty((0, 6)),
        run_times=np.array([[0.0, 3.0]]),
        sleep_box_immobile_times=np.empty((0, 2)),
        sleep_times=np.empty((0, 2)),
        rem_times=np.empty((0, 2)),
        well_sequence=None,
        metadata={},
        spike_marks=SpikeMarkData(
            times=mark_times,
            marks=one_dimensional_marks,
            source_file="synthetic.mat",
            source_variable="Spike_Amplitude_Marks",
            feature_names=("amp",),
            cell_ids=cell_ids,
        ),
    )
