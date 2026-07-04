import numpy as np
import pytest

from hipporeplayimm.clusterless import ClusterlessMarkConfig, fit_clusterless_mark_encoding
from hipporeplayimm.data import ReplaySession, SpikeMarkData
from hipporeplayimm.encoding import EncodingConfig


def test_clusterless_tetrode_grouping_rejects_misaligned_group_ids():
    session = _clusterless_session_with_group_ids(np.array([1, 1, 2], dtype=int))

    with pytest.raises(ValueError, match="clusterless tetrode group IDs must contain one value per spike mark row"):
        fit_clusterless_mark_encoding(
            session,
            ClusterlessMarkConfig(
                encoding=_encoding_config(),
                mark_likelihood="diagonal-gaussian",
                mark_group_by="tetrode",
                mark_smoothing_sigma_bins=0.0,
                mark_prior_count=0.1,
                mark_variance_floor=0.05,
            ),
        )


def test_clusterless_cell_grouping_rejects_misaligned_cell_ids():
    session = _clusterless_session_with_group_ids(np.array([1, 1, 2, 2], dtype=int))
    old_marks = session.spike_marks
    assert old_marks is not None
    session.spike_marks = SpikeMarkData(
        times=old_marks.times,
        marks=old_marks.marks,
        source_file=old_marks.source_file,
        source_variable=old_marks.source_variable,
        feature_names=old_marks.feature_names,
        cell_ids=np.array([1, 1, 2], dtype=int),
        group_ids=old_marks.group_ids,
    )

    with pytest.raises(ValueError, match="clusterless cell group IDs must contain one value per spike mark row"):
        fit_clusterless_mark_encoding(
            session,
            ClusterlessMarkConfig(
                encoding=_encoding_config(),
                mark_likelihood="diagonal-gaussian",
                mark_group_by="cell",
                mark_smoothing_sigma_bins=0.0,
                mark_prior_count=0.1,
                mark_variance_floor=0.05,
            ),
        )


def _encoding_config() -> EncodingConfig:
    return EncodingConfig(
        bin_size_cm=10.0,
        smoothing_sigma_bins=0.0,
        min_speed_cm_s=0.0,
        arena_padding_cm=5.0,
    )


def _clusterless_session_with_group_ids(group_ids: np.ndarray) -> ReplaySession:
    position_times = np.linspace(0.0, 2.0, 21)
    x = np.where(position_times < 1.0, 0.0, 10.0)
    y = np.zeros_like(x)
    position = np.column_stack([position_times, x, y, np.zeros_like(x)])
    mark_times = np.array([0.2, 0.4, 1.2, 1.4])
    cell_ids = np.array([1, 1, 2, 2], dtype=int)
    marks = np.array([[0.0], [0.1], [1.0], [1.1]])
    spikes = np.column_stack([mark_times, cell_ids])
    return ReplaySession(
        rat="RatX",
        name="OpenGroupedMisaligned",
        path=None,
        position=position,
        spikes=spikes,
        tetrode_cell_ids=np.array([[1, 1], [2, 2]]),
        excitatory_neurons=np.array([1, 2]),
        inhibitory_neurons=np.array([]),
        ripple_events=np.array([[1.6, 1.8, 1.7, 0.0, 0.0, 0.0]]),
        run_times=np.array([[0.0, 2.0]]),
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
