from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hipporeplayimm.clusterless import ClusterlessMarkConfig, ClusterlessMarkEncoding, _mark_group_ids_for_config
from hipporeplayimm.data import ReplaySession, SpikeMarkData



def _encoding_with_group_ids(group_ids: np.ndarray) -> ClusterlessMarkEncoding:
    return ClusterlessMarkEncoding(
        x_edges=np.array([0.0, 1.0]),
        y_edges=np.array([0.0, 1.0]),
        bin_centers=np.array([[0.0, 0.0]]),
        rate_hz=np.array([1.0]),
        occupancy_s=np.array([1.0]),
        effective_spike_count=np.array([1.0]),
        mark_mean=np.array([[0.0]]),
        mark_variance=np.array([[1.0]]),
        mark_feature_names=("mark0",),
        spike_mark_source="synthetic:marks",
        config=ClusterlessMarkConfig(mark_likelihood="diagonal-gaussian", mark_group_by="tetrode"),
        mark_likelihood="diagonal-gaussian",
        group_ids=np.asarray(group_ids),
        group_rate_hz=np.ones((len(group_ids), 1)),
        group_effective_spike_count=np.ones((len(group_ids), 1)),
        group_mark_mean=np.zeros((len(group_ids), 1, 1)),
        group_mark_variance=np.ones((len(group_ids), 1, 1)),
    )



def _session_with_mark_groups(group_ids: np.ndarray) -> ReplaySession:
    return ReplaySession(
        rat="RatX",
        name="OpenX",
        path=Path("synthetic"),
        position=np.array([[0.0, 0.0, 0.0, 0.0]]),
        spikes=np.array([[0.0, 1.0], [1.0, 2.0]]),
        tetrode_cell_ids=np.empty((0, 2)),
        excitatory_neurons=np.array([], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.empty((0, 6)),
        run_times=np.empty((0, 2)),
        sleep_box_immobile_times=np.empty((0, 2)),
        sleep_times=np.empty((0, 2)),
        rem_times=np.empty((0, 2)),
        well_sequence=None,
        metadata={},
        spike_marks=SpikeMarkData(
            times=np.array([0.0, 1.0]),
            marks=np.array([[0.0], [1.0]]),
            source_file="synthetic.mat",
            source_variable="Marks",
            feature_names=("mark0",),
            cell_ids=np.array([1, 2]),
            group_ids=np.asarray(group_ids),
        ),
    )



def test_clusterless_mark_likelihood_rejects_boolean_group_ids():
    encoding = _encoding_with_group_ids(np.array([0, 1]))

    with pytest.raises(ValueError, match="boolean"):
        encoding.log_mark_likelihood(np.array([[0.0]]), group_ids=np.array([True]))



def test_clusterless_mark_likelihood_rejects_mixed_python_boolean_group_ids():
    encoding = _encoding_with_group_ids(np.array([0, 1, 2]))

    with pytest.raises(ValueError, match="boolean"):
        encoding._coerce_group_indices([True, 2], n_marks=2)



def test_clusterless_mark_likelihood_rejects_out_of_range_group_ids():
    encoding = _encoding_with_group_ids(np.array([0, 1]))

    with pytest.raises(ValueError, match="integer identifier range"):
        encoding.log_mark_likelihood(np.array([[0.0]]), group_ids=np.array([1e100]))



def test_clusterless_mark_likelihood_maps_object_string_group_ids_numerically():
    encoding = _encoding_with_group_ids(np.array(["1", "10", "2"], dtype=object))

    group_indices = encoding._coerce_group_indices(np.array(["2", "10", "1"], dtype=object), n_marks=3)

    assert group_indices.tolist() == [2, 1, 0]



def test_clusterless_tetrode_group_extraction_rejects_boolean_group_ids():
    session = _session_with_mark_groups(np.array([True, False]))

    with pytest.raises(ValueError, match="boolean"):
        _mark_group_ids_for_config(session, ClusterlessMarkConfig(mark_group_by="tetrode"))



def test_clusterless_tetrode_group_extraction_rejects_mixed_python_boolean_group_ids():
    session = _session_with_mark_groups(np.array([1, 2]))
    assert session.spike_marks is not None
    session.spike_marks = replace(session.spike_marks, group_ids=[True, 2])

    with pytest.raises(ValueError, match="boolean"):
        _mark_group_ids_for_config(session, ClusterlessMarkConfig(mark_group_by="tetrode"))



def test_clusterless_mark_group_validation_patch_refreshes_stale_helpers(monkeypatch):
    import hipporeplayimm
    import hipporeplayimm.clusterless as clusterless

    def stale_mark_group_ids_for_config(session, config):
        return np.asarray([True, False])

    def stale_coerce_group_indices(self, group_ids, n_marks: int):
        return np.full(int(n_marks), -1, dtype=int)

    monkeypatch.setattr(clusterless, "_mark_group_ids_for_config", stale_mark_group_ids_for_config)
    monkeypatch.setattr(clusterless.ClusterlessMarkEncoding, "_coerce_group_indices", stale_coerce_group_indices)

    hipporeplayimm.apply_runtime_patches()

    session = _session_with_mark_groups(np.array([True, False]))
    with pytest.raises(ValueError, match="boolean"):
        clusterless._mark_group_ids_for_config(session, ClusterlessMarkConfig(mark_group_by="tetrode"))

    encoding = _encoding_with_group_ids(np.array([0, 1]))
    with pytest.raises(ValueError, match="boolean"):
        encoding._coerce_group_indices([True], n_marks=1)
