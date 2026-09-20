"""Exercise the complete small-file preflight, including JSON serialization."""

import json

import h5py
import numpy as np

from scripts.preflight_dandi000978 import inspect


def test_nwb_metadata_report_is_json_serializable_and_keeps_mapping_warning(tmp_path):
    path = tmp_path / "sample.nwb"
    with h5py.File(path, "w") as f:
        f.create_dataset("general/subject/subject_id", data="synthetic")
        epochs = f.create_group("intervals/epoch intervals")
        epochs.create_dataset("start_time", data=[0.0])
        epochs.create_dataset("stop_time", data=[10.0])
        trials = f.create_group("intervals/trials")
        for key, values in {
            "id": [0, 1],
            "start_time": [1.0, 4.0],
            "stop_time": [3.0, 6.0],
            "correct": [1, 1],
            "start_well": [1, 2],
            "end_well": [2, 1],
            "trajectory_type": [0, 1],
        }.items():
            trials.create_dataset(key, data=values)
        pos = f.create_group("processing/behavior/Position/SpatialSeries")
        pos.create_dataset("timestamps", data=np.linspace(0.5, 9.5, 20))
        pos.create_dataset("data", data=np.ones((20, 3)))
        pos["data"].attrs["unit"] = "centimeters; centimeters/second"
        units = f.create_group("units")
        units.create_dataset("id", data=[10, 11])
        units.create_dataset("spike_times_index", data=[3, 6])
        units.create_dataset("spike_times", data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        refs = units.create_dataset("electrodes", data=[0, 1])
        table = f.create_group("general/extracellular_ephys/electrodes")
        table.create_dataset("location", data=[b"CA1", b"PFC"])
        table.create_dataset("group_name", data=[b"tetrode1", b"tetrode2"])
        refs.attrs["table"] = table.ref
        lfp = f.create_group("processing/ecephys/LFP/ElectricalSeries")
        lfp.create_dataset("timestamps", data=np.linspace(0.0, 10.0, 1000))
        lfp.create_dataset("data", data=np.zeros((1000, 2)))
    with h5py.File(path, "r") as f:
        report, epochs, trials, units, arrays = inspect(f)
    restored = json.loads(json.dumps(report))
    assert restored["position_in_lfp_clock_support"]
    assert restored["spikes_in_lfp_clock_support"]
    assert restored["unit_reference_hypotheses_disagree_count"] == 1
    assert not restored["region_specific_RUN_scoring_allowed"]
    assert not restored["replay_scored"]
    assert len(epochs) == 1 and len(trials) == 2 and len(units) == 2
    assert len(arrays["spikes"]) == 6
