import h5py
import numpy as np
import pytest
from scipy.io import savemat

from scripts.verify_tirole_first_pair_preflight import clock_check, read_clock


def test_clock_coverage_is_not_synchronization():
    result = clock_check(np.arange(0, 5, 0.001), 1000, 0, 4)
    assert result["required_interval_covered"]
    assert result["clock_offset_applied_s"] == 0
    assert not result["independent_spike_lfp_synchronization_established"]


def test_no_guessed_offset_or_missing_gap():
    t = np.arange(0, 5, 0.001)
    assert not clock_check(t + 100, 1000, 0, 4)["required_interval_covered"]
    assert not clock_check(t[(t < 2) | (t > 2.1)], 1000, 0, 4)["required_interval_covered"]


@pytest.mark.parametrize("times,sr", [([0, 1, 0.5], 1), ([0, float("nan"), 2], 1), ([0, 1, 2], 1000), ([0], 1), ([0, 1], 0)])
def test_invalid_clocks_fail(times, sr):
    with pytest.raises(ValueError):
        clock_check(times, sr, 0, 1)


def channels():
    return [
        {"channel_label": label, "time_scale": "seconds", "SR": 1000, "CSCtime": np.arange(0, 5, 0.001), "ripple_zscore": np.zeros(5000), "channel": i, "filename": f"CSC{i}.ncs"}
        for i, label in enumerate(["best_theta", "best_ripple"])
    ]


@pytest.mark.parametrize("storage", ["classic", "hdf5"])
def test_both_matlab_formats(tmp_path, storage):
    path = tmp_path / "clock.mat"
    data = channels()
    if storage == "classic":
        savemat(path, {"CSC": np.array(data, dtype=object)})
    else:
        with h5py.File(path, "w") as f:
            c = f.create_group("CSC")
            for field in data[0]:
                refs = c.create_dataset(field, (1, 2), dtype=h5py.ref_dtype)
                for k, row in enumerate(data):
                    value = row[field]
                    value = [ord(x) for x in value] if isinstance(value, str) else value
                    array = np.asarray(value).reshape(-1, 1)
                    refs[0, k] = f.create_dataset(f"{field}_{k}", data=array).ref
    result = read_clock(path, 0, 4)
    assert result["required_interval_covered"]
    assert result["channel"] == 1
    assert result["filename"] == "CSC1.ncs"


@pytest.mark.parametrize("case", ["unit", "length", "duplicate"])
def test_classic_metadata_fail_closed(tmp_path, case):
    data = channels()
    if case == "unit":
        data[1]["time_scale"] = "milliseconds"
    elif case == "length":
        data[1]["ripple_zscore"] = np.zeros(5)
    else:
        data[0]["channel_label"] = "best_ripple"
    path = tmp_path / "bad.mat"
    savemat(path, {"CSC": np.array(data, dtype=object)})
    with pytest.raises(ValueError):
        read_clock(path, 0, 4)
