import h5py
import numpy as np
import pytest

from hipporeplayimm.tirole_ripple import load_supplied_ripple, window_ripple_stats


def make_file(path, spacing=0.001, units="seconds", mismatch=False):
    with h5py.File(path, "w") as f:
        g = f.create_group("CSC")
        vals = {
            "channel_label": "best_ripple",
            "time_scale": units,
            "SR": 1000.0,
            "channel": 58.0,
            "filename": "CSC58.ncs",
            "CSCtime": np.arange(1000) * spacing + 10.0,
            "ripple_zscore": np.linspace(-1, 5, 999 if mismatch else 1000),
        }
        for name, value in vals.items():
            if isinstance(value, str):
                data = np.array([ord(c) for c in value], dtype=np.uint16)[:, None]
            else:
                data = np.atleast_1d(value).reshape(1, -1)
            d = f.create_dataset("value_" + name, data=data)
            ref = g.create_dataset(name, (1, 1), dtype=h5py.ref_dtype)
            ref[0, 0] = d.ref


def test_read_supplied_envelope_and_crop(tmp_path):
    p = tmp_path / "ripple.mat"
    make_file(p)
    r = load_supplied_ripple(p, 10.2, 10.5)
    assert r["channel"] == 58
    assert r["times"][0] < 10.2 and r["times"][-1] >= 10.5
    assert r["full_samples"] == 1000
    stats = window_ripple_stats(r, 10.25, 10.4)
    assert stats["ripple_supported"]
    assert np.isfinite(stats["ripple_peak_z"])


@pytest.mark.parametrize("kwargs", [{"spacing": 0.002}, {"units": "milliseconds"}, {"mismatch": True}])
def test_invalid_sampling_rejected(tmp_path, kwargs):
    p = tmp_path / "ripple.mat"
    make_file(p, **kwargs)
    with pytest.raises(ValueError):
        load_supplied_ripple(p)


def test_no_interpolation_over_lfp_gaps():
    t = np.r_[np.arange(0, 0.04, 0.001), np.arange(0.06, 0.1, 0.001)]
    r = {"times": t, "zscore": np.ones(len(t)), "sample_rate_hz": 1000}
    stats = window_ripple_stats(r, 0.02, 0.08)
    assert not stats["ripple_supported"]
    assert np.isnan(stats["ripple_peak_z"])


def test_no_extrapolation_outside_lfp():
    r = {"times": np.arange(0.1, 0.2, 0.001), "zscore": np.ones(100), "sample_rate_hz": 1000}
    assert not window_ripple_stats(r, 0.0, 0.2)["ripple_supported"]


def test_mismatched_recording_clock_is_not_silently_cropped(tmp_path):
    p = tmp_path / "ripple.mat"
    make_file(p)
    with pytest.raises(ValueError, match="does not overlap"):
        load_supplied_ripple(p, 100.0, 110.0)
