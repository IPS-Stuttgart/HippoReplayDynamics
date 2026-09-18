"""Read supplied Hilbert-envelope z scores, not filtered LFP voltage.

Pinned author code uses 125-300 Hz and a 15-sample envelope smoother.
These files do not independently establish spike/LFP synchronization.
"""

from pathlib import Path

import h5py
import numpy as np


def _vector(dataset, start=None, stop=None):
    if dataset.ndim == 1:
        return np.asarray(dataset[slice(start, stop)])
    if dataset.ndim == 2 and dataset.shape[0] == 1:
        return np.asarray(dataset[0, slice(start, stop)])
    if dataset.ndim == 2 and dataset.shape[1] == 1:
        return np.asarray(dataset[slice(start, stop), 0])
    raise ValueError("expected MATLAB vector")


def _text(dataset):
    return "".join(chr(int(x)) for x in _vector(dataset))


def load_supplied_ripple(path, start_s=None, end_s=None):
    with h5py.File(Path(path), "r") as f:
        c = f["CSC"]
        labels = [_text(f[r]) for r in c["channel_label"][:].ravel()]
        if labels.count("best_ripple") != 1:
            raise ValueError("exactly one author-selected best_ripple channel required")
        k = labels.index("best_ripple")
        get = lambda name: f[c[name][:].ravel()[k]]
        if _text(get("time_scale")) != "seconds":
            raise ValueError("LFP timestamps must be explicitly labelled seconds")
        sr = float(_vector(get("SR"))[0])
        times = _vector(get("CSCtime"))
        if sr <= 0 or not np.isfinite(sr) or len(times) < 2 or not np.isfinite(times).all() or not (np.diff(times) > 0).all():
            raise ValueError("invalid LFP sampling rate or timestamps")
        if not np.isclose(np.median(np.diff(times)), 1 / sr, rtol=0.02):
            raise ValueError("LFP timestamp spacing does not match the stated sample rate")
        z = get("ripple_zscore")
        if z.size != len(times):
            raise ValueError("ripple/timestamp length mismatch")
        left = 0 if start_s is None else max(0, int(np.searchsorted(times, start_s)) - 1)
        right = len(times) if end_s is None else min(len(times), int(np.searchsorted(times, end_s)) + 1)
        values = _vector(z, left, right)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite supplied ripple envelope")
        return {
            "times": times[left:right],
            "zscore": values,
            "sample_rate_hz": sr,
            "channel": int(_vector(get("channel"))[0]),
            "channel_label": labels[k],
            "full_start_s": float(times[0]),
            "full_end_s": float(times[-1]),
            "full_samples": len(times),
            "filename": _text(get("filename")),
            "source_signal": "author_supplied_smoothed_Hilbert_envelope_zscore",
        }


def window_ripple_stats(ripple, start, end):
    t, z, sr = ripple["times"], ripple["zscore"], ripple["sample_rate_hz"]
    if not np.isfinite([start, end]).all() or end <= start:
        raise ValueError("finite positive event interval required")
    a, b = np.searchsorted(t, [start, end], side="left")
    n = b - a
    coverage = min(1.0, n / (sr * (end - start)))
    supported = n >= 2 and start >= t[0] and end <= t[-1] + 1 / sr and coverage >= 0.98
    if supported:
        expanded = t[max(0, a - 1) : min(len(t), b + 1)]
        supported = bool(np.max(np.diff(expanded)) <= 2.01 / sr)
    return {
        "ripple_supported": bool(supported),
        "ripple_coverage_fraction": coverage,
        "ripple_peak_z": float(np.max(z[a:b])) if supported else np.nan,
        "ripple_mean_z": float(np.mean(z[a:b])) if supported else np.nan,
    }
