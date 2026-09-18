"""Reader and RUN-only calibration for the pinned Tirole two-track release.

Position.linear.linear is already in cm; position.linear.length is in metres.
This module does not infer replay identity or anatomical labels from filenames.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import filtfilt
from scipy.special import logsumexp


@dataclass
class TrackSession:
    name: str
    times: np.ndarray
    speed: np.ndarray
    sleepbox: np.ndarray
    positions: np.ndarray
    lengths_cm: np.ndarray
    unit_ids: np.ndarray
    original_ids: np.ndarray
    spike_times: np.ndarray
    spike_units: np.ndarray
    spike_samples: np.ndarray

    @property
    def n_tracks(self):
        return len(self.positions)

    def epoch_mask(self):
        return np.isfinite(self.positions).any(axis=0)

    def run_mask(self):
        return self.epoch_mask() & np.isfinite(self.speed) & (self.speed > 5) & (self.speed < 50)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nearest_samples(times, query):
    right = np.searchsorted(times, query).clip(1, len(times) - 1)
    left = right - 1
    index = np.where(query - times[left] < times[right] - query, left, right)
    return np.where((query >= times[0]) & (query <= times[-1]), index, -1)


def load_session(root, name):
    root = Path(root)
    c = loadmat(root / f"{name}_extracted_clusters.mat", simplify_cells=True)["clusters"]
    p = loadmat(root / f"{name}_extracted_position.mat", simplify_cells=True)["position"]
    tracks = p["linear"] if isinstance(p["linear"], list) else [p["linear"]]
    times = np.asarray(p["t"], float).reshape(-1)
    if times.size < 2 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError("position timestamps must be finite and strictly increasing")
    dt = np.diff(times)
    if not np.allclose(dt, np.median(dt), rtol=1e-4, atol=1e-6):
        raise ValueError("regular position timestamps required for occupancy integration")
    positions = np.stack([np.asarray(t["linear"], float).reshape(-1) for t in tracks])
    if np.isinf(positions).any():
        raise ValueError("infinite track position")
    lengths = np.array([float(t["length"]) * 100 for t in tracks])
    speed = np.asarray(p["v_cm"], float).reshape(-1)
    sleepbox = np.isfinite(np.asarray(p["sleepbox"], float).reshape(-1))
    if positions.shape[1] != len(times) or len(speed) != len(times) or len(sleepbox) != len(times):
        raise ValueError("position arrays do not share the regular timestamp grid")
    if (np.isfinite(positions).sum(axis=0) > 1).any():
        raise ValueError("overlapping track epochs")
    for x, length in zip(positions, lengths, strict=True):
        finite = x[np.isfinite(x)]
        if not finite.size or not np.isfinite(length) or length <= 0 or finite.min() < -1e-6 or finite.max() > length + 1e-6:
            raise ValueError("empty or out-of-range track position")
    conversion = np.atleast_2d(c["id_conversion"])
    if conversion.shape[1] != 2 or not np.isfinite(conversion).all() or not np.equal(conversion, np.floor(conversion)).all():
        raise ValueError("invalid unit identity conversion")
    conversion = conversion[np.argsort(conversion[:, 0])].astype(int)
    if len(np.unique(conversion[:, 0])) != len(conversion):
        raise ValueError("duplicate unit IDs")
    st = np.asarray(c["spike_times"], float).reshape(-1)
    ids = np.asarray(c["spike_id"]).reshape(-1)
    if len(st) != len(ids) or not np.isfinite(st).all() or not np.isin(ids, conversion[:, 0]).all():
        raise ValueError("invalid spike timestamps or unit IDs")
    order = np.argsort(st, kind="stable")
    st, ids = st[order], ids[order]
    units = np.searchsorted(conversion[:, 0], ids)
    return TrackSession(name, times, speed, sleepbox, positions, lengths, conversion[:, 0], conversion[:, 1], st, units, nearest_samples(times, st))


def _sample_mask_at_spikes(session, mask):
    good = session.spike_samples >= 0
    result = np.zeros(len(session.spike_times), bool)
    result[good] = mask[session.spike_samples[good]]
    return result


def fit_maps(session, training_samples=None, bin_cm=10.0, min_occupancy_s=0.2):
    """Source-aligned rate maps; QC and all count terms use training RUN only."""
    train = np.ones(len(session.times), bool) if training_samples is None else np.asarray(training_samples, bool)
    if train.shape != session.times.shape:
        raise ValueError("training mask shape mismatch")
    if not np.isfinite(bin_cm) or bin_cm <= 0 or min_occupancy_s <= 0:
        raise ValueError("positive spatial bin and occupancy required")
    if not np.allclose(session.lengths_cm, session.lengths_cm[0]):
        raise ValueError("equal-length tracks required for uniform joint track/position prior")
    nbins = round(session.lengths_cm[0] / bin_cm)
    if nbins < 5 or not np.isclose(nbins * bin_cm, session.lengths_cm[0]):
        raise ValueError("track length must be an integer number of at least five bins")
    edges = np.arange(nbins + 1) * bin_cm
    dt = float(np.median(np.diff(session.times)))
    n_units = len(session.unit_ids)
    rates, raw_rates, occupancy, counts = [], [], [], []
    for x in session.positions:
        use = train & session.run_mask() & np.isfinite(x)
        occ = np.histogram(x[use], edges)[0] * dt
        spike_use = _sample_mask_at_spikes(session, use)
        bins = np.searchsorted(edges, x[session.spike_samples[spike_use]], side="right") - 1
        bins = bins.clip(0, nbins - 1)
        hist = np.bincount(session.spike_units[spike_use] * nbins + bins, minlength=n_units * nbins).reshape(n_units, nbins)
        raw = np.divide(hist, occ, out=np.zeros_like(hist, float), where=occ > 0)
        # Forward/backward [1/2,1/2] is the author's 10-cm map smoother.
        smooth = np.maximum(filtfilt([0.5, 0.5], [1.0], raw, axis=1, padlen=3), 0)
        rates.append(smooth)
        raw_rates.append(raw)
        occupancy.append(occ)
        counts.append(hist)
    rates, raw_rates, occupancy, counts = map(np.asarray, (rates, raw_rates, occupancy, counts))
    epoch = train & session.epoch_mask()
    seconds = epoch.sum() * dt
    epoch_spikes = _sample_mask_at_spikes(session, epoch)
    mean_rate = np.bincount(session.spike_units[epoch_spikes], minlength=n_units) / max(seconds, dt)
    probabilities = np.divide(occupancy, occupancy.sum(axis=1, keepdims=True), out=np.zeros_like(occupancy), where=occupancy.sum(axis=1, keepdims=True) > 0)
    rate_mean = (rates * probabilities[:, None, :]).sum(axis=-1, keepdims=True)
    ratio = np.divide(rates, rate_mean, out=np.zeros_like(rates), where=rate_mean > 0)
    information = (probabilities[:, None, :] * ratio * np.log2(np.maximum(ratio, 1e-100))).sum(axis=-1)
    passing = (rates.max(axis=-1) >= 0.5) & (raw_rates.max(axis=-1) >= 1) & (mean_rate[None, :] <= 5) & (information > 0)
    return {
        "rates": rates,
        "raw_rates": raw_rates,
        "occupancy_s": occupancy,
        "spike_counts": counts,
        "mean_rate_hz": mean_rate,
        "information": information,
        "passing_by_track": passing,
        "common_units": np.flatnonzero(passing.all(axis=0)),
        "union_units": np.flatnonzero(passing.any(axis=0)),
        "valid_bins": occupancy >= min_occupancy_s,
        "bin_centers_cm": (edges[1:] + edges[:-1]) / 2,
    }


def decode_counts(counts, rates, duration_s, valid_bins, rate_floor_hz=1e-4):
    """Independent-bin Poisson posterior on track x position, no dynamics prior."""
    counts, rates = np.asarray(counts), np.asarray(rates)
    if counts.ndim != 2 or rates.ndim != 3 or counts.shape[1] != rates.shape[1] or not counts.shape[1]:
        raise ValueError("nonempty compatible time x unit counts and track x unit x position rates required")
    if not np.isfinite(counts).all() or (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError("counts must be finite nonnegative integers")
    if not np.isfinite(rates).all() or (rates < 0).any() or not np.isfinite(duration_s) or duration_s <= 0 or rate_floor_hz <= 0:
        raise ValueError("invalid rates or duration")
    valid = np.asarray(valid_bins, bool)
    if valid.shape != (rates.shape[0], rates.shape[2]) or not valid.any(axis=1).all():
        raise ValueError("each track requires valid position bins")
    lam = np.maximum(rates, rate_floor_hz)
    ll = np.einsum("tu,kup->tkp", counts, np.log(lam)) - duration_s * lam.sum(axis=1)[None, :, :]
    # Equal track prior, uniform over valid bins within each track.
    ll -= np.log(valid.sum(axis=1))[None, :, None]
    ll[:, ~valid] = -np.inf
    norm = logsumexp(ll, axis=(1, 2), keepdims=True)
    return np.exp(ll - norm)


def blocked_training_mask(times, origin, fold, n_folds=5, block_s=10.0, guard_s=1.0):
    """Exclude test blocks and one-second neighbours without reading spikes."""
    if n_folds < 2 or not 0 <= fold < n_folds or not 0 <= guard_s < block_s / 2:
        raise ValueError("invalid blocked-fold definition")
    relative = np.asarray(times) - origin
    block = np.floor(relative / block_s).astype(int)
    phase = relative - block * block_s
    test = block % n_folds == fold
    near_previous = (phase < guard_s) & ((block - 1) % n_folds == fold)
    near_next = (phase > block_s - guard_s) & ((block + 1) % n_folds == fold)
    return ~(test | near_previous | near_next)


def run_crossvalidation(session, n_folds=5, window_s=0.25):
    """RUN calibration with fold-internal cell QC and maps; no replay samples."""
    centers = np.arange(session.times[0] + window_s / 2, session.times[-1] - window_s / 2, window_s)
    start, end = centers - window_s / 2, centers + window_s / 2
    left = np.searchsorted(session.times, start)
    right = np.searchsorted(session.times, end)
    valid_window = right > left
    truth_track = np.full(len(centers), -1, int)
    truth_x = np.full(len(centers), np.nan)
    for k, x in enumerate(session.positions):
        ok = session.run_mask() & np.isfinite(x)
        bad = np.r_[0, np.cumsum(~ok)]
        good = valid_window & (bad[right] == bad[left])
        truth_track[good] = k
        cs = np.r_[0, np.cumsum(np.nan_to_num(x))]
        truth_x[good] = (cs[right[good]] - cs[left[good]]) / (right[good] - left[good])
    counts = np.zeros((len(centers), len(session.unit_ids)), int)
    edges = np.r_[start, end[-1]]
    for u in range(len(session.unit_ids)):
        counts[:, u] = np.histogram(session.spike_times[session.spike_units == u], edges)[0]
    test_fold = np.floor((centers - session.times[0]) / 10).astype(int) % n_folds
    rows = []
    for fold in range(n_folds):
        training = blocked_training_mask(session.times, session.times[0], fold, n_folds)
        maps = fit_maps(session, training)
        units = maps["common_units"]
        # Test windows cannot straddle a fold boundary.
        same_block = np.floor((start - session.times[0]) / 10) == np.floor((end - session.times[0] - 1e-7) / 10)
        use = np.flatnonzero((truth_track >= 0) & (test_fold == fold) & same_block)
        if len(units) < 5 or not maps["valid_bins"].any(axis=1).all():
            for k in range(session.n_tracks):
                rows.append(
                    {
                        "fold": fold,
                        "track": k + 1,
                        "n_test_windows": int((truth_track[use] == k).sum()),
                        "n_scored_windows": 0,
                        "n_units": len(units),
                        "status": "insufficient_training_maps",
                    }
                )
            continue
        posterior = decode_counts(counts[use][:, units], maps["rates"][:, units], window_s, maps["valid_bins"])
        track_mass = posterior.sum(axis=2)
        for k in range(session.n_tracks):
            take = truth_track[use] == k
            conditional = posterior[take, k] / np.maximum(track_mass[take, k, None], 1e-300)
            mean = conditional @ maps["bin_centers_cm"]
            errors = np.abs(mean - truth_x[use[take]])
            rows.append(
                {
                    "fold": fold,
                    "track": k + 1,
                    "n_test_windows": int(take.sum()),
                    "n_scored_windows": int(take.sum()),
                    "n_units": len(units),
                    "context_accuracy": float((track_mass[take].argmax(axis=1) == k).mean()) if take.any() else np.nan,
                    "median_position_error_cm": float(np.median(errors)) if len(errors) else np.nan,
                    "p90_position_error_cm": float(np.quantile(errors, 0.9)) if len(errors) else np.nan,
                    "valid_bin_fraction": float(maps["valid_bins"][k].mean()),
                    "status": "complete",
                }
            )
    return rows
