"""PF-style geometric screening and two pre-decoding map-shuffle controls."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.sparse import csr_matrix

FAMILIES = ["cell_identity", "independent_xy_roll"]
CRITERIA = [(False, 10), (True, 10), (False, 11), (True, 11)]


def edge_support(counts, filtered):
    counts = np.asarray(counts)
    if counts.ndim != 2:
        raise ValueError("counts must be frames by cells")
    edges = np.flatnonzero(counts.sum(axis=1) >= 2)
    valid = np.zeros(len(counts), bool)
    if len(edges):
        valid[edges[0]:edges[-1] + 1] = True
    if filtered:
        valid &= (counts.sum(axis=1) >= 3) & ((counts > 0).sum(axis=1) >= 2)
    return valid


def screen_plan(counts, offsets):
    counts, offsets = np.asarray(counts), np.asarray(offsets, int)
    if offsets.ndim != 1 or len(offsets) < 1 or offsets[0] != 0 or offsets[-1] != len(counts) or (np.diff(offsets) < 0).any():
        raise ValueError("invalid observation offsets")
    masks = np.zeros((2, len(counts)), bool)
    for a, b in itertools.pairwise(offsets):
        masks[0, a:b] = edge_support(counts[a:b], False)
        masks[1, a:b] = edge_support(counts[a:b], True)
    return {"offsets": offsets, "masks": masks}


def screen_batch(map_indices, grid, plan):
    """Earliest longest runs, vectorized over shuffles and concatenated events."""
    paths, grid = np.asarray(map_indices, int), np.asarray(grid, float)
    offsets, masks = plan["offsets"], plan["masks"]
    if paths.ndim != 2 or paths.shape[1] != offsets[-1]:
        raise ValueError("paths must be shuffles by frames")
    output = np.zeros((len(paths), len(offsets) - 1, len(CRITERIA)), bool)
    if not paths.shape[1]:
        return output
    if grid.ndim != 2 or not np.isfinite(grid).all() or (paths < 0).any() or (paths >= len(grid)).any():
        raise ValueError("invalid spatial states")
    nonempty = np.flatnonzero(np.diff(offsets))
    boundaries = offsets[nonempty]
    adjacent = np.zeros(paths.shape, bool)
    # Decimal-offset grids can put exact 20/40 cm separations one ULP either side.
    adjacent[:, 1:] = np.linalg.norm(grid[paths[:, 1:]] - grid[paths[:, :-1]], axis=-1) < 20 - 1e-9
    adjacent[:, boundaries] = False
    frame = np.arange(paths.shape[1])
    for filtered, valid in enumerate(masks):
        continues = adjacent & valid[None] & np.r_[False, valid[:-1]][None]
        starts = np.maximum.accumulate(np.where(valid[None] & ~continues, frame[None], -1), axis=1)
        lengths = np.where(valid[None], frame[None] - starts + 1, 0)
        best = np.maximum.reduceat(lengths, boundaries, axis=1)
        expanded = np.repeat(best, np.diff(offsets)[nonempty], axis=1)
        ends = np.minimum.reduceat(np.where(lengths == expanded, frame[None], len(frame)), boundaries, axis=1)
        begins = np.maximum(0, ends - best + 1)
        # Zero-length runs cannot pass; clamp their unused endpoint for indexing.
        begins = np.minimum(begins, len(frame) - 1)
        row = np.arange(len(paths))[:, None]
        displacement = np.linalg.norm(grid[paths[row, ends]] - grid[paths[row, begins]], axis=-1)
        for c, (f, frames) in enumerate(CRITERIA):
            if f == bool(filtered):
                output[:, nonempty, c] = (best >= frames) & (displacement >= 40 - 1e-9)
    return output


def screen_maps(map_indices, grid, counts):
    return screen_batch(map_indices, grid, screen_plan(counts, [0, len(counts)]))[:, 0]


def shuffle_bank(n_cells, grid_shape, n_shuffles, seed, family):
    nx, ny = map(int, grid_shape)
    if min(nx, ny, n_cells, n_shuffles) < 1 or nx * ny < 2:
        raise ValueError("invalid population, grid or shuffle count")
    rng = np.random.default_rng(seed)
    if family == "cell_identity":
        return np.stack([rng.permutation(n_cells) for _ in range(n_shuffles)])
    if family == "independent_xy_roll":
        flat = rng.integers(1, nx * ny, size=(n_shuffles, n_cells))
        return np.stack([flat // ny, flat % ny], axis=-1)
    raise ValueError(f"unknown shuffle family {family}")


def bank_rates(rates, support, grid_shape, bank, family):
    rates, support = np.asarray(rates, float), np.asarray(support, bool)
    nx, ny = map(int, grid_shape)
    if rates.shape[1] != nx * ny or support.shape != (nx * ny,):
        raise ValueError("shuffle requires the complete rectangular map")
    if family == "cell_identity":
        return rates[bank][:, :, support]
    if family != "independent_xy_roll":
        raise ValueError("unknown shuffle family")
    x, y = np.unravel_index(np.flatnonzero(support), (nx, ny))
    indices = ((x[None, None, :] - bank[:, :, 0, None]) % nx) * ny + ((y[None, None, :] - bank[:, :, 1, None]) % ny)
    return rates[np.arange(len(rates))[None, :, None], indices]


def sparse_map(counts, rate_bank, frame_batch=1024):
    """Float64 sparse Poisson MAP; uniform prior, no temporal dependence."""
    counts, rates = np.asarray(counts), np.asarray(rate_bank, float)
    if rates.ndim != 3 or counts.ndim != 2 or counts.shape[1] != rates.shape[1] or min(rates.shape) < 1:
        raise ValueError("expected counts(time,cells), rates(shuffle,cells,states)")
    if not np.isfinite(rates).all() or (rates <= 0).any() or not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any() or frame_batch < 1:
        raise ValueError("positive rates, integer counts and positive batches required")
    k, cells, states = rates.shape
    weights = np.log(rates).transpose(1, 0, 2).reshape(cells, k * states)
    penalty = (.02 * rates.sum(axis=1)).reshape(-1)
    decoded = np.empty((k, len(counts)), np.int32)
    for a in range(0, len(counts), frame_batch):
        ll = csr_matrix(counts[a:a + frame_batch], dtype=np.float64) @ weights
        ll -= penalty
        decoded[:, a:a + frame_batch] = ll.reshape(-1, k, states).argmax(axis=2).T
    return decoded


def shuffle_test(counts, offsets, grid, rates, support, grid_shape, n_shuffles, seed, batch_size=8):
    """Keep all observations; skip Monte Carlo only if no criterion can pass."""
    if batch_size < 1 or n_shuffles < 1:
        raise ValueError("positive shuffle count and batch size required")
    counts, offsets = np.asarray(counts), np.asarray(offsets, int)
    rates, support = np.asarray(rates, float), np.asarray(support, bool)
    plan = screen_plan(counts, offsets)
    original_path = sparse_map(counts, rates[:, support][None])[0]
    original = screen_batch(original_path[None], grid, plan)[0]
    needed = original.any(axis=1)
    indices = np.flatnonzero(needed)
    lengths = np.diff(offsets)[indices]
    selected_offsets = np.r_[0, np.cumsum(lengths)]
    selected_frames = np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in indices]) if len(indices) else np.empty(0, int)
    selected_counts = counts[selected_frames]
    selected_plan = screen_plan(selected_counts, selected_offsets)
    null_pass = np.zeros((2, n_shuffles, len(indices), 4), bool)
    banks, samples = {}, {}
    for f, family in enumerate(FAMILIES):
        bank = shuffle_bank(len(rates), grid_shape, n_shuffles, seed + f, family)
        banks[family] = bank
        samples[family] = np.empty((0, len(selected_frames)), np.int32)
        if not len(indices):
            continue
        for a in range(0, n_shuffles, batch_size):
            b = min(a + batch_size, n_shuffles)
            paths = sparse_map(selected_counts, bank_rates(rates, support, grid_shape, bank[a:b], family))
            if a < 3:
                samples[family] = np.concatenate([samples[family], paths[:max(0, 3 - a)]])
            null_pass[f, a:b] = screen_batch(paths, grid, selected_plan)
    successes = null_pass.sum(axis=1)
    p = np.full((len(original), 4, 2), np.nan)
    p[indices] = (successes.transpose(1, 2, 0) + 1) / (n_shuffles + 1)
    p[~original] = np.nan
    accepted = original & np.all(p < .02, axis=-1)
    return {"original_path": original_path, "geometric_pass": original, "tested_observations": indices,
        "selected_frame_indices": selected_frames, "selected_offsets": selected_offsets, "null_pass": null_pass,
        "p_values": p, "accepted": accepted, "banks": banks, "sample_paths": samples}
