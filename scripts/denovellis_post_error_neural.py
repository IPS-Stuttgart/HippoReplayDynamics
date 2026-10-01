"""Flat-prior graph clusterless decoding, using preceding RUN marks only."""

from dataclasses import dataclass
import hashlib
import re

import numpy as np
from scipy.io import loadmat
from scipy.spatial import cKDTree, distance
from scipy.sparse.csgraph import shortest_path

try:
    from scripts.denovellis_post_error_core import day_epochs, field, normalize_likelihood
except ModuleNotFoundError:
    from denovellis_post_error_core import day_epochs, field, normalize_likelihood


@dataclass
class Graph:
    xy: np.ndarray
    segment: np.ndarray
    center_distance: np.ndarray
    distance: np.ndarray
    unique_masks: tuple
    routes: tuple
    coordinates: np.ndarray


def make_graph(coordinates, wells, center, outers, bin_cm=3.0):
    coords = np.asarray(coordinates, float)
    if coords.shape != (5, 4) or np.any(~np.isfinite(coords)):
        raise ValueError("Expected the released five-segment W-track")
    nodes = []
    def node(xy):
        for i, x in enumerate(nodes):
            if np.linalg.norm(x-xy) < 1e-6:
                return i
        nodes.append(xy)
        return len(nodes)-1
    edges = [(node(x[:2]), node(x[2:])) for x in coords]
    adj = np.full((len(nodes), len(nodes)), np.inf)
    np.fill_diagonal(adj, 0)
    for (a, b), x in zip(edges, coords, strict=True):
        adj[a, b] = adj[b, a] = np.linalg.norm(x[2:]-x[:2])
    distances = shortest_path(adj, directed=False)
    c = int(np.argmin(np.linalg.norm(np.asarray(nodes)-wells[center-1], axis=1)))
    centers, segs, offsets = [], [], []
    for i, x in enumerate(coords):
        length = np.linalg.norm(x[2:]-x[:2])
        n = int(np.ceil(length/bin_cm))
        for s in (np.arange(n)+.5)*length/n:
            centers.append(x[:2]+(x[2:]-x[:2])*s/length)
            segs.append(i)
            offsets.append(s)
    segs, offsets = np.asarray(segs), np.asarray(offsets)
    endpoint_a = np.array([edges[s][0] for s in segs])
    endpoint_b = np.array([edges[s][1] for s in segs])
    lengths = np.linalg.norm(coords[:, 2:]-coords[:, :2], axis=1)[segs]
    to_a, to_b = offsets, lengths-offsets
    geo = np.minimum.reduce([to_a[:, None]+distances[endpoint_a[:, None], endpoint_a[None, :]]+to_a[None, :],
                             to_a[:, None]+distances[endpoint_a[:, None], endpoint_b[None, :]]+to_b[None, :],
                             to_b[:, None]+distances[endpoint_b[:, None], endpoint_a[None, :]]+to_a[None, :],
                             to_b[:, None]+distances[endpoint_b[:, None], endpoint_b[None, :]]+to_b[None, :]])
    same = segs[:, None] == segs[None, :]
    geo[same] = np.abs(offsets[:, None]-offsets[None, :])[same]
    cd = np.minimum(distances[c, endpoint_a]+to_a, distances[c, endpoint_b]+to_b)
    masks = []
    for well in outers:
        leaf = int(np.argmin(np.linalg.norm(np.asarray(nodes)-wells[well-1], axis=1)))
        wd = np.minimum(distances[leaf, endpoint_a]+to_a, distances[leaf, endpoint_b]+to_b)
        masks.append(np.isclose(cd+wd, distances[c, leaf], atol=1e-6))
    if not np.all(np.isfinite(geo)) or not all(m.any() for m in masks):
        raise ValueError("Disconnected graph or missing center-to-outer routes")
    shared = masks[0] & masks[1]
    return Graph(np.asarray(centers), segs, cd, geo, tuple(m & ~shared for m in masks),
                 tuple((m, cd[m]) for m in masks), coords)


def marked_file(path):
    loaded = loadmat(path, squeeze_me=True, struct_as_record=False)
    data = loaded.get("filedata")
    if data is None:
        raise ValueError("Unsupported mark wrapper; no silent sorted-unit substitution")
    values = np.asarray(field(data, "params", []), float)
    names = [str(x).strip().lower() for x in np.asarray(field(data, "paramnames", []), object).reshape(-1)]
    wanted = ["time", *[f"channel {i} max" for i in range(1, 5)]]
    if values.ndim != 2 or len(names) != values.shape[1] or any(x not in names for x in wanted):
        raise ValueError("Named time and four waveform maxima are required")
    selected = values[:, [names.index(x) for x in wanted]]
    if np.any(~np.isfinite(selected)):
        raise ValueError("Nonfinite mark times/features; no silent row deletion")
    t = selected[:, 0]/10000.0
    if np.any(np.diff(t) < 0):
        raise ValueError("Mark timestamps reversed")
    return t, selected[:, 1:]


def hippocampal_tetrodes(folder, animal, day, epoch, areas):
    root = loadmat(folder/f"{animal}tetinfo.mat", squeeze_me=True, struct_as_record=False)["tetinfo"]
    days = np.asarray(root, object).reshape(-1)
    if not 1 <= day <= len(days):
        raise ValueError("Tetrode day metadata absent")
    epochs = np.asarray(days[day-1], object).reshape(-1)
    if not 1 <= epoch <= len(epochs):
        raise ValueError("Tetrode epoch metadata absent")
    return {i: str(field(x, "area", "")).upper() for i, x in enumerate(np.asarray(epochs[epoch-1], object).reshape(-1), 1)
            if str(field(x, "area", "")).upper() in areas}


def load_marks(folder, animal, day, epoch, areas):
    allowed = hippocampal_tetrodes(folder, animal, day, epoch, areas)
    groups = {}
    sources = []
    for path in sorted((folder/"EEG").glob(f"{animal}marks{day:02d}-*.mat")):
        match = re.fullmatch(rf"{animal}marks{day:02d}-(\d+)\.mat", path.name)
        tet = int(match[1]) if match else -1
        if tet in allowed:
            groups[tet] = marked_file(path)
            sources.append(path)
    if len(groups) < 2:
        raise ValueError("Fewer than two verified hippocampal mark tetrodes")
    return groups, sources


@dataclass
class Encoding:
    graph: Graph
    occupied: np.ndarray
    occupancy: np.ndarray
    rate: dict
    features: dict
    spatial: dict
    trees: dict
    mark_sigma: float

    def likelihood(self, starts, ends, marks):
        starts, ends = np.asarray(starts), np.asarray(ends)
        ll = -(ends-starts)[:, None]*sum(self.rate.values())[None, :]
        counts = np.zeros(len(starts), int)
        active = np.zeros(len(starts), int)
        for tet, reference in self.features.items():
            times, features = marks[tet]
            keep = (times >= starts[0]) & (times < ends[-1])
            times, features = times[keep], features[keep]
            bins = np.searchsorted(starts, times, side="right")-1
            keep = (bins >= 0) & (times < ends[np.maximum(bins, 0)])
            features, bins = features[keep], bins[keep]
            for start in range(0, len(features), 64):
                queries = features[start:start+64]
                d2 = distance.cdist(queries/self.mark_sigma, reference/self.mark_sigma, "sqeuclidean")
                neighbors = d2 <= 36
                neighbors[~neighbors.any(axis=1)] = True
                weight = np.exp(-.5*d2)*neighbors
                intensity = weight @ self.spatial[tet] / self.occupancy
                # Batched sums are the same six-bandwidth KDE, without per-spike gathers.
                np.add.at(ll, bins[start:start+64], np.log(np.maximum(intensity, np.finfo(float).tiny)))
            counts += np.bincount(bins, minlength=len(starts))
            for b in np.unique(bins):
                active[b] += 1
        ll[:, ~self.occupied] = -np.inf
        return normalize_likelihood(ll), counts, active


def fit_encoding(graph, time, xy, speed, marks, *, start, end, spatial_sigma=6., mark_sigma=24.):
    time = np.asarray(time)
    good = (time >= start) & (time < end) & np.isfinite(xy).all(axis=1) & np.isfinite(speed) & (speed > 4)
    dt = np.diff(time, append=time[-1])
    good &= (dt > 0) & (dt <= .25)
    if not good.any():
        raise ValueError("No valid moving RUN exposure")
    index = cKDTree(graph.xy).query(xy[good])[1]
    kernel = np.exp(-.5*(graph.distance/spatial_sigma)**2)
    occupancy = dt[good] @ kernel[index]
    occupied = occupancy >= .1
    occupancy = np.maximum(occupancy, 1e-12)
    rate, refs, spatial, trees = {}, {}, {}, {}
    for tet, (t, f) in marks.items():
        row = np.searchsorted(time, t, side="right")-1
        row = np.clip(row, 0, len(time)-1)
        keep = (t >= start) & (t < end) & good[row] & (t-time[row] <= .25)
        if not keep.any():
            continue
        f = f[keep]
        x = np.column_stack([np.interp(t[keep], time, xy[:, j]) for j in range(2)])
        where = cKDTree(graph.xy).query(x)[1]
        weights = kernel[where]
        rate[tet] = weights.sum(axis=0)/occupancy
        refs[tet], spatial[tet] = f, weights
        trees[tet] = cKDTree(f/mark_sigma)
    if len(refs) < 2:
        raise ValueError("Fewer than two tetrodes have moving-RUN encoding marks")
    return Encoding(graph, occupied, occupancy, rate, refs, spatial, trees, mark_sigma)


def preceding_run(folder, animal, day, epoch):
    path = folder/f"{animal}task{day:02d}.mat"
    epochs = day_epochs(loadmat(path, squeeze_me=True, struct_as_record=False)["task"], day)
    preceding = [i for i, x in enumerate(epochs, 1) if i < epoch and str(field(x, "type", "")).lower() == "run"]
    return max(preceding) if preceding else None


def event_seed(seed, *identity):
    return int.from_bytes(hashlib.sha256((str(seed)+":"+":".join(map(str, identity))).encode()).digest()[:8], "little")


def run_arm_qc(encoding, time, xy, speed, marks, *, start, end, dt=.020):
    starts = np.arange(start, end-dt+1e-9, dt)
    ends = starts+dt
    mid = starts+dt/2
    rows = np.clip(np.searchsorted(time, mid)-1, 0, len(time)-2)
    position = np.column_stack([np.interp(mid, time, xy[:, j]) for j in range(2)])
    finite = np.isfinite(position).all(axis=1)
    nearest = np.zeros(len(position), int)
    nearest[finite] = cKDTree(encoding.graph.xy).query(position[finite])[1]
    truth = np.full(len(starts), -1)
    for arm, mask in enumerate(encoding.graph.unique_masks):
        truth[mask[nearest]] = arm
    usable = (truth >= 0) & finite & np.isfinite(speed[rows]) & (speed[rows] > 4) & (time[rows+1]-time[rows] <= .25)
    # Decode all bins, including zero-spike bins; do not condition RUN quality on a sharp posterior.
    posterior, counts, active = encoding.likelihood(starts, ends, marks)
    masses = np.column_stack([posterior[:, mask].sum(axis=1) for mask in encoding.graph.unique_masks])
    predicted = masses.argmax(axis=1)
    recall = [float(np.mean(predicted[usable & (truth == arm)] == arm)) if np.any(usable & (truth == arm)) else None for arm in (0, 1)]
    return {"n_run_windows": len(starts), "n_arm_windows": int(usable.sum()), "arm0_recall": recall[0], "arm1_recall": recall[1],
            "balanced_accuracy": float(np.mean(recall)) if all(x is not None for x in recall) else None,
            "median_spikes_per_arm_window": float(np.median(counts[usable])) if usable.any() else None,
            "median_active_tetrodes_per_arm_window": float(np.median(active[usable])) if usable.any() else None}
