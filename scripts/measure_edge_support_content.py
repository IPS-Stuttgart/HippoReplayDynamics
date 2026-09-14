#!/usr/bin/env python3
"""Paired fixed/trimmed endpoint readouts and known-path falsification controls."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial.distance import cdist

from hipporeplayimm.encoding import _speed_cm_s
from hipporeplayimm.replay_coverage_data import count_candidate_bins, index_spike_times
from scripts._provenance import build_script_provenance, file_sha256
from scripts.measure_population_content_stability import decode, seed_for

POLICIES = ("raw_endpoint", "pooled_two_spike_edge", "a_supported_edge", "joint_supported_edge")
SIMULATORS = ("sim_stationary", "sim_moving", "sim_moving_gain", "sim_late_jump")


def window_sums(base):
    base = np.asarray(base)
    if base.ndim != 2 or len(base) < 4 or (base < 0).any() or (base != np.floor(base)).any():
        raise ValueError("need >=4 full nonnegative integer base bins")
    cumulative = np.vstack([np.zeros((1, base.shape[1]), dtype=np.int64), np.cumsum(base, axis=0)])
    return cumulative[4:] - cumulative[:-4]


def select_windows(a_counts, b_counts):
    if a_counts.shape != b_counts.shape or a_counts.ndim != 2 or not len(a_counts):
        raise ValueError("invalid equal-population count matrices")
    a_ok = (a_counts.sum(axis=1) >= 3) & ((a_counts > 0).sum(axis=1) >= 2)
    b_ok = (b_counts.sum(axis=1) >= 3) & ((b_counts > 0).sum(axis=1) >= 2)
    masks = (np.ones(len(a_counts), bool), (a_counts+b_counts).sum(axis=1) >= 2, a_ok, a_ok & b_ok)
    return {name: int(np.flatnonzero(mask)[-1]) if mask.any() else -1
            for name, mask in zip(POLICIES, masks, strict=True)}


def occupied_graph(grid):
    distances = cdist(grid, grid)
    edges = np.where((distances > 0) & (distances <= 8*np.sqrt(2)+1e-7), distances, 0)
    graph = csr_matrix(edges)
    _, labels = connected_components(graph, directed=False)
    component = np.flatnonzero(labels == np.argmax(np.bincount(labels)))
    if len(component) < 2:
        raise ValueError("no connected occupied-grid simulation path")
    distance, predecessors = shortest_path(graph, directed=False, return_predecessors=True)
    return component, distance, predecessors


def known_path(grid, rates, graph, n_base, kind, seed):
    """1-ms interpolation on a fixed occupied-bin geodesic, not arena-wall truth."""
    component, distances, previous = graph
    rng = np.random.default_rng(seed)
    origin = int(rng.choice(component))
    target = int(component[np.argmax(distances[origin, component])])
    reverse = [target]
    while reverse[-1] != origin:
        reverse.append(int(previous[origin, reverse[-1]]))
        if reverse[-1] < 0 or len(reverse) > len(grid):
            raise ValueError("invalid geodesic predecessor chain")
    route = np.asarray(reverse[::-1])
    lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(grid[route], axis=0), axis=1))]
    length = lengths[-1]
    fine_t = (np.arange(n_base*5)+.5)*.001
    phase = (rng.uniform(0, 2*length) + 1000*fine_t) % (2*length)
    travelled = length-np.abs(phase-length)
    right = np.clip(np.searchsorted(lengths, travelled, side="right"), 1, len(route)-1)
    fraction = (travelled-lengths[right-1])/(lengths[right]-lengths[right-1])
    left_ids, right_ids = route[right-1], route[right]
    xy = grid[left_ids]*(1-fraction[:, None])+grid[right_ids]*fraction[:, None]
    intensity = rates[:, left_ids].T*(1-fraction[:, None])+rates[:, right_ids].T*fraction[:, None]
    if kind == "sim_stationary":
        xy[:] = grid[origin]
        intensity[:] = rates[:, origin]
    elif kind == "sim_late_jump":
        earlier = xy[max(0, len(xy)-21)]
        jump = int(component[np.argmax(np.linalg.norm(grid[component]-earlier, axis=1))])
        xy[-20:] = grid[jump]
        intensity[-20:] = rates[:, jump]
    elif kind not in ("sim_moving", "sim_moving_gain"):
        raise ValueError("unknown simulation")
    return xy.reshape(n_base, 5, 2).mean(axis=1), intensity.reshape(n_base, 5, len(rates)).mean(axis=1)


def conditional_draw(totals, intensities, seed, gains=False):
    rng = np.random.default_rng(seed)
    rates = np.array(intensities, copy=True)
    if gains:
        rates *= rng.lognormal(0, .4, rates.shape[1])[None, :]
    probabilities = rates/rates.sum(axis=1, keepdims=True)
    sample = np.asarray([rng.multinomial(int(n), p) for n, p in zip(totals, probabilities, strict=True)])
    if not np.array_equal(sample.sum(axis=1), totals):
        raise ValueError("simulation did not preserve whole-population totals")
    return sample


def read_event(base, rates, grid, groups, start, event_id, truth_base=None):
    counts = window_sums(base)
    a_ids, b_ids = groups
    a, b = counts[:, a_ids], counts[:, b_ids]
    selected = select_windows(a, b)
    raw = len(counts)-1
    wanted = sorted(set([raw]+[i for i in selected.values() if i >= 0]))
    indices = {i: j for j, i in enumerate(wanted)}
    da, db = decode(a[wanted], rates[a_ids], grid), decode(b[wanted], rates[b_ids], grid)
    truths = None if truth_base is None else (np.cumsum(np.vstack([np.zeros((1, 2)), truth_base]), axis=0)[4:]
                                             -np.cumsum(np.vstack([np.zeros((1, 2)), truth_base]), axis=0)[:-4])/4
    raw_truth = None if truths is None else truths[-1]

    def metrics(index):
        at = indices[index]
        result = dict(a_spikes=int(a[index].sum()), b_spikes=int(b[index].sum()),
            a_active=int((a[index] > 0).sum()), b_active=int((b[index] > 0).sum()),
            a_entropy=float(da["entropy"][at]), b_entropy=float(db["entropy"][at]),
            a_width_cm=float(da["width"][at]), b_width_cm=float(db["width"][at]),
            separation_cm=float(np.linalg.norm(da["mean"][at]-db["mean"][at])),
            regional_tv=float(.5*np.abs(da["regional"][at]-db["regional"][at]).sum()))
        for name, decoded in (("a", da), ("b", db)):
            for j in range(9):
                result[f"{name}_region{j}"] = float(decoded["regional"][at, j])
            for dim, label in enumerate(("x", "y")):
                result[f"{name}_{label}_cm"] = float(decoded["mean"][at, dim])
            result[f"{name}_selected_truth_error_cm"] = float(np.linalg.norm(decoded["mean"][at]-truths[index])) if truths is not None else np.nan
            result[f"{name}_original_truth_error_cm"] = float(np.linalg.norm(decoded["mean"][at]-raw_truth)) if truths is not None else np.nan
        return result

    baseline = metrics(raw)
    rows = []
    for policy, index in selected.items():
        record = dict(policy=policy, event_index=int(event_id), raw_window_index=raw,
            selected_window_index=index, status="available" if index >= 0 else "abstain",
            candidate_start_s=start, raw_start_s=start+.005*raw, raw_end_s=start+.005*raw+.02,
            selected_start_s=start+.005*index if index >= 0 else np.nan,
            selected_end_s=start+.005*index+.02 if index >= 0 else np.nan,
            shift_earlier_ms=5*(raw-index) if index >= 0 else np.nan,
            raw_a_activity_inadequate=baseline["a_spikes"] < 3 or baseline["a_active"] < 2,
            n_cells_per_group=len(a_ids))
        record.update({f"raw_{key}": value for key, value in baseline.items()})
        record.update(metrics(index) if index >= 0 else {key: np.nan for key in baseline})
        record["truth_shift_cm"] = float(np.linalg.norm(truths[index]-raw_truth)) if index >= 0 and truths is not None else np.nan
        rows.append(record)
    return rows


def q4_run_events(data, ids):
    start, end = data["run_bounds_s"]
    start = start+.75*(end-start)
    t, xy = data["position"][:, 0], data["position"][:, 1:3]
    speed = _speed_cm_s(t, xy)
    intervals = data["supported_run_intervals"]
    valid = []
    for a in np.arange(start, end-.2, .5):
        b = a+.2
        left, right = np.searchsorted(t, [a, b])
        left, right = max(0, left-1), min(len(t), right+1)
        if (right-left < 2 or t[left] > a or t[right-1] < b or np.max(np.diff(t[left:right])) > .1
                or not np.isfinite(xy[left:right]).all()
                or not ((speed[left:right] >= 10) & (speed[left:right] <= 200)).all()
                or not np.any((intervals[:, 0] <= a) & (intervals[:, 1] >= b))):
            continue
        valid.append(a)
    if len(valid) < 32:
        raise ValueError(f"insufficient q4 supported RUN pseudo-events: {len(valid)}")
    selected = np.asarray(valid)[np.linspace(0, len(valid)-1, min(200, len(valid)), dtype=int)]
    indexed = index_spike_times(data["spikes"], ids)
    results = []
    for i, a in enumerate(selected):
        edges, counts = count_candidate_bins(indexed, ids, a, a+.2, .005)
        fine = a+(np.arange(200)+.5)*.001
        truth = np.column_stack([np.interp(fine, t, xy[:, d]) for d in range(2)]).reshape(40, 5, 2).mean(axis=1)
        if len(counts) != 40:
            raise ValueError("q4 pseudo-event base clock mismatch")
        results.append((i, a, counts, truth))
    return results


def measure_session(row, output, seed):
    path = Path(row.artifact_path)
    if file_sha256(path) != row.artifact_sha256:
        raise ValueError("encoding input changed")
    with np.load(path, allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    keep, support = data["unit_qc_mask"].astype(bool), data["valid_spatial_bins"].astype(bool)
    ids = data["cell_ids"][keep]
    grid = data["bin_centers_cm"][support]
    rates = np.maximum(data["rates_hz"][keep][:, support], 1e-4)
    if len(ids) < 10 or len(grid) < 2:
        raise ValueError("insufficient training-QC cells or spatial bins")
    groups = []
    for split in range(3):
        order = np.random.default_rng(seed_for(seed, "edge_support_partition", row.dataset, row.session, split)).permutation(len(ids))
        k = len(ids)//2
        groups.append((np.sort(order[:k]), np.sort(order[k:2*k])))
    graph = occupied_graph(grid)
    target = output/(row.animal+"__"+row.session.replace("/", "_"))
    target.mkdir(exist_ok=False)
    freeze = dict(dataset=row.dataset, animal=row.animal, session=row.session, seed=seed,
        encoding_path=str(path), encoding_sha256=row.artifact_sha256,
        groups=[dict(split=i, a_ids=ids[a].tolist(), b_ids=ids[b].tolist()) for i, (a, b) in enumerate(groups)],
        simulation_largest_component_fraction=len(graph[0])/len(grid),
        created_at_utc=datetime.now(UTC).isoformat(), selected_candidates=int(row.selected_candidates))
    (target/"frozen_measurement.json").write_text(json.dumps(freeze, indent=2)+"\n")
    raw_events = []
    for i, event_id in enumerate(data["candidate_event_indices"]):
        left, right = data["candidate_offsets"][i:i+2]
        durations = data["candidate_base_durations_s"][left:right]
        full = np.flatnonzero(np.isclose(durations, .005, atol=1e-9, rtol=0))
        if len(full) < 4 or not np.array_equal(full, np.arange(len(full))):
            raise ValueError("sampled candidate has invalid complete-bin clock")
        base = data["candidate_base_counts"][left+full][:, keep]
        raw_events.append((int(event_id), float(data["candidate_start_s"][i]), base, None))
    failures = []
    cases = {"real": raw_events}
    try:
        cases["run_q4"] = q4_run_events(data, ids)
    except ValueError as exc:
        failures.append(dict(source="run_q4", reason=str(exc)))
    for kind in SIMULATORS:
        events = []
        for event_id, start, base, _ in raw_events:
            truth, intensity = known_path(grid, rates, graph, len(base), kind,
                seed_for(seed, "edge_support_path", row.dataset, row.session, event_id))
            generated = conditional_draw(base.sum(axis=1), intensity,
                seed_for(seed, "edge_support_draw", row.dataset, row.session, event_id, kind),
                gains=kind == "sim_moving_gain")
            events.append((event_id, start, generated, truth))
        cases[kind] = events
    rows = []
    for source, events in cases.items():
        counts, truths, offsets = [], [], [0]
        for event_id, start, base, truth in events:
            counts.append(base)
            truths.append(np.full((len(base), 2), np.nan) if truth is None else truth)
            offsets.append(offsets[-1]+len(base))
            for split, pair in enumerate(groups):
                local = read_event(base, rates, grid, pair, start, event_id, truth)
                for record in local:
                    record.update(dataset=row.dataset, animal=row.animal, session=row.session, split=split, source=source)
                rows.extend(local)
        np.savez_compressed(target/f"{source}_audit.npz", counts=np.concatenate(counts), truth_base_cm=np.concatenate(truths),
                            offsets=np.asarray(offsets), event_ids=np.asarray([e[0] for e in events]),
                            starts_s=np.asarray([e[1] for e in events]), cell_ids=ids, rates_hz=rates, grid_cm=grid)
    pd.DataFrame(rows).to_csv(target/"edge_readouts.csv.gz", index=False)
    (target/"source_failures.json").write_text(json.dumps(failures, indent=2)+"\n")
    (target/"outputs.json").write_text(json.dumps({p.name: file_sha256(p) for p in target.iterdir() if p.is_file()}, indent=2)+"\n")
    if file_sha256(path) != row.artifact_sha256:
        raise ValueError("encoding cache changed during measurement")
    return dict(dataset=row.dataset, animal=row.animal, session=row.session,
        status="complete" if not failures else "partial", selected_candidates=len(raw_events),
        source_candidates=int(row.source_candidates), training_qc_cells=len(ids),
        simulation_largest_component_fraction=len(graph[0])/len(grid),
        sources=len(cases), rows=len(rows), artifact_dir=str(target), source_failures=failures)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260914)
    args = p.parse_args()
    inputs = dict(catalog=args.input_dir/"encoding_sessions.csv", script=Path(__file__),
                  protocol=ROOT/"docs/edge_support_content_protocol.md",
                  decoder=ROOT/"scripts/measure_population_content_stability.py")
    manifest = build_script_provenance(input_paths=inputs, cwd=ROOT)
    manifest.update(status="running", seed=args.seed, created_at_utc=datetime.now(UTC).isoformat())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    rows = []
    for row in pd.read_csv(inputs["catalog"]).itertuples(index=False):
        try:
            if row.status != "cached":
                raise ValueError("encoding unavailable")
            result = measure_session(row, args.output_dir, args.seed)
        except (ValueError, OSError, KeyError) as exc:
            result = dict(dataset=row.dataset, animal=row.animal, session=row.session, status="failed", reason=str(exc),
                          source_candidates=row.source_candidates, selected_candidates=row.selected_candidates)
        rows.append(result)
        print(json.dumps(result), flush=True)
        pd.DataFrame(rows).to_csv(args.output_dir/"measurement_sessions.csv", index=False)
    unchanged = all(file_sha256(path) == manifest["input_file_sha256"][key] for key, path in inputs.items())
    manifest.update(status="complete" if unchanged else "failed", inputs_unchanged=unchanged, results=rows,
                    completed_at_utc=datetime.now(UTC).isoformat())
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    if not unchanged:
        raise ValueError("source changed during measurement")


if __name__ == "__main__":
    main()
