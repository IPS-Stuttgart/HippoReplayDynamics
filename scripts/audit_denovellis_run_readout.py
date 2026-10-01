#!/usr/bin/env python3
"""Bounded RUN-only decoder diagnostics; never qualifies a biological cohort."""

from __future__ import annotations

import argparse
import ast
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import signal
import subprocess
import time as clock

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.spatial import cKDTree, distance

try:
    from scripts._provenance import build_script_provenance, file_sha256
    from scripts.analyze_denovellis_post_error_content import ANIMALS, atomic_json, epoch_data
    from scripts.denovellis_post_error_core import day_epochs, field, normalize_likelihood
    from scripts.denovellis_post_error_neural import Encoding, hippocampal_tetrodes, load_marks, make_graph
except ModuleNotFoundError:
    from _provenance import build_script_provenance, file_sha256
    from analyze_denovellis_post_error_content import ANIMALS, atomic_json, epoch_data
    from denovellis_post_error_core import day_epochs, field, normalize_likelihood
    from denovellis_post_error_neural import Encoding, hippocampal_tetrodes, load_marks, make_graph

PREFIX = "denovellis_run_audit_"


def matching_context(source, target, tolerance=10., check_segments=True):
    se = str(field(source[8], "environment", field(source[8], "description", "")))
    te = str(field(target[8], "environment", field(target[8], "description", "")))
    if not se or not te:
        return False, "missing_context"
    if se != te:
        return False, "environment_mismatch"
    if source[5:7] != target[5:7]:
        return False, "well_identity_mismatch"
    if np.any(~np.isfinite(source[3])) or np.any(~np.isfinite(target[3])):
        return False, "nonfinite_wells"
    if np.max(np.linalg.norm(source[3]-target[3], axis=1)) > tolerance:
        return False, "well_geometry_mismatch"
    if not check_segments:
        return True, "match"
    if np.any(~np.isfinite(source[4])) or np.any(~np.isfinite(target[4])):
        return False, "nonfinite_segments"
    if np.max(np.linalg.norm(source[4].reshape(-1, 2)-target[4].reshape(-1, 2), axis=1)) > tolerance:
        return False, "segment_geometry_mismatch"
    return True, "match"


def choose_matching(candidates):
    """Latest verified matching predecessor, with no performance inputs."""
    matching = [x["source_epoch"] for x in candidates if x["context_matches"]]
    return max(matching) if matching else None


def sample_windows(graph, time, xy, speed, start, end, dt, cap):
    starts = np.arange(start, end-dt+1e-9, dt)
    mid = starts+dt/2
    row = np.clip(np.searchsorted(time, mid)-1, 0, len(time)-2)
    position = np.column_stack([np.interp(mid, time, xy[:, j]) for j in range(2)])
    finite = np.isfinite(position).all(axis=1)
    nearest = np.zeros(len(mid), int)
    nearest[finite] = cKDTree(graph.xy).query(position[finite])[1]
    truth = np.full(len(mid), -1)
    for arm, mask in enumerate(graph.unique_masks):
        truth[mask[nearest]] = arm
    usable = finite & np.isfinite(speed[row]) & (speed[row] > 4) & (time[row+1]-time[row] <= .25)
    pools = [np.flatnonzero(usable & (truth == arm)) for arm in (0, 1)]
    selected = np.sort(np.concatenate([pool[np.linspace(0, len(pool)-1, min(cap, len(pool)), dtype=int)]
                                       for pool in pools if len(pool)])) if any(len(x) for x in pools) else np.array([], int)
    if not len(selected) or not all(len(x) for x in pools):
        raise ValueError("Both held-out unique arms must have moving windows")
    return starts[selected], starts[selected]+dt, truth[selected], nearest[selected], [len(x) for x in pools]


def gapped_coordinate(graph, gap):
    shared = ~(graph.unique_masks[0] | graph.unique_masks[1])
    stem_end = graph.center_distance[shared].max()
    coordinate = graph.center_distance.copy()
    offset = gap
    for mask in graph.unique_masks:
        coordinate[mask] += offset
        offset += graph.center_distance[mask].max()-stem_end+gap
    return coordinate


def deduplicate_marks(marks):
    result, removed = {}, 0
    for tet, (time, features) in marks.items():
        _, first = np.unique(time, return_index=True)
        first.sort()
        removed += len(time)-len(first)
        result[tet] = time[first], features[first]
    return result, removed


def fit_variant(graph, time, xy, speed, marks, start, end, variant, protocol):
    if variant not in protocol["variants"]:
        raise ValueError("Unknown frozen audit variant")
    dt = np.diff(time, append=time[-1])
    good = (time >= start) & (time < end) & np.isfinite(xy).all(axis=1) & np.isfinite(speed) & (speed > 4) & (dt > 0) & (dt <= .25)
    if not good.any():
        raise ValueError("No moving training exposure")
    metric = graph.distance
    if variant.endswith("gapped_linear"):
        coordinate = gapped_coordinate(graph, protocol["gap_cm"])
        metric = np.abs(coordinate[:, None]-coordinate[None, :])
    kernel = np.exp(-.5*(metric/protocol["spatial_sigma_cm"])**2)
    nearest = cKDTree(graph.xy)
    train_bin = nearest.query(xy[good])[1]
    occupancy = dt[good] @ kernel[train_bin]
    occupied = occupancy >= .1
    occupancy = np.maximum(occupancy, 1e-12)
    rate, refs, spatial, trees, support = {}, {}, {}, {}, []
    for tet, (times, features) in marks.items():
        row = np.clip(np.searchsorted(time, times, side="right")-1, 0, len(time)-1)
        keep = (times >= start) & (times < end) & good[row] & (times-time[row] <= .25)
        if not keep.any():
            continue
        locations = xy[row[keep]]
        if variant == "frozen_geodesic":
            locations = np.column_stack([np.interp(times[keep], time, xy[:, j]) for j in range(2)])
        bins = nearest.query(locations)[1]
        weights = kernel[bins]
        rate[tet] = weights.sum(axis=0)/occupancy
        refs[tet], spatial[tet] = features[keep], weights
        trees[tet] = cKDTree(refs[tet]/protocol["mark_sigma"])
        for arm, mask in enumerate(graph.unique_masks):
            support.append({"tetrode": tet, "arm": arm, "n_training_marks": int(mask[bins].sum()),
                            "median_feature_max": float(np.median(features[keep][mask[bins]])) if mask[bins].any() else None})
    if len(refs) < 2:
        raise ValueError("Fewer than two encoding tetrodes")
    return Encoding(graph, occupied, occupancy, rate, refs, spatial, trees, protocol["mark_sigma"]), support


def author_equation(reference_repo, protocol):
    commit, path = protocol["reference_commit"], protocol["reference_path"]
    source = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=reference_repo, text=True)
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == protocol["reference_function"]]
    if len(nodes) != 1:
        raise ValueError("Pinned reference equation not unique")
    module = ast.Module(body=nodes, type_ignores=[])
    namespace = {"np": np}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[protocol["reference_function"]], {"commit": commit, "path": path,
            "file_sha256": hashlib.sha256(source.encode()).hexdigest(), "function_source": ast.get_source_segment(source, nodes[0])}


def parity_check(encoding, starts, ends, marks, equation):
    """Independent accumulation of the author's equation, same fitted KDE arrays."""
    ll = np.zeros((len(starts), len(encoding.occupied)))
    for tet, reference in encoding.features.items():
        ll += equation(np.ones((len(starts), len(encoding.occupied))), encoding.rate[tet][None, :], (ends-starts)[:, None])
        times, features = marks[tet]
        for i, (start, end) in enumerate(zip(starts, ends, strict=True)):
            features_in_bin = features[(times >= start) & (times < end)]
            if not len(features_in_bin):
                continue
            d2 = distance.cdist(features_in_bin/encoding.mark_sigma, reference/encoding.mark_sigma, "sqeuclidean")
            neighbors = d2 <= 36
            neighbors[~neighbors.any(axis=1)] = True
            intensity = (np.exp(-.5*d2)*neighbors) @ encoding.spatial[tet]/encoding.occupancy
            ll[i] += equation(intensity, np.zeros((1, len(encoding.occupied)))).sum(axis=0)
    ll[:, ~encoding.occupied] = -np.inf
    expected = normalize_likelihood(ll)
    actual, _, _ = encoding.likelihood(starts, ends, marks)
    return float(np.max(np.abs(expected-actual).sum(axis=1)))


def decode_metrics(encoding, starts, ends, truth, marks):
    posterior, counts, active = encoding.likelihood(starts, ends, marks)
    masses = np.column_stack([posterior[:, mask].sum(axis=1) for mask in encoding.graph.unique_masks])
    predicted = masses.argmax(axis=1)
    confusion = np.array([[np.sum((truth == a) & (predicted == b)) for b in (0, 1)] for a in (0, 1)])
    recalls = np.diag(confusion)/confusion.sum(axis=1)
    entropy = -(posterior*np.log(np.maximum(posterior, np.finfo(float).tiny))).sum(axis=1)
    return posterior, {"balanced_accuracy": float(recalls.mean()), "arm0_recall": float(recalls[0]), "arm1_recall": float(recalls[1]),
                       "n_windows": len(starts), "n_encoding_tetrodes": len(encoding.features),
                       "n_encoding_marks": sum(len(x) for x in encoding.features.values()), "zero_spike_windows": int((counts == 0).sum()),
                       "median_spikes": float(np.median(counts)), "median_active_tetrodes": float(np.median(active)),
                       "median_posterior_entropy": float(np.median(entropy))}, confusion, counts, active, entropy, predicted


def selection_audit(args, inputs):
    inventory = pd.read_csv(args.cohort_inventory)
    targets = inventory[inventory.n_eligible_transitions > 0]
    cache, candidates, decisions = {}, [], []
    protocol = {"position_alignment_tolerance_s": .001}

    def data(animal, day, epoch):
        key = (animal, day, epoch)
        if key not in cache:
            cache[key] = epoch_data(args.dataset_root/ANIMALS[animal], day, epoch, protocol)
            for path in cache[key][9].values():
                inputs[str(path)] = path
        return cache[key]

    for target in targets.itertuples(index=False):
        animal, day, epoch = target.animal, int(target.day), int(target.epoch)
        folder = args.dataset_root/ANIMALS[animal]
        rows = []
        task_path = folder/f"{animal}task{day:02d}.mat"
        inputs[str(task_path)] = task_path
        tasks = day_epochs(loadmat(task_path, squeeze_me=True, struct_as_record=False)["task"], day)
        preceding = [i for i, task in enumerate(tasks, 1) if i < epoch and str(field(task, "type", "")).lower() == "run"]
        for source_epoch in preceding:
            row = {"animal": animal, "day": day, "target_epoch": epoch, "source_epoch": source_epoch, "context_matches": False,
                   "original_context_matches": False, "metadata_error": "", "reason": "unresolved", "shared_tetrodes": None}
            try:
                source, dest = data(animal, day, source_epoch), data(animal, day, epoch)
                row["context_matches"], row["reason"] = matching_context(source, dest)
                row["original_context_matches"], _ = matching_context(source, dest, check_segments=False)
                a = hippocampal_tetrodes(folder, animal, day, source_epoch, {"CA1", "CA2", "CA3"})
                b = hippocampal_tetrodes(folder, animal, day, epoch, {"CA1", "CA2", "CA3"})
                row["shared_tetrodes"] = len(set(a) & set(b))
                inputs[str(folder/f"{animal}tetinfo.mat")] = folder/f"{animal}tetinfo.mat"
            except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
                row["metadata_error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
        candidates.extend(rows)
        old = max(preceding) if preceding else None
        old_matches = any(r["source_epoch"] == old and r["original_context_matches"] for r in rows)
        new = choose_matching(rows)
        decisions.append({"animal": animal, "day": day, "target_epoch": epoch, "session": target.session,
                          "latest_preceding_epoch": old, "latest_preceding_matches": old_matches,
                          "latest_matching_epoch": new, "newly_metadata_eligible": new is not None and not old_matches,
                          "decoder_qc_evaluated": False, "biological_cohort_promoted": False})
    return candidates, decisions


def figure(out, identity, graph, xy, train, windows, posterior, truth, variant):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(xy[train, 0], xy[train, 1], ".", ms=1, alpha=.15)
    for segment in graph.coordinates:
        axes[0].plot(segment[[0, 2]], segment[[1, 3]], "k-", lw=1)
    axes[0].set(title="Training RUN coverage", xlabel="x (cm)", ylabel="y (cm)", aspect="equal")
    axes[1].imshow(posterior.T, aspect="auto", origin="lower", interpolation="nearest")
    axes[1].set(title="Selected held-out RUN posteriors", xlabel="Chronological sampled window (not contiguous)", ylabel="Graph bin")
    masses = [posterior[:, mask].sum(axis=1) for mask in graph.unique_masks]
    axes[2].plot(windows, masses[0], ".", ms=2, label="Arm 0 mass")
    axes[2].plot(windows, truth, ".", ms=1, label="True arm")
    axes[2].set(title="Flat-prior arm decoding", xlabel="RUN time (s)", ylabel="Probability / arm label")
    axes[2].legend(fontsize=7)
    fig.suptitle(f"{identity} / {variant}")
    fig.tight_layout()
    path = out/f"{identity}_{variant}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("dataset-root", "cohort-inventory", "old-run-qc", "reference-repo", "protocol", "output-dir"):
        parser.add_argument("--"+option, type=Path, required=True)
    args = parser.parse_args()
    p = json.loads(args.protocol.read_text())
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/"checkpoints").mkdir()
    inputs = {"protocol": args.protocol, "cohort_inventory": args.cohort_inventory, "old_run_qc": args.old_run_qc}
    original_hashes = {str(path): file_sha256(path) for path in (args.cohort_inventory, args.old_run_qc)}
    equation, reference = author_equation(args.reference_repo, p)
    atomic_json(out/(PREFIX+"reference.json"), reference)
    began = clock.monotonic()
    metrics, windows, supports, confusions, coverage, parities, figures = [], [], [], [], [], [], []
    old_qc = pd.read_csv(args.old_run_qc)

    def timeout(_signum, _frame):
        raise TimeoutError("Frozen two-hour RUN-only runtime cap reached")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(p["max_runtime_s"])
    try:
        candidates, decisions = selection_audit(args, inputs)
        pd.DataFrame(candidates).to_csv(out/(PREFIX+"encoder_candidates.csv"), index=False)
        pd.DataFrame(decisions).to_csv(out/(PREFIX+"encoder_selection.csv"), index=False)
        for case in p["cases"]:
            animal, day, epoch = case["animal"], case["day"], case["epoch"]
            identity = f"{animal}-{day:02d}-{epoch:02d}"
            print(f"RUN-only case {identity}", flush=True)
            folder = args.dataset_root/ANIMALS[animal]
            time, xy, speed, wells, coords, center, outers, sm, _, paths = epoch_data(folder, day, epoch, {"position_alignment_tolerance_s": .001})
            for path in paths.values():
                inputs[str(path)] = path
            inputs[str(folder/f"{animal}tetinfo.mat")] = folder/f"{animal}tetinfo.mat"
            marks, sources = load_marks(folder, animal, day, epoch, {"CA1", "CA2", "CA3"})
            for path in sources:
                inputs[str(path)] = path
            graph = make_graph(coords, wells, center, outers, p["graph_bin_cm"])
            split = time[0]+p["training_fraction"]*(time[-1]-time[0])
            starts, ends, truth, nearest, pools = sample_windows(graph, time, xy, speed, split, time[-1], p["time_bin_s"], p["max_windows_per_arm"])
            dt = np.diff(time, append=time[-1])
            valid = np.isfinite(xy).all(axis=1) & np.isfinite(speed) & (speed > 4) & (dt > 0) & (dt <= .25)
            position_bins = np.zeros(len(time), int)
            position_bins[valid] = cKDTree(graph.xy).query(xy[valid])[1]
            native_segment = np.asarray(field(sm, "segmentIndex", []), float).reshape(-1)
            for arm, mask in enumerate(graph.unique_masks):
                on_arm = valid & mask[position_bins]
                coverage.append({"case": identity, "arm": arm, "train_exposure_s": float(dt[on_arm & (time < split)].sum()),
                                 "heldout_exposure_s": float(dt[on_arm & (time >= split)].sum()), "available_heldout_windows": pools[arm],
                                 "selected_heldout_windows": int((truth == arm).sum()),
                                 "native_segment_agreement": float(np.mean(native_segment[on_arm] == graph.segment[position_bins[on_arm]]+1))})
            deduplicated, duplicates = deduplicate_marks(marks)
            original = old_qc[(old_qc.animal == animal) & (old_qc.day == day) & (old_qc.preceding_run_epoch == epoch)]
            for variant in p["variants"]:
                used_marks = deduplicated if variant.startswith("deduplicated") else marks
                encoding, support = fit_variant(graph, time, xy, speed, used_marks, time[0], split, variant, p)
                posterior, metric, confusion, counts, active, entropy, predicted = decode_metrics(encoding, starts, ends, truth, used_marks)
                metric.update(case=identity, variant=variant, reason=case["reason"], duplicates_removed=duplicates if variant.startswith("deduplicated") else 0,
                              train_end_s=float(split), first_test_start_s=float(starts[0]),
                              diagnostic_threshold_met=metric["balanced_accuracy"] >= p["min_run_balanced_accuracy"] and min(metric["arm0_recall"], metric["arm1_recall"]) >= p["min_run_arm_recall"],
                              biological_cohort_promoted=False,
                              previous_full_run_balanced_accuracy=float(original.balanced_accuracy.iloc[0]) if len(original) else None)
                metrics.append(metric)
                for row in support:
                    supports.append({"case": identity, "variant": variant, **row})
                for a in (0, 1):
                    for b in (0, 1):
                        confusions.append({"case": identity, "variant": variant, "true_arm": a, "predicted_arm": b, "n_windows": int(confusion[a, b])})
                for i in range(len(starts)):
                    windows.append({"case": identity, "variant": variant, "start_time_s": starts[i], "end_time_s": ends[i],
                                    "true_arm": int(truth[i]), "predicted_arm": int(predicted[i]), "true_graph_bin": int(nearest[i]),
                                    "n_spikes": int(counts[i]), "n_active_tetrodes": int(active[i]), "posterior_entropy": entropy[i],
                                    "true_arm_mass": float(posterior[i, graph.unique_masks[truth[i]]].sum())})
                if variant == "frozen_geodesic":
                    subset = np.linspace(0, len(starts)-1, min(p["parity_max_windows"], len(starts)), dtype=int)
                    error = parity_check(encoding, starts[subset], ends[subset], used_marks, equation)
                    parities.append({"case": identity, "n_windows": len(subset), "max_posterior_l1_error": error, "passed": error <= p["parity_posterior_l1_tolerance"]})
                figures.append({"case": identity, "variant": variant, "path": figure(out, identity, graph, xy, valid & (time < split), starts, posterior, truth, variant)})
                atomic_json(out/"checkpoints"/f"{identity}_{variant}.json", metric)
                atomic_json(out/"progress.json", {"status": "running", "completed_variants": len(metrics), "elapsed_s": clock.monotonic()-began})
                print(json.dumps(metric), flush=True)
            del marks, deduplicated, encoding
    except Exception as exc:
        atomic_json(out/"progress.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "completed_variants": len(metrics)})
        raise
    finally:
        signal.alarm(0)
    for name, rows in (("metrics", metrics), ("windows", windows), ("training_support", supports), ("confusion", confusions),
                       ("coverage", coverage), ("parity", parities), ("figures_manifest", figures)):
        pd.DataFrame(rows).to_csv(out/(PREFIX+name+".csv"), index=False)
    provenance = build_script_provenance(input_paths=inputs, cwd=Path(__file__).resolve().parents[1])
    gates = [
        {"gate": "four_frozen_cases_complete", "passed": len(metrics) == len(p["cases"])*len(p["variants"])},
        {"gate": "author_equation_numerical_parity", "passed": bool(parities) and all(x["passed"] for x in parities)},
        {"gate": "heldout_window_cap", "passed": all(x["selected_heldout_windows"] <= p["max_windows_per_arm"] for x in coverage)},
        {"gate": "chronological_separation", "passed": all(x["first_test_start_s"] >= x["train_end_s"] for x in metrics)},
        {"gate": "original_input_tables_unchanged", "passed": all(file_sha256(Path(path)) == digest for path, digest in original_hashes.items())},
        {"gate": "no_biological_cohort_promoted", "passed": not any(x["biological_cohort_promoted"] for x in metrics+decisions)},
    ]
    gates.append({"gate": "overall_audit_execution", "passed": all(x["passed"] for x in gates)})
    pd.DataFrame(gates).to_csv(out/(PREFIX+"gate_summary.csv"), index=False)
    atomic_json(out/(PREFIX+"manifest.json"), {"scope": p["scope"], "protocol": p, "provenance": provenance, "reference": reference,
                "created_at_utc": datetime.now(UTC).isoformat(), "runtime_s": clock.monotonic()-began,
                "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "matplotlib")},
                "replay_decoded": False, "behavioral_association_tested": False, "original_no_go_preserved": True})
    rescued = sum(x["newly_metadata_eligible"] for x in decisions)
    lines = ["# Bounded Denovellis RUN audit", "", "Diagnostic only. Original post-error no-go remains in force.",
             f"Audited {len(decisions)} target epochs; {rescued} become metadata-eligible under latest matching-context selection (not decoder-qualified).",
             "", "The author comparison is an equation check plus author-inspired preprocessing sensitivities, NOT an exact reproduction of their complete decoder.",
             "", "| RUN | Variant | Balanced accuracy | Arm 0 recall | Arm 1 recall | Diagnostic threshold met |", "|---|---|---:|---:|---:|---|"]
    lines += [f"| {x['case']} | {x['variant']} | {x['balanced_accuracy']:.3f} | {x['arm0_recall']:.3f} | {x['arm1_recall']:.3f} | {x['diagnostic_threshold_met']} |" for x in metrics]
    lines += ["", "Passing a capped diagnostic subset cannot satisfy full chronological RUN QC or repair the original cohort gate.",
              "No replay windows, content-outcome models, threshold changes or automatic follow-on jobs were run."]
    (out/(PREFIX+"report.md")).write_text("\n".join(lines)+"\n")
    atomic_json(out/"progress.json", {"status": "completed", "completed_variants": len(metrics), "elapsed_s": clock.monotonic()-began})


if __name__ == "__main__":
    main()
