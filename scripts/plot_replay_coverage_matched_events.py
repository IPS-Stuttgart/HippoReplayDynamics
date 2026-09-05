#!/usr/bin/env python3
"""Illustrate frozen full/half-cell decisions, without changing event selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from scripts._provenance import build_script_provenance, file_sha256

KEY = ["dataset", "animal", "session", "window_uid"]
CATEGORIES = ["lost", "retained", "gained", "rejected"]


def pair_decisions(table):
    data = table[table.observation.eq("original_order") & table.detector.eq("source_high_mua")
        & table.bin_filter.eq("edge_only") & table.min_frames.eq(10) & table.population_replicate.eq(0)].copy()
    cols = KEY + ["accepted_alpha_0.02", "geometric_pass", "cell_identity_p", "xy_roll_p", "retained_cells"]
    full, half = [data[data.cell_fraction.eq(f)][cols] for f in [1, .5]]
    pairs = full.merge(half, on=KEY, how="outer", validate="one_to_one", indicator=True, suffixes=("_full", "_half"))
    if not pairs._merge.eq("both").all() or pairs.empty:
        raise ValueError("nonempty paired full/half observations required")
    a, b = pairs["accepted_alpha_0.02_full"], pairs["accepted_alpha_0.02_half"]
    pairs["category"] = np.select([a & ~b, a & b, ~a & b], CATEGORIES[:3], default="rejected")
    return pairs.drop(columns="_merge")


def choose_examples(pairs):
    """Middle lexicographic member per animal/category, no speed-based ranking."""
    rows, availability = [], []
    for (dataset, animal), part in pairs.groupby(["dataset", "animal"], sort=True):
        for category in CATEGORIES:
            subset = part[part.category.eq(category)].sort_values(["session", "window_uid"])
            availability.append({"dataset": dataset, "animal": animal, "category": category,
                "eligible_examples": len(subset), "status": "available" if len(subset) else "unavailable"})
            if len(subset):
                row = subset.iloc[(len(subset) - 1) // 2].to_dict()
                row["category_denominator"] = len(subset)
                rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(availability)


def posterior_diagnostics(counts, rates, grid):
    counts, rates, grid = np.asarray(counts), np.asarray(rates), np.asarray(grid)
    if counts.ndim != 2 or rates.ndim != 2 or counts.shape[1] != rates.shape[0] or rates.shape[1] != len(grid):
        raise ValueError("invalid decoder dimensions")
    if not np.isfinite(rates).all() or (rates <= 0).any() or (counts < 0).any() or (counts != np.floor(counts)).any():
        raise ValueError("positive rates and integer counts required")
    ll = counts @ np.log(.02 * rates) - .02 * rates.sum(axis=0)
    posterior = np.exp(ll - logsumexp(ll, axis=1, keepdims=True))
    mean = posterior @ grid
    radius = np.sqrt(np.maximum(0, posterior @ (grid**2).sum(axis=1) - (mean**2).sum(axis=1)))
    marginals = []
    for dim in range(2):
        coordinate = np.unique(grid[:, dim])
        marginal = np.column_stack([posterior[:, grid[:, dim] == c].sum(axis=1) for c in coordinate])
        marginals.append((coordinate, marginal))
    return posterior, mean, radius, marginals


def continuous_core(path, grid, counts):
    support = np.flatnonzero(counts.sum(axis=1) >= 2)
    if not len(support):
        return None
    best, start = (int(support[0]), int(support[0])), int(support[0])
    for t in range(start + 1, int(support[-1]) + 1):
        if np.linalg.norm(grid[path[t]] - grid[path[t - 1]]) >= 20 - 1e-9:
            start = t
        if t - start > best[1] - best[0]:
            best = (start, t)
    a, b = best
    return best if b - a + 1 >= 10 and np.linalg.norm(grid[path[b]] - grid[path[a]]) >= 40 - 1e-9 else None


def draw_example(row, arrays, specs, raw, output):
    index = np.flatnonzero(arrays["window_uids"] == row.window_uid)
    if len(index) != 1:
        raise ValueError("unique event identity required")
    wi = int(index[0])
    lo, hi = arrays["frame_offsets"][2 * wi:2 * wi + 2]
    blo, bhi = arrays["base_offsets"][wi:wi + 2]
    start = arrays["base_starts_s"][blo]
    end = arrays["base_starts_s"][bhi - 1] + arrays["base_durations_s"][bhi - 1]
    counts = arrays["frame_counts"][lo:hi]
    grid = arrays["grid_cm"]
    rates = arrays["rates_hz"][:, arrays["support"]]
    peak = grid[rates.argmax(axis=1)]
    order = np.lexsort((arrays["cell_ids"], peak[:, 1], peak[:, 0]))
    cell_rank = np.empty(len(order), int)
    cell_rank[order] = np.arange(len(order))
    time_ms = 1000 * (.01 + .005 * np.arange(len(counts)))
    duration_ms = 1000 * (end - start)
    decoded = []
    for spec in specs:
        subset = np.asarray(spec["indices"], int)
        p, mean, radius, marginal = posterior_diagnostics(counts[:, subset], rates[subset], grid)
        with np.load(spec["path"], allow_pickle=False) as z:
            path = z["original_path"][lo:hi]
        # Match likelihood, not argmax identity: equal-probability bins can tie.
        if not np.allclose(p[np.arange(len(p)), path], p.max(axis=1), rtol=1e-10, atol=1e-12):
            raise ValueError("posterior does not reproduce frozen MAP likelihood")
        decoded.append((subset, mean, radius, marginal, path))
    fig, axes = plt.subplots(2, 4, figsize=(17, 7), layout="constrained", squeeze=False)
    result = {"start_s": start, "end_s": end, "duration_ms": duration_ms, "decoding_frames": len(counts)}
    spikes = raw[(raw[:, 0] >= start) & (raw[:, 0] < end)]
    for r, (subset, mean, radius, marginal, path) in enumerate(decoded):
        label = "full" if r == 0 else "half"
        accepted = getattr(row, "accepted_alpha_0_02_" + label, None)
        if accepted is None:
            accepted = row.category in (["lost", "retained"] if r == 0 else ["gained", "retained"])
        ax = axes[r, 0]
        core = continuous_core(path, grid, counts[:, subset])
        n_spikes = 0
        for cell in subset:
            t = 1000 * (spikes[spikes[:, 1] == arrays["cell_ids"][cell], 0] - start)
            n_spikes += len(t)
            ax.scatter(t, np.full(len(t), cell_rank[cell]), marker="|", s=12, linewidths=.5, color="black")
        for cell in np.setdiff1d(np.arange(len(order)), subset):
            ax.axhspan(cell_rank[cell] - .5, cell_rank[cell] + .5, color=".92", linewidth=0)
        ax.set(xlim=(0, duration_ms), ylim=(-1, len(order)), ylabel="RUN-peak x/y sorted cell index",
            xlabel="Time from candidate start (ms)", title=f"{label.title()}: {len(subset)} cells, {n_spikes} spikes\nFrozen two-shuffle decision: {'pass' if accepted else 'fail'}")
        if core is not None:
            ax.axvspan(time_ms[core[0]], time_ms[core[1]], color="#287c80", alpha=.12, zorder=-1)
        for dim in range(2):
            ax = axes[r, dim + 1]
            coordinate, prob = marginal[dim]
            edges = np.r_[coordinate - 4, coordinate[-1] + 4]
            times = np.r_[time_ms - 2.5, time_ms[-1] + 2.5]
            vmax = max(d[3][dim][1].max() for d in decoded)
            mesh = ax.pcolormesh(times, edges, prob.T, cmap="viridis", vmin=0, vmax=vmax, rasterized=True)
            ax.plot(time_ms, grid[path, dim], color="#e76f51", lw=.8, label="MAP")
            ax.plot(time_ms, mean[:, dim], color="white", lw=.8, label="Mean")
            if core is not None:
                ix = slice(core[0], core[1] + 1)
                ax.plot(time_ms[ix], grid[path[ix], dim], color="black", lw=2.2, label="Geometric core")
            ax.set(xlim=(0, duration_ms), xlabel="Time (ms)", ylabel=f"{'xy'[dim]} coordinate (cm)",
                title=f"Marginal over {'xy'[dim]} (not a 1D path)")
            fig.colorbar(mesh, ax=ax, fraction=.04, label="Probability per 8 cm strip")
            if r == 0 and dim == 0:
                ax.legend(fontsize=7, loc="upper right")
        ax = axes[r, 3]
        ax.scatter(grid[:, 0], grid[:, 1], s=3, c=".85", label="Decoded states")
        ax.plot(grid[path, 0], grid[path, 1], color="#e76f51", lw=.8, label="MAP")
        ax.plot(mean[:, 0], mean[:, 1], color="#277b7a", lw=1, label="Mean")
        if core is not None:
            core_path = grid[path[core[0]:core[1] + 1]]
            ax.plot(core_path[:, 0], core_path[:, 1], color="black", lw=2.2, label="Geometric core")
        ax.scatter(*mean[0], marker="o", c="#277b7a", s=25)
        ax.scatter(*mean[-1], marker="x", c="#277b7a", s=25)
        ax.set(aspect="equal", xlabel="x (cm)", ylabel="y (cm)", title=f"2D decoded path, no temporal prior\nMedian posterior RMS radius: {np.median(radius):.1f} cm")
        ax.legend(fontsize=7)
        result[f"{label}_spikes"] = n_spikes
        result[f"{label}_median_posterior_radius_cm"] = np.median(radius)
        result[f"{label}_core_start_ms"] = time_ms[core[0]] if core else np.nan
        result[f"{label}_core_end_ms"] = time_ms[core[1]] if core else np.nan
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{row.dataset} / {row.animal} / {row.session} / {row.category}\n{row.window_uid}\nSame fixed candidate, subset 0; category-selected illustration, not replay ground truth. Gray raster rows are hidden cells; state support is not a verified wall outline.", fontsize=10)
    path = output / f"event_{row.dataset}_{row.animal}_{row.category}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    result["figure"] = str(path)
    return result


def run(root, output):
    manifest = root / "coverage_shuffle_baseline_manifest.json"
    audit_path = root / "coverage_shuffle_baseline_reconstruction_audit.json"
    meta, audit = json.loads(manifest.read_text()), json.loads(audit_path.read_text())
    if meta["status"] != "complete" or audit["status"] != "pass" or audit["input_file_sha256"]["scoring_manifest"] != file_sha256(manifest):
        raise ValueError("completed scoring with linked passing audit required")
    inputs = {"manifest": manifest, "audit": audit_path, "script": Path(__file__)}
    def checked(path, digest):
        path = Path(path)
        if file_sha256(path) != digest:
            raise ValueError(f"changed input: {path}")
        inputs[str(path)] = path
        return path
    tables = [pd.read_csv(checked(r["metrics_path"], r["metrics_sha256"])) for r in meta["results"]]
    pairs = pair_decisions(pd.concat(tables, ignore_index=True))
    examples, availability = choose_examples(pairs)
    if examples.empty:
        raise ValueError("no representative examples")
    output.mkdir(parents=True, exist_ok=False)
    rendered = []
    for record in meta["results"]:
        selected = examples[(examples[KEY[:3]] == [record[k] for k in KEY[:3]]).all(axis=1)]
        if selected.empty:
            continue
        arrays = dict(np.load(checked(record["input_arrays_path"], record["input_arrays_sha256"]), allow_pickle=False))
        source = record["source"]
        with np.load(checked(source["source_cache_path"], source["source_cache_sha256"]), allow_pickle=False) as z:
            raw = z["spikes"]
        specs = []
        for fraction in [1, .5]:
            spec = next(s for s in record["shuffle_files"] if s["cell_fraction"] == fraction and s["population_replicate"] == 0)
            checked(spec["path"], spec["sha256"])
            specs.append(spec)
        for row in selected.itertuples(index=False):
            rendered.append({**{k: getattr(row, k) for k in KEY}, **draw_example(row, arrays, specs, raw, output)})
    examples = examples.merge(pd.DataFrame(rendered), on=KEY, validate="one_to_one")
    examples.to_csv(output / "coverage_matched_event_figures.csv", index=False)
    availability.to_csv(output / "coverage_matched_event_categories.csv", index=False)
    (output / "coverage_matched_event_figures.md").write_text("# Matched Recording-Coverage Examples\n\n"
        "The same frozen high-MUA candidates are decoded with full cells and half-cell subset 0. "
        "One middle-lexicographic event is shown per animal and decision category (lost, retained, gained, rejected), where available. "
        "Examples are selected by category for illustration, not random estimates of prevalence. All category denominators and missing categories are retained.\n\n"
        "Raster dots use actual spike timestamps; hidden cells are shaded. Cells are sorted by RUN peak x then y, which is not a trajectory ordering. "
        "Independent 20 ms Poisson decoding advances by 5 ms under a uniform prior. The x/y heatmaps are marginals of a 2D posterior, not linearized tracks. "
        "The original saved MAP is checked against the reconstructed posterior maximum; ties need not choose the same bin. "
        "Mean paths, MAP paths and RMS posterior radii are diagnostic and do not alter the frozen 5,000-per-family shuffle decisions. "
        "Black MAP segments mark the longest geometric core when it passes the edge-only continuity screen; the rest of an accepted event can still jump. "
        "Connecting adjacent coordinates is visualization, not temporal smoothing. State support is not a measured arena boundary.\n")
    provenance = build_script_provenance(input_paths=inputs, cwd=ROOT)
    provenance.update(status="complete", examples=len(examples), animals=examples[KEY[:2]].drop_duplicates().shape[0],
        output_sha256={p.name: file_sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
    (output / "coverage_matched_event_figures_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"status": "complete", "examples": len(examples), "animals": provenance["animals"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.input_dir, args.output_dir)
