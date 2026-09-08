#!/usr/bin/env python3
"""Independently reconstruct arena-scale summaries without producer imports."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import build_script_provenance, file_sha256

PRIMARY = ("geometric_fraction", "imm_order_advantage_per_spike", "imm_order_map_interaction_per_spike")
METRICS = (*PRIMARY, "imm_order_advantage", "imm_order_map_interaction", "imm_minus_event_global", "imm_minus_event_global_per_spike", "diffusion_order_advantage_per_spike")
KEY = ["dataset", "animal", "session", "event_id", "split"]


def require_close(actual, expected):
    if not np.allclose(actual, expected, atol=1e-10, rtol=1e-9, equal_nan=True):
        raise ValueError(f"reconstruction mismatch: {actual!r} != {expected!r}")


def independent_slope(x, y):
    return float(np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0][1])


def reference_probability(outcomes):
    outcomes = np.asarray(outcomes, float)
    if outcomes.shape != (5, 3) or not np.isfinite(outcomes).all():
        raise ValueError("five animals by three finite environments required")
    real = np.mean((outcomes[:, 2] - outcomes[:, 0]) / 2)
    indices = np.array(list(itertools.permutations(range(3))))
    possible = (outcomes[:, indices[:, 2]] - outcomes[:, indices[:, 0]]) / 2
    combinations = np.indices((6,) * 5).reshape(5, -1)
    null = possible[np.arange(5)[:, None], combinations].mean(axis=0)
    return float((np.abs(null) >= abs(real) - 1e-12).mean())


def run(args):
    root, out = Path(args.run_dir).resolve(), Path(args.output_dir).resolve()
    path = root / "tanni_environment_scale_manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "complete" or manifest.get("events") != 5224 or not manifest.get("non_rescoring"):
        raise ValueError("complete frozen analysis required")
    for name, digest in manifest["output_sha256"].items():
        if file_sha256(root / name) != digest:
            raise ValueError("output artifact changed: " + name)
    for name, value in manifest["input_file_paths"].items():
        if file_sha256(Path(value)) != manifest["input_file_sha256"][name]:
            raise ValueError("input changed: " + name)
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing overwrite")
    meta = pd.read_csv(root / "tanni_environment_scale_metadata.csv")
    events = pd.read_csv(root / "tanni_environment_scale_events.csv")
    sessions = pd.read_csv(root / "tanni_environment_scale_sessions.csv")
    summary = pd.read_csv(root / "tanni_environment_scale_summary.csv")
    parent = Path(manifest["input_file_paths"]["parent_manifest"]).parent
    predictions = pd.read_csv(parent / "geometry_predictive_predictions.csv")
    predictions = predictions[predictions.dataset.eq("tanni2022")]
    labels = pd.read_csv(parent / "geometry_predictive_labels.csv")
    labels = labels[labels.dataset.eq("tanni2022") & labels.criterion.eq("edge10")]
    if len(events) != 26120 or len(sessions) != 125 or len(summary) != 80 or len(meta) != 25:
        raise ValueError("incomplete analysis tables")
    e = events.set_index(KEY).sort_index()
    p = predictions.set_index(KEY).sort_index()
    l = labels.set_index(KEY).sort_index()
    if not e.index.is_unique or not e.index.equals(p.index) or not e.index.equals(l.index):
        raise ValueError("event/split identity mismatch")
    for name in p:
        require_close(e[name], p[name])
    require_close(e.geometric_fraction, l.geometric_pass.astype(float))
    if not l.heldout_used_for_label.eq(False).all():
        raise ValueError("heldout leakage")
    metadata = meta.set_index(["animal", "session"])
    for r in meta.itertuples(index=False):
        with h5py.File(r.native_nwb, "r") as f:
            size = np.asarray(f["/general/data_collection/Settings/General/arena_size"][()])
            animal = f["/general/data_collection/Settings/General/animal"][()].decode()
        if animal != r.animal:
            raise ValueError("native animal mismatch")
        require_close(np.prod(size) / 10000, r.arena_area_m2)
        require_close(np.log2(r.arena_area_m2 / 1.09375), r.area_level)
    source_rows = pd.read_csv(manifest["input_file_paths"]["source_sessions"])
    source_rows = source_rows[source_rows.dataset.eq("tanni2022")].set_index(["animal", "session"])
    for keys, g in events.groupby(["animal", "session", "split"], sort=True):
        s = sessions[sessions.animal.eq(keys[0]) & sessions.session.eq(keys[1]) & sessions.split.eq(keys[2])]
        if len(s) != 1:
            raise ValueError("session key duplicated or absent")
        row = s.iloc[0]
        if row.events != len(g) or row.zero_heldout_events != sum(g.n_heldout_spikes.eq(0)):
            raise ValueError("session denominator mismatch")
        for metric in METRICS:
            values = [float(v) for v in g[metric] if pd.notna(v)]
            expected = math.fsum(values) / len(values) if values else np.nan
            require_close(row[metric], expected)
            if row[metric + "_events"] != len(values):
                raise ValueError("ratio denominator mismatch")
        for metric, raw in (("median_train_spikes", "n_train_spikes"), ("median_heldout_spikes", "n_heldout_spikes"), ("median_duration_s", "duration_s")):
            require_close(row[metric], np.median(g[raw]))
        m = metadata.loc[keys[:2]]
        with np.load(source_rows.loc[keys[:2]].artifact_path) as z:
            bins = np.count_nonzero(z["valid_spatial_bins"])
        require_close(m.n_valid_bins, bins)
        require_close(row.mean_normalized_training_entropy, g.mean_training_entropy_nats.mean() / np.log(bins))
    checked = 0
    for r in summary.itertuples(index=False):
        subset = sessions[sessions.split.eq(r.split)]
        slopes, bcd = [], []
        for _, g in subset.groupby("animal", sort=True):
            if r.scope == "BCD_primary":
                g = g[g.area_level.gt(0)]
            means = g.groupby("area_level")[r.metric].mean().sort_index()
            slopes.append(independent_slope(means.index.to_numpy(float), means.to_numpy()))
            if r.scope == "BCD_primary":
                bcd.append(means.to_numpy())
        mean = math.fsum(slopes) / 5
        half = np.std(slopes, ddof=1) / np.sqrt(5) * t.ppf(0.975, 4)
        require_close([r.mean_slope, r.ci_low, r.ci_high], [mean, mean - half, mean + half])
        require_close(r.leave_one_animal_out_min, min(np.mean(np.delete(slopes, j)) for j in range(5)))
        require_close(r.leave_one_animal_out_max, max(np.mean(np.delete(slopes, j)) for j in range(5)))
        if bcd:
            require_close(r.exact_label_reference_p, reference_probability(bcd))
        checked += 1
    prim = summary[summary.primary].sort_values("exact_label_reference_p")
    running = 0.0
    for i, r in enumerate(prim.itertuples(index=False)):
        running = max(running, min(1, (3 - i) * r.exact_label_reference_p))
        require_close(r.primary_holm_reference_p, running)
    adjusted = pd.read_csv(root / "tanni_environment_scale_adjusted.csv")
    for r in adjusted.itertuples(index=False):
        g = sessions[sessions.split.eq(0) & sessions.area_level.gt(0)]
        if r.omitted_animal != "none":
            g = g[~g.animal.eq(r.omitted_animal)]
        controls = [np.log(g.n_train_cells), np.log1p(g.median_train_spikes), np.log(g.median_duration_s)]
        if r.entropy_control:
            controls.append(g.mean_normalized_training_entropy)
        design = np.column_stack([pd.get_dummies(g.animal).to_numpy(float), *controls])
        q, rank = np.linalg.qr(design, mode="reduced"), np.linalg.matrix_rank(design)
        y, x = g[r.metric].to_numpy(), g.area_level.to_numpy(float)
        xr = x - q[0] @ (q[0].T @ x)
        yr = y - q[0] @ (q[0].T @ y)
        if rank == design.shape[1] and r.status == "ok":
            require_close(r.coefficient, xr @ yr / (xr @ xr))
    out.mkdir(parents=True, exist_ok=True)
    result = build_script_provenance(input_paths={"run_manifest": path})
    result.update(
        status="pass",
        events=5224,
        event_split_rows=26120,
        native_arenas=25,
        session_rows=125,
        summary_rows=checked,
        adjusted_rows=len(adjusted),
        primary_rows=3,
        scope="reconstruct parent prediction/label joins, native arena metadata, all session means and denominators, area slopes/intervals, exact reference probabilities, adjusted FWL coefficients; no predictive rescore",
    )
    (out / "tanni_environment_scale_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())
