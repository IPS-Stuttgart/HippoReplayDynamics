"""Recount unchanged trains, rerun native detection, and verify the crossed algebra."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from hipporeplayimm.regional_content_mua_null import load_detector
from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_edge_support_content import load_npz


def verify_hashes(folder, hashes):
    for name, digest in hashes.items():
        if file_sha256(folder / name) != digest:
            raise ValueError(f"changed file: {folder / name}")


def recount_independent(spikes, ids, windows):
    ordered = spikes[np.argsort(spikes[:, 0], kind="stable")]
    lookup = {int(cell): j for j, cell in enumerate(ids)}
    counts = np.zeros((len(windows), len(ids)), np.int64)
    for j, (a, b) in enumerate(windows):
        left, right = np.searchsorted(ordered[:, 0], [a, b], side="left")
        cells = [lookup[int(cell)] for cell in ordered[left:right, 1] if int(cell) in lookup]
        counts[j] = np.bincount(cells, minlength=len(ids))
    return counts


def truth_independent(labels, windows):
    result = np.zeros((len(windows), 9))
    for j, (a, b) in enumerate(windows):
        indices = np.arange(max(0, int(np.floor(a / 0.005))), min(len(labels), int(np.ceil(b / 0.005))))
        exposure = np.maximum(0, np.minimum((indices + 1) * 0.005, b) - np.maximum(indices * 0.005, a))
        result[j] = np.bincount(labels[indices], weights=exposure, minlength=9) / (b - a)
    return result


def assert_columns(actual, expected, names):
    if len(actual) != len(expected):
        raise ValueError("row count changed")
    for name in names:
        np.testing.assert_allclose(actual[name].to_numpy(float), expected[name].to_numpy(float), atol=2e-9, rtol=2e-9, equal_nan=True, err_msg=name)


def audit_session(row, detector):
    folder = Path(row.folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("incomplete session")
    verify_hashes(folder, manifest["outputs"])
    for name, digest in manifest["source_manifests"].items():
        if file_sha256(folder / name / "outputs.json") != digest:
            raise ValueError("changed source manifest")
    for name, path in manifest["input_file_paths"].items():
        if file_sha256(path) != manifest["input_file_sha256"][name]:
            raise ValueError("changed source cache")
    data = load_npz(manifest["input_file_paths"]["cache"])
    encoding = load_npz(folder / "encoding.npz")
    population = json.loads((folder / "populations.json").read_text())
    pops = {p["name"]: p for p in population["populations"]}
    ids, rates, grid = encoding["ids"], encoding["rates"], encoding["grid"]
    mask, valid = data["unit_qc_mask"].astype(bool), data["valid_spatial_bins"].astype(bool)
    np.testing.assert_array_equal(ids, data["cell_ids"][mask])
    np.testing.assert_array_equal(rates, np.maximum(data["rates_hz"][mask][:, valid], 1e-4))
    np.testing.assert_array_equal(grid, data["bin_centers_cm"][valid])
    scaled = np.clip(np.floor(3 * (grid - grid.min(axis=0)) / np.ptp(grid, axis=0)).astype(int), 0, 2)
    tiles = scaled[:, 0] * 3 + scaled[:, 1]
    np.testing.assert_array_equal(tiles, encoding["tiles"])
    score = rates[:, tiles == 4].mean(axis=1) / rates.mean(axis=1)
    order = np.lexsort((ids, score))
    np.testing.assert_array_equal(sorted(order[: len(ids) // 2]), pops["targeted_low"]["indices"])
    np.testing.assert_array_equal(sorted(order[-(len(ids) // 2) :]), pops["targeted_high"]["indices"])
    for p in pops.values():
        if p["family"] == "whole_tetrode":
            tt = encoding["tetrodes"]
            np.testing.assert_array_equal(np.flatnonzero(np.isin(tt, tt[p["indices"]])), p["indices"])
    sources = pd.read_csv(folder / "sources.csv")
    rows = []
    for source in sources.itertuples():
        local = Path(source.folder)
        verify_hashes(local, json.loads((local / "outputs.json").read_text()))
        raw = load_npz(folder / (source.source + "_train.npz")) if source.source != "real" else None
        spikes = raw["spikes"] if raw is not None else data["spikes"]
        if raw is None:
            times, speeds = detector.position_speed(data["position"], 0.1)
            intervals = data["supported_run_intervals"]
        else:
            xy = raw["truth_xy_cm"]
            qt = np.clip(np.floor(3 * (xy - grid.min(axis=0)) / np.ptp(grid, axis=0)).astype(int), 0, 2)
            np.testing.assert_array_equal(raw["truth_tiles"], qt[:, 0] * 3 + qt[:, 1])
            times, speeds = np.array([0.0, 0.005 * len(xy)]), np.zeros(2)
            intervals = np.array([times])
            if not np.isin(spikes[:, 1], ids).all() or not np.all(np.diff(spikes[:, 0]) >= 0):
                raise ValueError("simulation contains unknown cells or clock disorder")
        saved_events = pd.read_csv(local / "detector_events.csv")
        saved = load_npz(local / "endpoint_counts.npz")
        readouts = pd.read_csv(local / "window_readouts.csv.gz")
        summary = pd.read_csv(local / "readout_summary.csv")
        checked, detected = 0, 0
        for name, p in pops.items():
            keep = np.isin(spikes[:, 1], ids[p["indices"]])
            expected = []
            for lo, hi in intervals:
                if hi - lo < 0.05:
                    continue
                expected.extend(
                    detector.detect_high_mua_in_interval(
                        spikes[keep],
                        times,
                        speeds,
                        lo,
                        hi,
                        bin_s=0.001,
                        gaussian_sd_s=0.01,
                        z_threshold=3.0,
                        maximum_speed_cm_s=5.0,
                        minimum_duration_s=0.05,
                        maximum_duration_s=2.0,
                        minimum_active_cells=max(1, int(np.ceil(0.1 * len(p["indices"])))),
                    )
                )
            ev = saved_events.loc[saved_events.detector.eq(name)]
            if len(expected) != len(ev):
                raise ValueError("native detector rerun differs")
            if expected:
                assert_columns(ev, pd.DataFrame(expected), ["event_start_s", "event_end_s", "event_peak_s", "n_spikes", "n_active_cells"])
            detected += len(ev)
            expected_windows = []
            for e in expected:
                a, b = e["event_start_s"], e["event_end_s"]
                n = int(np.floor((b - a) / 0.005 + 1e-8))
                expected_windows.append([a + (n - 4) * 0.005, a + n * 0.005])
            np.testing.assert_allclose(saved[name + "__windows"], np.array(expected_windows).reshape(-1, 2), atol=1e-10, rtol=0)
        for detection in list(pops) + (["fixed"] if raw is not None else []):
            w, counts = saved[detection + "__windows"], saved[detection + "__counts"]
            independent = recount_independent(spikes, ids, w)
            np.testing.assert_array_equal(counts, independent)
            truth = truth_independent(raw["truth_tiles"], w) if raw is not None else np.full((len(w), 9), np.nan)
            np.testing.assert_allclose(saved[detection + "__truth"], truth, atol=2e-9, rtol=0, equal_nan=True)
            decoders = list(pops) if detection in ("full", "fixed") else ["full", detection]
            for decoding in decoders:
                n, r = counts[:, pops[decoding]["indices"]], rates[pops[decoding]["indices"]]
                for likelihood in ("poisson", "conditional_multinomial"):
                    ll = np.einsum("wc,cx->wx", n, np.log(r), optimize=True)
                    ll -= 0.02 * r.sum(axis=0) if likelihood == "poisson" else n.sum(axis=1)[:, None] * np.log(r.sum(axis=0))
                    posterior = np.exp(ll - logsumexp(ll, axis=1)[:, None])
                    regional = np.column_stack([posterior[:, tiles == k].sum(axis=1) for k in range(9)])
                    got = readouts.loc[readouts.detector.eq(detection) & readouts.decoder.eq(decoding) & readouts.likelihood.eq(likelihood)].sort_values("event_id")
                    if len(got) != len(w):
                        raise ValueError("missing crossed model rows")
                    np.testing.assert_allclose(got[[f"mass_{k}" for k in range(9)]], regional, atol=2e-9, rtol=2e-9)
                    np.testing.assert_allclose(got[[f"truth_{k}" for k in range(9)]], truth, atol=2e-9, rtol=0, equal_nan=True)
                    np.testing.assert_allclose(got[["start_s", "end_s"]], w, atol=1e-10, rtol=0)
                    s = summary.loc[summary.detector.eq(detection) & summary.decoder.eq(decoding) & summary.likelihood.eq(likelihood)].sort_values("region")
                    if len(s) != 9 or not s.n_windows.eq(len(w)).all():
                        raise ValueError("summary denominator or regions incomplete")
                    np.testing.assert_allclose(s.posterior_mass, regional.mean(axis=0) if len(w) else np.full(9, np.nan), atol=2e-9, rtol=0, equal_nan=True)
                    np.testing.assert_allclose(s.true_mass, truth.mean(axis=0) if len(w) and raw is not None else np.full(9, np.nan), atol=2e-9, rtol=0, equal_nan=True)
                    checked += len(w)
        contrast = pd.read_csv(local / "factorial_contrasts.csv")
        if len(contrast) != (len(pops) - 1) * 18:
            raise ValueError("missing factorial cells")
        np.testing.assert_allclose(contrast.total, contrast.detection + contrast.decoding + contrast.interaction, atol=2e-12, rtol=0, equal_nan=True)
        for r in contrast.itertuples():
            s = summary.loc[summary.region.eq(r.region) & summary.likelihood.eq(r.likelihood)].set_index(["detector", "decoder"])
            for field, key in [
                ("q_full_full", ("full", "full")),
                ("q_subset_full", (r.population, "full")),
                ("q_full_subset", ("full", r.population)),
                ("q_subset_subset", (r.population, r.population)),
            ]:
                np.testing.assert_allclose(getattr(r, field), s.loc[key, "posterior_mass"], atol=1e-12, equal_nan=True)
            assert (r.status == "complete") == (r.n_full > 0 and r.n_subset > 0)
        rows.append(
            {
                "dataset": row.dataset,
                "animal": row.animal,
                "session": row.session,
                "source": source.source,
                "detector_events": detected,
                "decoded_windows_checked": checked,
                "status": "pass",
            }
        )
        print("AUDITED", row.session, source.source, flush=True)
    return rows


def audit_worker(record, detector_path):
    return audit_session(pd.Series(record), load_detector(Path(detector_path)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.experiment / "manifest.json").read_text())
    sessions = pd.read_csv(args.experiment / "sessions.csv")
    if manifest["status"] != "complete_pending_audit" or not sessions.status.eq("complete").all():
        raise ValueError("experiment incomplete")
    for name, path in manifest["input_file_paths"].items():
        if file_sha256(path) != manifest["input_file_sha256"][name]:
            raise ValueError("producer input/code changed: " + name)
    frozen = pd.read_csv(args.experiment / "frozen_sessions.csv")
    keys = ["dataset", "animal", "session"]
    if set(map(tuple, sessions[keys].values)) != set(map(tuple, frozen[keys].values)) or sessions.duplicated(keys).any():
        raise ValueError("requested cohort differs from finished cohort")
    expected_sources = {f"{g}_rep{r}_gain{p}" for g in ("stationary", "moving") for r in range(manifest["parameters"]["replicates"]) for p in (3, 6)}
    if not manifest["parameters"]["skip_real"]:
        expected_sources.add("real")
    for row in sessions.itertuples():
        source = pd.read_csv(Path(row.folder) / "sources.csv")
        if set(source.source) != expected_sources or source.source.duplicated().any():
            raise ValueError("source panel missing or duplicated")
    if args.workers < 1:
        raise ValueError("positive audit workers required")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(audit_worker, record, manifest["input_file_paths"]["detector"]) for record in sessions.to_dict("records")]
        for job in as_completed(jobs):
            rows.extend(job.result())
            pd.DataFrame(rows).sort_values(["dataset", "animal", "session", "source"]).to_csv(args.output_dir / "source_audit.csv", index=False)
    result = build_script_provenance(input_paths={"experiment": args.experiment / "manifest.json", "auditor": Path(__file__)}, cwd=ROOT)
    result.update(status="technical_pass", sessions=len(sessions), sources=len(rows), output_sha256=file_sha256(args.output_dir / "source_audit.csv"))
    (args.output_dir / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
