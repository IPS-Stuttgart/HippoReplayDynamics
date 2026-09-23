#!/usr/bin/env python3
"""Verify RUN outputs and independently check raw-spike maps and predictions."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _provenance import file_sha256
from validate_kleinman_run_decoder import (
    PARAMETERS,
    align_behavior,
    interval_data,
    make_traversals,
    score_session,
    session_pass,
    split_units,
)


def independent_map(trains, t, dt, direction, spatial, mask, n_bins):
    occupancy = np.zeros((2, n_bins))
    counts = np.zeros((2, n_bins, len(trains)))
    for d in (0, 1):
        keep = mask & (direction == d)
        occupancy[d] = np.bincount(spatial[keep], weights=dt[keep], minlength=n_bins)
    for u, spikes in enumerate(trains):
        index = np.searchsorted(t, spikes, side="right") - 1
        valid = (index >= 0) & (index < len(dt))
        index = index[valid]
        index = index[mask[index]]
        for d in (0, 1):
            counts[d, :, u] = np.bincount(spatial[index[direction[index] == d]], minlength=n_bins)
    sigma = PARAMETERS["smooth_cm"] / PARAMETERS["bin_cm"]
    smooth_occ = gaussian_filter1d(occupancy, sigma, axis=1, mode="constant", truncate=4)
    rates = gaussian_filter1d(counts, sigma, axis=1, mode="constant", truncate=4) / np.maximum(smooth_occ[:, :, None], 1e-12)
    units = ((counts.sum(axis=(0, 1)) >= PARAMETERS["minimum_run_spikes"])
             & (rates.max(axis=(0, 1)) >= PARAMETERS["minimum_peak_hz"]))
    return (np.maximum(rates[:, :, units], PARAMETERS["rate_floor_hz"]).reshape(2 * n_bins, -1),
            (occupancy >= PARAMETERS["minimum_occupancy_s"]).reshape(-1), units)


def check_independent_windows(folder, windows):
    info = loadmat(folder / "session_info.mat", simplify_cells=True)["session_info"]
    t, x, speed, _, visits, epochs = align_behavior(info)
    runs = make_traversals(visits, epochs)
    _, trains, _ = split_units(loadmat(folder / "spike_data.mat", simplify_cells=True)["spike_data"])
    width = PARAMETERS["bin_cm"]
    edges = np.arange(np.floor(x.min() / width) * width, np.ceil(x.max() / width) * width + width, width)
    centers = np.tile((edges[:-1] + edges[1:]) / 2, 2)
    dt, _, trainable, fold, direction, spatial = interval_data(t, x, speed, runs, edges)
    checked = 0
    for f, rows in windows.groupby("fold"):
        rates, support, units = independent_map(trains, t, dt, direction, spatial, trainable & (fold != f), len(edges) - 1)
        selected = [s for s, keep in zip(trains, units, strict=True) if keep]
        for row in rows.iloc[:32].itertuples():
            observed = np.array([np.count_nonzero((s >= row.start_s) & (s < row.end_s)) for s in selected])
            assert observed.sum() == row.n_spikes and (observed > 0).sum() == row.n_active_units
            assert len(selected) == row.n_units
            logits = np.array([np.sum(observed * np.log(rate)) - PARAMETERS["test_window_s"] * np.sum(rate) for rate in rates])
            logits[~support] = -np.inf
            prob = np.exp(logits - logsumexp(logits))
            mean = float(np.sum(prob * centers))
            map_x = centers[np.argmax(logits)]
            np.testing.assert_allclose(mean, row.posterior_mean_cm, rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(map_x, row.map_cm, rtol=0, atol=1e-9)
            np.testing.assert_allclose(prob[len(edges) - 1:].sum(), row.right_direction_probability, rtol=1e-9, atol=1e-9)
            checked += 1
    return checked


def plot(sessions, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    animals = sorted(sessions.animal.unique())
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharex=True, sharey=True)
    for ax, animal in zip(axes.flat, animals, strict=True):
        rows = sessions[sessions.animal == animal]
        for passed, color in [(True, "#157c85"), (False, "#b74445")]:
            sub = rows[rows.decoder_pass == passed]
            ax.scatter(sub.mean_posterior_mean_error_cm, sub.mean_direction_correct, s=24, color=color)
        ax.axvline(35, color="0.6", linestyle=":")
        ax.axhline(.6, color="0.6", linestyle=":")
        ax.set_title(f"{animal}: {int(rows.decoder_pass.sum())}/{len(rows)} pass", fontsize=11)
        ax.set_xlim(0, 105)
        ax.set_ylim(.45, 1.02)
    for ax in axes[-1]:
        ax.set_xlabel("Mean held-out position error (cm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Held-out direction accuracy")
    fig.suptitle("Whole-lap RUN validation; teal = pass, red = fail; source failures have no point", fontsize=11)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def verify(result_dir):
    manifest = json.loads((result_dir / "manifest.json").read_text())
    for name, digest in manifest["outputs"].items():
        if Path(name).name != name or file_sha256(result_dir / name) != digest:
            raise ValueError("Changed output: " + name)
    for key, path in manifest["input_file_paths"].items():
        if file_sha256(path) != manifest["input_file_sha256"][key]:
            raise ValueError("Changed input: " + key)
    root = Path(manifest["input_file_paths"]["readme"]).parent
    sessions = pd.read_csv(result_dir / "kleinman_run_sessions.csv")
    folds = pd.read_csv(result_dir / "kleinman_run_folds.csv")
    windows = pd.read_csv(result_dir / "kleinman_run_windows.csv")
    visits = pd.read_csv(result_dir / "kleinman_run_visit_alignment.csv")
    assert len(sessions) == manifest["n_spike_sessions"] == len(list(root.glob("Experiment_1/*/*/spike_data.mat")))
    assert not sessions.duplicated(["animal", "session"]).any()
    assert not windows.duplicated(["animal", "session", "start_s"]).any()
    assert (windows.groupby(["animal", "session", "lap_group"]).fold.nunique() == 1).all()
    rerun_sessions, independently_checked, independent_animals = 0, 0, set()
    for row in sessions.itertuples():
        folder = root / "Experiment_1" / row.animal / row.session
        try:
            ss, ff, ww, vv = score_session(folder)
        except (ValueError, KeyError, IndexError) as exc:
            assert row.status == "failed" and row.failure_reason == str(exc)
            continue
        assert row.status == "scored"
        for name, value in ss.items():
            if isinstance(value, (str, bool)):
                assert getattr(row, name) == value
            else:
                np.testing.assert_allclose(getattr(row, name), value, atol=1e-9, rtol=1e-9)
        for actual, reconstructed in [(folds, ff), (windows, ww), (visits, vv)]:
            sub = actual[(actual.animal == row.animal) & (actual.session == row.session)]
            columns = list(reconstructed[0]) if reconstructed else []
            if columns:
                pd.testing.assert_frame_equal(sub[columns].reset_index(drop=True), pd.DataFrame(reconstructed), check_dtype=False, atol=1e-9, rtol=1e-9)
            else:
                assert not len(sub)
        sub = windows[(windows.animal == row.animal) & (windows.session == row.session)]
        expected = session_pass(sub.posterior_mean_error_cm.mean(), sub.direction_correct.mean(), len(sub), row.minimum_fold_units, row.n_scored_folds)
        assert row.decoder_pass == expected
        if row.animal not in independent_animals and len(sub):
            independently_checked += check_independent_windows(folder, sub)
            independent_animals.add(row.animal)
        rerun_sessions += 1
    assert int(sessions.decoder_pass.sum()) == manifest["n_decoder_pass_sessions"]
    assert not manifest["replay_scored"] and not manifest["biological_contrast_scored"]
    summary = sessions.groupby("animal").agg(
        n_sessions=("session", "size"), decoder_pass_sessions=("decoder_pass", "sum"),
        median_session_mean_error_cm=("mean_posterior_mean_error_cm", "median"),
        median_session_direction_accuracy=("mean_direction_correct", "median"),
        n_run_test_windows=("n_test_windows", "sum"), native_sdes=("native_sdes", "sum"))
    summary.to_csv(result_dir / "kleinman_run_by_animal.csv")
    plot(sessions, result_dir / "kleinman_run_qc.png")
    result = {
        "status": "passed", "sessions_recomputed": rerun_sessions,
        "source_failures_reproduced": len(sessions) - rerun_sessions,
        "all_window_outputs_recomputed": len(windows),
        "raw_spike_map_and_likelihood_independent_windows": independently_checked,
        "raw_spike_independent_animals": sorted(independent_animals),
        "all_windows_independently_recomputed": False,
        "decoder_pass_sessions": int(sessions.decoder_pass.sum()),
        "new_biological_result": False, "result_manifest_sha256": file_sha256(result_dir / "manifest.json"),
        "verification_script_sha256": file_sha256(Path(__file__)),
        "report_outputs": {name: file_sha256(result_dir / name) for name in ["kleinman_run_by_animal.csv", "kleinman_run_qc.png"]},
    }
    (result_dir / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    verify(parser.parse_args().result_dir)
