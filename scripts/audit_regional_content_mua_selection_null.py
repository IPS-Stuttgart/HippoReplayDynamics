"""Independently recount, redetect, and verify the fixed-tuning null artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from hipporeplayimm.regional_content_mua_null import load_detector, stable_seed
from scripts._provenance import file_sha256


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def recount_independent(spikes, ids, windows):
    result = np.zeros((len(windows), len(ids)), int)
    for j, cell in enumerate(ids):
        t = np.sort(spikes[spikes[:, 1] == cell, 0])
        result[:, j] = np.searchsorted(t, windows[:, 1], side="left")-np.searchsorted(t, windows[:, 0], side="left")
    return result


def independent_exposure(windows, gain):
    edges = np.arange(len(gain)+1)*.001
    local = windows-np.floor(windows[:, :1]/2+1e-10)*2
    overlap = np.maximum(0., np.minimum(local[:, 1, None], edges[1:])-np.maximum(local[:, 0, None], edges[:-1]))
    return overlap @ gain


def verify_hashes(folder):
    for name, digest in json.loads((folder/"outputs.json").read_text()).items():
        if file_sha256(folder/name) != digest:
            raise AssertionError(f"output hash changed: {folder/name}")


def audit(root):
    manifest = json.loads((root/"manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["tuning_changes"] is False
    for key, path in manifest["input_file_paths"].items():
        assert file_sha256(Path(path)) == manifest["input_file_sha256"][key], key
    detector = load_detector(manifest["input_file_paths"]["detector_script"])
    params = manifest["parameters"]
    sessions = pd.read_csv(root/"sessions.csv")
    assert len(sessions) and not sessions.session.duplicated().any()
    rows = []
    for session in sessions.itertuples():
        folder = Path(session.folder)
        verify_hashes(folder)
        c = arrays(folder/"calibration.npz")
        encoder = arrays(Path(manifest["input_file_paths"][session.session+":encoder"]))
        np.testing.assert_array_equal(c["rates_all"], np.maximum(encoder["rates_hz"][:, encoder["valid_spatial_bins"]], 1e-4))
        np.testing.assert_array_equal(c["keep"], encoder["unit_qc_mask"])
        rates, region = c["rates_all"][c["keep"]], c["region"]
        event = pd.read_csv(folder/"event_readouts.csv.gz")
        detection = pd.read_csv(folder/"detection.csv")
        assert len(detection) == params["replicates"]*3
        for row in detection.itertuples():
            sub = folder/f"rep{row.replicate}_peak{row.peak_gain}"
            verify_hashes(sub)
            a = arrays(sub/"spike_train.npz")
            spikes = a["spikes"]
            assert np.isfinite(spikes).all() and (np.diff(spikes[:, 0]) >= 0).all()
            assert np.isin(spikes[:, 1], c["ids"]).all()
            assert (spikes[:, 0] >= 0).all() and (spikes[:, 0] < params["duration_s"]).all()
            assert len(spikes) == row.total_generated_spikes
            rerun = detector.detect_high_mua_in_interval(spikes, np.array([0., params["duration_s"]]), np.zeros(2), 0., params["duration_s"],
                bin_s=.001, gaussian_sd_s=.010, z_threshold=3., maximum_speed_cm_s=5.,
                minimum_duration_s=.050, maximum_duration_s=2., minimum_active_cells=max(1, int(np.ceil(.1*len(c["ids"])))))
            assert len(rerun) == row.detector_events
            if rerun:
                saved = pd.read_csv(sub/"detector_events.csv")
                pd.testing.assert_frame_equal(pd.DataFrame(rerun), saved, check_dtype=False, atol=1e-9)
            w = a["selected_windows"]
            assert len(w) == row.selected_endpoints
            for window, j in zip(w, a["selected_event_ids"], strict=True):
                ev = rerun[j]
                complete = int(np.floor((ev["event_end_s"]-ev["event_start_s"])/.005+1e-8))
                expected_window = ev["event_start_s"]+np.array([complete-4, complete])*.005
                np.testing.assert_allclose(window, expected_window, atol=1e-9)
            for windows, key in ((w, "selected_counts"), (a["fixed_windows"], "fixed_counts")):
                np.testing.assert_array_equal(recount_independent(spikes, c["ids"][c["keep"]], windows), a[key])
                np.testing.assert_allclose(windows[:, 1]-windows[:, 0], .02, atol=1e-9)
            exposures = independent_exposure(w, a["gain"])
            np.testing.assert_allclose(exposures, a["exposures"], atol=1e-12)
            state = a["states"][np.floor(w[:, 0]/2+1e-10).astype(int)]
            np.testing.assert_array_equal(state, a["selected_locations"])
            means = rates[:, state].T*exposures[:, None]
            np.testing.assert_allclose(means, a["expected_counts"], atol=1e-10)
            independent = np.random.default_rng(stable_seed(params["seed"], session.session, "independent", row.replicate, row.peak_gain)).poisson(a["expected_counts"])
            np.testing.assert_array_equal(independent, a["independent_counts"])
            observed = event.loc[(event.replicate == row.replicate) & (event.peak_gain == row.peak_gain)]
            for cohort, n, windows, locations in (
                ("fixed", a["fixed_counts"], a["fixed_windows"], a["fixed_locations"]),
                ("selected", a["selected_counts"], w, state),
                ("independent_same_window", independent, w, state)):
                exp = independent_exposure(windows, a["gain"])
                for readout in ("frozen_poisson", "oracle_gain"):
                    frame = observed.loc[(observed.cohort == cohort) & (observed.readout == readout)]
                    assert len(frame) == len(n)
                    if not len(n):
                        continue
                    ll = n @ np.log(rates)
                    ll -= (exp[:, None] if readout == "oracle_gain" else .02)*rates.sum(axis=0)
                    bf = logsumexp(ll[:, region], axis=1)-logsumexp(ll[:, ~region], axis=1)+np.log((~region).sum()/region.sum())
                    np.testing.assert_allclose(frame.log_bf, bf, atol=1e-8)
                    np.testing.assert_array_equal(frame.true_home, region[locations])
                    call = np.where(bf > np.log(3), 2, np.where(bf < -np.log(3), 0, 1))
                    call[n.sum(axis=1) == 0] = 1
                    np.testing.assert_array_equal(frame.category, call)
            rows.append({"session": session.session, "replicate": row.replicate, "peak_gain": row.peak_gain,
                             "spikes": len(spikes), "selected_windows": len(w), "status": "pass"})
    for name in ("calibration_fits", "compatibility", "detection", "call_distributions"):
        expected = pd.concat([pd.read_csv(Path(s.folder)/(name+".csv")) for s in sessions.itertuples()], ignore_index=True)
        pd.testing.assert_frame_equal(expected, pd.read_csv(root/(name+".csv")), check_dtype=False, atol=1e-9)
    return {"status": "pass", "sessions": len(sessions), "simulation_runs": len(rows),
                "generated_spikes": sum(r["spikes"] for r in rows), "selected_windows": sum(r["selected_windows"] for r in rows),
                "verified": "input/output hashes, unchanged maps, original detector, recounts, endpoint timing, oracle exposure, independent draws, likelihoods, calls, pooled tables", "runs": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input_dir)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print({k: v for k, v in result.items() if k != "runs"})


if __name__ == "__main__":
    main()
