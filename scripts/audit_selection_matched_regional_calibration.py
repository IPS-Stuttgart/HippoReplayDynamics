"""Independent artifact, likelihood, clock and detector audit for matched calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp


def read_csv(path, **kwargs):
    return pd.read_csv(path, keep_default_na=False, na_values=[""], **kwargs)


from hipporeplayimm.data import load_mat_variable
from hipporeplayimm.regional_content_mua_null import load_detector
from hipporeplayimm.selection_matched_regional import draw_panel
from scripts._provenance import file_sha256
from scripts.measure_edge_support_content import occupied_graph
from scripts.measure_selection_matched_regional_calibration import detector_run


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {key:z[key] for key in z.files}


def verify_manifest(path):
    m = json.loads(path.read_text())
    assert m["status"] == "complete"
    for key, source in m["input_file_paths"].items():
        assert file_sha256(Path(source)) == m["input_file_sha256"][key], source
    for name, digest in m.get("outputs", {}).items():
        assert file_sha256(path.parent/name) == digest, name
    return m


def audit(root):
    manifest = verify_manifest(root/"manifest.json")
    sessions = read_csv(root/"sessions.csv")
    assert len(sessions) > 0 and sessions.status.eq("complete").all()
    results = []
    for s in sessions.itertuples():
        folder = Path(s.folder)
        m = verify_manifest(folder/"manifest.json")
        frozen, bank, readouts = [arrays(folder/name) for name in ("frozen_inputs.npz", "simulation_counts.npz", "calibration_and_readouts.npz")]
        jobs = read_csv(folder/"simulation_jobs.csv", dtype={"seed":str})
        assert len(jobs) == len(bank["counts"])
        totals = frozen["real_counts"].sum(axis=1)
        np.testing.assert_array_equal(bank["counts"].sum(axis=2), np.broadcast_to(totals, bank["counts"].shape[:2]))
        np.testing.assert_array_equal(bank["labels"], frozen["region"][bank["targets"]])
        np.testing.assert_array_equal(bank["accepted"], bank["active"] >= int(np.ceil(.1*len(frozen["cell_ids"]))))
        templates = [{"start": float(a), "end": float(b), "times": frozen["template_times"][lo:hi]}
            for a,b,lo,hi in zip(frozen["candidate_start"], frozen["candidate_end"], frozen["template_offsets"][:-1], frozen["template_offsets"][1:], strict=True)]
        graph = occupied_graph(frozen["grid"])
        sample = jobs.loc[jobs.replica.eq(0)]
        for job in sample.itertuples():
            p = draw_panel(templates, job.requested_prevalence, job.generator, frozen["grid"], frozen["rates"], graph,
                frozen["region"], np.random.default_rng(int(job.seed)), job.perturbation)
            for key in ("counts", "targets", "labels", "active"):
                np.testing.assert_array_equal(p[key], bank[key][job.job])
        for job in jobs.loc[jobs.perturbation.eq("global_gain_x2")].itertuples():
            control = jobs.loc[jobs.phase.eq("validation") & jobs.requested_prevalence.eq(.30) & jobs.generator.eq("mix") & jobs.replica.eq(job.replica)].iloc[0]
            assert job.seed == control.seed
            np.testing.assert_array_equal(bank["counts"][job.job], bank["counts"][int(control.job)])
        shared = jobs.loc[jobs.perturbation.ne("global_gain_x2")]
        assert not shared.seed.duplicated().any()
        populations = json.loads((folder/"population_definitions.json").read_text())
        id_lookup = {int(c):i for i,c in enumerate(frozen["cell_ids"])}
        for j, _ in enumerate(populations):
            ids, rates, region = [frozen[f"pop{j}_{name}"] for name in ("ids", "rates", "region")]
            ix = [id_lookup[int(c)] for c in ids]
            counts = bank["counts"].reshape(-1, len(id_lookup))[:, ix]
            expected = readouts[f"pop{j}_bf"].ravel()
            for start in range(0, len(counts), 1024):
                n = counts[start:start+1024]
                ll = n @ np.log(rates)-.02*rates.sum(axis=0)
                bf = logsumexp(ll[:, region], axis=1)-logsumexp(ll[:, ~region], axis=1)+np.log((~region).sum()/region.sum())
                np.testing.assert_allclose(bf, expected[start:start+1024], atol=1e-8, rtol=1e-10)
            calls = np.where(expected > np.log(3), 2, np.where(expected < -np.log(3), 0, 1))
            calls[counts.sum(axis=1) == 0] = 1
            np.testing.assert_array_equal(calls.reshape(bank["labels"].shape), readouts[f"pop{j}_calls"])
            mask = jobs.phase.eq("calibration").to_numpy()[:, None] & bank["accepted"]
            actual = np.zeros((2, 3), int)
            np.add.at(actual, (bank["labels"][mask], calls.reshape(mask.shape)[mask]), 1)
            np.testing.assert_array_equal(actual, readouts[f"pop{j}_calibration"])
        raw = np.asarray(load_mat_variable(Path(m["input_file_paths"]["Spike_Data.mat"]), "Spike_Data"), float).reshape(-1, 2)
        raw = raw[np.argsort(raw[:, 0], kind="stable")]
        position = np.asarray(load_mat_variable(Path(m["input_file_paths"]["Position_Data.mat"]), "Position_Data"), float).reshape(-1, 4)
        intervals = np.asarray(load_mat_variable(Path(m["input_file_paths"]["Epochs.mat"]), "Run_Times"), float).reshape(-1, 2)
        detector = load_detector(Path(manifest["input_file_paths"]["detector"]))
        st, speed = detector.position_speed(position, .10)
        for kind in ("stationary", "moving", "late_jump"):
            saved = arrays(folder/f"detector_audit_{kind}.npz")
            changed = raw.copy()
            for event, lo, hi in zip(templates, frozen["template_offsets"][:-1], frozen["template_offsets"][1:], strict=True):
                left,right = np.searchsorted(raw[:, 0], [event["start"], event["end"]], side="left")
                np.testing.assert_array_equal(raw[left:right, 0], event["times"])
                changed[left:right, 1] = frozen["cell_ids"][saved["identities"][lo:hi]]
            np.testing.assert_array_equal(changed[:, 0], raw[:, 0])
            redetected = detector_run(detector, changed, st, speed, intervals, len(id_lookup))
            starts = {round(float(e["event_start_s"]), 6) for e in redetected}
            actual = np.array([round(e["start"], 6) in starts for e in templates])
            np.testing.assert_array_equal(actual, bank["accepted"][int(saved["job"])])
        results.append({"session": s.session, "panels": len(jobs), "templates": len(templates),
            "redrawn_panels": len(sample), "populations": len(populations), "status": "pass"})
        print("AUDITED", s.session, flush=True)
    for name in ("validation_and_null", "real_descriptive"):
        expected = pd.concat([read_csv(Path(s.folder)/(name+".csv")) for s in sessions.itertuples()], ignore_index=True)
        actual = read_csv(root/(name+".csv"))
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False, atol=1e-9)
    return {"status": "pass", "sessions": len(results), "panels": sum(r["panels"] for r in results),
        "population_definitions": sum(r["populations"] for r in results),
        "checks": "hashes, endpoint count traces, accepted support, known truth, seed separation, redraws, all likelihoods, all calls, calibration counts, gain invariance, full original detector reruns, pooled tables", "runs": results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit(args.input_dir)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print({k:v for k,v in result.items() if k != "runs"})


if __name__ == "__main__":
    main()
