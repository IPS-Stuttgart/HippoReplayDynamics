"""Independent persisted-count, likelihood, estimator and denominator audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from hipporeplayimm.regional_readout_endpoint import class_likelihoods
from scripts._provenance import build_script_provenance, file_sha256
from scripts.diagnose_regional_readout_endpoint import load_npz, read_csv, validation_mask


def check_hashes(folder, manifest):
    for name, expected in manifest["outputs"].items():
        assert file_sha256(folder/name) == expected, folder/name


def independent_bf(counts, rates, region):
    result = []
    for start in range(0, len(counts), 256):
        likelihood = counts[start:start+256]@np.log(rates)-.005*rates.sum(axis=0)
        result.extend(logsumexp(likelihood[:, region], axis=1)-np.log(region.sum())
                      -logsumexp(likelihood[:, ~region], axis=1)+np.log((~region).sum()))
    return np.array(result)


def audit(root):
    manifest = json.loads((root/"manifest.json").read_text())
    assert manifest["status"] == "complete"
    check_hashes(root, manifest)
    for name, path in manifest["input_file_paths"].items():
        assert file_sha256(Path(path)) == manifest["input_file_sha256"][name], path
    joined = read_csv(root/"validation_panels.csv")
    keys = ["session", "population", "window_ms", "readout", "calibration_mode", "job"]
    assert not joined.duplicated(keys).any()
    assert len(manifest["sessions"]) > 0
    checks = []
    for s in manifest["sessions"]:
        folder = root/s["session"].replace("/", "_")
        saved_audit = json.loads((folder/"audit.json").read_text())
        check_hashes(folder, saved_audit)
        source_folder = Path(s["source"]).parent
        assert file_sha256(Path(s["source"])) == s["source_sha256"]
        e = load_npz(source_folder/"frozen_inputs.npz")
        original = load_npz(source_folder/"simulation_counts.npz")
        short = load_npz(folder/"terminal_5ms_audit.npz")
        meta = read_csv(folder/"terminal_jobs.csv")
        jobs = short["jobs"]
        np.testing.assert_array_equal(jobs, meta.job)
        np.testing.assert_array_equal(short["labels"], original["labels"][jobs])
        np.testing.assert_array_equal(short["accepted"], original["accepted"][jobs])
        assert (short["counts"] <= original["counts"][jobs]).all()
        ends = e["windows"][:, 1]
        offsets = e["template_offsets"]
        totals = []
        for k, end in enumerate(ends):
            t = e["template_times"][offsets[k]:offsets[k+1]]
            totals.append(np.sum((t >= end-.005) & (t < end)))
        np.testing.assert_array_equal(short["counts"].sum(axis=2), np.broadcast_to(totals, short["counts"].shape[:2]))
        lookup = {int(c): i for i, c in enumerate(e["cell_ids"])}
        ix = [lookup[int(c)] for c in e["pop0_ids"]]
        n = short["counts"][:, :, ix]
        bf = independent_bf(n.reshape(-1, len(ix)), e["pop0_rates"], e["pop0_region"]).reshape(n.shape[:2])
        np.testing.assert_allclose(bf, short["bf"], atol=1e-10, rtol=1e-10)
        calls = np.where(bf > np.log(3), 2, np.where(bf < -np.log(3), 0, 1))
        calls[n.sum(axis=2) == 0] = 1
        np.testing.assert_array_equal(calls, short["calls"])
        frame = read_csv(folder/"validation_panels.csv")
        subset = joined.loc[joined.session.eq(s["session"])]
        pd.testing.assert_frame_equal(frame.sort_values(keys).reset_index(drop=True), subset.sort_values(keys).reset_index(drop=True), check_exact=False, atol=1e-12)
        expected = s["populations"]*320*6+160*3
        assert len(frame) == expected, (s["session"], len(frame), expected)
        assert np.isfinite(frame.estimate).all(), "unidentified fits require explicit failure review"
        np.testing.assert_allclose(frame.absolute_error, np.abs(frame.estimate-frame.truth), atol=1e-12)
        np.testing.assert_array_equal(frame.within_5pp, frame.absolute_error.le(.05))
        # Refit continuous short-window examples using a different optimizer.
        cal = meta.phase.eq("calibration").to_numpy()[:, None] & short["accepted"]
        vm = validation_mask(meta)
        selected = meta.loc[vm].groupby(["generator", "requested_prevalence"]).head(1)
        refits = 0
        for r in selected.itertuples():
            local = int(np.flatnonzero(jobs == r.job)[0])
            cm = cal & np.broadcast_to(meta.generator.eq(r.generator).to_numpy()[:, None], cal.shape)
            ok = short["accepted"][local]
            for readout in ("continuous_neutral", "continuous_native"):
                score = np.where(n.sum(axis=2) == 0, 0., bf) if readout == "continuous_neutral" else bf
                f = class_likelihoods(score[cm], short["labels"][cm], score[local, ok], readout)
                f /= f.max(axis=1, keepdims=True)
                def loss(p, f=f):
                    return -np.log(np.maximum((1-p)*f[:, 0]+p*f[:, 1], 1e-250)).sum()
                fit = minimize_scalar(loss, bounds=(0., 1.), method="bounded", options={"xatol": 1e-10})
                estimate = min([0., float(fit.x), 1.], key=loss)
                row = frame.loc[frame.window_ms.eq(5) & frame.job.eq(r.job) & frame.readout.eq(readout)].iloc[0]
                assert abs(estimate-row.estimate) < 1e-5
                assert row.events == int(ok.sum())
                refits += 1
        checks.append({"session": s["session"], "rows": len(frame), "independent_refits": refits,
                       "short_windows_redecoded": int(np.prod(bf.shape)), "status": "pass"})
        print("AUDIT", s["session"], "pass", flush=True)
    return {"status": "pass", "sessions": checks, "rows": len(joined),
            "checks": "hashes, exact 5ms totals, unchanged truth and acceptance, every 5ms likelihood/call, pooled rows, denominators, independent MLE refits"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("audit output already exists")
    result = audit(args.input_dir)
    result["provenance"] = build_script_provenance(input_paths={"manifest": args.input_dir/"manifest.json", "audit_script": Path(__file__)}, cwd=ROOT)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
