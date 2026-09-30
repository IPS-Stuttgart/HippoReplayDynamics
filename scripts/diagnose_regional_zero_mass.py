"""Post-hoc zero-mass control for the frozen known-truth endpoint diagnostic."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]

import numpy as np
import pandas as pd

from hipporeplayimm.regional_content_frontier import discrimination
from hipporeplayimm.regional_readout_endpoint import ContinuousDensity, fit_prevalence, oracle_prevalence
from scripts._provenance import build_script_provenance, file_sha256
from scripts.diagnose_regional_readout_endpoint import load_npz, read_csv, summarize, validation_mask


def mixed_measure_likelihoods(cal_score, cal_truth, query):
    y, z, q = np.asarray(cal_score), np.asarray(cal_truth), np.asarray(query)
    if y.shape != z.shape or not np.isfinite(y).all() or not np.isfinite(q).all() or not np.isin(z, [0, 1]).all():
        raise ValueError("invalid scores or labels")
    columns = []
    for c in (0, 1):
        values = y[z == c]
        if not len(values):
            raise ValueError("both calibration classes required")
        zero = values == 0
        p0 = (zero.sum()+.5)/(len(values)+1)
        f = np.full(len(q), p0)
        positive_query = q != 0
        if positive_query.any():
            if zero.all():
                raise ValueError("nonzero queries without nonzero class calibration")
            f[positive_query] = (1-p0)*ContinuousDensity(values[~zero])(q[positive_query])
        columns.append(f)
    return np.column_stack(columns)


def analyze(meta, labels, accepted, kinds, bf, totals, session, animal, window_ms):
    score = np.where(totals == 0, 0., bf)
    cal = meta.phase.eq("calibration").to_numpy()[:, None] & accepted
    rows = []
    for mode in (("pooled", "oracle_generator") if window_ms == 20 else ("oracle_generator",)):
        f = np.zeros((*score.shape, 2))
        if mode == "pooled":
            f[:] = mixed_measure_likelihoods(score[cal], labels[cal], score.ravel()).reshape(*score.shape, 2)
        else:
            for kind in np.unique(kinds):
                use, query = cal & (kinds == kind), kinds == kind
                f[query] = mixed_measure_likelihoods(score[use], labels[use], score[query])
        for i in np.flatnonzero(validation_mask(meta)):
            ok = accepted[i]
            estimate = fit_prevalence(f[i, ok]) if mode == "pooled" else oracle_prevalence(f[i, ok], kinds[i, ok])
            truth = float(labels[i, ok].mean())
            error = abs(estimate-truth)
            job = meta.iloc[i]
            lr = np.log(f[i, ok, 1])-np.log(f[i, ok, 0])
            rows.append({"session": session, "animal": animal, "population": "full", "family": "full", "window_ms": window_ms,
                "readout": "continuous_zero_mass", "calibration_mode": mode, "generator": job.generator,
                "requested_prevalence": job.requested_prevalence, "replica": int(job.replica), "job": int(job.job),
                "estimate": estimate, "truth": truth, "absolute_error": error, "within_5pp": bool(np.isfinite(error) and error <= .05),
                "status": "fit" if np.isfinite(estimate) else "unidentified", "events": int(ok.sum()),
                "raw_bf_auc": discrimination(bf[i, ok], labels[i, ok])["auc"],
                "calibrated_readout_auc": discrimination(lr, labels[i, ok])["auc"],
                "mean_spikes": float(totals[i, ok].mean()), "silent_fraction": float((totals[i, ok] == 0).mean())})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnostic-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.diagnostic_dir/"manifest.json").read_text())
    assert manifest["status"] == "complete"
    inputs = {"primary": args.diagnostic_dir/"manifest.json", "script": Path(__file__),
              "core": ROOT/"src/hipporeplayimm/regional_readout_endpoint.py",
              "protocol": ROOT/"docs/regional_readout_zero_mass_control.md"}
    rows = []
    for s in manifest["sessions"]:
        source = Path(s["source"]).parent
        folder = args.diagnostic_dir/s["session"].replace("/", "_")
        sm = json.loads((source/"manifest.json").read_text())
        for name in ("frozen_inputs.npz", "simulation_counts.npz", "calibration_and_readouts.npz", "simulation_jobs.csv"):
            assert file_sha256(source/name) == sm["outputs"][name]
            inputs[s["session"]+":"+name] = source/name
        assert file_sha256(folder/"terminal_5ms_audit.npz") == s["outputs"]["terminal_5ms_audit.npz"]
        inputs[s["session"]+":5ms"] = folder/"terminal_5ms_audit.npz"
        inputs[s["session"]+":5ms_jobs"] = folder/"terminal_jobs.csv"
        e = load_npz(source/"frozen_inputs.npz")
        bank = load_npz(source/"simulation_counts.npz")
        stored = load_npz(source/"calibration_and_readouts.npz")
        lookup = {int(c): i for i, c in enumerate(e["cell_ids"])}
        ix = [lookup[int(c)] for c in e["pop0_ids"]]
        meta = read_csv(source/"simulation_jobs.csv")
        rows.extend(analyze(meta, bank["labels"], bank["accepted"], bank["generators"], stored["pop0_bf"],
            bank["counts"][:, :, ix].sum(axis=2), s["session"], s["animal"], 20))
        short = load_npz(folder/"terminal_5ms_audit.npz")
        rows.extend(analyze(read_csv(folder/"terminal_jobs.csv"), short["labels"], short["accepted"],
            bank["generators"][short["jobs"]], short["bf"], short["counts"][:, :, ix].sum(axis=2), s["session"], s["animal"], 5))
        print("ZERO-MASS", s["session"], "complete", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir/"validation_panels.csv", index=False)
    summary = summarize(frame)
    summary.to_csv(args.output_dir/"validation_summary.csv", index=False)
    keys = ["window_ms", "generator", "calibration_mode", "readout"]
    metrics = ["mean_absolute_error", "within_5pp_fraction", "calibrated_readout_auc", "silent_fraction"]
    by_animal = summary.groupby(["animal"]+keys)[metrics].mean().reset_index()
    by_animal.to_csv(args.output_dir/"by_animal.csv", index=False)
    pooled = by_animal.groupby(keys)[metrics].mean().reset_index()
    pooled.to_csv(args.output_dir/"descriptive_summary.csv", index=False)
    provenance = build_script_provenance(input_paths=inputs, cwd=ROOT)
    provenance.update(status="complete", posthoc_control=True,
        outputs={f.name:file_sha256(f) for f in args.output_dir.iterdir() if f.is_file()})
    (args.output_dir/"manifest.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(pooled.to_string(index=False))


if __name__ == "__main__":
    main()
