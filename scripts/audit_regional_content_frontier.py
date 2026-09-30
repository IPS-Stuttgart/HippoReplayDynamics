#!/usr/bin/env python3
"""Reconstruct frontier counts, likelihood ratios and metrics independently."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]

import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp
from scipy.stats import mannwhitneyu

from scripts._provenance import build_script_provenance, file_sha256


def independent_bf(n, r, region, exposure, conditional=False):
    result = []
    for start in range(0, len(n), 512):
        counts = n[start:start+512]
        ll = counts @ np.log(r)
        if conditional:
            ll -= np.sum(counts, axis=1)[:, None]*np.log(np.sum(r, axis=0))
        else:
            e = np.broadcast_to(exposure, (len(n),))[start:start+512]
            ll -= e[:, None]*np.sum(r, axis=0)
        result.extend(logsumexp(ll[:, region], axis=1)-np.log(np.count_nonzero(region))
                      -logsumexp(ll[:, ~region], axis=1)+np.log(np.count_nonzero(~region)))
    return np.asarray(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.input_dir/"manifest.json").read_text())
    assert manifest["status"] == "complete"
    for key, path in manifest["input_file_paths"].items():
        assert file_sha256(path) == manifest["input_file_sha256"][key], path
    sessions = pd.read_csv(args.input_dir/"sessions.csv")
    checked = []
    for row in sessions.itertuples():
        folder = Path(row.folder)
        outputs = json.loads((folder/"outputs.json").read_text())
        for name, digest in outputs.items():
            assert file_sha256(folder/name) == digest, name
        a = np.load(folder/"run_calibration_audit.npz", allow_pickle=False)
        r, region, ids = a["rates"], a["region"], a["cell_ids"]
        with np.load(manifest["input_file_paths"][row.session+":encoder"], allow_pickle=False) as archive:
            source = {key: archive[key] for key in archive.files}
        win, counts = a["run_windows"], a["run_counts"]
        assert (win[:, 0] >= float(source["training_end_s"])).all()
        assert (np.diff(win[:, 0]) >= .02-1e-8).all()
        np.testing.assert_allclose(win[:, 1]-win[:, 0], .02, atol=1e-9)
        independently_counted = np.empty_like(counts)
        for j, cell in enumerate(ids):
            times = np.sort(source["spikes"][source["spikes"][:, 1] == cell, 0])
            independently_counted[:, j] = np.searchsorted(times, win[:, 1], side="left")-np.searchsorted(times, win[:, 0], side="left")
        np.testing.assert_array_equal(counts, independently_counted)
        pos = source["position"]
        times = win[:, :1]+(np.arange(20)+.5)*.001
        xy = np.stack([np.interp(times, pos[:, 0], pos[:, d]) for d in (1, 2)], axis=-1)
        np.testing.assert_allclose(xy.mean(axis=1), win[:, 2:4], atol=1e-9)
        prior_source = json.loads(Path(manifest["input_file_paths"][row.session+":source"]).read_text())
        mask = np.linalg.norm(xy-np.array(prior_source["home_xy"]), axis=2) <= 20
        assert (mask.all(axis=1) | (~mask).all(axis=1)).all()
        np.testing.assert_array_equal(mask[:, 0], win[:, 4])
        np.testing.assert_array_equal(a["run_bouts"], np.cumsum(np.r_[True, np.diff(win[:, 0]) > 1.0])-1)
        np.testing.assert_array_equal(a["calibration"], a["run_bouts"] % 2 == 0)
        assert not set(a["run_bouts"][a["calibration"]]) & set(a["run_bouts"][~a["calibration"]])
        retention = np.minimum(1, np.divide(a["desired_counts"], counts.sum(axis=1),
            out=np.zeros(len(counts)), where=counts.sum(axis=1) > 0))
        np.testing.assert_allclose(retention, a["retention_probability"])
        assert np.all(a["thinned_counts"] <= counts)
        for name, n, exposure in (("native_run", counts, .02),
                                  ("thinned_frozen_decoder", a["thinned_counts"], .02),
                                  ("thinned_exposure_adjusted", a["thinned_counts"], .02*retention),
                                  ("real_log_bf", a["real_counts"], .02)):
            np.testing.assert_allclose(independent_bf(n, r, region, exposure), a[name], atol=1e-9)
        frontier = pd.read_csv(folder/"discrimination_frontier.csv")
        s = np.load(folder/"synthetic_frontier_audit.npz", allow_pickle=False)
        metrics_checked, synthetic_counts = 0, 0
        for line in frontier.itertuples():
            key = f"{line.population}__{line.duration_ms}__{line.multiplier}__{line.generator}___"
            n, z, indices = (s[key+x] for x in ("counts", "truth", "indices"))
            rr = r[indices]
            matched = line.readout == "matched_likelihood"
            exposure = line.duration_ms/1000*(row.simulation_gain*line.multiplier if matched else 1)
            bf = independent_bf(n, rr, region, exposure, matched and line.generator == "conditional_multinomial")
            np.testing.assert_allclose(bf, s[key+("matched_log_bf" if matched else "frozen_log_bf")], atol=1e-9)
            auc = mannwhitneyu(bf[z == 1], bf[z == 0]).statistic/((z == 1).sum()*(z == 0).sum())
            error = np.where(bf == 0, .5, (bf > 0) != z).mean()
            q = expit(bf)
            np.testing.assert_allclose([auc, error, np.mean((q-z)**2), np.mean(np.logaddexp(0, bf)-bf*z)],
                [line.auc, line.balanced_error, line.balanced_brier, line.balanced_log_loss], atol=1e-9)
            metrics_checked += 1
            if matched:
                rng = np.random.default_rng(line.simulation_seed)
                states = np.r_[rng.choice(np.flatnonzero(~region), len(z)//2), rng.choice(np.flatnonzero(region), len(z)//2)]
                np.testing.assert_array_equal(states, s[key+"states"])
                if line.generator == "poisson":
                    expected = rng.poisson(rr[:, states].T*line.duration_ms/1000*row.simulation_gain*line.multiplier)
                else:
                    # Reconstruct this population/segment's observed-total pool.
                    pool = []
                    raw_ids = np.flatnonzero(np.isin(source["cell_ids"], ids))[indices]
                    for j in range(len(source["candidate_event_indices"])):
                        lo, hi = source["candidate_offsets"][j:j+2]
                        good = np.flatnonzero(np.isclose(source["candidate_base_durations_s"][lo:hi], .005, atol=1e-9, rtol=0))
                        width = line.duration_ms//5
                        if len(good) >= width:
                            pool.append(source["candidate_base_counts"][lo+good[-width:]][:, raw_ids].sum())
                    totals = rng.choice(pool, len(z))*line.multiplier
                    np.testing.assert_array_equal(totals, s[key+"requested"])
                    probabilities = rr[:, states].T.copy()
                    probabilities /= probabilities.sum(axis=1)[:, None]
                    expected = np.array([rng.multinomial(int(k), prob) for k, prob in zip(totals, probabilities, strict=True)])
                    np.testing.assert_array_equal(n.sum(axis=1), totals)
                np.testing.assert_array_equal(n, expected)
                synthetic_counts += n.size
        checked.append({"session": row.session, "run_cell_counts": counts.size,
                            "synthetic_cell_counts": synthetic_counts, "frontier_rows": metrics_checked})
        print(row.session, "independent audit passed", flush=True)
    for name in ("discrimination_frontier", "run_calibration", "prevalence_fits", "prevalence_compatibility", "populations"):
        pooled = pd.read_csv(args.input_dir/(name+".csv"))
        joined = pd.concat([pd.read_csv(Path(row.folder)/(name+".csv")) for row in sessions.itertuples()], ignore_index=True)
        pd.testing.assert_frame_equal(pooled, joined)
    out = build_script_provenance(input_paths={"manifest": args.input_dir/"manifest.json", "auditor": Path(__file__)}, cwd=ROOT)
    out.update(status="pass", checked=checked, scope="source RUN counts/truth; saved likelihood ratios; synthetic RNG reconstruction; AUC/error/losses; pooled tables and hashes. Bootstrap bounds are sensitivity sets, not audited coverage guarantees.",
               result_sha256={p.name: file_sha256(p) for p in args.input_dir.glob("*.csv")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2)+"\n")


if __name__ == "__main__":
    main()

