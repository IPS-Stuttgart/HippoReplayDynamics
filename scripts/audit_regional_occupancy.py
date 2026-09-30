"""Independent arithmetic, raw-spike and numerical-geometry audit."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_edge_support_content import load_npz
from scripts.build_regional_blind_bank import STRATA, read_csv, write_json


def close(actual, expected, tolerance=1e-10):
    if not np.isclose(actual, expected, atol=tolerance, rtol=tolerance, equal_nan=True):
        raise AssertionError((actual, expected))


def audit_task(task):
    _root, bank, row = task
    folder = Path(row["folder"])
    bank = Path(bank)/row["session"].replace("/", "_")
    manifest = json.loads((folder/"manifest.json").read_text())
    for name, expected in manifest["outputs"].items():
        if file_sha256(folder/name) != expected:
            raise AssertionError("changed task output")
    a = load_npz(folder/"readouts.npz")
    meta = read_csv(folder/"panels.csv")
    cal, val = a["calibration_indices"], a["validation_indices"]
    if len(cal) != 4 or len(val) != 16 or not meta.iloc[cal].phase.eq("calibration").all() or not meta.iloc[val].phase.eq("validation").all():
        raise AssertionError("wrong calibration/validation split")
    o = a["occupancy"]
    e = read_csv(folder/"emissions.csv")
    if len(e) != int(row["populations"])*4*10*7:
        raise AssertionError("missing emission rows")
    for r in e.itertuples():
        g = STRATA.index((r.generator, r.scale))
        if r.scope == "calibration_all":
            ix = cal
        elif r.scope == "validation_all":
            ix = val
        elif r.scope.startswith("calibration_leave_out_"):
            ix = np.delete(cal, int(r.scope.rsplit("_", 1)[1]))
        else:
            quota = float(r.scope.rsplit("_", 1)[1])
            ix = np.flatnonzero(meta.phase.eq("validation") & meta.prevalence.eq(quota))
        q, truth = a[f"pop{r.population_index}_{r.silence_policy}"][g, ix].ravel(), o[g, ix].ravel()
        wi, wo = (truth, 1-truth) if r.definition == "time_conditional" else (truth >= 1-1e-8, truth <= 1e-8)
        expected_s = sum(q*wi)/sum(wi) if sum(wi) else np.nan
        expected_f = sum(q*wo)/sum(wo) if sum(wo) else np.nan
        close(r.s, expected_s)
        close(r.f, expected_f)
        if r.definition == "time_conditional":
            close(q.mean(), truth.mean()*r.s+(1-truth.mean())*r.f)
    checks = read_csv(folder/"invariance.csv")
    for r in checks.itertuples():
        z = e[(e.population_index==r.population_index)&e.silence_policy.eq(r.silence_policy)&e.definition.eq(r.definition)&e.scope.eq(r.scope)]
        finite = len(z)==7 and np.isfinite(z[["s", "f"]].to_numpy()).all()
        sr, fr = (z.s.max()-z.s.min(), z.f.max()-z.f.min()) if finite else (np.nan, np.nan)
        close(r.s_range, sr)
        close(r.f_range, fr)
        assert bool(r.invariance_pass) == bool(finite and sr<=.02 and fr<=.02)
    v = read_csv(folder/"validation.csv")
    for r in v.itertuples():
        g = STRATA.index((r.generator, r.scale))
        ix = np.flatnonzero(meta.phase.eq("validation") & meta.prevalence.eq(r.binary_label_quota) & meta.replica.eq(r.replica))
        assert len(ix)==1
        q = a[f"pop{r.population_index}_{r.silence_policy}"]
        truth = float(o[g, ix].mean())
        mean_mass = float(q[g, ix].mean())
        close(r.mean_mass, mean_mass)
        close(r.true_occupancy, truth)
        cq, co = (q[:, cal].ravel(), o[:, cal].ravel()) if r.calibration=="pooled" else (q[g, cal].ravel(), o[g, cal].ravel())
        wi, wo = (co, 1-co) if r.definition=="time_conditional" else (co>=1-1e-8, co<=1e-8)
        s, f = (float(np.mean(cq*wi)/np.mean(wi)) if np.any(wi) else np.nan,
                float(np.mean(cq*wo)/np.mean(wo)) if np.any(wo) else np.nan)
        close(r.s, s)
        close(r.f, f)
        estimate = (mean_mass-f)/(s-f) if np.isfinite([s, f]).all() and s-f>1e-12 else np.nan
        close(r.estimate, estimate)
        close(r.absolute_error, abs(estimate-truth))
    data, templates = load_npz(bank/"source_inputs.npz"), load_npz(bank/"templates.npz")
    center = np.array(json.loads((bank/"config.json").read_text())["center"])
    raw_indices = {int(cell): i for i, cell in enumerate(data["cell_ids"])}
    ix = [raw_indices[int(cell)] for cell in data["pop0_ids"]]
    rates, region = data["pop0_rates"], data["pop0_region"]
    k = int(row["delta_ms"])//20
    maximum_numeric, maximum_mass, reconstructed = 0., 0., 0
    pidx = int(val[0])
    for g, (kind, scale) in enumerate(STRATA):
        p = load_npz(bank/f"{kind}_scale{scale:g}_delta{row['delta_ms']}"/f"panel_{int(meta.iloc[pidx].panel):03}.npz")
        for event in range(min(3, len(templates["event_ids"]))):
            lo, hi = p["path_offsets"][event:event+2]
            ages, nodes = p["path_ages"][lo:hi], p["path_nodes"][lo:hi]
            points = data["grid"][nodes]
            for b in range(k):
                start_age = (k-b-1)*.02
                samples = start_age+(np.arange(1000)+.5)*.00002
                segments = np.clip(np.searchsorted(ages, samples, side="right")-1, 0, len(ages)-2)
                pos = points[segments]
                if kind=="moving":
                    fraction = (samples-ages[segments])/(ages[segments+1]-ages[segments])
                    pos = pos+(points[segments+1]-pos)*fraction[:, None]
                numeric = (np.linalg.norm(pos-center, axis=1)<=20).mean()
                error = abs(numeric-o[g, pidx, event, b])
                maximum_numeric = max(maximum_numeric, float(error))
                if error > .002*len(np.unique(segments)):
                    raise AssertionError("numeric dwell mismatch")
            off0, off1 = templates["offsets"][event:event+2]
            times, ids = templates["times"][off0:off1], p["identities"][off0:off1]
            origin, endpoint = templates["starts"][event], templates["endpoints"][event]
            step = round((endpoint-origin)/.005)
            edges = origin+.005*(step-4*np.arange(k,-1,-1))
            edges[-1]=endpoint
            edges[-2]=data["windows"][templates["source_indices"][event],0]
            for b in range(k):
                counts = np.bincount(ids[(times>=edges[b])&(times<edges[b+1])], minlength=len(data["cell_ids"]))[ix]
                ll = counts@np.log(rates)-.02*rates.sum(axis=0)
                direct = float(np.exp(logsumexp(ll[region])-logsumexp(ll)))
                for policy in ("poisson_silence", "neutral_silence"):
                    expected = float(region.mean()) if policy=="neutral_silence" and counts.sum()==0 else direct
                    actual = a[f"pop0_{policy}"][g,pidx,event,b]
                    maximum_mass=max(maximum_mass, abs(expected-actual))
                    close(expected, actual)
                    reconstructed += 1
    return {"session": row["session"], "delta_ms": row["delta_ms"], "status": "pass",
            "emissions_checked": len(e), "invariance_rows_checked": len(checks), "validation_rows_checked": len(v),
            "raw_masses_checked": reconstructed, "max_numeric_occupancy_error": maximum_numeric,
            "max_raw_mass_error": maximum_mass}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root, out = Path(args.experiment), Path(args.output_dir)
    m = json.loads((root/"manifest.json").read_text())
    assert m["status"]=="complete_pending_independent_audit" and m["tasks"]==32
    for name, expected in m["outputs"].items():
        assert file_sha256(root/name)==expected
    tasks = read_csv(root/"tasks.csv")
    assert not tasks.duplicated(["session", "delta_ms"]).any() and len(tasks)==32 and tasks.session.nunique()==8
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(audit_task, (str(root), m["parameters"]["bank"], row)) for row in tasks.to_dict("records")]):
            row=future.result()
            rows.append(row)
            print("AUDIT", row["session"], row["delta_ms"], "pass", flush=True)
    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(out/"audit.csv", index=False)
    provenance=build_script_provenance(input_paths={"experiment": root/"manifest.json", "audit_code": __file__})
    provenance.update(status="technical_pass", outputs={"audit.csv":file_sha256(out/"audit.csv")})
    write_json(out/"manifest.json", provenance)


if __name__=="__main__":
    main()
