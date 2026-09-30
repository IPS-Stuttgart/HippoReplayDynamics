"""Frozen terminal-segment readout calibration on a known-truth shared bank."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd

from hipporeplayimm.regional_content_frontier import discrimination
from hipporeplayimm.regional_readout_endpoint import fit_prevalence
from hipporeplayimm.regional_terminal_mixture import (
    Calibration,
    count_subwindows,
    empirical_envelope,
    regional_mass,
    represented_scores,
    segment_features,
    simplex_grid,
    weighted_mixture_estimates,
)
from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_edge_support_content import load_npz
from scripts.build_regional_blind_bank import DELTAS, STRATA, read_csv, write_json

READOUTS = ("independent_bin_any_home", "pooled_counts")
REPRESENTATIONS = ("continuous_zero_mass", "ternary")


def panel_fit(f, truth):
    estimate = fit_prevalence(f)
    actual = float(np.mean(truth))
    error = abs(estimate-actual)
    return {"estimate": estimate, "truth": actual, "absolute_error": error,
            "within_5pp": bool(np.isfinite(error) and error <= .05),
            "status": "fit" if np.isfinite(estimate) else "unidentified"}


def load_panels(bank, session, duration):
    parent = bank/session.replace("/", "_")
    source, templates = load_npz(parent/"source_inputs.npz"), load_npz(parent/"templates.npz")
    config = json.loads((parent/"config.json").read_text())
    original = load_npz(Path(config["source_bank"])/"frozen_inputs.npz")
    if source.keys() != original.keys():
        raise ValueError("source arrays differ")
    for key in original:
        np.testing.assert_array_equal(source[key], original[key])
    counts, labels, occupancy, late, metadata, inputs = [], [], [], [], [], {}
    for kind, scale in STRATA:
        folder = parent/f"{kind}_scale{scale:g}_delta{duration}"
        manifest = json.loads((folder/"manifest.json").read_text())
        inputs[folder.name+":manifest"] = folder/"manifest.json"
        meta = read_csv(folder/"panels.csv")
        meta = meta.loc[meta.phase.isin(["calibration", "validation"])].reset_index(drop=True)
        if len(meta) != 20 or meta.phase.eq("calibration").sum() != 4:
            raise ValueError("expected frozen 4-calibration/16-validation panel layout")
        gc, gz, go, gl = [], [], [], []
        for row in meta.itertuples():
            path = folder/f"panel_{row.panel:03}.npz"
            if file_sha256(path) != manifest["sha256"][path.name]:
                raise ValueError("bank panel changed")
            p = load_npz(path)
            windows = count_subwindows(p["identities"], templates, len(source["cell_ids"]), duration,
                source["windows"][templates["source_indices"], 0])
            ix = DELTAS.index(duration/1000.)
            gc.append(windows)
            gz.append(p["label"][:, ix])
            go.append(p["occupancy_fraction"][:, ix])
            gl.append(p["late_crossing"][:, ix])
        counts.append(gc)
        labels.append(gz)
        occupancy.append(go)
        late.append(gl)
        metadata.append(meta)
    for meta in metadata[1:]:
        pd.testing.assert_frame_equal(meta[["phase", "prevalence", "replica"]], metadata[0][["phase", "prevalence", "replica"]])
    labels = np.asarray(labels)
    np.testing.assert_array_equal(labels.sum(axis=2), np.broadcast_to(labels[0].sum(axis=1), labels.shape[:2]))
    return source, templates, np.asarray(counts), labels, np.asarray(occupancy), np.asarray(late), metadata[0], inputs


def worker(task):
    bank, destination, session, duration = task
    bank, out = Path(bank), Path(destination)/session.replace("/", "_")/f"delta{duration}"
    out.mkdir(parents=True)
    data, templates, counts, labels, occupancy, late, meta, inputs = load_panels(bank, session, duration)
    n = counts.shape[2]
    val, cal = np.flatnonzero(meta.phase.eq("validation")), np.flatnonzero(meta.phase.eq("calibration"))
    truth = labels[:, val]
    population_definitions = json.loads((bank/session.replace("/", "_")/"population_definitions.json").read_text())
    lookup = {int(cell): i for i, cell in enumerate(data["cell_ids"])}
    fits, envelopes, mixtures, strata_rows, support = [], [], [], [], []
    arrays = {"truth": labels, "occupancy": occupancy, "late_crossing": late,
              "validation_indices": val, "calibration_indices": cal, "event_ids": templates["event_ids"]}
    for j, definition in enumerate(population_definitions):
        name = definition["name"]
        ix = [lookup[int(c)] for c in data[f"pop{j}_ids"]]
        raw = counts[..., ix]
        features = segment_features(raw.reshape(-1, duration//20, len(ix)), data[f"pop{j}_rates"], data[f"pop{j}_region"])
        features = {key: value.reshape(counts.shape[:3]) for key, value in features.items()}
        totals = features["totals"]
        base = {"session": session, "animal": session.split("/")[0], "population": name,
                "family": definition["family"], "delta_ms": duration, "legacy_eligibility": definition["legacy_eligibility"],
                "shared_cells": definition["shared_cells"]}
        arrays[f"pop{j}_totals"] = totals
        arrays[f"pop{j}_silent_bin_fraction"] = features["silent_bin_fraction"]
        support.append(base | {"events": n, "mean_segment_spikes": float(totals[:, val].mean()),
                               "silent_segment_fraction": float((totals[:, val] == 0).mean()),
                               "silent_bin_fraction": float(features["silent_bin_fraction"][:, val].mean())})
        for readout in READOUTS:
            scores = features[readout]
            arrays[f"pop{j}_{readout}_score"] = scores
            for representation in REPRESENTATIONS:
                model = base | {"readout": readout, "representation": representation}
                x = represented_scores(scores, totals, representation)
                calibration = Calibration(x[:, cal].ravel(), labels[:, cal].ravel(), representation)
                f = calibration.likelihoods(x[:, val])
                arrays[f"pop{j}_{readout}_{representation}_likelihoods"] = f
                for g, (kind, scale) in enumerate(STRATA):
                    for v, panel in enumerate(val):
                        row = meta.iloc[panel]
                        fits.append(model | {"calibration_mode": "pooled", "generator": kind, "scale": scale,
                            "prevalence": float(row.prevalence), "replica": int(row.replica), "events": n,
                            **panel_fit(f[g, v], truth[g, v])})
                for v, panel in enumerate(val):
                    row = meta.iloc[panel]
                    result = empirical_envelope(f[:, v], float(truth[0, v].mean()))
                    pure_estimates = result.pop("pure_estimates")
                    envelope = model | {"prevalence": float(row.prevalence), "truth": float(truth[0, v].mean()),
                        "replica": int(row.replica), "events_per_stratum": n, **result}
                    envelopes.append(envelope)
                    if j == 0 and readout == READOUTS[0] and representation == REPRESENTATIONS[0] and row.replica == 0:
                        balanced = np.zeros(7)
                        balanced[[0, 2, 5]] = 1/3
                        weights = np.vstack([simplex_grid(), np.ones(7)/7, balanced])
                        estimates = weighted_mixture_estimates(f[:, v], weights)
                        finite = estimates[np.isfinite(estimates)]
                        assert len(finite) and finite.min() >= result["lower_estimate"]-1e-7
                        assert finite.max() <= result["upper_estimate"]+1e-7
                        if np.isfinite(pure_estimates).all():
                            assert abs(np.max(abs(finite-truth[0, v].mean()))-result["worst_absolute_error"]) < 1e-7
                        for k, (w, estimate) in enumerate(zip(weights, estimates, strict=True)):
                            mixtures.append(base | {"prevalence": float(row.prevalence), "truth": float(truth[0, v].mean()),
                                "replica": 0, "mixture": k, "weights": json.dumps(w.tolist()), "estimate": float(estimate),
                                "absolute_error": abs(float(estimate)-float(truth[0, v].mean()))})
                if representation == REPRESENTATIONS[0]:
                    for g, (kind, scale) in enumerate(STRATA):
                        oracle = Calibration(x[g, cal].ravel(), labels[g, cal].ravel(), representation)
                        of = oracle.likelihoods(x[g, val])
                        for v, panel in enumerate(val):
                            row = meta.iloc[panel]
                            fits.append(model | {"calibration_mode": "oracle_generator", "generator": kind, "scale": scale,
                                "prevalence": float(row.prevalence), "replica": int(row.replica), "events": n,
                                **panel_fit(of[v], truth[g, v])})
                        zs, os, ls = truth[g].ravel(), occupancy[g, val].ravel(), late[g, val].ravel()
                        score = scores[g, val].ravel()
                        ratio = (np.log(f[g, ..., 1])-np.log(f[g, ..., 0])).ravel()
                        calls = represented_scores(score, totals[g, val].ravel(), "ternary")
                        mass = regional_mass(score, float(data[f"pop{j}_region"].mean()), duration//20, readout)
                        bins = np.digitize(os, [.2, .5, .8])
                        masks = [("all", np.ones(len(zs), bool))] + [(f"occupancy_bin{b}", bins == b) for b in range(4)]
                        masks += [("late_crossing", ls), ("not_late_crossing", ~ls)]
                        for stratum, mask in masks:
                            occupied = bool(mask.any())
                            auc = discrimination(ratio[mask], zs[mask])["auc"] if occupied else np.nan
                            strata_rows.append(model | {"generator": kind, "scale": scale, "stratum": stratum,
                                "n": int(mask.sum()), "positive": int(zs[mask].sum()), "negative": int((~zs[mask]).sum()),
                                "mean_occupancy": float(os[mask].mean()) if occupied else np.nan,
                                "positive_readout_fraction": float((calls[mask] == 2).mean()) if occupied else np.nan,
                                "mean_raw_home_mass": float(mass[mask].mean()) if occupied else np.nan,
                                "calibrated_lr_auc": auc, "status": "available" if occupied else "empty"})
    tables = {"pure_validation.csv": fits, "mixture_envelopes.csv": envelopes, "mixture_grid_checks.csv": mixtures,
              "occupancy_and_crossing_readouts.csv": strata_rows, "spike_support.csv": support}
    for filename, rows in tables.items():
        pd.DataFrame(rows).to_csv(out/filename, index=False)
    meta.to_csv(out/"panels_used.csv", index=False)
    np.savez_compressed(out/"readouts_and_likelihoods.npz", **arrays)
    provenance = build_script_provenance(input_paths=inputs)
    provenance.update(status="complete", session=session, delta_ms=duration, populations=len(population_definitions),
                      events=n, null_panels_used=False, outputs={p.name: file_sha256(p) for p in out.iterdir()})
    write_json(out/"manifest.json", provenance)
    return {"session": session, "delta_ms": duration, "folder": str(out), "events": n,
            "populations": len(population_definitions), "status": "complete"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", default="/mnt/seagate10tb/florianpfaff/regional-blind-shared-bank-pf-20260916")
    parser.add_argument("--bank-audit", default="/mnt/seagate10tb/florianpfaff/regional-blind-shared-bank-pf-audit-20260916")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--session-limit", type=int)
    args = parser.parse_args()
    bank, audit, out = Path(args.bank), Path(args.bank_audit), Path(args.output_dir)
    quality = json.loads((audit/"manifest.json").read_text())
    if quality["status"] != "technical_pass" or quality["input_file_sha256"]["bank_manifest"] != file_sha256(bank/"manifest.json"):
        raise ValueError("bank is not the independently audited passing version")
    manifest = json.loads((bank/"manifest.json").read_text())
    for key, path in manifest["input_file_paths"].items():
        if file_sha256(path) != manifest["input_file_sha256"][key]:
            raise ValueError(f"changed bank input: {key}")
    out.mkdir(parents=True, exist_ok=False)
    inputs = {"bank": bank/"manifest.json", "bank_audit": audit/"manifest.json", "runner": Path(__file__),
              "core": ROOT/"src/hipporeplayimm/regional_terminal_mixture.py",
              "regional_bf": ROOT/"src/hipporeplayimm/regional_content_frontier.py",
              "density_and_fit": ROOT/"src/hipporeplayimm/regional_readout_endpoint.py",
              "protocol": ROOT/"docs/regional_terminal_mixture_protocol.md"}
    provenance = build_script_provenance(input_paths=inputs)
    provenance.update(status="running", parameters=vars(args), created_at_utc=datetime.now(UTC).isoformat(),
                      calibration_weights=[1/7]*7, inferred_generator_used=False, null_panels_used=False)
    write_json(out/"manifest.json", provenance)
    sessions = read_csv(bank/"sessions.csv").session.tolist()
    if args.session_limit:
        sessions = sessions[:args.session_limit]
    tasks = [(str(bank), str(out), s, d) for s in sessions for d in (20, 40, 60, 100)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(worker, t) for t in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
            print(len(rows), "/", len(tasks), rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out/"tasks.csv", index=False)
    for filename in ("pure_validation.csv", "mixture_envelopes.csv", "mixture_grid_checks.csv", "occupancy_and_crossing_readouts.csv", "spike_support.csv"):
        frames = [read_csv(Path(r["folder"])/filename) for r in rows]
        pd.concat(frames, ignore_index=True).to_csv(out/filename, index=False)
    provenance.update(status="complete_pending_report_audit", tasks=len(rows), finished_at_utc=datetime.now(UTC).isoformat(),
                      outputs={p.name: file_sha256(p) for p in out.iterdir() if p.is_file() and p.name != "manifest.json"})
    write_json(out/"manifest.json", provenance)


if __name__ == "__main__":
    main()
