"""Capped grid-level marginal-content inference on the frozen synthetic bank."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd

from hipporeplayimm.regional_grid_mixture import fit_grid, grid_likelihood, predictive_score, smoothing_matrix
from hipporeplayimm.regional_terminal_mixture import count_subwindows
from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_edge_support_content import load_npz
from scripts.build_regional_blind_bank import STRATA, read_csv, write_json

PENALTIES = (0.0, 0.001, 0.01, 0.1)
OBSERVATIONS = ("conditional_multinomial", "poisson", "fixed_total_censored")


def prepare(task):
    bank, destination, session = task
    bank, out = Path(bank), Path(destination) / session.replace("/", "_")
    parent = bank / session.replace("/", "_")
    source, template = load_npz(parent / "source_inputs.npz"), load_npz(parent / "templates.npz")
    config = json.loads((parent / "config.json").read_text())
    original = load_npz(Path(config["source_bank"]) / "frozen_inputs.npz")
    if source.keys() != original.keys():
        raise ValueError("source arrays changed")
    for key in source:
        np.testing.assert_array_equal(source[key], original[key])
    old_manifest = json.loads((Path(config["source_bank"]) / "manifest.json").read_text())
    reconstruction_inputs = {"source_manifest": Path(config["source_bank"]) / "manifest.json"}
    for key in ("encoder", "matched_archive"):
        path = Path(old_manifest["input_file_paths"][key])
        if file_sha256(path) != old_manifest["input_file_sha256"][key]:
            raise ValueError("native grid reconstruction input changed")
        reconstruction_inputs[key] = path
    encoder = load_npz(reconstruction_inputs["encoder"])
    old = load_npz(reconstruction_inputs["matched_archive"])
    definitions = json.loads((parent / "population_definitions.json").read_text())
    coordinate_lookup = {tuple(x): i for i, x in enumerate(encoder["bin_centers_cm"])}
    cell_lookup = {int(x): i for i, x in enumerate(encoder["cell_ids"])}
    coordinates, alignment = {}, []
    for j, definition in enumerate(definitions):
        grid = source["grid"] if definition["name"] == "full" else old["grid_cm"]
        gi = [coordinate_lookup[tuple(x)] for x in grid]
        ci = [cell_lookup[int(x)] for x in source[f"pop{j}_ids"]]
        full_rates = np.maximum(encoder["rates_hz"][:, gi], 1e-4)
        np.testing.assert_array_equal(full_rates[ci], source[f"pop{j}_rates"])
        np.testing.assert_array_equal(np.linalg.norm(grid - np.asarray(config["center"]), axis=1) <= 20, source[f"pop{j}_region"])
        coordinates[f"grid{j}"] = grid
        coordinates[f"full_rate_sum{j}"] = full_rates.sum(axis=0)
        overlap = len(set(map(tuple, grid)) & set(map(tuple, source["grid"])))
        alignment.append({"population": definition["name"], "decoder_bins": len(grid), "generator_bins": len(source["grid"]), "shared_coordinates": overlap})
    out.mkdir(parents=True)
    np.savez_compressed(out / "encoding_coordinates.npz", **coordinates)
    pd.DataFrame(alignment).to_csv(out / "grid_alignment.csv", index=False)
    counts, truths, metas, support = [], [], [], []
    inputs = {"source": parent / "source_inputs.npz", "templates": parent / "templates.npz", "definitions": parent / "population_definitions.json"}
    inputs.update(reconstruction_inputs)
    for kind, scale in STRATA:
        folder = parent / f"{kind}_scale{scale:g}_delta40"
        m = json.loads((folder / "manifest.json").read_text())
        inputs[folder.name] = folder / "manifest.json"
        meta = read_csv(folder / "panels.csv")
        meta = meta[meta.phase.isin(["calibration", "validation"]) & meta.replica.eq(0)].reset_index(drop=True)
        if len(meta) != 5 or meta.phase.eq("calibration").sum() != 1:
            raise ValueError("capped panel selection changed")
        gc, gt = [], []
        for row in meta.itertuples():
            path = folder / f"panel_{row.panel:03}.npz"
            if file_sha256(path) != m["sha256"][path.name]:
                raise ValueError("bank panel changed")
            p = load_npz(path)
            gc.append(count_subwindows(p["identities"], template, len(source["cell_ids"]), 40, source["windows"][template["source_indices"], 0]))
            gt.append(p["occupancy_fraction"][:, 1])
            support.append(
                {
                    "session": session,
                    "generator": kind,
                    "scale": scale,
                    "phase": row.phase,
                    "binary_quota": row.prevalence,
                    "spike_rejections": int(p["spike_rejections"].sum()),
                    "accepted_events": len(template["event_ids"]),
                    "attempts": int(p["attempts"].sum()),
                }
            )
        counts.append(gc)
        truths.append(gt)
        metas.append(meta)
    for meta in metas[1:]:
        pd.testing.assert_frame_equal(meta[["phase", "prevalence", "replica"]], metas[0][["phase", "prevalence", "replica"]])
    rng = np.random.default_rng(2026091701)
    folds = np.empty(len(template["event_ids"]), int)
    folds[rng.permutation(len(folds))] = np.arange(len(folds)) % 3
    np.savez_compressed(out / "inputs.npz", counts=counts, truth=truths, event_ids=template["event_ids"], folds=folds)
    metas[0].to_csv(out / "panels.csv", index=False)
    pd.DataFrame(support).to_csv(out / "selection_support.csv", index=False)
    defs = json.loads((parent / "population_definitions.json").read_text())
    write_json(out / "populations.json", defs)
    manifest = build_script_provenance(input_paths=inputs)
    manifest.update(status="prepared", session=session, outputs={p.name: file_sha256(p) for p in out.iterdir()})
    write_json(out / "manifest.json", manifest)
    return {"session": session, "folder": str(out), "source": str(parent / "source_inputs.npz"), "populations": len(defs)}


def population_input(task):
    parent, population = task
    folder = Path(parent["folder"])
    a, data = load_npz(folder / "inputs.npz"), load_npz(parent["source"])
    definition = json.loads((folder / "populations.json").read_text())[population]
    lookup = {int(cell): j for j, cell in enumerate(data["cell_ids"])}
    ix = [lookup[int(cell)] for cell in data[f"pop{population}_ids"]]
    counts = a["counts"][..., ix]
    rate = data[f"pop{population}_rates"]
    coordinates = load_npz(folder / "encoding_coordinates.npz")
    data["inference_grid"] = coordinates[f"grid{population}"]
    data["full_rate_sum_on_grid"] = coordinates[f"full_rate_sum{population}"]
    base = {
        "session": parent["session"],
        "animal": parent["session"].split("/")[0],
        "population": definition["name"],
        "population_index": population,
        "family": definition["family"],
        "source": parent["source"],
        "folder": str(folder),
        "area_fraction": float(data[f"pop{population}_region"].mean()),
    }
    return a, data, counts, rate, base, read_csv(folder / "panels.csv")


def cv_worker(task):
    a, data, counts, rates, base, meta = population_input(task)
    out = Path(base["folder"]) / f"pop{base['population_index']}"
    out.mkdir()
    c = int(np.flatnonzero(meta.phase.eq("calibration"))[0])
    training = counts[:, c].reshape(-1, len(rates))
    ll = grid_likelihood(training, rates, "conditional_multinomial")
    folds = np.broadcast_to(a["folds"][None, :, None], counts[:, c].shape[:-1]).ravel()
    smooth = smoothing_matrix(data["inference_grid"])
    rows, weights = [], []
    for penalty in PENALTIES:
        for fold in range(3):
            started = time.monotonic()
            fit = fit_grid(ll[folds != fold], smooth, penalty)
            weights.append(fit["pi"])
            rows.append(
                base
                | {
                    "penalty": penalty,
                    "fold": fold,
                    "heldout_windows": int((folds == fold).sum()),
                    "score": predictive_score(ll[folds == fold], fit["pi"]),
                    "dual_gap": fit["dual_gap"],
                    "converged": fit["converged"],
                    "iterations": fit["iterations"],
                    "runtime_s": time.monotonic() - started,
                }
            )
    pd.DataFrame(rows).to_csv(out / "cv.csv", index=False)
    np.savez_compressed(out / "cv_weights.npz", pi=weights)
    return str(out)


def choose_penalty(cv):
    if len(cv) != 22 * 4 * 3 or cv.duplicated(["session", "population", "penalty", "fold"]).any() or not cv.converged.all() or not np.isfinite(cv.score).all():
        raise ValueError("incomplete/nonconverged penalty-selection fits")
    per_session = cv.groupby(["penalty", "animal", "session"]).score.mean()
    per_rat = per_session.groupby(["penalty", "animal"]).mean()
    scores = per_rat.groupby("penalty").mean()
    best = scores.max()
    return float(scores[scores >= best - 1e-6].index.max()), scores


def evaluate_worker(task):
    job, penalty = task
    a, data, counts, rates, base, meta = population_input(job)
    out = Path(base["folder"]) / f"pop{base['population_index']}"
    smooth = smoothing_matrix(data["inference_grid"])
    region = data[f"pop{base['population_index']}_region"]
    rows, weights, first_weights = [], [], []
    for obs in OBSERVATIONS:
        shape = counts.shape[:-1]
        ll = grid_likelihood(counts.reshape(-1, len(rates)), rates, obs, a["counts"].sum(axis=-1).ravel(), data["full_rate_sum_on_grid"]).reshape(*shape, -1)
        for p in np.flatnonzero(meta.phase.eq("validation")):
            for g in range(8):
                selected = ll[g, p] if g < 7 else ll[:, p]
                truth = a["truth"][g, p] if g < 7 else a["truth"][:, p]
                observation = selected.reshape(-1, selected.shape[-1])
                kind, scale = STRATA[g] if g < 7 else ("equal_seven", np.nan)
                fitted = {}
                for fit_kind, lam in (("unpenalized", 0.0), ("selected_penalty", penalty)):
                    started = time.monotonic()
                    if lam not in fitted:
                        fitted[lam] = fit_grid(observation, smooth, lam)
                    fit = fitted[lam]
                    fit_id = len(rows)
                    weights.append(fit["pi"])
                    first_weights.append(fit["first"])
                    estimate = float(fit["pi"][region].sum())
                    first = float(fit["first"][region].sum())
                    rows.append(
                        base
                        | {
                            "fit_id": fit_id,
                            "observation": obs,
                            "fit_kind": fit_kind,
                            "penalty": lam,
                            "generator": kind,
                            "scale": scale,
                            "binary_label_quota": float(meta.iloc[p].prevalence),
                            "replica": 0,
                            "true_occupancy": float(truth.mean()),
                            "first_em_mass": first,
                            "em20_mass": float(fit["em20"][region].sum()),
                            "estimate": estimate,
                            "absolute_error": abs(estimate - float(truth.mean())),
                            "first_absolute_error": abs(first - float(truth.mean())),
                            "converged": fit["converged"],
                            "dual_gap": fit["dual_gap"],
                            "objective": fit["objective"],
                            "em_fixed_point_l1": fit["em_fixed_point_l1"],
                            "flat_likelihood": fit["flat_likelihood"],
                            "iterations": fit["iterations"],
                            "active_grid_bins": int((fit["pi"] > 1e-8).sum()),
                            "runtime_s": time.monotonic() - started,
                        }
                    )
    pd.DataFrame(rows).to_csv(out / "estimates.csv", index=False)
    np.savez_compressed(out / "fit_weights.npz", pi=weights, first=first_weights)
    write_json(out / "manifest.json", {"status": "complete", "outputs": {p.name: file_sha256(p) for p in out.iterdir()}})
    return str(out)


def summarize(out):
    d = read_csv(out / "estimates.csv")
    keys = ["observation", "fit_kind", "population", "family"]
    summary = (
        d.groupby(keys)
        .agg(
            fits=("estimate", "size"),
            converged=("converged", "sum"),
            mean_error=("absolute_error", "mean"),
            maximum_error=("absolute_error", "max"),
            mean_first_error=("first_absolute_error", "mean"),
            flat_fits=("flat_likelihood", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(out / "error_summary.csv", index=False)
    rat = d.groupby([*keys, "animal", "session"])[["absolute_error", "first_absolute_error"]].mean().groupby([*keys, "animal"]).mean().reset_index()
    rat.to_csv(out / "by_animal.csv", index=False)
    pairs = []
    pair_keys = ["session", "animal", "family", "observation", "fit_kind", "generator", "scale", "binary_label_quota"]
    for key, group in d[~d.population.eq("full")].groupby(pair_keys, dropna=False):
        if len(group) != 2:
            raise ValueError("missing paired population")
        hi = group[group.population.str.endswith("_high")].iloc[0]
        lo = group[group.population.str.endswith("_low")].iloc[0]
        if abs(hi.true_occupancy - lo.true_occupancy) > 1e-12:
            raise ValueError("paired populations have different truth")
        pairs.append(
            dict(zip(pair_keys, key, strict=True))
            | {
                "truth": hi.true_occupancy,
                "first_gap": hi.first_em_mass - lo.first_em_mass,
                "fitted_gap": hi.estimate - lo.estimate,
                "absolute_first_gap": abs(hi.first_em_mass - lo.first_em_mass),
                "absolute_fitted_gap": abs(hi.estimate - lo.estimate),
                "both_converged": bool(hi.converged and lo.converged),
                "both_within_5pp": bool(hi.absolute_error <= 0.05 and lo.absolute_error <= 0.05),
            }
        )
    pair = pd.DataFrame(pairs)
    pair.to_csv(out / "paired_population_gaps.csv", index=False)
    pair.groupby(["family", "observation", "fit_kind"]).agg(
        pairs=("fitted_gap", "size"),
        mean_first_gap=("absolute_first_gap", "mean"),
        mean_fitted_gap=("absolute_fitted_gap", "mean"),
        maximum_fitted_gap=("absolute_fitted_gap", "max"),
        both_converged=("both_converged", "sum"),
        both_within_5pp=("both_within_5pp", "sum"),
    ).reset_index().to_csv(out / "gap_summary.csv", index=False)
    primary = d[d.observation.eq("conditional_multinomial") & d.fit_kind.eq("selected_penalty")]
    paired = pair[pair.observation.eq("conditional_multinomial") & pair.fit_kind.eq("selected_penalty")]
    technical = len(d) == 4224 and d.converged.all() and not d.flat_likelihood.any()
    truth_pass = bool(primary.absolute_error.max() <= 0.05)
    gaps_pass = bool(len(paired) == 224 and paired.absolute_fitted_gap.max() <= 0.05)
    decision = "numerical_failure" if not technical else ("requires_broader_synthetic_validation" if truth_pass and gaps_pass else "synthetic_screen_failed_no_real_reconciliation")
    write_json(
        out / "decision.json",
        {
            "status": decision,
            "primary_truth_budget_pass": truth_pass,
            "primary_gap_budget_pass": gaps_pass,
            "real_data_authorized": False,
            "replicas_used": 1,
            "unknown_mixture_guarantee": False,
        },
    )
    gates = [
        {"gate": "all_4224_fits_valid", "pass": bool(technical)},
        {"gate": "primary_truth_error_within_5pp", "pass": truth_pass},
        {"gate": "primary_same_event_gap_within_5pp", "pass": gaps_pass},
    ]
    pd.DataFrame(gates).to_csv(out / "gate_summary.csv", index=False)
    lines = [
        "# Grid-level mixture pilot",
        "",
        f"Decision: **{decision}**.",
        "",
        "40-ms terminal occupancy, independent 20-ms windows; seven pure strata plus their equal mixture.",
        "Eight sessions/four rats, 22 frozen populations, validation replica 0 only.",
        "Grid weights are inferred before regional aggregation. The first EM iterate is the mean flat-prior posterior.",
        "One shared smoothing penalty was selected by event-wise CV, without Home truth.",
        "",
        "## Limits",
        "",
        "This is a capped synthetic falsification screen, not an all-mixtures bound or a bias confidence interval.",
        "Conditional multinomial removes position-dependent subset-count information; Poisson is not the bank's fixed-total generator.",
        "The censored-total diagnostic uses extra whole-population count/rate information and is not an isolated-subset decoder.",
        "Native active-cell acceptance and within-bin moving/jumping remain model-mismatch sources.",
        "Numerical convergence does not prove regional identification; a failure does not rule out every encoding-based estimator.",
        "No real population correction, perturbation panel, own-population detection run or real-data bootstrap was performed.",
        "See source protocol, per-fit dual gaps and the independent audit before interpretation.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines))
    print(summary[summary.population.eq("full")].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", default="/mnt/seagate10tb/florianpfaff/regional-blind-shared-bank-pf-20260916")
    parser.add_argument("--bank-audit", default="/mnt/seagate10tb/florianpfaff/regional-blind-shared-bank-pf-audit-20260916")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    bank, out = Path(args.bank), Path(args.output_dir)
    checked = json.loads((Path(args.bank_audit) / "manifest.json").read_text())
    if checked["status"] != "technical_pass" or checked["input_file_sha256"]["bank_manifest"] != file_sha256(bank / "manifest.json"):
        raise ValueError("audited unchanged bank required")
    out.mkdir(parents=True, exist_ok=False)
    source = {
        "bank": bank / "manifest.json",
        "bank_audit": Path(args.bank_audit) / "manifest.json",
        "runner": __file__,
        "core": ROOT / "src/hipporeplayimm/regional_grid_mixture.py",
        "counting": ROOT / "src/hipporeplayimm/regional_terminal_mixture.py",
        "protocol": ROOT / "docs/regional_grid_mixture_protocol.md",
    }
    m = build_script_provenance(input_paths=source)
    m.update(status="running", parameters=vars(args), no_real_data=True)
    write_json(out / "manifest.json", m)
    sessions = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(prepare, (str(bank), str(out), s)) for s in read_csv(bank / "sessions.csv").session]):
            row = future.result()
            sessions.append(row)
            print("PREPARED", row["session"], flush=True)
    pd.DataFrame(sessions).to_csv(out / "sessions.csv", index=False)
    jobs = [(parent, j) for parent in sessions for j in range(parent["populations"])]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(cv_worker, job) for job in jobs]):
            print("CV", future.result(), flush=True)
    folders = [Path(parent["folder"]) / f"pop{j}" for parent, j in jobs]
    cv = pd.concat([read_csv(f / "cv.csv") for f in folders], ignore_index=True)
    cv.to_csv(out / "cross_validation.csv", index=False)
    penalty, scores = choose_penalty(cv)
    scores.to_csv(out / "penalty_scores.csv")
    write_json(out / "frozen_penalty.json", {"penalty": penalty, "scores": scores.to_dict(), "truth_used": False})
    print("FROZEN PENALTY", penalty, flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(evaluate_worker, (job, penalty)) for job in jobs]):
            print("VALIDATED", future.result(), flush=True)
    pd.concat([read_csv(f / "estimates.csv") for f in folders], ignore_index=True).to_csv(out / "estimates.csv", index=False)
    pd.concat([read_csv(Path(s["folder"]) / "selection_support.csv") for s in sessions], ignore_index=True).to_csv(out / "selection_support.csv", index=False)
    summarize(out)
    m.update(status="complete_pending_audit", outputs={p.name: file_sha256(p) for p in out.iterdir() if p.is_file() and p.name != "manifest.json"})
    write_json(out / "manifest.json", m)


if __name__ == "__main__":
    main()
