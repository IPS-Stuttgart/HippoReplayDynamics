"""Test a two-mean regional occupancy calibration before any real-data correction."""
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

from hipporeplayimm.regional_occupancy import (
    bin_masses,
    correct_occupancy,
    emission_means,
    invariance,
    window_occupancy,
)
from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_edge_support_content import load_npz
from scripts.build_regional_blind_bank import DELTAS, STRATA, read_csv, write_json
from scripts.test_regional_terminal_mixtures import load_panels

DEFINITIONS = ("time_conditional", "pure_window")
POLICIES = ("neutral_silence", "poisson_silence")


def load_occupancies(bank, session, duration, meta, data):
    parent = bank/session.replace("/", "_")
    center = json.loads((parent/"config.json").read_text())["center"]
    all_o = []
    maximum_error = 0.
    for kind, scale in STRATA:
        folder = parent/f"{kind}_scale{scale:g}_delta{duration}"
        panels = []
        for row in meta.itertuples():
            p = load_npz(folder/f"panel_{row.panel:03}.npz")
            result = []
            for a, b in zip(p["path_offsets"][:-1], p["path_offsets"][1:], strict=True):
                result.append(window_occupancy(data["grid"], p["path_ages"][a:b], p["path_nodes"][a:b], kind, center, duration))
            o = np.asarray(result)
            for delta in DELTAS:
                if delta <= duration/1000:
                    actual = o[:, -round(delta/.02):].mean(axis=1)
                    error = np.max(np.abs(actual-p["occupancy_fraction"][:, DELTAS.index(delta)]))
                    maximum_error = max(maximum_error, float(error))
                    if error > 1e-8:
                        raise AssertionError("bin dwell disagrees with frozen path truth")
            panels.append(o)
        all_o.append(panels)
    return np.asarray(all_o), maximum_error


def worker(task):
    bank, destination, session, duration = task
    bank = Path(bank)
    out = Path(destination)/session.replace("/", "_")/f"delta{duration}"
    out.mkdir(parents=True)
    data, templates, counts, _, _, _, meta, inputs = load_panels(bank, session, duration)
    occ, truth_error = load_occupancies(bank, session, duration, meta, data)
    cal, val = np.flatnonzero(meta.phase.eq("calibration")), np.flatnonzero(meta.phase.eq("validation"))
    definitions = json.loads((bank/session.replace("/", "_")/"population_definitions.json").read_text())
    lookup = {int(cell): i for i, cell in enumerate(data["cell_ids"])}
    arrays = {"occupancy": occ, "event_ids": templates["event_ids"], "calibration_indices": cal, "validation_indices": val}
    emissions, checks, validation, window_stats = [], [], [], []
    for j, definition in enumerate(definitions):
        ix = [lookup[int(cell)] for cell in data[f"pop{j}_ids"]]
        raw = counts[..., ix]
        masses = bin_masses(raw, data[f"pop{j}_rates"], data[f"pop{j}_region"])
        arrays[f"pop{j}_totals"] = raw.sum(axis=-1)
        base = {"session": session, "animal": session.split("/")[0], "population": definition["name"],
                "population_index": j, "family": definition["family"], "delta_ms": duration,
                "area_fraction": float(data[f"pop{j}_region"].mean()), "events_per_panel": len(templates["event_ids"])}
        for policy, q in masses.items():
            arrays[f"pop{j}_{policy}"] = q
            for definition_name in DEFINITIONS:
                model = base | {"silence_policy": policy, "definition": definition_name}
                pooled = emission_means(q[:, cal], occ[:, cal], definition_name)
                oracle = [emission_means(q[g, cal], occ[g, cal], definition_name) for g in range(7)]
                selections = [("calibration_all", cal), ("validation_all", val)]
                selections += [(f"calibration_leave_out_{i}", np.delete(cal, i)) for i in range(4)]
                selections += [(f"validation_binary_quota_{p:g}", np.flatnonzero(meta.phase.eq("validation") & meta.prevalence.eq(p))) for p in (.05, .15, .3, .5)]
                for scope, indices in selections:
                    values = []
                    for g, (kind, scale) in enumerate(STRATA):
                        stats = emission_means(q[g, indices], occ[g, indices], definition_name)
                        values.append(stats)
                        emissions.append(model | {"scope": scope, "generator": kind, "scale": scale, **stats})
                    checks.append(model | {"scope": scope, **invariance([x["s"] for x in values], [x["f"] for x in values])})
                for g, (kind, scale) in enumerate(STRATA):
                    for p in val:
                        qbar, truth = float(q[g, p].mean()), float(occ[g, p].mean())
                        row = meta.iloc[p]
                        for mode, fit in (("pooled", pooled), ("oracle_generator", oracle[g])):
                            estimate = correct_occupancy(qbar, fit["s"], fit["f"])
                            validation.append(model | {"generator": kind, "scale": scale, "calibration": mode,
                                "binary_label_quota": float(row.prevalence), "replica": int(row.replica),
                                "true_occupancy": truth, "mean_mass": qbar, "estimate": estimate,
                                "absolute_error": abs(estimate-truth), "signed_error": estimate-truth,
                                "naive_absolute_error": abs(qbar-truth), "s": fit["s"], "f": fit["f"],
                                "s_minus_f": fit["s_minus_f"], "low_occupancy_2_to_10_percent": .02 <= truth <= .1,
                                "out_of_range": bool(np.isfinite(estimate) and not 0 <= estimate <= 1),
                                "identified": bool(np.isfinite(estimate))})
            for g, (kind, scale) in enumerate(STRATA):
                o, mass, totals = occ[g, val].ravel(), q[g, val].ravel(), raw[g, val].sum(axis=-1).ravel()
                # Boundary snapping affects diagnostic bins only, not geometric truth or fits.
                bins = np.digitize(np.round(o, 8), [.00000001, .2, .5, .8, .99999999])
                for b in range(6):
                    mask = bins == b
                    window_stats.append(base | {"silence_policy": policy, "generator": kind, "scale": scale,
                        "occupancy_bin": b, "windows": int(mask.sum()),
                        "mean_true_occupancy": float(o[mask].mean()) if mask.any() else np.nan,
                        "mean_mass": float(mass[mask].mean()) if mask.any() else np.nan,
                        "mean_spikes": float(totals[mask].mean()) if mask.any() else np.nan})
    for name, rows in (("emissions.csv", emissions), ("invariance.csv", checks), ("validation.csv", validation), ("window_diagnostics.csv", window_stats)):
        pd.DataFrame(rows).to_csv(out/name, index=False)
    meta.to_csv(out/"panels.csv", index=False)
    np.savez_compressed(out/"readouts.npz", **arrays)
    m = build_script_provenance(input_paths=inputs)
    m.update(status="complete", session=session, delta_ms=duration, events=len(templates["event_ids"]), populations=len(definitions),
             maximum_geometric_truth_error=truth_error, null_panels_used=False,
             outputs={p.name: file_sha256(p) for p in out.iterdir()})
    write_json(out/"manifest.json", m)
    return {"session": session, "delta_ms": duration, "folder": str(out), "populations": len(definitions), "status": "complete"}


def report(out):
    checks, val = read_csv(out/"invariance.csv"), read_csv(out/"validation.csv")
    keys = ["population", "family", "delta_ms", "silence_policy", "definition"]
    summary = checks[checks.scope.eq("calibration_all")].groupby(keys, observed=True).agg(
        sessions=("session", "size"), animals=("animal", "nunique"), passing_sessions=("invariance_pass", "sum"),
        median_s_range=("s_range", "median"), max_s_range=("s_range", "max"),
        median_f_range=("f_range", "median"), max_f_range=("f_range", "max"),
        missing_support=("support_complete", lambda x: int((~x).sum()))).reset_index()
    summary["all_sessions_pass"] = summary.passing_sessions.eq(summary.sessions) & summary.missing_support.eq(0)
    summary.to_csv(out/"invariance_summary.csv", index=False)
    transfer = val.groupby([*keys, "calibration"], observed=True).agg(
        panels=("estimate", "size"), identified=("identified", "sum"), mean_error=("absolute_error", "mean"),
        maximum_error=("absolute_error", "max"), mean_naive_error=("naive_absolute_error", "mean"),
        out_of_range=("out_of_range", "sum"), mean_s_minus_f=("s_minus_f", "mean")).reset_index()
    transfer.to_csv(out/"transfer_summary.csv", index=False)
    low = val[val.low_occupancy_2_to_10_percent]
    low.groupby([*keys, "calibration"], observed=True).agg(
        panels=("estimate", "size"), identified=("identified", "sum"), mean_error=("absolute_error", "mean"),
        max_error=("absolute_error", "max"), minimum_truth=("true_occupancy", "min"),
        maximum_truth=("true_occupancy", "max")).reset_index().to_csv(out/"low_occupancy_summary.csv", index=False)
    primary = summary[summary.population.eq("full") & summary.silence_policy.eq("neutral_silence") & summary.definition.eq("time_conditional")]
    if len(primary) != 4 or not primary.sessions.eq(8).all():
        raise ValueError("incomplete full-population primary coverage")
    status = "invariance_failed_no_real_correction" if not primary.all_sessions_pass.any() else "candidate_requires_independent_accuracy_validation"
    write_json(out/"decision.json", {"status": status, "tolerance": .02, "real_correction_authorized": False,
        "perturbation_panel_run": False, "binary_result_comparison": "different_estimand_not_head_to_head",
        "claim_boundary": "tests_two_mean_calibration_not_all_occupancy_estimators"})
    text = ["# Regional occupancy invariance", "", "Known-truth synthetic diagnostic; no real content correction.", "",
            f"Decision: **{status}**.", "", "Primary: mean independent-bin Home mass with silent bins neutral;",
            "exact time-conditional s/f from fractional geometric dwell.", "",
            "| Delta | Sessions passing both 0.02 tolerances | Median s range | Median f range |",
            "| --- | ---: | ---: | ---: |"]
    for r in primary.sort_values("delta_ms").itertuples():
        text.append(f"| {r.delta_ms} ms | {r.passing_sessions}/{r.sessions} | {r.median_s_range:.4f} | {r.median_f_range:.4f} |")
    text += ["", "## Interpretation", "", "Linearity of occupancy does not make region-conditional emissions invariant.",
             "They also average over within-region position, spike support and within-bin motion.",
             "Time-conditional s/f reconstruct their own calibration mean by algebra; that is not validation.",
             "Even q=occupancy can have dynamics-dependent time-conditional s/f for mixed windows.",
             "Pure-window diagnostics and held-out transfer are provided to distinguish these issues.", "",
             "Transfer tables report unclipped corrections, oracle diagnostics and actual 2-10% occupancy",
             "panels. The bank's binary label quotas are not occupancy quotas. No new occupancy error",
             "budget or real-data remedy is certified. The previous 40-ms binary 5.68-pp error concerns",
             "a different estimand and is not a like-for-like baseline.", "",
             "The 11.38%/2.80% endpoint mass aggregates were not relabeled as terminal-time occupancy.",
             "Real-population reconciliation and perturbation/held-out-cell checks remain untested.", "",
             "Four calibration replicas per stratum; four validation replicas per binary quota.",
             "Session/rat replication, legacy population overlap, fixed-total allocation, selected",
             "decoder-derived anchors and truncated large jumps retain the shared-bank limitations.",
             "See the frozen protocol and independent audit before drawing conclusions.", ""]
    (out/"report.md").write_text("\n".join(text))
    figure(summary, transfer, out)
    print(primary.to_string(index=False), flush=True)


def figure(summary, transfer, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    selected = summary[summary.population.eq("full") & summary.silence_policy.eq("neutral_silence")]
    for definition, group in selected.groupby("definition"):
        group = group.sort_values("delta_ms")
        ax[0].plot(group.delta_ms, group.median_s_range, marker="o", label=definition)
        ax[1].plot(group.delta_ms, group.median_f_range, marker="o", label=definition)
    for a, name in zip(ax[:2], ["Home response s", "Non-Home response f"], strict=True):
        a.axhline(.02, color="black", linestyle="--")
        a.set(title=f"{name}: across-dynamics range", xlabel="Terminal segment (ms)", ylabel="Median session range")
    ax[0].legend(fontsize=8)
    t = transfer[transfer.population.eq("full") & transfer.silence_policy.eq("neutral_silence") & transfer.definition.eq("time_conditional")]
    for mode, group in t.groupby("calibration"):
        group = group.sort_values("delta_ms")
        ax[2].plot(group.delta_ms, 100*group.mean_error, marker="o", label=mode)
    group = t[t.calibration.eq("pooled")].sort_values("delta_ms")
    ax[2].plot(group.delta_ms, 100*group.mean_naive_error, marker="o", label="uncorrected mass")
    ax[2].set(title="Independent validation: occupancy error", xlabel="Terminal segment (ms)", ylabel="Mean absolute error (occupancy pp)")
    ax[2].legend(fontsize=8)
    fig.suptitle("Linear occupancy readout: synthetic PF bank, 8 sessions / 4 rats")
    fig.savefig(out/"occupancy_invariance.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", default="/mnt/seagate10tb/florianpfaff/regional-blind-shared-bank-pf-20260916")
    parser.add_argument("--bank-audit", default="/mnt/seagate10tb/florianpfaff/regional-blind-shared-bank-pf-audit-20260916")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    bank, out = Path(args.bank), Path(args.output_dir)
    audit = json.loads((Path(args.bank_audit)/"manifest.json").read_text())
    if audit["status"] != "technical_pass" or audit["input_file_sha256"]["bank_manifest"] != file_sha256(bank/"manifest.json"):
        raise ValueError("audited frozen bank required")
    original = json.loads((bank/"manifest.json").read_text())
    for name, path in original["input_file_paths"].items():
        if file_sha256(path) != original["input_file_sha256"][name]:
            raise ValueError("bank input changed")
    out.mkdir(parents=True, exist_ok=False)
    inputs = {"bank": bank/"manifest.json", "bank_audit": Path(args.bank_audit)/"manifest.json",
              "runner": __file__, "core": ROOT/"src/hipporeplayimm/regional_occupancy.py",
              "loader": ROOT/"scripts/test_regional_terminal_mixtures.py",
              "counting": ROOT/"src/hipporeplayimm/regional_terminal_mixture.py",
              "regional_bf": ROOT/"src/hipporeplayimm/regional_content_frontier.py",
              "protocol": ROOT/"docs/regional_occupancy_invariance_protocol.md"}
    m = build_script_provenance(input_paths=inputs)
    m.update(status="running", parameters=vars(args), null_panels_used=False, real_data_used=False)
    write_json(out/"manifest.json", m)
    sessions = read_csv(bank/"sessions.csv").session.tolist()
    tasks = [(str(bank), str(out), session, duration) for session in sessions for duration in (20, 40, 60, 100)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(worker, t) for t in tasks]):
            row = future.result()
            rows.append(row)
            print(len(rows), "/", len(tasks), row["session"], row["delta_ms"], flush=True)
    pd.DataFrame(rows).to_csv(out/"tasks.csv", index=False)
    for name in ("emissions.csv", "invariance.csv", "validation.csv", "window_diagnostics.csv"):
        pd.concat([read_csv(Path(row["folder"])/name) for row in rows], ignore_index=True).to_csv(out/name, index=False)
    report(out)
    m.update(status="complete_pending_independent_audit", tasks=len(rows), outputs={p.name: file_sha256(p) for p in out.iterdir() if p.is_file() and p.name != "manifest.json"})
    write_json(out/"manifest.json", m)


if __name__ == "__main__":
    main()
