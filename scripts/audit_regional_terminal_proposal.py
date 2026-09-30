"""Non-rescoring feasibility and generator-stratified population-gap audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]

import numpy as np
import pandas as pd
from scipy.special import expit

from scripts._provenance import build_script_provenance, file_sha256
from scripts.diagnose_regional_readout_endpoint import load_npz, read_csv, validation_mask
from scripts.measure_edge_support_content import occupied_graph

LENGTHS = (20, 40, 60, 100)


def anywhere_label(origin_in_home, endpoint_in_home, duration_ms, jump_before_end_ms=5.):
    if duration_ms <= 0 or jump_before_end_ms <= 0:
        raise ValueError("positive window and jump time required")
    endpoint = np.asarray(endpoint_in_home, bool)
    origin = np.asarray(origin_in_home, bool)
    if origin.shape != endpoint.shape:
        raise ValueError("label shape mismatch")
    return endpoint | (origin & (duration_ms > jump_before_end_ms))


def conditional_identity_probability(rates, gain, nonspatial):
    if gain <= 0 or np.any(np.asarray(nonspatial) < 0):
        raise ValueError("positive gain and nonnegative component required")
    intensity = gain*np.asarray(rates)+np.asarray(nonspatial)
    return intensity/intensity.sum(axis=-1, keepdims=True)


def run(source, dest):
    sessions = read_csv(source/"sessions.csv")
    if len(sessions) != 8 or sessions.animal.nunique() != 4:
        raise ValueError("expected frozen eight-session/four-rat bank")
    rows, refs, gaps, audit = [], [], [], []
    inputs = {"source_manifest": source/"manifest.json", "sessions": source/"sessions.csv", "script": Path(__file__),
              "protocol": ROOT/"docs/regional_terminal_proposal_protocol.md", "generator": ROOT/"src/hipporeplayimm/selection_matched_regional.py"}
    for s in sessions.itertuples():
        folder = source/s.session.replace("/", "_")
        m = json.loads((folder/"manifest.json").read_text())
        for name, expected in m["outputs"].items():
            assert file_sha256(folder/name) == expected, folder/name
        inputs[s.session+":manifest"] = folder/"manifest.json"
        e = load_npz(folder/"frozen_inputs.npz")
        with np.load(folder/"simulation_counts.npz", allow_pickle=False) as b:
            labels, accepted, targets = b["labels"], b["accepted"], b["targets"]
        np.testing.assert_array_equal(labels, e["region"][targets])
        meta = read_csv(folder/"simulation_jobs.csv")
        assert len(meta) == len(labels)
        bf = load_npz(folder/"calibration_and_readouts.npz")
        populations = json.loads((folder/"population_definitions.json").read_text())
        duration = e["windows"][:, 1]-e["candidate_start"]
        component = occupied_graph(e["grid"])[0]
        area = float(e["region"][component].mean())
        late = meta.generator.eq("late_jump").to_numpy() & (meta.phase.eq("calibration").to_numpy() | validation_mask(meta))
        for length in LENGTHS:
            available = duration+1e-9 >= length/1000
            terminal = labels[late]
            truth = anywhere_label(~terminal.astype(bool), terminal, length)
            take = accepted[late] & available[None, :]
            if not take.any():
                raise ValueError("no eligible late-jump validation windows")
            assert truth[take].all()
            rows.append({"animal": s.animal, "session": s.session, "duration_ms": length,
                "frozen_events": len(duration), "in_event_windows": int(available.sum()),
                "missing_in_event_windows": int((~available).sum()), "all_frozen_events_available": bool(available.all()),
                "late_jump_accepted_simulated_windows": int(take.sum()),
                "late_jump_any_home_positives": int(truth[take].sum()), "late_jump_any_home_negatives": int((~truth[take]).sum()),
                "late_jump_any_home_prevalence": float(truth[take].mean()),
                "supports_non_degenerate_late_jump_validation": False,
                "calibration_status": "blocked_degenerate_anywhere_truth"})
            for kind, reference in (("stationary", area), ("late_jump", 1.), ("moving", np.nan)):
                refs.append({"animal": s.animal, "session": s.session, "duration_ms": length, "generator": kind,
                    "anywhere_home_reference": reference, "reference_definition": "uniform terminal target over connected simulation grid; before activity selection",
                    "status": "analytic" if np.isfinite(reference) else "requires_path_reconstruction",
                    "biological_chance_claim": False})
        original = read_csv(folder/"validation_and_null.csv")
        index = {p["name"]: j for j, p in enumerate(populations)}
        for family in ("targeted", "whole_tetrode"):
            if family+"_high" not in index:
                continue
            h, l = index[family+"_high"], index[family+"_low"]
            ah, al = populations[h]["area_fraction"], populations[l]["area_fraction"]
            high = expit(bf[f"pop{h}_bf"]+np.log(ah/(1-ah)))
            low = expit(bf[f"pop{l}_bf"]+np.log(al/(1-al)))
            hc = original.loc[original.population.eq(family+"_high")].set_index("job")
            lc = original.loc[original.population.eq(family+"_low")].set_index("job")
            for j in np.flatnonzero(validation_mask(meta)):
                ok = accepted[j]
                job = meta.iloc[j]
                pooled_delta = float(hc.loc[j, "prevalence"]-lc.loc[j, "prevalence"])
                for condition, select in (("all", ok), ("endpoint_home", ok & (labels[j] == 1)),
                                          ("endpoint_elsewhere", ok & (labels[j] == 0))):
                    if not select.any():
                        raise ValueError("missing endpoint class in gap diagnostic")
                    signed = float((high[j, select]-low[j, select]).mean())*100
                    gaps.append({"animal": s.animal, "session": s.session, "family": family, "generator": job.generator,
                        "requested_endpoint_prevalence": float(job.requested_prevalence), "job": int(job.job), "replica": int(job.replica),
                        "endpoint_condition": condition, "retained_events": int(select.sum()),
                        "actual_endpoint_prevalence": float(labels[j, ok].mean()),
                        "high_naive_mass": float(high[j, select].mean()), "low_naive_mass": float(low[j, select].mean()),
                        "signed_naive_gap_pp": signed, "absolute_naive_gap_pp": abs(signed),
                        "signed_pooled_calibrated_gap_pp": 100*pooled_delta if condition == "all" else np.nan,
                        "shared_cells": populations[h]["shared_cells"], "high_area_fraction": ah, "low_area_fraction": al})
        # Common gain and absolute additive scale cannot be separated at fixed totals.
        rates = e["rates"].T
        bias = np.where(e["region"][np.argmax(e["rates"], axis=1)], 2., 0.)
        np.testing.assert_allclose(conditional_identity_probability(rates, 1., bias),
                                   conditional_identity_probability(rates, 3., 3*bias), atol=1e-15, rtol=1e-14)
        audit.append({"session": s.session, "output_hashes_verified": len(m["outputs"]),
                          "exact_labels_verified": True, "gain_additive_scale_invariance": True, "status": "pass"})
        print("AUDIT", s.session, "complete", flush=True)
    feasibility = pd.DataFrame(rows)
    feasibility.to_csv(dest/"terminal_segment_feasibility.csv", index=False)
    pd.DataFrame(refs).to_csv(dest/"terminal_segment_references.csv", index=False)
    gap = pd.DataFrame(gaps)
    gap.to_csv(dest/"generator_population_gap_panels.csv", index=False)
    keys = ["animal", "session", "family", "generator", "endpoint_condition", "requested_endpoint_prevalence"]
    measures = ["signed_naive_gap_pp", "absolute_naive_gap_pp", "signed_pooled_calibrated_gap_pp", "actual_endpoint_prevalence"]
    per_pair = gap.groupby(keys)[measures].mean().reset_index()
    per_pair.to_csv(dest/"generator_population_gap_by_pair.csv", index=False)
    per_animal = per_pair.groupby(["animal", "family", "generator", "endpoint_condition"])[measures].mean().reset_index()
    per_animal.to_csv(dest/"generator_population_gap_by_animal.csv", index=False)
    desc = per_animal.groupby(["family", "generator", "endpoint_condition"])[measures].mean().reset_index()
    desc.to_csv(dest/"generator_population_gap_summary.csv", index=False)
    base = per_pair.loc[per_pair.generator.eq("stationary")]
    comp = per_pair.loc[~per_pair.generator.eq("stationary")].merge(base,
        on=["animal", "session", "family", "endpoint_condition", "requested_endpoint_prevalence"], suffixes=("", "_stationary"), validate="many_to_one")
    comp["absolute_gap_increase_vs_stationary_pp"] = comp.absolute_naive_gap_pp-comp.absolute_naive_gap_pp_stationary
    comp.to_csv(dest/"generator_gap_contrasts.csv", index=False)
    totals = feasibility.groupby("duration_ms")[["frozen_events", "in_event_windows", "missing_in_event_windows"]].sum()
    lines = ["# Terminal-content proposal feasibility and generator-gap audit", "", "## Decision", "",
        ("Do not choose Delta* or apply terminal-anywhere calibration to real events from this simulation bank. "
        "The existing late-jump generator forces an opposite-region origin and terminal target. Every available 20-100-ms segment includes Home, even when the endpoint is elsewhere. "
        "The negative class is absent, so low error would not validate the proposed range of anywhere-content prevalence."), "",
        "This is an estimand/generator mismatch, not evidence that terminal-anywhere estimation is impossible.", "",
        "## Same-event window support", "", "| Terminal length | In-event windows | Unavailable |", "|---|---:|---:|"]
    for length, r in totals.iterrows():
        lines.append(f"| {length} ms | {int(r.in_event_windows)}/{int(r.frozen_events)} | {int(r.missing_in_event_windows)} |")
    lines += ["", "No pre-event padding, truncation or silent removal was used. A common cohort for a length sweep must be declared explicitly.", "",
        "## Generator-stratified endpoint population gaps", "",
        ("Original 20-ms endpoint readouts, not newly defined anywhere-content. Known-truth validation only. "
        "Signed gap is Home-rich minus Home-poor regional posterior mass; absolute gap is the absolute panel-level prevalence difference. "
        "Averages give animals equal weight within each matching family. Targeted groups cover three rats; whole-tetrode groups cover two. "
        "Subsets can share cells and are not independent detectors."), "",
        "| Family | Generator | Signed naive gap (pp) | Absolute naive gap (pp) |", "|---|---|---:|---:|"]
    for r in desc.loc[desc.endpoint_condition.eq("all")].itertuples():
        lines.append(f"| {r.family} | {r.generator} | {r.signed_naive_gap_pp:.2f} | {r.absolute_naive_gap_pp:.2f} |")
    lines += ["", ("Per-endpoint-class results and stationary-reference contrasts are retained in the CSVs. "
        "These simulated gaps cannot establish that the same mechanism produces the real population difference."), "",
        "## Required design corrections", "",
        "- A new segment generator must permit both Home-present and Home-absent paths within every trajectory family, including within-region jumps. Instantaneous endpoint labels cannot be reused as anywhere labels.",
        "- A uniform terminal target does not give one common segment reference: it is the Home area fraction for stationary paths but 1 for the existing forced-crossing jumps. Moving references require path reconstruction. None of these is automatically a biological null.",
        "- A finite composition grid gives worst error among tested mixtures, not a proof covering every possible dynamics distribution.",
        "- With exact whole-population counts fixed, multiplying both global gain and additive rates by the same factor preserves all cell-identity probabilities. The held-out detector must target relative additive participation, or use an unconditional-count validation to identify absolute Hz.",
        "- Inferred IMM-mode calibration is not guaranteed to fall between pooled and oracle errors. Evaluate it as an additional candidate method; wrong dynamics can make it worse. A dynamics prior supplies assumptions, not missing observations.", "",
        "No real-data correction, IMM inference, delta selection or hc-11 transfer was performed. Existing reports and frozen settings remain unchanged."]
    (dest/"report.md").write_text("\n".join(lines)+"\n")
    pd.DataFrame(audit).to_csv(dest/"technical_audit.csv", index=False)
    provenance = build_script_provenance(input_paths=inputs, cwd=ROOT)
    provenance.update(status="complete", non_rescoring=True, delta_star_ms=None,
        segment_validation_status="blocked_generator_and_estimand_mismatch",
        outputs={f.name:file_sha256(f) for f in dest.iterdir() if f.is_file()})
    (dest/"manifest.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print((dest/"report.md").read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=Path("/mnt/seagate10tb/florianpfaff/selection-matched-regional-pf-v3-20260915"))
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    run(a.input_dir, a.output_dir)


if __name__ == "__main__":
    main()
