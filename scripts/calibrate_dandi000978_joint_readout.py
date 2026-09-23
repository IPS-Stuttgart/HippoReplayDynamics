#!/usr/bin/env python3
"""Non-rescoring joint RUN-readout calibration; no sleep-content claims."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from acquire_dandi000978 import write_json
from dandi000978_coverage_content import composition_scores, provenance
from validate_dandi000978_run_readouts import seed_for

PARAMETERS = {"seed": 20260924, "virtual_studies": 500, "null_draws": 199, "alpha": 0.05, "sizes": [10, 25, 50], "planning_power": 0.8, "planning_max_false_positive": 0.075}
KEYS = ["target_index", "repeat", "scope", "route"]


def validate_pair(ca, pf, ca_z, pf_z):
    if ca.duplicated(KEYS).any() or pf.duplicated(KEYS).any():
        raise ValueError("Duplicate anchor identities")
    if not ca[KEYS].equals(pf[KEYS]):
        raise ValueError("Regional anchors must share target/repeat/route/scope order")
    if not np.array_equal(ca_z["target_event_ids"], pf_z["target_event_ids"]):
        raise ValueError("Regional target event order differs")
    if set(ca_z["source_unit_ids"]) & set(pf_z["source_unit_ids"]):
        raise ValueError("Regional unit overlap")
    for frame, z in ((ca, ca_z), (pf, pf_z)):
        if set(z["training_trial_ids"]) & set(z["test_trial_ids"]):
            raise ValueError("Training/test trial leakage")
        if not set(frame.trial_id).issubset(z["test_trial_ids"]):
            raise ValueError("Anchor not held out")
        if (z["sparse"] < 0).any() or (z["sparse"] > z["native"]).any():
            raise ValueError("Invalid thinning")
        matched = frame.matched.to_numpy(bool)
        if not np.array_equal(frame.target_count[matched], z["sparse"][matched].sum(axis=1)):
            raise ValueError("Count matching failed")
        if not np.array_equal(matched, frame.target_count <= z["native"].sum(axis=1)):
            raise ValueError("Unsupported counts hidden")


def score_matrix(pfc_vectors, references):
    """Block x CA1 truth x PFC donor truth, using each CA1 reference route."""
    v = np.asarray(pfc_vectors)
    r = np.asarray(references)
    if v.ndim != 3 or v.shape[1:] != (4, 4) or r.shape != v.shape[:2]:
        raise ValueError("Four-route blocks required")
    return np.take_along_axis(np.broadcast_to(v[:, None], (len(v), 4, 4, 4)), np.broadcast_to(r[:, :, None, None], (len(v), 4, 4, 1)), axis=3)[..., 0]


def build_bank(ca, pf, ca_z, pf_z):
    validate_pair(ca, pf, ca_z, pf_z)
    blocks, groups = [], []
    joint = ca.matched.to_numpy(bool) & pf.matched.to_numpy(bool)
    for (target, repeat, scope), g in ca.groupby(["target_index", "repeat", "scope"], sort=True):
        if len(g) != 4 or set(g.route) != set(range(4)):
            raise ValueError("Incomplete route block")
        ix = g.sort_values("route").index.to_numpy()
        ok = bool(joint[ix].all())
        blocks.append(
            {
                "target_index": int(target),
                "event_id": ca_z["target_event_ids"][target],
                "repeat": int(repeat),
                "scope": scope,
                "matched_block": ok,
                "target_ca1_spikes": int(ca_z["target_counts"][target]),
                "target_pfc_spikes": int(pf_z["target_counts"][target]),
            }
        )
        groups.append(ix)
    blocks = pd.DataFrame(blocks)
    ix = np.array(groups)[blocks.matched_block.to_numpy()].reshape(-1)
    supported = blocks[blocks.matched_block].copy().reset_index(drop=True)
    bank = {}
    for regime, field in (("native", "native"), ("sleep_matched", "sparse")):
        _, ca_prob = composition_scores(ca_z[field][ix], ca_z["rates"])
        vectors, _ = composition_scores(pf_z[field][ix], pf_z["rates"])
        ca_reference = ca_prob.argmax(axis=1).reshape(-1, 4)
        vectors = vectors.reshape(-1, 4, 4)
        bank[regime + "_decoded"] = score_matrix(vectors, ca_reference)
        bank[regime + "_oracle"] = score_matrix(vectors, np.broadcast_to(np.arange(4), ca_reference.shape))
        bank[regime + "_ca1_accuracy"] = (ca_reference == np.arange(4)).mean(axis=1)
    return blocks, supported, bank


def group_simulation(matrices, targets, requested, key):
    """Resample a conditional calibration bank, never infer a biological p-value."""
    groups = {tid: np.flatnonzero(targets == tid) for tid in np.unique(targets)}
    fixed = not isinstance(requested, int)
    n = len(requested) if fixed else requested
    absent = sorted(set(requested) - set(groups)) if fixed else []
    if n < 1 or absent or (not fixed and n > len(groups)):
        return {
            "status": "unsupported",
            "n_requested": n,
            "missing_target_count": len(absent),
            "failure_reason": "unmatched_primary_targets" if absent else "insufficient_unique_targets",
        }, None
    rng = np.random.default_rng(seed_for(PARAMETERS["seed"], *key))
    candidates = np.array(sorted(groups))
    trials, k = PARAMETERS["virtual_studies"], PARAMETERS["null_draws"]
    observed, null_observed, pvalues, null_pvalues, excess = [], [], [], [], []
    selected, chosen_routes = [], []
    for _ in range(trials):
        ids = np.asarray(requested) if fixed else rng.choice(candidates, n, replace=False)
        rows = np.array([rng.choice(groups[tid]) for tid in ids])
        truth = rng.integers(4, size=n)
        controls = rng.integers(4, size=(k, n))
        null = matrices[rows[None, :], truth[None, :], controls].mean(axis=1)
        actual = matrices[rows, truth, truth].mean()
        randomized = matrices[rows, truth, rng.integers(4, size=n)].mean()
        observed.append(actual)
        null_observed.append(randomized)
        pvalues.append((1 + (null >= actual).sum()) / (1 + k))
        null_pvalues.append((1 + (null >= randomized).sum()) / (1 + k))
        excess.append(actual - null.mean())
        selected.append(rows)
        chosen_routes.append(truth)
    rejection = np.mean(np.asarray(pvalues) <= PARAMETERS["alpha"])
    false_positive = np.mean(np.asarray(null_pvalues) <= PARAMETERS["alpha"])
    result = {
        "status": "simulated",
        "n_requested": n,
        "missing_target_count": 0,
        "failure_reason": "",
        "virtual_studies": trials,
        "positive_control_rejection": float(rejection),
        "null_false_positive_rate": float(false_positive),
        "rejection_monte_carlo_se": float(np.sqrt(rejection * (1 - rejection) / trials)),
        "median_score": float(np.median(observed)),
        "median_excess": float(np.median(excess)),
    }
    return result, {"observed": observed, "randomized_observed": null_observed, "pvalues": pvalues, "null_pvalues": null_pvalues, "rows": selected, "truth": chosen_routes}


def summaries(blocks, supported, bank, primary, out):
    coverage, signal, power, expected = [], [], [], []
    for (animal, scope), group in blocks.groupby(["animal", "scope"], sort=True):
        available = group[group.matched_block]
        missing = sorted(set(primary[primary.animal == animal].event_id) - set(available.event_id))
        coverage.append(
            {
                "animal": animal,
                "scope": scope,
                "blocks": len(group),
                "matched_blocks": len(available),
                "block_match_fraction": len(available) / len(group),
                "targets": group.event_id.nunique(),
                "supported_targets": available.event_id.nunique(),
                "primary_targets": len(primary[primary.animal == animal]),
                "missing_primary_targets": len(missing),
            }
        )
        rows = supported.index[(supported.animal == animal) & (supported.scope == scope)].to_numpy()
        for regime in ("native", "sleep_matched"):
            for reference in ("decoded", "oracle"):
                key = regime + "_" + reference
                m = bank[key][rows]
                targets = supported.loc[rows, "event_id"].to_numpy()
                diagonal = np.diagonal(m, axis1=1, axis2=2).mean(axis=1)
                excess = diagonal - m.mean(axis=(1, 2))
                values = supported.loc[rows, ["event_id", "heldout_epoch", "file"]].copy()
                values["excess"] = excess
                values["score"] = diagonal
                per_event = values.groupby(["file", "heldout_epoch", "event_id"])[["excess", "score"]].mean()
                per_epoch = per_event.groupby(["file", "heldout_epoch"]).mean()
                base = {"animal": animal, "scope": scope, "regime": regime, "reference": reference}
                signal.append(
                    {
                        **base,
                        "equal_epoch_mean_score": per_epoch.score.mean(),
                        "equal_epoch_mean_excess": per_epoch.excess.mean(),
                        "n_epochs": len(per_epoch),
                        "n_targets": len(per_event),
                    }
                )
                designs = [("original_primary", primary[primary.animal == animal].event_id.to_numpy())]
                designs += [(f"sensitivity_n{n}", n) for n in PARAMETERS["sizes"]]
                for design, requested in designs:
                    expected.append((*base.values(), design))
                    result, audit = group_simulation(m, targets, requested, (*base.values(), design))
                    power.append({**base, "design": design, **result})
                    if audit is not None:
                        np.savez_compressed(out / "simulations" / ("_".join(map(str, (*base.values(), design))) + ".npz"), **audit)
    coverage, signal, power = map(pd.DataFrame, (coverage, signal, power))
    coverage.to_csv(out / "joint_coverage.csv", index=False)
    signal.to_csv(out / "joint_signal.csv", index=False)
    power.to_csv(out / "conditional_sensitivity.csv", index=False)
    pick = power[(power.scope == "window_250ms") & (power.regime == "sleep_matched") & (power.reference == "decoded") & (power.design == "original_primary")]
    adequate = bool(
        set(pick.animal) == {"JS14", "ZT2"}
        and (pick.status == "simulated").all()
        and (pick.positive_control_rejection >= PARAMETERS["planning_power"]).all()
        and (pick.null_false_positive_rate <= PARAMETERS["planning_max_false_positive"]).all()
    )
    technical = bool(len(expected) == 64 and len(power) == 64 and set(blocks.animal) == {"JS14", "ZT2"})
    gates = [
        {"gate": "all_conditional_designs_explicit", "passed": technical},
        {"gate": "original_primary_short_window_calibration_adequate", "passed": adequate},
        {"gate": "sleep_content_validated", "passed": False},
        {"gate": "paper_ready", "passed": False},
    ]
    pd.DataFrame(gates).to_csv(out / "gate_summary.csv", index=False)
    decision = "joint_readout_sensitivity_adequate_transfer_unresolved" if adequate else "primary_sleep_endpoint_not_calibrated_for_adequate_sensitivity"
    write_json(out / "decision.json", {"decision": decision, "technical_complete": technical, "paper_ready": False})
    (out / "joint_readout_summary.md").write_text(
        "# Joint readout calibration\n\nDecision: `" + decision + "`.\n\n"
        "This is conditional RUN sensitivity, not sleep evidence or a biological power guarantee.\n"
        "CA1 and PFC donors share a known route but are independent windows/trials.\n"
        "Short-window matching requires all four routes and both regions; unsupported primary targets remain explicit.\n"
        "Whole-trial and oracle conditions are optimistic controls, not replay analyses.\n\n```csv\n"
        + coverage.to_csv(index=False)
        + "```\n\n```csv\n"
        + power[power.design == "original_primary"].to_csv(index=False)
        + "```\n\n500 virtual studies do not create new animals. No thresholds were relaxed.\n"
    )
    return technical


def run(args):
    src, out = args.calibration_dir, args.output_dir
    source_manifest = json.loads((src / "manifest.json").read_text())
    verified = json.loads((src / "independent_verification.json").read_text())
    if not verified["verified"] or verified["regional_holdouts"] != 28 or verified["producer_commit"] != source_manifest["code_commit"]:
        raise ValueError("Unverified source calibration")
    sleep = pd.read_csv(args.sleep_events)
    if len(sleep) != 1354 or sleep.event_id.duplicated().any():
        raise ValueError("Frozen sleep cohort changed")
    primary = sleep[sleep.primary_lost_half].copy()
    if primary.groupby("animal").size().to_dict() != {"JS14": 3, "ZT2": 11}:
        raise ValueError("Frozen primary group changed")
    inputs = {"calibration_manifest": src / "manifest.json", "calibration_verification": src / "independent_verification.json", "sleep_events": args.sleep_events}
    paths = sorted(src.glob("*/CA1/epoch_*/audit.npz"))
    if len(paths) != 14:
        raise ValueError("Fourteen expected RUN holdouts required")
    for j, path in enumerate(paths):
        for region in ("CA1", "PFC"):
            folder = path.parent.parent.parent / region / path.parent.name
            for name in ("audit.npz", "anchors.csv"):
                inputs[f"fold{j}_{region}_{name}"] = folder / name
    prov = provenance(inputs)
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    (out / "simulations").mkdir()
    write_json(out / "manifest.json", {**prov, "parameters": PARAMETERS, "scope": "joint_RUN_calibration_not_replay", "independent_same_route_donors": True})
    blocks_all, supported_all, bank_all = [], [], []
    metrics = pd.read_csv(src / "fold_metrics.csv")
    source_assets = json.loads(Path(source_manifest["input_file_paths"]["assets"]).read_text())["assets"]
    lookup = {a["asset_id"]: Path(a["path"]).name for a in source_assets}
    for path in paths:
        other = path.parent.parent.parent / "PFC" / path.parent.name
        ca = pd.read_csv(path.parent / "anchors.csv")
        pf = pd.read_csv(other / "anchors.csv")
        ca_z, pf_z = dict(np.load(path)), dict(np.load(other / "audit.npz"))
        blocks, supported, bank = build_bank(ca, pf, ca_z, pf_z)
        filename = lookup[path.parent.parent.parent.name]
        animal = metrics.loc[metrics.file == filename, "animal"].iloc[0]
        for table in (blocks, supported):
            table["file"], table["animal"] = filename, animal
            table["heldout_epoch"] = int(path.parent.name.split("_")[1])
        blocks_all.append(blocks)
        supported_all.append(supported)
        bank_all.append(bank)
    blocks = pd.concat(blocks_all, ignore_index=True)
    supported = pd.concat(supported_all, ignore_index=True)
    bank = {key: np.concatenate([b[key] for b in bank_all]) for key in bank_all[0]}
    if set(blocks.event_id) != set(sleep.event_id):
        raise ValueError("Target coverage differs from frozen sleep cohort")
    blocks.to_csv(out / "block_eligibility.csv", index=False)
    supported.to_csv(out / "supported_blocks.csv", index=False)
    np.savez_compressed(out / "joint_bank.npz", **bank)
    complete = summaries(blocks, supported, bank, primary, out)
    write_json(out / "terminal_status.json", {"status": "complete" if complete else "incomplete"})
    print((out / "decision.json").read_text(), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibration-dir", type=Path, required=True)
    p.add_argument("--sleep-events", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    try:
        run(args)
    except Exception as exc:
        if args.output_dir.exists() and not (args.output_dir / "terminal_status.json").exists():
            write_json(args.output_dir / "terminal_status.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
