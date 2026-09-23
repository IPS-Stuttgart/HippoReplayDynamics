"""Compare fixed uncertainty-aware RUN readouts; never rescore sleep events."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from acquire_dandi000978 import write_json
from calibrate_dandi000978_joint_readout import PARAMETERS, group_simulation
from dandi000978_coverage_content import provenance

CONDITIONS = ("synchronous", "position_matched", "independent")
UNITS = {
    "soft_predictive": "nats_per_window",
    "hard_predictive": "nats_per_window",
    "oracle_predictive": "nats_per_window",
    "legacy_centered": "nats_per_PFC_spike",
    "soft_centered": "nats_per_PFC_spike",
}
PARTITIONS = ("both_correct", "ca1_only_correct", "pfc_only_correct", "both_wrong_agree", "both_wrong_disagree", "ambiguous_top_choice")


def likelihoods(counts, rates):
    c, r = np.asarray(counts), np.asarray(rates)
    if c.ndim != 2 or len(c) % 4 or r.shape != (4, c.shape[1]):
        raise ValueError("Four-route blocks with compatible units required")
    if not np.isfinite(c).all() or (c < 0).any() or not np.equal(c, np.floor(c)).all():
        raise ValueError("Finite nonnegative integer counts required")
    if not np.isfinite(r).all() or (r <= 0).any():
        raise ValueError("Finite positive rates required")
    ll = c @ np.log(r / r.sum(axis=1, keepdims=True)).T
    logp = ll - logsumexp(ll, axis=1, keepdims=True)
    centered = (ll - ll.mean(axis=1, keepdims=True)) / np.maximum(c.sum(axis=1, keepdims=True), 1)
    return logp.reshape(-1, 4, 4), centered.reshape(-1, 4, 4)


def take_reference(values, refs):
    return np.take_along_axis(np.broadcast_to(values[:, None], (len(values), 4, 4, 4)), np.broadcast_to(refs[:, :, None, None], (len(values), 4, 4, 1)), axis=3)[..., 0]


def partition_labels(ca, pf):
    ci, pi = ca.argmax(axis=2), pf.argmax(axis=2)
    tied_c = np.isclose(ca, ca.max(axis=2, keepdims=True), rtol=0, atol=1e-12).sum(axis=2) != 1
    tied_p = np.isclose(pf, pf.max(axis=2, keepdims=True), rtol=0, atol=1e-12).sum(axis=2) != 1
    cgood, pgood = ci == np.arange(4), pi == np.arange(4)
    label = np.full(ci.shape, 4, dtype=np.int8)
    label[cgood & pgood] = 0
    label[cgood & ~pgood] = 1
    label[~cgood & pgood] = 2
    label[~cgood & ~pgood & (ci == pi)] = 3
    label[tied_c | tied_p] = 5
    return label


def score_readouts(ca_counts, pfc_counts, ca_rates, pfc_rates):
    ca, _ = likelihoods(ca_counts, ca_rates)
    pf, centered = likelihoods(pfc_counts, pfc_rates)
    if ca.shape != pf.shape:
        raise ValueError("Regional block counts differ")
    ref = ca.argmax(axis=2)
    scores = {
        "soft_predictive": np.log(4) + logsumexp(ca[:, :, None, :] + pf[:, None, :, :], axis=3),
        "hard_predictive": np.log(4) + take_reference(pf, ref),
        "oracle_predictive": np.log(4) + take_reference(pf, np.broadcast_to(np.arange(4), ref.shape)),
        "legacy_centered": take_reference(centered, ref),
        "soft_centered": np.einsum("bir,bjr->bij", np.exp(ca), centered),
    }
    if not all(np.isfinite(s).all() for s in scores.values()):
        raise ValueError("Nonfinite predictive score")
    return scores, partition_labels(ca, pf)


def hierarchy_weights(blocks):
    if blocks.empty:
        return np.empty((0, 4))
    keys = ["file", "heldout_epoch"]
    epoch_count = blocks[keys].drop_duplicates().shape[0]
    targets = blocks.groupby(keys).event_id.transform("nunique").to_numpy()
    repeats = blocks.groupby(keys + ["event_id"]).event_id.transform("size").to_numpy()
    w = 1 / (epoch_count * targets * repeats * 4)
    return np.broadcast_to(w[:, None], (len(w), 4))


def summarize(blocks, bank, partitions, primary, out):
    signal, partition_rows, powers, paired = [], [], [], []
    for animal, animal_blocks in blocks.groupby("animal", sort=True):
        for regime in ("native", "sleep_matched"):
            for support in ("all", "common") if regime == "native" else ("common",):
                group = animal_blocks if support == "all" else animal_blocks[animal_blocks.common_matched]
                w = hierarchy_weights(group)
                for condition in CONDITIONS:
                    labels = partitions[regime + "_" + condition][group.index]
                    for readout, units in UNITS.items():
                        m = bank[f"{regime}_{condition}_{readout}"][group.index]
                        diagonal = np.diagonal(m, axis1=1, axis2=2)
                        base = {"animal": animal, "regime": regime, "support": support, "condition": condition, "readout": readout, "units": units}
                        signal.append(
                            {
                                **base,
                                "mean_score": float(np.sum(w * diagonal)) if len(group) else np.nan,
                                "mean_excess": float(np.sum(w * (diagonal - m.mean(axis=2)))) if len(group) else np.nan,
                                "n_targets": group.event_id.nunique(),
                                "anchor_route_draws": len(group) * 4,
                            }
                        )
                        for label, name in enumerate(PARTITIONS):
                            mask = labels == label
                            fraction = float(w[mask].sum())
                            contribution = float(np.sum((w * diagonal)[mask]))
                            partition_rows.append(
                                {
                                    **base,
                                    "partition": name,
                                    "weighted_fraction": fraction,
                                    "score_contribution": contribution,
                                    "conditional_mean_score": contribution / fraction if fraction else np.nan,
                                    "anchor_route_draws": int(mask.sum()),
                                }
                            )
                    if regime != "sleep_matched":
                        continue
                    desired = primary.loc[primary.animal == animal, "event_id"].to_numpy()
                    for design, requested in [("original_primary", desired), *[(f"sensitivity_n{n}", n) for n in PARAMETERS["sizes"]]]:
                        simulated = {}
                        for readout, units in UNITS.items():
                            m = bank[f"{regime}_{condition}_{readout}"][group.index]
                            result, audit = group_simulation(m, group.event_id.to_numpy(), requested, (animal, "synchronized_common", design))
                            powers.append({"animal": animal, "condition": condition, "readout": readout, "units": units, "design": design, **result})
                            if audit is not None:
                                np.savez_compressed(out / "simulations" / f"{animal}_{condition}_{readout}_{design}.npz", **audit)
                                simulated[readout] = np.asarray(audit["pvalues"]) <= PARAMETERS["alpha"]
                        if simulated:
                            differences = simulated["soft_predictive"].astype(float) - simulated["hard_predictive"].astype(float)
                            paired.append(
                                {
                                    "animal": animal,
                                    "condition": condition,
                                    "design": design,
                                    "soft_minus_hard_rejection": differences.mean(),
                                    "paired_monte_carlo_se": differences.std(ddof=1) / np.sqrt(len(differences)),
                                    "both_reject": np.mean(simulated["soft_predictive"] & simulated["hard_predictive"]),
                                    "neither_reject": np.mean(~simulated["soft_predictive"] & ~simulated["hard_predictive"]),
                                }
                            )
    pd.DataFrame(signal).to_csv(out / "signal_summary.csv", index=False)
    pd.DataFrame(partition_rows).to_csv(out / "score_partition_summary.csv", index=False)
    power = pd.DataFrame(powers)
    power.to_csv(out / "conditional_sensitivity.csv", index=False)
    pd.DataFrame(paired).to_csv(out / "paired_readout_comparison.csv", index=False)
    plan = power[(power.condition == "synchronous") & (power.readout == "soft_predictive") & (power.design == "sensitivity_n50")]
    adequate = (
        len(plan) == 2
        and set(plan.animal) == {"JS14", "ZT2"}
        and (plan.status == "simulated").all()
        and (plan.positive_control_rejection >= PARAMETERS["planning_power"]).all()
        and (plan.null_false_positive_rate <= PARAMETERS["planning_max_false_positive"]).all()
    )
    decision = {
        "technical_complete": True,
        "conditional_planning_criterion_passed": bool(adequate),
        "sleep_content_validated": False,
        "paper_ready": False,
        "decision": "independent_validation_required" if adequate else "stop_incremental_four_route_assay_tuning",
    }
    write_json(out / "decision.json", decision)


def run(args):
    source, out = args.source_dir, args.output_dir
    verification = json.loads((source / "independent_verification.json").read_text())
    if not verification["verified"] or verification["raw_source_folds"] != 14:
        raise ValueError("Verified synchronized predecessor required")
    previous = json.loads((source / "manifest.json").read_text())
    inputs = {
        "source_manifest": source / "manifest.json",
        "source_verification": source / "independent_verification.json",
        "source_blocks": source / "blocks.csv",
        "source_bank": source / "joint_bank.npz",
        "source_coverage": source / "coverage.csv",
        "source_folds": source / "fold_status.csv",
        "sleep_events": Path(previous["input_file_paths"]["sleep_events"]),
    }
    folds = pd.read_csv(inputs["source_folds"])
    if len(folds) != 14 or folds.duplicated(["file", "heldout_epoch"]).any():
        raise ValueError("Expected 14 unique source folds")
    by_name = {Path(a["path"]).name: a["asset_id"] for a in json.loads(Path(previous["input_file_paths"]["assets"]).read_text())["assets"]}
    for i, row in enumerate(folds.itertuples(index=False)):
        inputs[f"fold{i}_audit"] = source / by_name[row.file] / f"epoch_{row.heldout_epoch}" / "audit.npz"
    prov = provenance(inputs)
    if prov["input_file_sha256"]["sleep_events"] != previous["input_file_sha256"]["sleep_events"]:
        raise ValueError("Frozen sleep group input changed")
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    (out / "simulations").mkdir()
    write_json(
        out / "manifest.json",
        {
            **prov,
            "scope": "fixed_bank_uncertainty_RUN_calibration_no_sleep",
            "study_parameters": PARAMETERS,
            "readouts": UNITS,
            "conditions": CONDITIONS,
            "no_fitted_hyperparameters": True,
        },
    )
    banks, partitions = {}, {}
    for i, row in enumerate(folds.itertuples(index=False)):
        audit = dict(np.load(inputs[f"fold{i}_audit"]))
        ix = audit["block_indices"].reshape(-1)
        for regime in ("native", "sleep_matched"):
            ca = audit["ca1_native" if regime == "native" else "ca1_sparse"][ix]
            for condition in CONDITIONS:
                pf = audit[regime + "_" + condition + "_pfc_counts"][ix]
                scores, labels = score_readouts(ca, pf, audit["ca1_rates"], audit["pfc_rates"])
                np.testing.assert_allclose(scores["legacy_centered"], audit[regime + "_" + condition + "_decoded"], atol=1e-12)
                for name, m in scores.items():
                    banks.setdefault(f"{regime}_{condition}_{name}", []).append(m)
                partitions.setdefault(f"{regime}_{condition}", []).append(labels)
        print(json.dumps({"file": row.file, "epoch": row.heldout_epoch, "status": "complete"}), flush=True)
    bank = {k: np.concatenate(v) for k, v in banks.items()}
    partitions = {k: np.concatenate(v) for k, v in partitions.items()}
    blocks = pd.read_csv(inputs["source_blocks"])
    if any(len(m) != len(blocks) for m in bank.values()):
        raise ValueError("Global block ordering/size mismatch")
    original_bank = dict(np.load(inputs["source_bank"]))
    for regime in ("native", "sleep_matched"):
        for condition in CONDITIONS:
            np.testing.assert_allclose(bank[f"{regime}_{condition}_legacy_centered"], original_bank[f"{regime}_{condition}_decoded"], atol=1e-12)
    blocks.to_csv(out / "blocks.csv", index=False)
    pd.read_csv(inputs["source_coverage"]).to_csv(out / "coverage.csv", index=False)
    np.savez_compressed(out / "score_bank.npz", **bank)
    np.savez_compressed(out / "partition_bank.npz", **partitions)
    primary = pd.read_csv(inputs["sleep_events"])
    summarize(blocks, bank, partitions, primary[primary.primary_lost_half], out)
    write_json(out / "terminal_status.json", {"status": "complete"})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    try:
        run(args)
    except Exception as exc:
        if args.output_dir.exists():
            write_json(args.output_dir / "terminal_status.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
