#!/usr/bin/env python3
"""Audit the accuracy cost of changing a model-correct Home posterior."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp, softmax
from scipy.stats import poisson

from scripts._provenance import build_script_provenance, file_sha256
from scripts.audit_joint_exchange_truth import checked_manifest
from scripts.report_content_screening_bound import markdown

SESSIONS = ("Rat1/Open1", "Rat1/Open2", "Rat2/Open1", "Rat4/Open2")
METHODS = ("baseline", "temperature_2", "uniform_mixture_half")
BANK = "cal_poisson_gain1"
DT = 0.02


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def home_posterior(counts, rates, near, indices, temperature=1.0):
    ll = counts[:, indices] @ np.log(rates[indices]) - DT * rates[indices].sum(axis=0)
    return softmax(ll / temperature, axis=1)[:, near].sum(axis=1)


def independent_home(counts, rates, near, indices):
    # Include all Poisson constants and accumulate individual cells separately.
    ll = np.zeros((len(counts), rates.shape[1]))
    for cell in indices:
        mu = DT * rates[cell]
        n = counts[:, cell, None]
        ll += n * np.log(mu) - mu - gammaln(n + 1)
    ll -= logsumexp(ll, axis=1, keepdims=True)
    return np.exp(ll[:, near]).sum(axis=1)


def verify_generator(enc, bank, session):
    rates = enc["early_run"]
    near = enc["near"]
    if rates.ndim != 2 or near.dtype != bool or near.shape != (rates.shape[1],):
        raise ValueError("invalid encoding shapes or region mask")
    if not np.isfinite(rates).all() or np.any(rates <= 0) or not 0 < near.mean() < 1:
        raise ValueError("invalid rates or empty true class")
    digest = hashlib.sha256(f"20260914|regional_prevalence|{session}|{BANK}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    position = np.r_[rng.choice(np.flatnonzero(~near), 2000), rng.choice(np.flatnonzero(near), 2000)]
    count = rng.poisson(DT * rates[:, position].T)
    np.testing.assert_array_equal(bank["position_ids"], position)
    np.testing.assert_array_equal(bank["counts"], count)
    np.testing.assert_array_equal(bank["labels"], near[position])
    np.testing.assert_array_equal(bank["truth_cm"], enc["grid_cm"][position])
    np.testing.assert_array_equal(bank["drawn_totals"], count.sum(axis=1))
    for side in ("high", "low"):
        ix = enc[f"{side}_indices"]
        if ix.ndim != 1 or not np.issubdtype(ix.dtype, np.integer) or len(np.unique(ix)) != len(ix) or not len(ix) or ix.min() < 0 or ix.max() >= len(rates):
            raise ValueError("invalid observation subset")
    if len(enc["high_indices"]) != len(enc["low_indices"]):
        raise ValueError("original population counts differ")


def risks(q, a, labels, prior):
    q, a, labels = np.asarray(q), np.asarray(a), np.asarray(labels)
    if not 0 < prior < 1 or q.shape != a.shape or q.shape != labels.shape or q.ndim != 1:
        raise ValueError("invalid risk inputs")
    if not np.isfinite(q).all() or not np.isfinite(a).all() or np.any((q < 0) | (q > 1)) or np.any((a < 0) | (a > 1)) or not np.isin(labels, (0, 1)).all():
        raise ValueError("invalid probabilities or labels")
    if np.sum(labels == 0) != np.sum(labels == 1) or len(labels) < 4:
        raise ValueError("expected balanced class-stratified bank")
    # The bank has 50/50 classes, whereas the decoder prior is uniform in space.
    w = np.where(labels == 1, 2 * prior, 2 * (1 - prior))
    marginal_ratio = 2 / (q / prior + (1 - q) / (1 - prior))
    baseline = (q - labels) ** 2
    changed = (a - labels) ** 2
    penalty = (a - q) ** 2
    cross = 2 * (q - labels) * (a - q)
    np.testing.assert_allclose(changed - baseline, penalty + cross, atol=1e-14, rtol=1e-12)
    conditional_base = q * (1 - q)
    conditional_new = q * (1 - a) ** 2 + (1 - q) * a**2
    np.testing.assert_allclose(conditional_new - conditional_base, penalty, atol=1e-14, rtol=1e-12)
    se = np.sqrt(sum(p**2 * np.var(cross[labels == k], ddof=1) / np.sum(labels == k) for k, p in ((0, 1 - prior), (1, prior))))
    result = dict(
        baseline_brier_importance=float(np.mean(w * baseline)),
        changed_brier_importance=float(np.mean(w * changed)),
        observed_brier_change=float(np.mean(w * (changed - baseline))),
        squared_change_importance=float(np.mean(w * penalty)),
        cross_term_mean=float(np.mean(w * cross)),
        cross_term_se=float(se),
        baseline_brier_rao_blackwell=float(np.mean(marginal_ratio * conditional_base)),
        changed_brier_rao_blackwell=float(np.mean(marginal_ratio * conditional_new)),
        expected_brier_cost_rao_blackwell=float(np.mean(marginal_ratio * penalty)),
        max_conditional_identity_error=float(np.max(np.abs(conditional_new - conditional_base - penalty))),
        unchanged=bool(np.array_equal(q, a)),
    )
    for k in (0, 1):
        result[f"baseline_brier_class{k}"] = float(baseline[labels == k].mean())
        result[f"changed_brier_class{k}"] = float(changed[labels == k].mean())
    return result


def agreement_bound(q, a, labels, prior):
    w = np.where(labels == 1, prior, 1 - prior).astype(float)
    w /= w.sum()
    d0 = float(np.sqrt(np.sum(w * (q[:, 0] - q[:, 1]) ** 2)))
    d1 = float(np.sqrt(np.sum(w * (a[:, 0] - a[:, 1]) ** 2)))
    cost = np.sum(w[:, None] * (a - q) ** 2, axis=0)
    lower = max(0.0, d0 - np.sqrt(cost).sum())
    minimum_cost_for_observed_reduction = max(0.0, d0 - d1) ** 2 / 2
    if d1 < lower - 1e-12 or cost.sum() < minimum_cost_for_observed_reduction - 1e-12:
        raise ValueError("L2 accuracy/agreement bound violated")
    return dict(
        baseline_home_probability_rms_disagreement=d0,
        changed_home_probability_rms_disagreement=d1,
        rms_disagreement_reduction_fraction=1 - d1 / d0 if d0 else 0.0,
        squared_adjustment_high=float(cost[0]),
        squared_adjustment_low=float(cost[1]),
        rms_disagreement_lower_bound=float(lower),
        minimum_summed_brier_cost_for_observed_rms_reduction=float(minimum_cost_for_observed_reduction),
        minimum_summed_brier_cost_for_20pct_rms_reduction=float(0.2**2 * d0**2 / 2),
    )


def toy_check():
    means = np.array([[0.8, 0.1], [0.1, 0.8], [0.1, 0.3], [0.3, 0.1]])
    region = np.array([True, True, False, False])
    observations = np.array(list(product(range(21), repeat=2)))
    conditional = np.prod(poisson.pmf(observations[:, None, :], means[None, :, :]), axis=2)
    joint = conditional / len(means)
    weights = joint.sum(axis=1)
    tail_bound = float(np.sum(poisson.sf(20, means), axis=1).mean())
    if abs(weights.sum() - 1) > 1e-12 or tail_bound > 1e-12:
        raise ValueError("toy count truncation is not negligible")
    posterior = joint / weights[:, None]
    q_full = posterior[:, region].sum(axis=1)
    partial = []
    for cell in range(2):
        likelihood = poisson.pmf(observations[:, cell, None], means[None, :, cell])
        partial.append(likelihood[:, region].sum(axis=1) / likelihood.sum(axis=1))
    q = np.column_stack(partial)
    alternative = dict(baseline=q, uniform_mixture_half=0.5 * q + 0.25, shared_full_observation=np.column_stack((q_full, q_full)))
    rows = []
    for name, a in alternative.items():
        row = dict(method=name, uses_additional_observations=name == "shared_full_observation", rms_disagreement=float(np.sqrt(np.sum(weights * (a[:, 0] - a[:, 1]) ** 2))))
        for cell in range(2):
            b0 = np.sum(joint * (q[:, cell, None] - region) ** 2)
            b1 = np.sum(joint * (a[:, cell, None] - region) ** 2)
            penalty = np.sum(weights * (a[:, cell] - q[:, cell]) ** 2)
            change = b1 - b0
            expected = -penalty if name == "shared_full_observation" else penalty
            np.testing.assert_allclose(change, expected, atol=1e-12, rtol=1e-10)
            row.update({f"brier_change_{cell}": float(change), f"squared_adjustment_{cell}": float(penalty), f"identity_error_{cell}": float(abs(change - expected))})
        rows.append(row)
    return pd.DataFrame(rows), dict(count_vectors=len(observations), retained_probability=float(weights.sum()), omitted_probability_upper_bound=tail_bound)


def evaluate_session(enc, bank, session):
    verify_generator(enc, bank, session)
    near, prior = enc["near"], float(enc["near"].mean())
    rows, frame = [], dict(observation_index=np.arange(len(bank["counts"])), true_home=bank["labels"].astype(int))
    predictions = {}
    for column, side in enumerate(("high", "low")):
        indices = enc[f"{side}_indices"]
        q = home_posterior(bank["counts"], enc["early_run"], near, indices)
        check = independent_home(bank["counts"], enc["early_run"], near, indices)
        np.testing.assert_allclose(q, check, atol=1e-10, rtol=1e-9)
        np.testing.assert_allclose(q, bank["early_run_scores"][:, column], atol=1e-10, rtol=1e-9)
        values = dict(baseline=q, temperature_2=home_posterior(bank["counts"], enc["early_run"], near, indices, 2), uniform_mixture_half=0.5 * q + 0.5 * prior)
        predictions[side] = values
        for name, a in values.items():
            rows.append(
                dict(
                    session=session,
                    animal=session.split("/")[0],
                    side=side,
                    method=name,
                    observations=len(q),
                    home_spatial_prior=prior,
                    cells=len(indices),
                    independent_posterior_max_error=float(np.max(np.abs(q - check))),
                    **risks(q, a, bank["labels"], prior),
                )
            )
            frame[f"{side}_{name}"] = a
    original = np.column_stack([predictions[s]["baseline"] for s in ("high", "low")])
    pairs = []
    for method in METHODS:
        changed = np.column_stack([predictions[s][method] for s in ("high", "low")])
        pairs.append(dict(session=session, method=method, **agreement_bound(original, changed, bank["labels"], prior)))
    return pd.DataFrame(rows), pd.DataFrame(pairs), pd.DataFrame(frame)


def run(reference_dir, reference_audit, output_dir):
    reference = checked_manifest(reference_dir, reference_audit)
    source = Path(reference["source_dir"])
    inputs = dict(reference["input_file_sha256"])
    inputs.update(
        {
            str(p): file_sha256(p)
            for p in (
                reference_dir / "manifest.json",
                reference_audit,
                Path(__file__),
                ROOT / "scripts/audit_joint_exchange_truth.py",
                ROOT / "scripts/_provenance.py",
                ROOT / "scripts/report_content_screening_bound.py",
                ROOT / "docs/content_bayes_risk_protocol.md",
            )
        }
    )
    if any(v is None for v in inputs.values()):
        raise ValueError("missing provenance input")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = {**build_script_provenance(), "input_file_sha256": inputs, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    (output_dir / "pre_analysis.json").write_text(json.dumps(started, indent=2) + "\n")
    all_risks, all_pairs = [], []
    toy, toy_metadata = toy_check()
    toy.to_csv(output_dir / "exact_toy.csv", index=False)
    for session in SESSIONS:
        folder = source / session.replace("/", "_")
        enc_path, bank_path = folder / "encoding.npz", folder / f"{BANK}.npz"
        for path in (enc_path, bank_path):
            if inputs.get(str(path)) != file_sha256(path):
                raise ValueError(f"source not linked to audited reference: {path}")
        risk, pairs, frames = evaluate_session(load(enc_path), load(bank_path), session)
        all_risks.append(risk)
        all_pairs.append(pairs)
        frames.to_csv(output_dir / f"{session.replace('/', '_')}_probabilities.csv.gz", index=False)
        print("audited", session, "observations", len(frames), flush=True)
    risk, pairs = pd.concat(all_risks, ignore_index=True), pd.concat(all_pairs, ignore_index=True)
    risk.to_csv(output_dir / "accuracy_costs.csv", index=False)
    pairs.to_csv(output_dir / "agreement_cost_bounds.csv", index=False)
    checks = dict(
        all_original_pairs_present=set(risk.session) == set(SESSIONS) and len(risk) == 24,
        all_generator_counts_reconstructed=True,
        all_baseline_posteriors_independently_reconstructed=bool(risk.independent_posterior_max_error.max() < 1e-9),
        conditional_risk_identity=bool(risk.max_conditional_identity_error.max() < 1e-12),
        nontrivial_adjustments_have_positive_expected_cost=bool((risk.loc[risk.method.ne("baseline"), "expected_brier_cost_rao_blackwell"] > 0).all()),
        same_information_toy_adjustment_cost_positive=bool((toy.loc[toy.method.eq("uniform_mixture_half"), ["brier_change_0", "brier_change_1"]] > 0).all().all()),
        additional_information_toy_cost_negative=bool((toy.loc[toy.method.eq("shared_full_observation"), ["brier_change_0", "brier_change_1"]] < 0).all().all()),
    )
    pd.DataFrame([dict(gate=k, passed=v) for k, v in checks.items()] + [dict(gate="numerical_audit", passed=all(checks.values()))]).to_csv(output_dir / "gates.csv", index=False)
    text = "# Model-correct posterior risk diagnostic\n\n"
    text += "Development-only analysis of previously inspected gain-1 Poisson calibration banks. No native RUN, Q4, test banks or replay were decoded. No independent-recording validation.\n\n"
    text += "The baseline is the exact Home posterior under the audited generator and uniform spatial prior. For an adjustment using the SAME observations, expected Brier increase is E[(adjusted - baseline)^2]. Both true-class Brier risks cannot be nonworse in expectation unless the Home probabilities are unchanged. This is a standard conditional-expectation identity, not a novel theorem.\n\n"
    text += "The bank draws 50/50 Home classes. Label importance weights and marginal likelihood ratios restore the decoder's uniform-bin prior; unweighted balanced risk is not the Bayes-risk target. Reported Rao-Blackwell costs use exact conditional risks but Monte Carlo integration over counts.\n\n"
    text += (
        markdown(risk[["session", "side", "method", "home_spatial_prior", "observed_brier_change", "expected_brier_cost_rao_blackwell", "cross_term_mean", "cross_term_se"]])
        + "\n\n"
    )
    text += "## Accuracy/agreement trade-off\n\nFor fixed information, RMS regional-probability disagreement can fall by at most sqrt(cost_high) + sqrt(cost_low). The same weighted finite-sample L2 inequality is audited here. This metric is NOT the native replay Home mean gap, the prior J, trajectory acceptance or full regional TV.\n\n"
    text += markdown(pairs) + "\n\n## Exact toy controls\n\n" + markdown(toy) + "\n\n"
    text += "A shared full observation improves expected Brier and removes disagreement in the toy. It changes the available information and is NOT a proposed equal-observation remedy or an empirical validation. The result does not rule out better encodings, more information, misspecification corrections or whole-cell exchanges. It does not prove the observed exchange failures are unavoidable.\n\n"
    text += "The actual replay state is unknown; these Bayes-optimality conclusions cannot be transferred to biological replay without the correct-model assumption. No accuracy safeguards were weakened, no method was selected, and no validated remedy is claimed.\n"
    (output_dir / "report.md").write_text(text)
    for path, sha in inputs.items():
        if file_sha256(path) != sha:
            raise ValueError(f"input changed during audit: {path}")
    manifest = {
        **build_script_provenance(),
        "source_dir": str(source),
        "source_reference": str(reference_dir),
        "source_audit": str(reference_audit),
        "source_bank": BANK,
        "input_file_sha256": inputs,
        "toy": toy_metadata,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "numerical_audit_passed": all(checks.values()),
        "native_data_decoded": False,
        "heldout_banks_decoded": False,
        "replay_decoded": False,
        "external_validation": False,
        "validated_remedy": False,
        "independent_predictive_diagnostic_validated": False,
        "output_sha256": {p.name: file_sha256(p) for p in output_dir.iterdir() if p.is_file()},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not all(checks.values()):
        raise ValueError("risk diagnostic numerical audit failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("reference-dir", "reference-audit", "output-dir"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args()
    run(args.reference_dir, args.reference_audit, args.output_dir)
