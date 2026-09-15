import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from scripts import audit_content_bayes_risk as m


def fixture(session="Rat1/Open1"):
    rng = np.random.default_rng(440)
    enc = dict(
        early_run=rng.lognormal(2, 1, (6, 5)),
        near=np.array([True, False, False, False, False]),
        grid_cm=np.column_stack((np.arange(5) * 10, np.zeros(5))),
        high_indices=np.array([0, 1, 2]),
        low_indices=np.array([2, 3, 4]),
    )
    seed = int.from_bytes(m.hashlib.sha256(f"20260914|regional_prevalence|{session}|{m.BANK}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    position = np.r_[rng.choice(np.flatnonzero(~enc["near"]), 2000), rng.choice(np.flatnonzero(enc["near"]), 2000)]
    counts = rng.poisson(m.DT * enc["early_run"][:, position].T)
    bank = dict(counts=counts, position_ids=position, truth_cm=enc["grid_cm"][position], labels=enc["near"][position], drawn_totals=counts.sum(axis=1))
    bank["early_run_scores"] = np.column_stack([m.independent_home(counts, enc["early_run"], enc["near"], enc[f"{s}_indices"]) for s in ("high", "low")])
    return enc, bank


def test_full_poisson_reconstruction_and_generator_tampering():
    enc, bank = fixture()
    m.verify_generator(enc, bank, "Rat1/Open1")
    for field in ("counts", "position_ids", "labels", "drawn_totals"):
        corrupt = deepcopy(bank)
        corrupt[field].flat[0] = corrupt[field].flat[0] + 1 if field != "labels" else not corrupt[field].flat[0]
        with pytest.raises(AssertionError):
            m.verify_generator(enc, corrupt, "Rat1/Open1")
    corrupt = deepcopy(enc)
    corrupt["early_run"] *= 4
    with pytest.raises(AssertionError):
        m.verify_generator(corrupt, bank, "Rat1/Open1")
    risk, pair, frame = m.evaluate_session(enc, bank, "Rat1/Open1")
    assert len(risk) == 6 and len(pair) == 3 and len(frame) == 4000
    assert risk.independent_posterior_max_error.max() < 1e-12
    assert (risk.loc[risk.method.ne("baseline"), "expected_brier_cost_rao_blackwell"] > 0).all()
    assert risk.loc[risk.method.eq("baseline"), "expected_brier_cost_rao_blackwell"].eq(0).all()
    mixture = pair.set_index("method").loc["uniform_mixture_half"]
    assert mixture.rms_disagreement_reduction_fraction == pytest.approx(0.5)


def test_prior_density_ratio_restores_uniform_not_balanced_bayes_risk():
    means = np.array([0.2, 0.7, 0.1, 0.4, 0.8])
    n = np.arange(25)
    likelihood = poisson.pmf(n[:, None], means)
    p_uniform = likelihood.mean(axis=1)
    p_balanced = 0.5 * likelihood[:, 0] + 0.5 * likelihood[:, 1:].mean(axis=1)
    q = 0.2 * likelihood[:, 0] / p_uniform
    ratio = 2 / (q / 0.2 + (1 - q) / 0.8)
    np.testing.assert_allclose(p_balanced * ratio, p_uniform, atol=1e-15)
    assert np.sum(p_balanced * q) != pytest.approx(0.2, abs=1e-3)
    assert np.sum(p_balanced * ratio * q) == pytest.approx(0.2, abs=1e-12)


def test_conditional_identity_is_exact_but_finite_sample_cross_term_need_not_vanish():
    q = np.array([0.1, 0.2, 0.6, 0.8])
    a = 0.5 * q + 0.1
    result = m.risks(q, a, np.array([0, 0, 1, 1]), 0.2)
    assert result["max_conditional_identity_error"] < 1e-14
    assert abs(result["cross_term_mean"]) > 0.001
    assert result["observed_brier_change"] == pytest.approx(result["squared_change_importance"] + result["cross_term_mean"])
    assert result["changed_brier_rao_blackwell"] - result["baseline_brier_rao_blackwell"] == pytest.approx(result["expected_brier_cost_rao_blackwell"])


@pytest.mark.parametrize("bad_prior", [0, 1, -0.1, np.nan])
def test_invalid_priors_fail(bad_prior):
    with pytest.raises(ValueError):
        m.risks(np.array([0.1, 0.2, 0.3, 0.4]), np.ones(4) / 2, np.array([0, 0, 1, 1]), bad_prior)


def test_l2_tradeoff_bound_and_information_positive_control():
    rng = np.random.default_rng(110)
    q = rng.uniform(0, 1, (100, 2))
    labels = np.tile([0, 1], 50)
    for alpha in (0, 0.1, 0.5, 1):
        a = (1 - alpha) * q + alpha * 0.2
        result = m.agreement_bound(q, a, labels, 0.2)
        assert result["rms_disagreement_reduction_fraction"] == pytest.approx(alpha)
        assert result["squared_adjustment_high"] + result["squared_adjustment_low"] >= result["minimum_summed_brier_cost_for_observed_rms_reduction"] - 1e-12
    toy, meta = m.toy_check()
    assert meta["omitted_probability_upper_bound"] < 1e-12
    toy = toy.set_index("method")
    assert toy.loc["uniform_mixture_half", "brier_change_0"] > 0
    assert toy.loc["shared_full_observation", "brier_change_0"] < 0
    assert toy.loc["shared_full_observation", "rms_disagreement"] == 0
    assert toy.loc["shared_full_observation", "uses_additional_observations"]


def test_incorrect_model_can_improve_without_contradicting_identity():
    true_q = np.array([0.2, 0.8])
    wrong_q = np.array([0.9, 0.1])
    true_risk = true_q * (1 - true_q)
    wrong_risk = true_q * (1 - wrong_q) ** 2 + (1 - true_q) * wrong_q**2
    assert np.all(true_risk < wrong_risk)
    np.testing.assert_allclose(wrong_risk - true_risk, (wrong_q - true_q) ** 2)


def test_rare_count_change_still_costs_risk_under_full_poisson_support():
    n = 8
    likelihood = poisson.pmf(n, [0.2, 0.7])
    probability = likelihood.mean()
    q = likelihood[1] / likelihood.sum()
    a = q - 0.2
    baseline = q * (1 - q)
    changed = q * (1 - a) ** 2 + (1 - q) * a**2
    assert probability > 0 and probability * (changed - baseline) > 0
    assert probability * (changed - baseline) == pytest.approx(probability * 0.2**2)
    assert (1 - probability) ** 4000 > 0.99


def test_end_to_end_all_pairs_preserve_nonvalidation_boundary(tmp_path):
    source, reference, output = tmp_path / "source", tmp_path / "reference", tmp_path / "output"
    reference.mkdir()
    inputs = {}
    for session in m.SESSIONS:
        folder = source / session.replace("/", "_")
        folder.mkdir(parents=True)
        enc, bank = fixture(session)
        np.savez(folder / "encoding.npz", **enc)
        np.savez(folder / f"{m.BANK}.npz", **bank)
        inputs.update({str(p): m.file_sha256(p) for p in folder.iterdir()})
    # No replay, Q3/Q4 or test-bank files exist in this fixture.
    manifest = reference / "manifest.json"
    manifest.write_text(json.dumps(dict(source_dir=str(source), input_file_sha256=inputs, output_sha256={})))
    audit = reference / "audit.json"
    audit.write_text(json.dumps(dict(status="pass", manifest_sha256=m.file_sha256(manifest))))
    m.run(reference, audit, output)
    record = json.loads((output / "manifest.json").read_text())
    assert record["numerical_audit_passed"]
    for field in ("validated_remedy", "external_validation", "native_data_decoded", "heldout_banks_decoded", "replay_decoded", "independent_predictive_diagnostic_validated"):
        assert not record[field]
    assert len(pd.read_csv(output / "accuracy_costs.csv")) == 24
    assert pd.read_csv(output / "gates.csv").passed.all()
    for name, sha in record["output_sha256"].items():
        assert m.file_sha256(output / name) == sha
    changed = source / "Rat1_Open1" / "encoding.npz"
    changed.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="changed source"):
        m.run(reference, audit, tmp_path / "bad_output")
