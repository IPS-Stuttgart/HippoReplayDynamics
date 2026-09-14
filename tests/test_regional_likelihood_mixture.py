"""Synthetic truth and non-vacuous gates for regional mixture estimation."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, softmax

from scripts.measure_regional_likelihood_mixture import (
    PREVALENCES,
    REAL,
    SESSIONS,
    TRUTH,
    fit_mixture,
    panel_weights,
    regional_log_bf,
    summarize,
    measure,
)
from scripts.audit_regional_likelihood_mixture import (
    audit,
    audit_tables,
    digest,
    independent_fit,
    independent_likelihood,
)


def test_known_binary_mixture_recovers_prevalence_not_prior():
    # P(symbol | Home)=.8, P(symbol | outside)=.2, true mixture p=.3.
    b = np.log([4.0, 0.25])
    fit = fit_mixture(b, [0.38, 0.62])
    assert fit["estimate"] == pytest.approx(0.3, abs=1e-10)
    assert fit["fit_status"] == "interior"
    raw = np.dot([0.38, 0.62], expit(b + np.log(0.03 / 0.97)))
    assert abs(raw - 0.3) > 0.2


def test_full_likelihood_differs_from_mean_posterior():
    b = np.log([9.0, 1 / 9])
    assert fit_mixture(b, [0.66, 0.34])["estimate"] == pytest.approx(0.7)
    assert np.dot([0.66, 0.34], expit(b)) == pytest.approx(0.628)


@pytest.mark.parametrize("value,status,estimate", [(2, "boundary_one", 1), (-2, "boundary_zero", 0)])
def test_boundaries_are_explicit(value, status, estimate):
    out = fit_mixture(np.full(100, value))
    assert out["fit_status"] == status
    assert out["estimate"] == estimate
    assert 0 <= out["profile_low"] <= out["profile_high"] <= 1


def test_uninformative_is_not_artificial_agreement():
    out = fit_mixture(np.zeros(30))
    assert np.isnan(out["estimate"])
    assert out["fit_status"] == "nonidentified"
    assert out["profile_width"] == 1


def test_extreme_likelihoods_and_zero_weight():
    assert fit_mixture([1000, -1000], [0.3, 0.7])["estimate"] == pytest.approx(0.3)
    assert fit_mixture([0, 1000], [1, 0])["fit_status"] == "nonidentified"


@pytest.mark.parametrize("b,w", [([], None), ([np.nan], None), ([0], [0]), ([0, 1], [-1, 2]), ([0], [1, 2])])
def test_invalid_fit_inputs_fail(b, w):
    with pytest.raises(ValueError):
        fit_mixture(b, w)


def test_regional_poisson_matches_uniform_prior_decoder():
    rng = np.random.default_rng(4)
    counts, rates = rng.poisson(2, (13, 7)), rng.uniform(0.01, 30, (7, 9))
    near = np.arange(9) < 2
    ll = counts @ np.log(rates) - 0.02 * rates.sum(axis=0)
    expected = softmax(ll, axis=1)[:, near].sum(axis=1)
    actual = expit(regional_log_bf(counts, rates, near) + np.log(2 / 7))
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_conditional_silence_has_no_information_but_poisson_silence_can():
    rates = np.array([[100.0, 1.0], [50.0, 2.0]])
    n, near = np.zeros((3, 2)), np.array([True, False])
    np.testing.assert_allclose(regional_log_bf(n, rates, near, True), 0)
    np.testing.assert_allclose(regional_log_bf(n, rates, near), -0.02 * 147)


def test_conditional_is_invariant_to_whole_map_gain():
    rates = np.array([[10.0, 1.0, 4.0], [1.0, 7.0, 5.0]])
    n, near = np.array([[2, 3], [0, 0]]), np.array([True, False, False])
    np.testing.assert_allclose(regional_log_bf(n, rates, near, True), regional_log_bf(n, 100 * rates, near, True), atol=1e-12)


def test_panel_weights_do_not_enter_estimator_as_labels():
    labels = np.array([False, False, False, True])
    w = panel_weights(labels, 0.7)
    assert w.sum() == pytest.approx(1)
    assert w[labels].sum() == pytest.approx(0.7)
    np.testing.assert_allclose(panel_weights(labels), 0.25)
    with pytest.raises(ValueError):
        panel_weights(np.ones(4, bool), 0.7)


def complete_fixture():
    rows = []
    for session in SESSIONS:
        for obs in ("poisson", "conditional"):
            for source in REAL + TRUTH:
                encodings = ("early_run", "full_run") if source in REAL else ("early_run",)
                for enc in encodings:
                    panels = [("real", "real", np.nan)] if source in REAL else [(f"p{p:.2f}", "fixed_prevalence", p) for p in PREVALENCES]
                    if source == "run_q4":
                        panels.append(("natural", "natural", 0.1))
                    for panel, kind, truth in panels:
                        for side in ("high", "low"):
                            raw = (0.3 if side == "high" else 0.1) if source in REAL else 0.02
                            estimate = (0.21 if side == "high" else 0.19) if source in REAL else truth
                            rows.append(
                                dict(
                                    session=session,
                                    animal=session.split("/")[0],
                                    observation=obs,
                                    source=source,
                                    encoding=enc,
                                    side=side,
                                    panel=panel,
                                    panel_type=kind,
                                    true_prevalence=truth,
                                    estimate=estimate,
                                    raw_posterior_mass=raw,
                                    fit_status="interior",
                                )
                            )
    return pd.DataFrame(rows)


def test_complete_good_fixture_passes_development_not_external_claim():
    g = summarize(complete_fixture())["gate_summary"].set_index("gate")
    assert g.passed.all()
    assert "external" in g.loc["development_numerical_screen", "detail"]


@pytest.mark.parametrize("damage", ["missing", "duplicate", "nonidentified", "boundary", "truth_harm", "one_rat_harm"])
def test_failure_modes_fail_gates(damage):
    d = complete_fixture()
    if damage == "missing":
        d = d.iloc[1:]
        with pytest.raises(ValueError):
            summarize(d)
        return
    if damage == "duplicate":
        d = pd.concat([d, d[d.source.eq("run_q4")].iloc[[0]]], ignore_index=True)
    if damage == "nonidentified":
        d.loc[0, ["estimate", "fit_status"]] = [np.nan, "nonidentified"]
    if damage == "boundary":
        m = d.source.isin(REAL)
        d.loc[m, ["estimate", "fit_status"]] = [0, "boundary_zero"]
    if damage in ("truth_harm", "one_rat_harm"):
        m = d.source.eq("test_conditional_shared_assembly")
        if damage == "one_rat_harm":
            m &= d.animal.eq("Rat4")
        d.loc[m, "estimate"] = 1
    assert not summarize(d)["gate_summary"].set_index("gate").loc["development_numerical_screen", "passed"]


def test_rat_weighting_not_session_weighting():
    d = complete_fixture()
    m = d.source.eq("run_q4") & d.animal.eq("Rat1")
    d.loc[m, "estimate"] += 0.3
    table = summarize(d)["truth_summary"]
    row = table[(table.source == "run_q4") & (table.observation == "poisson") & (table.panel_type == "fixed_prevalence")].iloc[0]
    assert row.mixture_error == pytest.approx(0.1)


def test_profile_contains_estimate_and_shrinks_with_information():
    a = fit_mixture(np.tile(np.log([4.0, 0.25]), 10))
    b = fit_mixture(np.tile(np.log([4.0, 0.25]), 1000))
    assert a["profile_low"] < 0.5 < a["profile_high"]
    assert b["profile_width"] < a["profile_width"]


def test_independent_optimizer_and_full_probability_reconstruction():
    b, w = np.log([4.0, 0.25]), np.array([0.38, 0.62])
    assert independent_fit(b, w)[0] == pytest.approx(0.3, abs=2e-7)
    rng = np.random.default_rng(5)
    rates, counts, mask = rng.uniform(0.1, 30, (8, 10)), rng.poisson(2, (11, 8)), np.arange(10) < 3
    for conditional in (False, True):
        bf, _ = independent_likelihood(counts, rates, mask, "conditional" if conditional else "poisson")
        np.testing.assert_allclose(bf, regional_log_bf(counts, rates, mask, conditional), atol=1e-10)


def test_truth_table_tamper_and_gate_tamper_rejected(tmp_path):
    d = complete_fixture()
    for name, table in summarize(d).items():
        table.to_csv(tmp_path / f"{name}.csv", index=False)
    assert audit_tables(tmp_path, d)["development_numerical_screen"]
    p = tmp_path / "truth_summary.csv"
    changed = pd.read_csv(p)
    changed.loc[0, "mixture_error"] = 0.123
    changed.to_csv(p, index=False)
    with pytest.raises(AssertionError):
        audit_tables(tmp_path, d)
    summarize(d)["truth_summary"].to_csv(p, index=False)
    p = tmp_path / "gate_summary.csv"
    changed = pd.read_csv(p)
    changed.loc[0, "passed"] = False
    changed.to_csv(p, index=False)
    with pytest.raises(AssertionError):
        audit_tables(tmp_path, d)


def test_full_artifact_roundtrip_and_rehashed_tampering(tmp_path):
    source, output = tmp_path / "source", tmp_path / "result"
    source.mkdir()
    rng = np.random.default_rng(823)
    for session in SESSIONS:
        folder = source / session.replace("/", "_")
        folder.mkdir()
        rates, near = rng.uniform(0.01, 40, (6, 8)), np.arange(8) < 2
        groups = [np.array([0, 1, 2]), np.array([3, 4, 5])]
        np.savez_compressed(folder / "encoding.npz", early_run=rates, full_run=rates * 1.1, near=near, high_indices=groups[0], low_indices=groups[1])
        for name in REAL + TRUTH:
            labels = np.arange(80) < 30
            states = np.where(labels, 0, 6)
            counts = rng.poisson(0.02 * rates[:, states].T)
            scores = {}
            for encoding, r in (("early_run", rates), ("full_run", 1.1 * rates)):
                scores[f"{encoding}_scores"] = np.column_stack([softmax(counts[:, g] @ np.log(r[g]) - 0.02 * r[g].sum(axis=0), axis=1)[:, near].sum(axis=1) for g in groups])
            np.savez_compressed(folder / f"{name}.npz", counts=counts, labels=labels, **scores)
        (folder / "outputs.json").write_text(json.dumps({p.name: digest(p) for p in folder.glob("*.npz")}))
    (source / "manifest.json").write_text(json.dumps({"synthetic_test_fixture": True}))
    source_audit = tmp_path / "synthetic_source_audit.json"
    source_audit.write_text(
        json.dumps({"status": "pass", "input_file_sha256": {"source": digest(source / "manifest.json")}, "results": [dict(session=s, status="pass") for s in SESSIONS]})
    )
    measure(SimpleNamespace(source_dir=source, source_audit=source_audit, output_dir=output))
    assert audit(output)["status"] == "pass"
    table = pd.read_csv(output / "truth_summary.csv")
    table.loc[0, "mixture_error"] += 0.1
    table.to_csv(output / "truth_summary.csv", index=False)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["output_sha256"]["truth_summary.csv"] = digest(output / "truth_summary.csv")
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        audit(output)
