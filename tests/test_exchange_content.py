import json
from argparse import Namespace
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from scripts import exchange_content as m
from scripts import audit_exchange_content as a
from scripts import report_exchange_content as r


def fixture():
    rng = np.random.default_rng(1762)
    grid = np.array([[0.0, 0.0], [0.0, 20.0], [20.0, 0.0], [20.0, 20.0]])
    rates = rng.uniform(2, 60, (16, 4))
    enc = dict(
        early_run=rates,
        full_run=rates * 1.07,
        grid_cm=grid,
        near=np.array([True, True, False, False]),
        high_indices=np.arange(8),
        low_indices=np.arange(4, 12),
        cell_ids=np.arange(16),
    )
    pos = np.tile(np.arange(4), 20)
    bank = dict(counts=rng.poisson(0.02 * rates[:, pos].T), truth_cm=grid[pos], labels=enc["near"][pos], parent_ids=np.arange(len(pos)))
    return enc, bank


def test_gradient_matches_direct_covariance_and_finite_difference():
    enc, bank = fixture()
    pair = {s: enc[f"{s}_indices"].tolist() for s in m.base.SIDES}
    ll = {s: m.likelihood(bank["counts"], enc["early_run"], pair[s]) for s in m.base.SIDES}
    state = m.readout(ll, enc["grid_cm"], enc["near"], bank["truth_cm"], bank["labels"])
    gradient = m.derivative(state, bank["counts"], enc["early_run"], enc["near"], bank["labels"])
    direct = a.direct_derivative(state["probabilities"], state["homes"], state["delta"], enc, bank)
    np.testing.assert_allclose(gradient, direct, atol=1e-12, rtol=1e-10)
    for i, j in ((0, 8), (3, 11)):
        contribution = (
            bank["counts"][:, j, None] * np.log(enc["early_run"][j])
            - 0.02 * enc["early_run"][j]
            - bank["counts"][:, i, None] * np.log(enc["early_run"][i])
            + 0.02 * enc["early_run"][i]
        )
        eps = 1e-5
        values = []
        for sign in (-1, 1):
            perturbed = dict(high=ll["high"] + sign * eps * contribution, low=ll["low"] - sign * eps * contribution)
            values.append(m.readout(perturbed, enc["grid_cm"], enc["near"], bank["truth_cm"], bank["labels"])["objective"])
        assert (values[1] - values[0]) / (2 * eps) == pytest.approx(gradient[j] - gradient[i], abs=1e-8)


def test_exact_assignment_audit_and_unchanged_overlap():
    enc, bank = fixture()
    choice = m.select_cells(enc, bank, "Rat1/Open1")
    a.check_assignment(enc, bank, choice)
    m.validate(enc, choice)
    assert choice["final_objective"] <= choice["baseline_objective"] + 1e-10
    assert np.all(np.array(choice["final_risks"]) <= np.array(choice["baseline_risks"]) + 1e-10)
    assert choice["accepted_steps"] <= 10
    changed = deepcopy(choice)
    changed["methods"]["targeted"]["high"][0] = 15
    with pytest.raises(ValueError, match="union|count|shared"):
        m.validate(enc, changed)
    bad = deepcopy(choice)
    bad["trace"][0]["proposals"][0]["objective"] += 0.2
    with pytest.raises(AssertionError):
        a.check_assignment(enc, bank, bad)


def test_no_exclusive_cells_preserves_zero_budget():
    enc, bank = fixture()
    enc["low_indices"] = enc["high_indices"].copy()
    choice = m.select_cells(enc, bank, "Rat1/Open1")
    assert choice["net_exchanged"] == 0 and choice["accepted_steps"] == 0
    assert choice["stop_reason"] == "no_admissible_proposal"
    assert all(p == choice["methods"]["baseline"] for p in choice["methods"].values())
    a.check_assignment(enc, bank, choice)


def test_acceptance_flow_with_constant_risk_oracle(monkeypatch):
    enc, bank = fixture()
    real = m.readout

    def unconstrained(*args):
        value = real(*args)
        value["risks"] = np.zeros(8)
        return value

    # Tests the search state machine, not a positive scientific recovery case.
    monkeypatch.setattr(m, "readout", unconstrained)
    choice = m.select_cells(enc, bank, "Rat1/Open1")
    assert choice["accepted_steps"] > 0
    assert choice["final_objective"] < choice["baseline_objective"]
    m.validate(enc, choice)
    assert all(len(set(pair["high"]) & set(pair["low"])) == 4 for pair in choice["methods"].values())


def test_missing_truth_class_rejected_and_noncalibration_data_irrelevant():
    enc, bank = fixture()
    changed = {**bank, "labels": np.zeros(len(bank["counts"]), bool)}
    with pytest.raises(ValueError, match="parents"):
        m.select_cells(enc, changed, "Rat1/Open1")
    choice = m.select_cells(enc, bank, "Rat1/Open1")
    other = {**enc, "full_run": enc["full_run"] * 1000}
    assert m.select_cells(other, bank, "Rat1/Open1") == choice


def test_full_artifact_audit_and_rehashed_tamper(tmp_path):
    source, previous = tmp_path / "source", tmp_path / "previous"
    source.mkdir()
    previous.mkdir()
    inputs = {}
    for session in m.base.SESSIONS:
        enc, cal = fixture()
        folder = source / session.replace("/", "_")
        folder.mkdir()
        np.savez(folder / "encoding.npz", **enc)
        np.savez(folder / "run_q3.npz", **cal)
        for src in m.base.REAL + m.base.TRUTH:
            bank = {k: v.copy() for k, v in cal.items() if k != "parent_ids"}
            bank["event_ids"] = np.array([f"e{j}" for j in range(len(bank["counts"]))])
            if src in m.base.REAL:
                bank.pop("labels")
                bank["truth_cm"][:] = np.nan
            for encoding in ("early_run", "full_run") if src in m.base.REAL else ("early_run",):
                values = m.base.score_pair(bank, enc, encoding, dict(high=[], low=[]))
                bank[f"{encoding}_scores"] = np.column_stack([values["high_home"], values["low_home"]])
            np.savez(folder / f"{src}.npz", **bank)
        inputs.update({str(p): a.base.sha(p) for p in folder.iterdir()})
    (previous / "manifest.json").write_text(json.dumps(dict(source_dir=str(source), input_file_sha256=inputs, output_sha256={}, synthetic_fixture=True)))
    audit_path = tmp_path / "source_audit.json"
    audit_path.write_text(json.dumps(dict(status="pass", manifest_sha256=a.base.sha(previous / "manifest.json"), synthetic_fixture=True)))
    root = tmp_path / "measurement"
    m.measure(Namespace(result_dir=previous, audit=audit_path, output_dir=root))
    verified = a.audit(root, tmp_path / "audit.json")
    assert verified["status"] == "pass" and not verified["validated_remedy"]
    assert not any(v for k, v in verified["gates"].items() if k.startswith("beats_equal_budget_random"))
    r.report(root, tmp_path / "audit.json", tmp_path / "report")
    assert (tmp_path / "report/exchange_content.png").stat().st_size > 1000
    assert "NOT ESTABLISHED" in (tmp_path / "report/report.md").read_text()
    path = root / "truth_by_class.csv"
    table = pd.read_csv(path)
    table.loc[0, "high_error"] += 100
    table.to_csv(path, index=False)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["output_sha256"][path.name] = a.base.sha(path)
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        a.audit(root, tmp_path / "bad_audit.json")
