import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from scripts import robust_exchange_content as m


def fixture():
    rng = np.random.default_rng(44)
    rates = rng.lognormal(2.5, 1.5, (12, 2))
    positions = np.repeat(np.tile(np.arange(2), 20), 2)
    grid = np.array([[0.0, 0.0], [20.0, 0.0]])
    enc = dict(early_run=rates, full_run=rates, grid_cm=grid, near=np.array([True, False]), high_indices=np.arange(6), low_indices=np.arange(4, 10), cell_ids=np.arange(12))
    bank = dict(counts=rng.poisson(0.02 * rates[:, positions].T), truth_cm=grid[positions], labels=positions == 0, parent_ids=np.repeat(np.arange(40), 2))
    pair = {s: enc[f"{s}_indices"].tolist() for s in ("high", "low")}
    return enc, bank, pair


def test_group_partition_is_atomic_stratified_and_ignores_held_counts():
    _, bank, _ = fixture()
    split = m.grouped_split(bank, "Rat1/Open1", "run_q3")
    train, valid = split["train"], split["validation"]
    assert len(train) == 60 and len(valid) == 20
    assert set(train) | set(valid) == set(range(80)) and not set(train) & set(valid)
    assert not set(bank["parent_ids"][train]) & set(bank["parent_ids"][valid])
    bad = deepcopy(bank)
    bad["counts"][valid] += 10000
    assert m.grouped_split(bad, "Rat1/Open1", "run_q3") == split
    for key in ("counts", "truth_cm", "labels"):
        np.testing.assert_array_equal(m.subset(bank, train)[key], m.subset(bad, train)[key])
    bank["labels"][:] = False
    with pytest.raises(ValueError, match="support"):
        m.grouped_split(bank, "Rat1/Open1", "run_q3")


def test_risk_and_objective_gradients_match_full_finite_differences():
    enc, bank, original = fixture()
    _, gradient, risks, high, low = m.coefficients(enc, bank, original)
    base_weights = {s: np.isin(np.arange(12), original[s]).astype(float) for s in original}

    def weighted(weights):
        ll = {s: (bank["counts"] * weights[s]) @ np.log(enc["early_run"]) - 0.02 * (weights[s] @ enc["early_run"]) for s in weights}
        return m.exact.readout(ll, enc["grid_cm"], enc["near"], bank["truth_cm"], bank["labels"])

    for j, cell in enumerate(high + low):
        sign = -1 if j < len(high) else 1
        results = []
        for eps in (-1e-5, 1e-5):
            w = deepcopy(base_weights)
            w["high"][cell] += sign * eps
            w["low"][cell] -= sign * eps
            results.append(weighted(w))
        fd = (results[1]["risks"] - results[0]["risks"]) / 2e-5
        np.testing.assert_allclose(fd, risks[:, j], atol=1e-6, rtol=1e-5)
        assert (results[1]["objective"] - results[0]["objective"]) / 2e-5 == pytest.approx(gradient[j], abs=1e-7, rel=1e-5)


def test_milp_risk_constraints_and_membership_exclusion():
    grad = np.array([-0.1, 0.1, -0.1, 0.1])
    risks = np.tile([-0.1, 1.0, -0.1, 1.0], (24, 1))
    rows = list(m.proposals(grad, risks, 1.0, np.ones(24), [0, 1], [2, 3]))
    first = rows[0]
    assert first["feasible"] and first["high"] == [0] and first["low"] == [2]
    assert np.max(first["predicted_risk_changes"]) <= 0
    assert not rows[1]["feasible"]
    assert not any(r["feasible"] for r in rows[2:])


def test_linearized_gain_cannot_override_exact_accuracy_failure(monkeypatch):
    enc, bank, original = fixture()
    actual = m.state

    def degraded(enc, bank, pair):
        result = actual(enc, bank, pair)
        if pair != original:
            result["objective"] = 0.0
            result["risks"][:] = 1000.0
        return result

    monkeypatch.setattr(m, "state", degraded)
    monkeypatch.setattr(m, "proposals", lambda *a: iter([dict(budget=1, index=0, feasible=True, high=[0], low=[6])]))
    choice = m.select(enc, {s: bank for s in m.SOURCES}, "Rat1/Open1")
    assert not choice["candidates"][0]["admissible"]
    assert choice["chosen"] is None and choice["final_pair"] == original


def test_actual_selector_finds_accuracy_guarded_synthetic_exchange():
    rng = np.random.default_rng(2)
    rates = rng.lognormal(2.5, 1.5, (12, 2))
    positions = np.tile(np.arange(2), 20)
    grid = np.array([[0.0, 0.0], [20.0, 0.0]])
    enc = dict(early_run=rates, grid_cm=grid, near=np.array([True, False]), high_indices=np.arange(6), low_indices=np.arange(4, 10), cell_ids=np.arange(12))
    bank = dict(counts=rng.poisson(0.02 * rates[:, positions].T), truth_cm=grid[positions], labels=positions == 0)
    # Repeated banks exercise the numerical selector, not independent validation.
    choice = m.select(enc, {s: bank for s in m.SOURCES}, "synthetic")
    assert choice["chosen"] is not None and choice["net_exchanged"] > 0
    before = m.state(enc, bank, choice["original"])
    after = m.state(enc, bank, choice["final_pair"])
    assert after["objective"] < before["objective"] - 1e-10
    assert np.all(after["risks"] <= before["risks"] + 1e-10)
    candidate = next(c for c in choice["candidates"] if c["name"] == choice["chosen"])
    assert len(candidate["risks"]) == 24
    assert np.all(np.asarray(candidate["risks"]) <= np.asarray(choice["baseline_risks"]) + 1e-10)
    for side in ("high", "low"):
        assert len(choice["final_pair"][side]) == len(choice["original"][side])
    assert set(choice["final_pair"]["high"]) | set(choice["final_pair"]["low"]) == set(choice["original"]["high"]) | set(choice["original"]["low"])
    assert set(choice["final_pair"]["high"]) & set(choice["final_pair"]["low"]) == set(choice["original"]["high"]) & set(choice["original"]["low"])


def test_artifacts_freeze_before_validation_and_no_vacuous_success(tmp_path, monkeypatch):
    source, reference, output = tmp_path / "source", tmp_path / "reference", tmp_path / "output"
    reference.mkdir()
    inputs, expected = {}, {}
    for session in m.exact.base.SESSIONS:
        folder = source / session.replace("/", "_")
        folder.mkdir(parents=True)
        enc, bank, original = fixture()
        np.savez(folder / "encoding.npz", **enc)
        for s in m.SOURCES:
            np.savez(folder / f"{s}.npz", **bank)
            expected[(session, s)] = m.subset(bank, m.grouped_split(bank, session, s)["train"])
        inputs.update({str(p): m.file_sha256(p) for p in folder.iterdir()})
    (reference / "manifest.json").write_text(json.dumps(dict(source_dir=str(source), input_file_sha256=inputs, output_sha256={})))
    audit = reference / "independent_audit.json"
    audit.write_text(json.dumps(dict(status="pass", manifest_sha256=m.file_sha256(reference / "manifest.json"))))

    def no_move(enc, banks, session):
        assert not (output / "frozen_assignments.json").exists()
        for s in m.SOURCES:
            for key in ("counts", "labels", "truth_cm"):
                np.testing.assert_array_equal(banks[s][key], expected[(session, s)][key])
        state = m.state(enc, banks["run_q3"], original)
        return dict(session=session, original=original, final_pair=original, net_exchanged=0, chosen=None, baseline_j=state["objective"], final_j=state["objective"], candidates=[])

    validate = m.validation

    def frozen_validation(enc, banks, choice):
        record = json.loads((output / "frozen_assignments.json").read_text())
        assert len(record["choices"]) == 4
        return validate(enc, banks, choice)

    monkeypatch.setattr(m, "select", no_move)
    monkeypatch.setattr(m, "validation", frozen_validation)
    m.run(source, reference, audit, output)
    gates = pd.read_csv(output / "gates.csv").set_index("gate").passed
    assert gates.internal_accuracy_nonworsening
    assert not gates.changed_populations_in_all_rats and not gates.ready_for_truth_preflight
    assert len(pd.read_csv(output / "internal_validation.csv")) == 96
    record = json.loads((output / "manifest.json").read_text())
    assert not any(record[k] for k in ("external_validation", "validated_remedy", "replay_scored", "q4_scored"))
    for name, sha in record["output_sha256"].items():
        assert m.file_sha256(output / name) == sha
