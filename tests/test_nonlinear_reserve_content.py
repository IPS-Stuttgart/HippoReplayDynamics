from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from scripts import nonlinear_reserve_content as m
from scripts import audit_nonlinear_reserve_content as verifier


def fixture():
    rates = np.array([[20, 1], [1, 1], [5, 1], [50, 1], [50, 1], [1, 50]], float)
    labels = np.repeat(np.tile([0, 1], 20), 2).astype(bool)
    counts = np.zeros((len(labels), 6), int)
    counts[labels, 0] = 1
    counts[labels, 3:5] = 3
    counts[~labels, 5] = 3
    grid = np.array([[0.0, 0.0], [20.0, 0.0]])
    enc = dict(early_run=rates, grid_cm=grid, near=np.array([True, False]), high_indices=np.array([0, 1]), low_indices=np.array([1, 2]), cell_ids=np.arange(6))
    bank = dict(counts=counts, labels=labels, truth_cm=grid[1 - labels.astype(int)], parent_ids=np.repeat(np.arange(40), 2))
    return enc, bank


def test_binary_states_match_full_poisson_and_gradients():
    enc, bank = fixture()
    loss = m.NonlinearLoss(enc, {s: bank for s in m.SOURCES})
    for added in (dict(high=[], low=[]), dict(high=[3], low=[5])):
        pair = {s: sorted(set(enc[f"{s}_indices"]) | set(added[s])) for s in m.base.SIDES}
        w = np.array([i in added[s] for s in m.base.SIDES for i in loss.reserve], float)
        actual, j, _, _ = loss(w)
        expected, expected_j = m.previous.measurements(enc, {s: bank for s in m.SOURCES}, pair)
        np.testing.assert_allclose(actual, expected, atol=1e-10)
        assert j == pytest.approx(expected_j, abs=1e-12)
    x = np.linspace(0.1, 0.7, 6)
    _, _, jac, dj = loss(x)
    for i in range(len(x)):
        step = np.eye(len(x))[i] * 1e-6
        left, lj, _, _ = loss(x - step)
        right, rj, _, _ = loss(x + step)
        np.testing.assert_allclose((right - left) / 2e-6, jac[:, i], atol=1e-6, rtol=1e-4)
        assert (rj - lj) / 2e-6 == pytest.approx(dj[i], abs=1e-7, rel=1e-4)


def test_rounding_is_integer_disjoint_balanced_and_seeded():
    for n in (3, 4, 9):
        reserve = np.arange(10, 10 + n)
        for budget in m.budgets(reserve):
            weights = np.full(2 * n, budget / n)
            for draw in range(len(m.ROUND_SCALES)):
                added = m.rounded(weights, reserve, budget, "s", "t", draw)
                assert all(len(added[s]) == budget for s in m.base.SIDES)
                assert not set(added["high"]) & set(added["low"])
                assert added == m.rounded(weights, reserve, budget, "s", "t", draw)
    enc, _ = fixture()
    for invalid in (dict(high=[3], low=[3]), dict(high=[0], low=[5]), dict(high=[], low=[])):
        with pytest.raises(ValueError):
            m.membership(enc, invalid, 1)


def test_distinct_sources_two_dimensional_gradient_and_full_constants():
    rng = np.random.default_rng(914)
    grid = np.array([(x, y) for x in range(5) for y in range(5)], float) * 8
    enc = dict(early_run=rng.uniform(0.01, 35, (12, 25)), grid_cm=grid, near=np.arange(25) < 3, high_indices=np.arange(5), low_indices=np.arange(2, 7), cell_ids=np.arange(12))
    labels = np.tile([False, True], 48)
    banks = {}
    for source in m.SOURCES:
        positions = np.where(labels, rng.integers(0, 3, 96), rng.integers(3, 25, 96))
        banks[source] = dict(labels=labels, truth_cm=grid[positions], counts=rng.poisson(0.02 * enc["early_run"][:, positions].T))
    loss = m.NonlinearLoss(enc, banks)
    weights = rng.uniform(0.1, 0.8, 2 * loss.n)
    risks, j, jac, dj = loss(weights)
    expected, ej = verifier.full_power_scores(enc, banks, loss.original, loss.reserve, weights)
    np.testing.assert_allclose(risks, expected, atol=1e-10)
    assert j == pytest.approx(ej, abs=1e-12)
    for i in range(len(weights)):
        h = np.eye(len(weights))[i] * 1e-6
        left, lj, _, _ = loss(weights - h)
        right, rj, _, _ = loss(weights + h)
        np.testing.assert_allclose((right - left) / 2e-6, jac[:, i], atol=1e-7, rtol=1e-4)
        assert (rj - lj) / 2e-6 == pytest.approx(dj[i], abs=1e-8, rel=1e-4)


def test_hard_evaluation_budget_and_exact_positive_selection(monkeypatch):
    enc, bank = fixture()
    loss = m.NonlinearLoss(enc, {s: bank for s in m.SOURCES})
    monkeypatch.setattr(m, "MAX_EVALUATIONS", 2)
    record = m.optimize(loss, np.full(6, 1 / 3), 1, "minimax")
    assert record["evaluations"] <= 2 and record["selected"]["linear_feasible"]
    monkeypatch.setattr(m, "MAX_EVALUATIONS", 120)
    choice = m.select(enc, {s: bank for s in m.SOURCES}, "synthetic")
    assert choice["chosen"] and choice["budget"] == 1
    chosen = next(c for c in choice["candidates"] if c["name"] == choice["chosen"])
    assert chosen["admissible"] and not chosen["failed_risks"]
    assert len(choice["searches"]) == 4 and len(choice["candidates"]) == 56


def inputs_fixture(tmp_path):
    source, reference, out = tmp_path / "source", tmp_path / "reference", tmp_path / "out"
    reference.mkdir()
    inputs, splits = {}, {}
    for session in m.base.SESSIONS:
        folder = source / session.replace("/", "_")
        folder.mkdir(parents=True)
        enc, bank = fixture()
        np.savez(folder / "encoding.npz", **enc)
        for s in m.SOURCES:
            np.savez(folder / f"{s}.npz", **bank)
        splits[session] = {s: m.robust.grouped_split(bank, session, s) for s in m.SOURCES}
        inputs.update({str(p): m.file_sha256(p) for p in folder.iterdir()})
    frozen = reference / "frozen_assignments.json"
    frozen.write_text(json.dumps(dict(splits=splits)))
    manifest = reference / "manifest.json"
    manifest.write_text(json.dumps(dict(input_file_sha256=inputs, output_sha256={frozen.name: m.file_sha256(frozen)})))
    audit = reference / "audit.json"
    audit.write_text(json.dumps(dict(status="pass", manifest_sha256=m.file_sha256(manifest))))
    return source, reference, audit, out


def test_no_validation_leakage(tmp_path):
    source, ref, _, _ = inputs_fixture(tmp_path)
    session = m.base.SESSIONS[0]
    split = json.loads((ref / "frozen_assignments.json").read_text())["splits"][session]
    folder = source / session.replace("/", "_")
    choices = []
    for i in (0, 1):
        out = tmp_path / str(i)
        out.mkdir()
        if i:
            for s in m.SOURCES:
                bank = deepcopy(m.base.read_npz(folder / f"{s}.npz"))
                bank["counts"][split[s]["validation"]] += 10000
                np.savez(folder / f"{s}.npz", **bank)
        choices.append(m.worker((source, session, split, out)))
    for key in ("chosen", "budget", "added", "final_pair", "baseline_risks", "final_j", "candidates"):
        assert choices[0][key] == choices[1][key]


def test_all_choices_freeze_before_validation(tmp_path, monkeypatch):
    source, ref, audit, out = inputs_fixture(tmp_path)
    validation = m.robust.validation

    def check(enc, banks, choice):
        assert len(json.loads((out / "frozen_assignments.json").read_text())["choices"]) == 4
        return validation(enc, banks, choice)

    monkeypatch.setattr(m.robust, "validation", check)
    m.run(source, ref, audit, out, workers=1)
    assert pd.read_csv(out / "gates.csv").passed.all()
    assert len(pd.read_csv(out / "internal_validation.csv")) == 96
    manifest = json.loads((out / "manifest.json").read_text())
    assert all(manifest[k] is False for k in ("q4_scored", "test_banks_scored", "replay_scored", "external_validation", "validated_remedy"))


def test_independent_fractional_and_integer_audit(tmp_path):
    source, ref, audit, out = inputs_fixture(tmp_path)
    m.run(source, ref, audit, out, workers=1)
    result = verifier.audit(source, out, tmp_path / "verification")
    assert result["status"] == "pass"
    assert result["integer_candidates_reconstructed"] == 224
    assert result["integer_risks_reconstructed"] == 224 * 24
    assert result["selected_fractional_states_reconstructed"] == 16
    assert result["validation_risks_reconstructed"] == 96
    assert result["ready_for_truth_preflight"]
    assert not result["validated_remedy"]
    assert not result["solver_optimality_certified"]


@pytest.mark.parametrize("damage", ["integer_risk", "fractional_risk", "validation"])
def test_semantic_audit_rejects_rehashed_errors(tmp_path, damage):
    source, ref, audit, out = inputs_fixture(tmp_path)
    m.run(source, ref, audit, out, workers=1)
    if damage == "validation":
        path = out / "internal_validation.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "targeted"] += 1
        frame.to_csv(path, index=False)
    else:
        frozen_path = out / "frozen_assignments.json"
        frozen = json.loads(frozen_path.read_text())
        choice = frozen["choices"][m.base.SESSIONS[0]]
        if damage == "integer_risk":
            choice["candidates"][0]["risks"][0] += 1
        else:
            choice["searches"][0]["selected"]["risks"][0] += 1
        frozen_path.write_text(json.dumps(frozen))
        (out / "Rat1_Open1_choice.json").write_text(json.dumps(choice))
    path = out / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["frozen_assignments_sha256"] = m.file_sha256(out / "frozen_assignments.json")
    manifest["output_sha256"] = {p.name: m.file_sha256(p) for p in out.iterdir() if p.name != "manifest.json"}
    path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, AssertionError)):
        verifier.audit(source, out, tmp_path / "verification")
