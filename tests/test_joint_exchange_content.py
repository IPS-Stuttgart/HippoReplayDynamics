import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import joint_exchange_content as m


def fixture():
    rng = np.random.default_rng(44)
    rates = rng.lognormal(2.5, 1.5, (12, 2))
    position = np.tile(np.arange(2), 20)
    grid = np.array([[0.0], [20.0]])
    enc = dict(early_run=rates, full_run=rates * 1.07, grid_cm=grid, near=np.array([True, False]), high_indices=np.arange(6), low_indices=np.arange(4, 10), cell_ids=np.arange(12))
    bank = dict(counts=rng.poisson(0.02 * rates[:, position].T), truth_cm=grid[position], labels=position == 0, parent_ids=np.arange(40))
    old = m.exact.select_cells(enc, bank, "Rat1/Open1")
    assert old["accepted_steps"] == 0
    singles = m.coverage.inspect_session(enc, bank, old)
    assert not singles.admissible.any()
    return enc, bank, singles, old


def test_genuine_joint_move_can_pass_when_every_single_swap_fails():
    enc, bank, singles, old = fixture()
    original = old["methods"]["baseline"]
    initial = dict(objective=old["baseline_objective"], risks=np.array(old["baseline_risks"]))
    result = m.exact_candidate(enc, bank, original, [1, 2], [7, 9], initial, "joint_known")
    assert result["admissible"]
    assert result["objective"] < initial["objective"]
    _, _, risks, _, objective = m.proof.state_for(result["pair"], enc, bank)
    np.testing.assert_allclose(risks, result["risks"], atol=1e-10)
    assert objective == pytest.approx(result["objective"])
    approximate = initial["risks"] + np.array([singles.set_index(["high", "low"]).loc[[(1, 7), (2, 9)], f"change_{r}"].sum() for r in m.coverage.RISK_NAMES])
    assert np.max(np.abs(approximate - risks)) > 1e-6


def toy_table():
    table = pd.DataFrame(dict(high=[0, 0, 1, 1], low=[2, 3, 2, 3], objective_improvement=[0.2, -0.1, -0.1, 0.2]))
    for n, key in enumerate(m.coverage.RISK_NAMES):
        table[f"change_{key}"] = [0.2, 1, 1, -0.3] if n == 0 else [-0.3, 1, 1, 0.2] if n == 1 else [0, 1, 1, 0]
    return table


def test_milp_compensating_risks_and_membership_no_good():
    table = toy_table()
    records = m.propose(table, np.ones(8), 1.0)
    first = records[0]
    assert first["feasible_incumbent"] and first["high"] == [0, 1] and first["low"] == [2, 3]
    assert first["edges"] == [[0, 2], [1, 3]]
    assert np.all(np.array(first["approximate_risks"]) <= 1)
    assert records[1]["status"] == 2 and not records[1]["feasible_incumbent"]
    assert [r["status"] for r in records[2:]] == ["budget_unavailable", "budget_unavailable"]
    _, a, lower, upper = m.problem(table, np.ones(8), 1, 2, [([0, 1], [2, 3])])
    assert (a @ np.array([1, 0, 0, 1]))[-1] > upper[-1]
    assert (a @ np.array([0, 1, 1, 0]))[-1] > upper[-1]


def test_solver_limit_fractional_and_infeasible_incumbents():
    _, a, lower, upper = m.problem(toy_table(), np.ones(8), 1, 2, [])
    good = np.array([1, 0, 0, 1])
    assert np.array_equal(m.feasible_incumbent(SimpleNamespace(status=1, x=good), a, lower, upper), good)
    for status, x in [(1, None), (2, good), (1, np.ones(4) / 2), (1, np.array([1, 1, 0, 0]))]:
        assert m.feasible_incumbent(SimpleNamespace(status=status, x=x), a, lower, upper) is None


def test_exact_memberships_and_no_foreign_or_shared_cells():
    original = dict(high=[0, 1, 2], low=[2, 3, 4])
    result = m.joint_pair(original, [0, 1], [3, 4])
    assert result == dict(high=[2, 3, 4], low=[0, 1, 2])
    for high, low in [([2], [3]), ([0, 0], [3, 4]), ([0], [5]), ([0, 1], [3])]:
        with pytest.raises(ValueError):
            m.joint_pair(original, high, low)


def test_real_likelihood_selector_verification_and_tamper():
    enc, bank, singles, _ = fixture()
    choice = m.select(enc, bank, singles, "Rat1/Open1")
    assert choice["chosen"].startswith("joint_2_") and choice["net_exchanged"] == 2
    m.verify(enc, bank, singles, choice)
    bad = deepcopy(choice)
    bad["baseline_j"] += 1
    with pytest.raises(AssertionError):
        m.verify(enc, bank, singles, bad)
    bad = deepcopy(choice)
    bad["final_pair"]["high"][0] = 11
    with pytest.raises(AssertionError):
        m.verify(enc, bank, singles, bad)
    if choice["candidates"]:
        bad = deepcopy(choice)
        bad["candidates"][0]["risks"][0] += 1
        with pytest.raises(AssertionError):
            m.verify(enc, bank, singles, bad)


def test_run_only_artifact_and_source_hash_guard(tmp_path):
    enc, bank, singles, _ = fixture()
    previous, source = tmp_path / "previous", tmp_path / "source"
    previous.mkdir()
    inputs = {}
    for session in m.exact.base.SESSIONS:
        name = session.replace("/", "_")
        folder = source / name
        folder.mkdir(parents=True)
        np.savez(folder / "encoding.npz", **enc)
        np.savez(folder / "run_q3.npz", **bank)
        inputs.update({str(p): m.file_sha256(p) for p in folder.iterdir()})
        singles.to_csv(previous / f"{name}_proposals.csv", index=False)
    source_record = dict(input_file_sha256=inputs, output_sha256={p.name: m.file_sha256(p) for p in previous.iterdir()})
    (previous / "manifest.json").write_text(json.dumps(source_record))
    (previous / "independent_audit.json").write_text(json.dumps(dict(status="pass", manifest_sha256=m.file_sha256(previous / "manifest.json"))))
    output = tmp_path / "output"
    m.run(previous, output)
    record = json.loads((output / "manifest.json").read_text())
    assert not record["replay_scored"] and not record["validated_remedy"] and not record["external_validation"]
    for name, value in record["output_sha256"].items():
        assert m.file_sha256(output / name) == value
    audit = json.loads((output / "independent_audit.json").read_text())
    assert audit["status"] == "pass" and audit["manifest_sha256"] == m.file_sha256(output / "manifest.json")
    assert len(pd.read_csv(output / "summary.csv")) == 4
    path = previous / "Rat1_Open1_proposals.csv"
    path.write_text(path.read_text() + "tampered")
    with pytest.raises(ValueError, match="artifact"):
        m.run(previous, tmp_path / "bad")
