import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from scripts import guarded_reserve_content as m
from scripts import audit_guarded_reserve_content as audit


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


def test_every_single_addition_and_disjointness_constraints():
    enc, bank = fixture()
    banks = {s: bank for s in m.SOURCES}
    table, risks, j = m.single_additions(enc, banks)
    assert len(table) == 6
    for row in table.itertuples(index=False):
        pair = {s: list(enc[f"{s}_indices"]) + ([row.cell] if s == row.side else []) for s in m.base.SIDES}
        _, _, actual, _, objective = m.robust.proof.state_for(pair, enc, bank)
        np.testing.assert_allclose([getattr(row, r) for r in m.RISK_COLUMNS], np.tile(actual, 3), atol=1e-12)
        assert row.objective == pytest.approx(objective)
    proposals = list(m.allocation_proposals(table, risks, j, m.base.reserve_indices(enc)))
    assert m.OPTIONS == dict(time_limit=20.0, node_limit=64, mip_rel_gap=0.001)
    for p in proposals:
        if p["feasible"]:
            pair = m.augmented(enc, p["added"])
            assert set(pair["high"]) & set(pair["low"]) == {1}
            assert len(pair["high"]) == len(pair["low"]) == 3
    for family in ("guarded_finite", "objective_finite"):
        sets = [(tuple(p["added"]["high"]), tuple(p["added"]["low"])) for p in proposals if p["family"] == family and p["feasible"]]
        assert len(set(sets)) == len(sets)
    with pytest.raises(ValueError, match="incomplete"):
        list(m.allocation_proposals(table.iloc[:-1], risks, j, m.base.reserve_indices(enc)))
    for invalid in ({"high": [3], "low": [3]}, {"high": [0], "low": [4]}, {"high": [], "low": []}):
        with pytest.raises(ValueError):
            m.augmented(enc, invalid)


def test_positive_exact_selection_and_accuracy_rejection(tmp_path):
    enc, bank = fixture()
    result = m.select(enc, {s: bank for s in m.SOURCES}, "synthetic", tmp_path)
    assert result["chosen"] is not None and result["final_j"] < result["baseline_j"]
    candidate = next(c for c in result["candidates"] if c["name"] == result["chosen"])
    assert candidate["admissible"] and candidate["failed_risks"] == 0
    assert np.all(np.array(candidate["risks"]) <= np.array(result["baseline_risks"]) + 1e-10)
    assert len([c for c in result["candidates"] if c["family"] == "random"]) == 20
    rejected = [{**c, "admissible": False} for c in result["candidates"]]
    assert m.winner(rejected) is None


def test_validation_counts_do_not_change_selection(tmp_path):
    source = tmp_path / "source"
    session = "Rat1/Open1"
    folder = source / "Rat1_Open1"
    folder.mkdir(parents=True)
    enc, bank = fixture()
    np.savez(folder / "encoding.npz", **enc)
    splits = {s: m.robust.grouped_split(bank, session, s) for s in m.SOURCES}
    results = []
    for run in (0, 1):
        for s in m.SOURCES:
            current = deepcopy(bank)
            if run:
                current["counts"][splits[s]["validation"]] += 10000
            np.savez(folder / f"{s}.npz", **current)
        out = tmp_path / str(run)
        out.mkdir()
        results.append(m.worker((source, session, splits, out)))
    for key in ("chosen", "final_pair", "added", "baseline_risks", "baseline_j", "final_j", "single_addition_sha256"):
        assert results[0][key] == results[1][key]


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


def test_end_to_end_all_choices_freeze_before_validation(tmp_path, monkeypatch):
    source, reference, audit_path, out = inputs_fixture(tmp_path)
    original_validation = m.robust.validation

    def check_freeze(enc, banks, choice):
        assert len(json.loads((out / "frozen_assignments.json").read_text())["choices"]) == 4
        return original_validation(enc, banks, choice)

    monkeypatch.setattr(m.robust, "validation", check_freeze)
    m.run(source, reference, audit_path, out, workers=1)
    gates = pd.read_csv(out / "gates.csv")
    assert gates.passed.all()
    assert len(pd.read_csv(out / "internal_validation.csv")) == 96
    assert len(pd.read_csv(out / "random_native_validation.csv")) == 80
    record = json.loads((out / "manifest.json").read_text())
    for field in ("q4_scored", "test_banks_scored", "replay_scored", "external_validation", "validated_remedy"):
        assert record[field] is False
    choices = json.loads((out / "frozen_assignments.json").read_text())["choices"]
    for c in choices.values():
        c["chosen"] = None
    failed = m.gate_values(pd.read_csv(out / "internal_validation.csv"), choices)
    assert not failed["all_pairs_augmented"] and not failed["ready_for_truth_preflight"]


def test_independent_full_reconstruction(tmp_path):
    source, reference, audit_path, out = inputs_fixture(tmp_path)
    m.run(source, reference, audit_path, out, workers=1)
    result = audit.audit(source, out, tmp_path / "audit")
    assert result["status"] == "pass" and result["ready_for_truth_preflight"]
    assert result["single_additions_reconstructed"] == 24
    assert result["validation_risks_reconstructed"] == 96
    assert result["random_validation_objectives_reconstructed"] == 80
    assert not result["validated_remedy"] and not result["external_validation"]


@pytest.mark.parametrize("tamper", ["risk", "shared_cell", "single_addition", "validation", "random", "empty_gates", "freeze_time"])
def test_rehashed_tampering_fails_independent_audit(tmp_path, tamper):
    source, reference, audit_path, out = inputs_fixture(tmp_path)
    m.run(source, reference, audit_path, out, workers=1)
    frozen_path = out / "frozen_assignments.json"
    frozen = json.loads(frozen_path.read_text())
    session = m.base.SESSIONS[0]
    choice = frozen["choices"][session]
    if tamper in ("risk", "shared_cell"):
        c = choice["candidates"][0]
        if tamper == "risk":
            c["risks"][0] += 1
        else:
            c["added"]["low"] = c["added"]["high"]
        (out / f"{session.replace('/', '_')}_choice.json").write_text(json.dumps(choice))
    elif tamper == "single_addition":
        p = out / f"{session.replace('/', '_')}_single_additions.csv"
        table = pd.read_csv(p)
        table.loc[0, "risk_0"] += 1
        table.to_csv(p, index=False)
        choice["single_addition_sha256"] = m.file_sha256(p)
        (out / f"{session.replace('/', '_')}_choice.json").write_text(json.dumps(choice))
    elif tamper == "freeze_time":
        frozen["created_at_utc"] = "2999-01-01T00:00:00+00:00"
    else:
        name, column = {"validation": ("internal_validation.csv", "targeted"), "random": ("random_native_validation.csv", "validation_j"), "empty_gates": ("gates.csv", None)}[
            tamper
        ]
        p = out / name
        table = pd.read_csv(p)
        if column:
            table.loc[0, column] += 1
        else:
            table = table.iloc[:0]
        table.to_csv(p, index=False)
    frozen_path.write_text(json.dumps(frozen))
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["frozen_assignments_sha256"] = m.file_sha256(frozen_path)
    manifest["output_sha256"] = {name: m.file_sha256(out / name) for name in manifest["output_sha256"]}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, AssertionError)):
        audit.audit(source, out, tmp_path / "audit")
