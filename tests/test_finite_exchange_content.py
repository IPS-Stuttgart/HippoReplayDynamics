import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from scripts import finite_exchange_content as m


def fixture(seed=2, grouped=False):
    rng = np.random.default_rng(seed)
    rates = rng.lognormal(2.5, 1.5, (12, 2))
    positions = np.tile(np.arange(2), 20)
    if grouped:
        positions = np.repeat(positions, 2)
    grid = np.array([[0.0, 0.0], [20.0, 0.0]])
    enc = dict(early_run=rates, grid_cm=grid, near=np.array([True, False]), high_indices=np.arange(6), low_indices=np.arange(4, 10), cell_ids=np.arange(12))
    bank = dict(counts=rng.poisson(0.02 * rates[:, positions].T), truth_cm=grid[positions], labels=positions == 0)
    if grouped:
        bank["parent_ids"] = np.repeat(np.arange(40), 2)
    return enc, bank


def test_all_finite_costs_match_full_poisson_reconstruction():
    enc, bank = fixture()
    banks = {s: bank for s in m.SOURCES}
    table, risks, j = m.single_costs(enc, banks, "synthetic")
    check = m.verify_singles(enc, banks, "synthetic", table, risks, j)
    assert len(table) == 16 and check["numerically_reconstructed"] == 16
    assert check["full_neighborhood_independently_reconstructed"]
    assert table.admissible.any()
    with pytest.raises(ValueError, match="incomplete"):
        m.verify_singles(enc, banks, "synthetic", table.iloc[:-1], risks, j)
    corrupt = table.copy()
    corrupt.loc[0, "risk_0"] += 1
    with pytest.raises(AssertionError):
        m.verify_singles(enc, banks, "synthetic", corrupt, risks, j)


def test_joint_matching_prevents_repeated_cells_and_complete_memberships():
    rows = [dict(high=h, low=lo, objective_improvement=0.1, **{k: 0.99 for k in m.RISK_COLUMNS}) for h in (0, 1) for lo in (3, 4)]
    table = pd.DataFrame(rows)
    result = list(m.joint_proposals(table, np.ones(24), 1.0))
    first = result[0]
    assert first["feasible"] and first["high"] == [0, 1] and first["low"] == [3, 4]
    assert len({e[0] for e in first["edges"]}) == len({e[1] for e in first["edges"]}) == 2
    assert not result[1]["feasible"]
    assert all(r["status"] == "budget_unavailable" for r in result[2:])


def test_finite_selection_returns_exactly_guarded_positive_example(tmp_path):
    enc, bank = fixture()
    choice = m.select(enc, {s: bank for s in m.SOURCES}, "synthetic", tmp_path)
    assert choice["chosen"] is not None and choice["final_j"] < choice["baseline_j"]
    best = next(c for c in choice["candidates"] if c["name"] == choice["chosen"])
    assert best["admissible"] and np.all(np.asarray(best["risks"]) <= np.asarray(choice["baseline_risks"]) + 1e-10)
    assert len(choice["final_pair"]["high"]) == len(choice["original"]["high"])
    assert len(choice["final_pair"]["low"]) == len(choice["original"]["low"])
    assert choice["finite_single_audit"]["complete_neighborhood"] == 16


def test_validation_spikes_cannot_influence_worker_choices(tmp_path):
    session = "Rat1/Open1"
    folder = tmp_path / "source" / "Rat1_Open1"
    folder.mkdir(parents=True)
    enc, bank = fixture(grouped=True)
    np.savez(folder / "encoding.npz", **enc)
    splits = {s: m.robust.grouped_split(bank, session, s) for s in m.SOURCES}
    outputs = []
    for iteration in (0, 1):
        for s in m.SOURCES:
            current = deepcopy(bank)
            if iteration:
                current["counts"][splits[s]["validation"]] += 10000
            np.savez(folder / f"{s}.npz", **current)
        result_dir = tmp_path / f"output{iteration}"
        result_dir.mkdir()
        outputs.append(m.worker((tmp_path / "source", session, splits, result_dir)))
    for field in ("baseline_j", "baseline_risks", "chosen", "final_pair", "final_j", "finite_cost_sha256"):
        assert outputs[0][field] == outputs[1][field]


def test_end_to_end_reuses_fixed_partitions_and_independent_auditor(tmp_path, monkeypatch):
    source, reference, output = tmp_path / "source", tmp_path / "reference", tmp_path / "output"
    reference.mkdir()
    inputs, choices, splits = {}, {}, {}
    for session in m.robust.exact.base.SESSIONS:
        folder = source / session.replace("/", "_")
        folder.mkdir(parents=True)
        enc, bank = fixture(grouped=True)
        np.savez(folder / "encoding.npz", **enc)
        for s in m.SOURCES:
            np.savez(folder / f"{s}.npz", **bank)
        choices[session] = {}
        splits[session] = {s: m.robust.grouped_split(bank, session, s) for s in m.SOURCES}
        inputs.update({str(p): m.file_sha256(p) for p in folder.iterdir()})
    frozen = reference / "frozen_assignments.json"
    frozen.write_text(json.dumps(dict(choices=choices, splits=splits)))
    manifest = reference / "manifest.json"
    manifest.write_text(json.dumps(dict(input_file_sha256=inputs, output_sha256={"frozen_assignments.json": m.file_sha256(frozen)})))
    audit_path = reference / "audit.json"
    audit_path.write_text(json.dumps(dict(status="pass", manifest_sha256=m.file_sha256(manifest))))
    validate = m.robust.validation

    def check_freeze(enc, banks, choice):
        record = json.loads((output / "frozen_assignments.json").read_text())
        assert len(record["choices"]) == 4 and record["splits"] == splits
        return validate(enc, banks, choice)

    monkeypatch.setattr(m.robust, "validation", check_freeze)
    m.run(source, reference, audit_path, output, workers=1)
    m.audit.audit(source, output, tmp_path / "audit")
    report = json.loads((tmp_path / "audit" / "independent_audit.json").read_text())
    assert report["status"] == "pass" and not report["validated_remedy"]
    assert len(pd.read_csv(output / "internal_validation.csv")) == 96
    summary = pd.read_csv(output / "selection_summary.csv")
    assert summary.legal_single_swaps.sum() == 64
