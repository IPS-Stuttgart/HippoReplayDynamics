from copy import deepcopy
from itertools import combinations, product
import json
from math import comb

import numpy as np
import pandas as pd
import pytest

from scripts import exact_small_reserve_content as m
from scripts import audit_exact_small_reserve_content as a


def fixture():
    rates = np.array([[20, 1], [1, 1], [5, 1], [50, 1], [50, 1], [1, 50], [1, 50], [1, 1]], float)
    labels = np.repeat(np.tile([False, True], 20), 2)
    counts = np.zeros((len(labels), len(rates)), int)
    counts[labels, 0] = 1
    counts[labels, 3:5] = 3
    counts[~labels, 5:7] = 3
    grid = np.array([[0.0, 0.0], [20.0, 0.0]])
    enc = dict(early_run=rates, grid_cm=grid, near=np.array([True, False]), high_indices=np.array([0, 1]), low_indices=np.array([1, 2]), cell_ids=np.arange(len(rates)))
    bank = dict(counts=counts, labels=labels, truth_cm=grid[1 - labels.astype(int)], parent_ids=np.repeat(np.arange(40), 2))
    return enc, bank


def inputs_fixture(tmp_path):
    source, ref, out = tmp_path / "source", tmp_path / "reference", tmp_path / "out"
    ref.mkdir()
    inputs, splits = {}, {}
    for session in m.base.SESSIONS:
        enc, bank = fixture()
        folder = source / session.replace("/", "_")
        folder.mkdir(parents=True)
        np.savez(folder / "encoding.npz", **enc)
        for s in m.SOURCES:
            np.savez(folder / f"{s}.npz", **bank)
        splits[session] = {s: m.robust.grouped_split(bank, session, s) for s in m.SOURCES}
        inputs.update({str(p): m.file_sha256(p) for p in folder.iterdir()})
    frozen = ref / "frozen_assignments.json"
    frozen.write_text(json.dumps(dict(splits=splits)))
    manifest = ref / "manifest.json"
    manifest.write_text(json.dumps(dict(input_file_sha256=inputs, output_sha256={frozen.name: m.file_sha256(frozen)})))
    audit = ref / "audit.json"
    audit.write_text(json.dumps(dict(status="pass", manifest_sha256=m.file_sha256(manifest))))
    return source, ref, audit, out


def test_factorized_enumeration_matches_every_bruteforce_disjoint_pair():
    enc, bank = fixture()
    banks = {s: bank for s in m.SOURCES}
    choice, states, pairs = m.select(enc, banks, "synthetic")
    original, reserve = choice["original"], choice["reserve"]
    assert len(states) == 2 * (comb(5, 1) + comb(5, 2))
    assert sum(2 * (comb(n, 1) + comb(n, 2)) for n in (34, 36, 30, 21)) == 3914
    r0, j0 = m.previous.measurements(enc, banks, original)
    expected = set()
    for b in (1, 2):
        for high, low in product(combinations(reserve, b), repeat=2):
            if set(high) & set(low):
                continue
            pair = dict(high=sorted(original["high"] + list(high)), low=sorted(original["low"] + list(low)))
            r, j = m.previous.measurements(enc, banks, pair)
            if np.all(r <= r0 + 1e-10) and j < j0 - 1e-10:
                expected.add((b, high, low))
    indexed = states.set_index("state_index").to_dict("index")
    actual = {(row.budget, tuple(m.cells(indexed[row.high_state])), tuple(m.cells(indexed[row.low_state]))) for row in pairs[pairs.admissible].itertuples()}
    assert expected and actual == expected
    assert choice["chosen"] is not None
    assert not set(choice["added"]["high"]) & set(choice["added"]["low"])
    assert set(choice["final_pair"]["high"]) & set(choice["final_pair"]["low"]) == set(original["high"]) & set(original["low"])


def test_full_constants_and_distinct_sources():
    enc, bank = fixture()
    rng = np.random.default_rng(18)
    banks = {s: {**bank, "counts": rng.poisson(1.5, bank["counts"].shape)} for s in m.SOURCES}
    original = {s: enc[f"{s}_indices"].tolist() for s in m.SIDES}
    scorer = m.SideScorer(enc, banks, original)
    for side in m.SIDES:
        for added in ([], [3], [4, 6]):
            r, means = scorer(side, added)
            expected, expected_means = a.full_side(enc, banks, sorted(original[side] + added))
            np.testing.assert_allclose(r, expected, atol=1e-10)
            np.testing.assert_allclose(means, expected_means, atol=1e-12)


def test_non_improving_safe_subsets_are_not_a_remedy():
    enc, bank = fixture()
    enc["early_run"][:] = 1
    choice, states, pairs = m.select(enc, {s: bank for s in m.SOURCES}, "constant")
    assert states.safe.all()
    assert not pairs.admissible.any()
    assert choice["chosen"] is None and choice["budget"] == 0
    assert choice["final_pair"] == choice["original"]
    assert pairs.disjoint.any() and (~pairs.disjoint).any()


def test_ties_prefer_budget_then_cell_indices():
    table = pd.DataFrame(
        [
            dict(state_index=0, budget=1, cell_1=9, cell_2=-1),
            dict(state_index=1, budget=1, cell_1=10, cell_2=-1),
            dict(state_index=2, budget=2, cell_1=3, cell_2=4),
            dict(state_index=3, budget=2, cell_1=5, cell_2=6),
        ]
    )
    pairs = pd.DataFrame(
        [dict(high_state=2, low_state=3, budget=2, objective=0.1, admissible=True), dict(high_state=0, low_state=1, budget=1, objective=0.1 + 5e-13, admissible=True)]
    )
    assert m.choose(table, pairs)["budget"] == 1
    assert m.choose(table, pairs.iloc[::-1])["high_state"] == 0


def test_validation_counts_cannot_change_selection(tmp_path):
    source, ref, _, _ = inputs_fixture(tmp_path)
    s = m.base.SESSIONS[0]
    split = json.loads((ref / "frozen_assignments.json").read_text())["splits"][s]
    folder = source / s.replace("/", "_")
    choices, frames = [], []
    for i in (0, 1):
        out = tmp_path / str(i)
        out.mkdir()
        if i:
            for name in m.SOURCES:
                bank = deepcopy(m.base.read_npz(folder / f"{name}.npz"))
                bank["counts"][split[name]["validation"]] += 10000
                np.savez(folder / f"{name}.npz", **bank)
        choice = m.worker((source, s, split, out))
        choice.pop("tables_sha256")
        choices.append(choice)
        frames.append(pd.read_csv(out / "Rat1_Open1_side_states.csv").drop(columns="runtime_s"))
    assert choices[0] == choices[1]
    pd.testing.assert_frame_equal(*frames)


def test_complete_freeze_and_independent_serialized_audit(tmp_path, monkeypatch):
    source, ref, audit, out = inputs_fixture(tmp_path)
    original = m.robust.validation

    def validate(enc, banks, choice):
        assert set(json.loads((out / "frozen_assignments.json").read_text())["choices"]) == set(m.base.SESSIONS)
        return original(enc, banks, choice)

    monkeypatch.setattr(m.robust, "validation", validate)
    m.run(source, ref, audit, out, workers=1)
    result = a.audit(source, out, tmp_path / "verified")
    assert result["status"] == "pass"
    assert result["population_subsets_reconstructed"] == 120
    assert result["population_risks_reconstructed"] == 1440
    assert result["disjoint_safe_pairs_full_likelihood_reconstructed"] > 0
    assert result["validation_risks_reconstructed"] == 96
    assert result["ready_for_truth_preflight"]
    assert not result["validated_remedy"] and not result["external_validation"]


@pytest.mark.parametrize("damage", ["missing_subset", "native_mean", "pair_objective", "validation"])
def test_audit_rejects_rehashed_semantic_corruption(tmp_path, damage):
    source, ref, audit, out = inputs_fixture(tmp_path)
    m.run(source, ref, audit, out, workers=1)
    if damage == "validation":
        path = out / "internal_validation.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "targeted"] += 1
        frame.to_csv(path, index=False)
    else:
        kind = "safe_pair_candidates" if damage == "pair_objective" else "side_states"
        path = out / f"Rat1_Open1_{kind}.csv"
        frame = pd.read_csv(path)
        if damage == "missing_subset":
            frame = frame.iloc[:-1]
        else:
            frame.loc[0, "objective" if damage == "pair_objective" else "native_mean_home"] += 0.1
        frame.to_csv(path, index=False)
        frozen_path = out / "frozen_assignments.json"
        frozen = json.loads(frozen_path.read_text())
        choice = frozen["choices"]["Rat1/Open1"]
        choice["tables_sha256"][kind] = m.file_sha256(path)
        (out / "Rat1_Open1_choice.json").write_text(json.dumps(choice))
        frozen_path.write_text(json.dumps(frozen))
    path = out / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["output_sha256"] = {p.name: m.file_sha256(p) for p in out.iterdir() if p.name != "manifest.json"}
    manifest["frozen_assignments_sha256"] = m.file_sha256(out / "frozen_assignments.json")
    path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, AssertionError)):
        a.audit(source, out, tmp_path / "verified")
