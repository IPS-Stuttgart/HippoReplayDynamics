from copy import deepcopy
from itertools import combinations, product
import json
from math import comb

import numpy as np
import pandas as pd
import pytest

from scripts import screened_reserve_content as m
from scripts import audit_screened_reserve_content as a


def fixture():
    rates = np.array([[20, 1], [1, 1], [5, 1]] + [[50, 1]] * 4 + [[1, 50]] * 4, float)
    labels = np.repeat(np.tile([False, True], 20), 2)
    counts = np.zeros((len(labels), len(rates)), int)
    counts[labels, 0] = 1
    counts[labels, 3:7] = 3
    counts[~labels, 7:11] = 3
    grid = np.array([[0.0, 0.0], [20.0, 0.0]])
    enc = dict(early_run=rates, grid_cm=grid, near=np.array([True, False]), high_indices=np.array([0, 1]), low_indices=np.array([1, 2]), cell_ids=np.arange(len(rates)))
    bank = dict(counts=counts, labels=labels, truth_cm=grid[1 - labels.astype(int)], parent_ids=np.repeat(np.arange(40), 2))
    return enc, bank


def inputs_fixture(root, noisy=False):
    source, ref, output = root / "source", root / "reference", root / "out"
    ref.mkdir()
    inputs, splits = {}, {}
    for session in m.base.SESSIONS:
        enc, bank = fixture()
        if noisy:
            bank["counts"] = np.random.default_rng(923).poisson(0.4, bank["counts"].shape)
        folder = source / session.replace("/", "_")
        folder.mkdir(parents=True)
        np.savez(folder / "encoding.npz", **enc)
        for name in m.SOURCES:
            np.savez(folder / f"{name}.npz", **bank)
        splits[session] = {s: m.robust.grouped_split(bank, session, s) for s in m.SOURCES}
        inputs.update({str(p): m.file_sha256(p) for p in folder.iterdir()})
    frozen = ref / "frozen_assignments.json"
    frozen.write_text(json.dumps(dict(splits=splits)))
    manifest = ref / "manifest.json"
    manifest.write_text(json.dumps(dict(input_file_sha256=inputs, output_sha256={frozen.name: m.file_sha256(frozen)})))
    audit = ref / "audit.json"
    audit.write_text(json.dumps(dict(status="pass", manifest_sha256=m.file_sha256(manifest))))
    return source, ref, audit, output


def test_all_three_four_subsets_and_pairs_match_unscreened_full_likelihood():
    enc, bank = fixture()
    banks = {s: bank for s in m.SOURCES}
    choice, table, pairs = m.select(enc, banks, "synthetic")
    assert len(table) == 2 * (comb(8, 3) + comb(8, 4))
    assert sum(2 * (comb(n, 3) + comb(n, 4)) for n in (34, 36, 30, 21)) == 314370
    baseline, states, _, count = a.states_for(enc, banks, choice["original"], choice["reserve"], table)
    assert count <= len(table) * 12 and states
    original = choice["original"]
    r0, j0 = m.previous.measurements(enc, banks, original)
    expected = set()
    for b in m.BUDGETS:
        for h, low in product(combinations(choice["reserve"], b), repeat=2):
            if set(h) & set(low):
                continue
            pair = dict(high=sorted(original["high"] + list(h)), low=sorted(original["low"] + list(low)))
            risk, j = m.previous.measurements(enc, banks, pair)
            if np.all(risk <= r0 + 1e-10) and j < j0 - 1e-10:
                expected.add((b, h, low))
    indexed = table.set_index("state_index").to_dict("index")
    actual = {(r.budget, tuple(m.cells(indexed[r.high_state])), tuple(m.cells(indexed[r.low_state]))) for r in pairs[pairs.admissible].itertuples()}
    assert expected and expected == actual
    a.pairs_for(enc, banks, original, choice["reserve"], states, baseline, pairs)


def test_stage_screening_has_exact_first_failure_and_no_false_safe_states():
    enc, bank = fixture()
    rng = np.random.default_rng(923)
    banks = {s: {**bank, "counts": rng.poisson(0.4, bank["counts"].shape)} for s in m.SOURCES}
    original = {s: enc[f"{s}_indices"].tolist() for s in m.SIDES}
    scorer = m.Screen(enc, banks, original)
    independent = a.IndependentStages(enc, banks, original)
    early_rejections = 0
    for side in m.SIDES:
        r0, _ = a.full.full_side(enc, banks, original[side])
        for added in combinations(range(3, 11), 4):
            risk, means, stages, safe = scorer.measure(side, list(added))
            full, full_means = a.full.full_side(enc, banks, sorted(original[side] + list(added)))
            computed = np.isfinite(risk)
            np.testing.assert_allclose(risk[computed], full[computed], atol=1e-10)
            assert safe == bool(np.all(full <= r0 + 1e-10))
            for i, (source, label) in enumerate(m.STAGES[:stages]):
                values, mean = independent.stage(side, list(added), i)
                j = 4 * m.SOURCES.index(source) + 2 * label
                np.testing.assert_allclose(values, full[j : j + 2], atol=1e-10)
                if source == "run_q3":
                    np.testing.assert_allclose(mean, full_means[label], atol=1e-10)
            if not safe:
                assert computed.sum() == stages * 2
                early_rejections += stages < 6
            else:
                assert computed.all() and np.isfinite(means).all()
    assert early_rejections > 0


def test_safe_constant_decoder_does_not_count_as_a_remedy():
    enc, bank = fixture()
    enc["early_run"][:] = 1
    choice, states, pairs = m.select(enc, {s: bank for s in m.SOURCES}, "constant")
    assert states.safe.all() and states.stages_evaluated.eq(6).all()
    assert pairs.disjoint.any() and (~pairs.disjoint).any()
    assert not pairs.admissible.any()
    assert choice["budget"] == 0 and choice["final_pair"] == choice["original"]


def test_validation_spikes_cannot_change_frozen_selection(tmp_path):
    source, ref, _, _ = inputs_fixture(tmp_path)
    session = m.base.SESSIONS[0]
    splits = json.loads((ref / "frozen_assignments.json").read_text())["splits"][session]
    folder = source / session.replace("/", "_")
    choices, states = [], []
    for i in (0, 1):
        output = tmp_path / str(i)
        output.mkdir()
        if i:
            for s in m.SOURCES:
                bank = deepcopy(m.base.read_npz(folder / f"{s}.npz"))
                bank["counts"][splits[s]["validation"]] += 10000
                np.savez(folder / f"{s}.npz", **bank)
        choice = m.worker((source, session, splits, output))
        choice.pop("tables_sha256")
        choices.append(choice)
        states.append(pd.read_csv(output / "Rat1_Open1_side_states.csv").drop(columns="runtime_s"))
    assert choices[0] == choices[1]
    pd.testing.assert_frame_equal(*states)


def test_full_cohort_freeze_audit_and_validation(tmp_path, monkeypatch):
    source, ref, reference_audit, output = inputs_fixture(tmp_path)
    original = m.robust.validation

    def validate(enc, banks, choice):
        assert set(json.loads((output / "frozen_assignments.json").read_text())["choices"]) == set(m.base.SESSIONS)
        return original(enc, banks, choice)

    monkeypatch.setattr(m.robust, "validation", validate)
    m.run(source, ref, reference_audit, output, workers=1)
    result = a.audit(source, output, tmp_path / "verified", workers=1)
    assert result["status"] == "pass"
    assert result["population_subsets_verified"] == 4 * 252
    assert result["validation_risks_reconstructed"] == 96
    assert result["safe_subsets_full_poisson_reconstructed"] > 0
    assert result["disjoint_safe_pairs_full_likelihood_reconstructed"] > 0
    assert result["ready_for_truth_preflight"]
    assert not result["validated_remedy"] and not result["external_validation"]


@pytest.mark.parametrize("damage", ["missing_subset", "stage_count", "risk", "fabricated_risk", "safe", "pair_objective", "validation"])
def test_audit_rejects_rehashed_semantic_corruption(tmp_path, damage):
    source, ref, reference_audit, output = inputs_fixture(tmp_path, noisy=damage == "fabricated_risk")
    m.run(source, ref, reference_audit, output, workers=1)
    if damage == "validation":
        path = output / "internal_validation.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "targeted"] += 1
        frame.to_csv(path, index=False)
    else:
        kind = "safe_pair_candidates" if damage == "pair_objective" else "side_states"
        path = output / f"Rat1_Open1_{kind}.csv"
        frame = pd.read_csv(path)
        if damage == "missing_subset":
            frame = frame.iloc[:-1]
        elif damage == "stage_count":
            frame.loc[0, "stages_evaluated"] = 0
        elif damage == "safe":
            frame.loc[0, "safe"] = not frame.loc[0, "safe"]
        elif damage == "fabricated_risk":
            missing = np.argwhere(frame[m.RISKS].isna().to_numpy())
            assert len(missing)
            row, col = missing[0]
            frame.loc[row, m.RISKS[col]] = 0.0
        else:
            frame.loc[0, "objective" if damage == "pair_objective" else "risk_4"] += 0.1
        frame.to_csv(path, index=False)
        frozen_path = output / "frozen_assignments.json"
        frozen = json.loads(frozen_path.read_text())
        choice = frozen["choices"]["Rat1/Open1"]
        choice["tables_sha256"][kind] = m.file_sha256(path)
        (output / "Rat1_Open1_choice.json").write_text(json.dumps(choice))
        frozen_path.write_text(json.dumps(frozen))
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["output_sha256"] = {p.name: m.file_sha256(p) for p in output.iterdir() if p.name != "manifest.json"}
    manifest["frozen_assignments_sha256"] = m.file_sha256(output / "frozen_assignments.json")
    path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, AssertionError)):
        a.audit(source, output, tmp_path / "verified", workers=1)


def test_missing_or_invalid_observations_fail():
    enc, bank = fixture()
    original = {s: enc[f"{s}_indices"].tolist() for s in m.SIDES}
    bad = deepcopy(bank)
    bad["labels"][:] = True
    with pytest.raises(ValueError, match="missing class"):
        m.Screen(enc, {s: bad for s in m.SOURCES}, original)
    enc["early_run"][0, 0] = 0
    with pytest.raises(ValueError, match="invalid rate"):
        m.Screen(enc, {s: bank for s in m.SOURCES}, original)


def test_tie_breaking_is_budget_then_cell_identity_not_row_order():
    table = pd.DataFrame(
        [
            dict(state_index=0, side="high", budget=3, added_cells="[8, 9, 10]", native_mean_nonhome=0.2, native_mean_home=0.7, safe=True),
            dict(state_index=1, side="low", budget=3, added_cells="[11, 12, 13]", native_mean_nonhome=0.1, native_mean_home=0.6, safe=True),
            dict(state_index=2, side="high", budget=4, added_cells="[1, 2, 3, 4]", native_mean_nonhome=0.2, native_mean_home=0.7, safe=True),
            dict(state_index=3, side="low", budget=4, added_cells="[5, 6, 7, 8]", native_mean_nonhome=0.1, native_mean_home=0.6, safe=True),
            dict(state_index=4, side="high", budget=3, added_cells="[7, 9, 10]", native_mean_nonhome=0.2, native_mean_home=0.7, safe=True),
        ]
    )
    for values in (table, table.iloc[::-1]):
        _, selected = m.pairs_and_choice(values, 0.2)
        assert selected["budget"] == 3
        assert selected["high_state"] == 4 and selected["low_state"] == 1
