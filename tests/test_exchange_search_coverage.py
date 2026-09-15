import json
from copy import deepcopy

import numpy as np
import pytest

from scripts import audit_exchange_search_coverage as m
from scripts import exchange_content as exchange
from tests.test_exchange_content import fixture


def test_complete_neighborhood_and_independent_full_likelihood():
    enc, bank = fixture()
    old = exchange.select_cells(enc, bank, "Rat1/Open1")
    assert old["accepted_steps"] == 0
    table = m.inspect_session(enc, bank, old)
    assert len(table) == 16
    audit = m.verify_session(enc, bank, old, table)
    assert audit["verified_pairs"] == 16 and audit["max_absolute_reconstruction_error"] < 1e-9
    assert table.previously_evaluated.all()
    summary, rejects = m.summarize(table, "Rat1/Open1")
    assert summary["legal_single_swaps"] == 16 and not summary["validated_remedy"]
    assert len(rejects) == 8


def test_missing_and_tampered_proposals_fail():
    enc, bank = fixture()
    old = exchange.select_cells(enc, bank, "Rat1/Open1")
    table = m.inspect_session(enc, bank, old)
    with pytest.raises(AssertionError, match="missing"):
        m.verify_session(enc, bank, old, table.iloc[1:])
    bad = table.copy()
    bad.loc[0, "high_class0_error_cm"] += 1
    with pytest.raises(AssertionError):
        m.verify_session(enc, bank, old, bad)
    bad = table.copy()
    bad.loc[0, "admissible"] = not bad.loc[0, "admissible"]
    with pytest.raises(AssertionError):
        m.verify_session(enc, bank, old, bad)


def test_guards_are_separate_and_do_not_trade_true_classes():
    original = np.ones(8)
    changed = original.copy()
    changed[0] -= 0.5
    changed[1] += 0.01
    improved, flags, allowed = m.feasibility(0.5, changed, 1.0, original)
    assert improved and flags.tolist() == [False, True, False, False, False, False, False, False]
    assert not allowed
    assert m.feasibility(0.5, original, 1.0, original)[2]
    assert not m.feasibility(1.0, original, 1.0, original)[2]


def test_noncalibration_inputs_do_not_change_search():
    enc, bank = fixture()
    old = exchange.select_cells(enc, bank, "Rat1/Open1")
    expected = m.inspect_session(enc, bank, old)
    changed = deepcopy(enc)
    changed["full_run"] *= 1000
    actual = m.inspect_session(changed, bank, old)
    assert expected.equals(actual)
    bad = deepcopy(old)
    bad["accepted_steps"] = 1
    with pytest.raises(ValueError, match="zero-swap"):
        m.inspect_session(enc, bank, bad)


def test_empty_neighborhood_is_explicit_not_a_positive():
    enc, bank = fixture()
    enc["low_indices"] = enc["high_indices"].copy()
    old = exchange.select_cells(enc, bank, "Rat1/Open1")
    table = m.inspect_session(enc, bank, old)
    assert table.empty
    m.verify_session(enc, bank, old, table)
    summary, _ = m.summarize(table, "Rat1/Open1")
    assert summary["no_legal_swap"] and summary["admissible"] == 0


def test_more_than32_options_and_serialized_source_workflow(tmp_path):
    enc, bank = fixture()
    rng = np.random.default_rng(1762)
    rates = rng.uniform(2, 60, (32, 4))
    enc.update(early_run=rates, full_run=rates * 1.07, cell_ids=np.arange(32), high_indices=np.arange(16), low_indices=np.arange(8, 24))
    positions = np.tile(np.arange(4), 20)
    bank["counts"] = rng.poisson(0.02 * rates[:, positions].T)
    old = exchange.select_cells(enc, bank, "Rat1/Open1")
    assert old["accepted_steps"] == 0
    table = m.inspect_session(enc, bank, old)
    assert len(table) == 64 and table.previously_evaluated.sum() == 32
    m.verify_session(enc, bank, old, table)
    # The missed-option summary is independent of the incidental fixture verdict.
    missed = table.copy()
    missed["admissible"] = False
    missed.loc[40, "admissible"] = True
    summary, _ = m.summarize(missed, "Rat1/Open1")
    assert summary["admissible_outside_original32"] == 1 and summary["first_admissible_rank"] == 41

    previous, source = tmp_path / "previous", tmp_path / "source"
    previous.mkdir()
    inputs, assignments = {}, {}
    for session in exchange.base.SESSIONS:
        folder = source / session.replace("/", "_")
        folder.mkdir(parents=True)
        np.savez(folder / "encoding.npz", **enc)
        np.savez(folder / "run_q3.npz", **bank)
        inputs.update({str(p): m.file_sha256(p) for p in folder.iterdir()})
        assignments[session] = {**deepcopy(old), "session": session}
    (previous / "pre_scoring.json").write_text(json.dumps(dict(assignments=assignments)))
    manifest = dict(source_dir=str(source), input_file_sha256=inputs, assignments_sha256=m.file_sha256(previous / "pre_scoring.json"))
    (previous / "manifest.json").write_text(json.dumps(manifest))
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(dict(status="pass", manifest_sha256=m.file_sha256(previous / "manifest.json"))))
    output = tmp_path / "output"
    m.run(previous, audit_path, output)
    verified = json.loads((output / "independent_audit.json").read_text())
    assert verified["status"] == "pass" and verified["reconstructed_pairs"] == 256
    saved = json.loads((output / "manifest.json").read_text())
    assert not saved["replay_scored"] and not saved["external_validation"] and not saved["validated_remedy"]
    for name, value in saved["output_sha256"].items():
        assert m.file_sha256(output / name) == value
    audit_path.write_text(json.dumps(dict(status="pass", manifest_sha256="changed")))
    with pytest.raises(ValueError, match="manifest"):
        m.run(previous, audit_path, tmp_path / "bad")
