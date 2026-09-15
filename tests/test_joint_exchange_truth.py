import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from scripts import audit_joint_exchange_truth as m


def fixture():
    rng = np.random.default_rng(44)
    rates = rng.lognormal(2.5, 1.5, (12, 2))
    position = np.tile(np.arange(2), 20)
    grid = np.array([[0.0, 0.0], [20.0, 0.0]])
    enc = dict(early_run=rates, full_run=rates * 1.07, grid_cm=grid, near=np.array([True, False]), high_indices=np.arange(6), low_indices=np.arange(4, 10), cell_ids=np.arange(12))
    bank = dict(counts=rng.poisson(0.02 * rates[:, position].T), truth_cm=grid[position], labels=position == 0, parent_ids=np.repeat(np.arange(20), 2))
    original = {s: enc[f"{s}_indices"].tolist() for s in ("high", "low")}
    return enc, bank, dict(original=original, final_pair=deepcopy(original), net_exchanged=0)


def test_independent_full_likelihood_and_real_joint_risks():
    enc, bank, choice = fixture()
    original = choice["original"]
    final = dict(high=[0, 3, 4, 5, 7, 9], low=[1, 2, 4, 5, 6, 8])
    m.validate_pair(enc, {**choice, "final_pair": final, "net_exchanged": 2})
    before, independent_before = m.score(bank, enc, original)
    after, independent_after = m.score(bank, enc, final)
    table = pd.DataFrame(m.class_table(before, after, "Rat1/Open1", "fixture", bank["parent_ids"]))
    check = pd.DataFrame(m.class_table(independent_before, independent_after, "Rat1/Open1", "fixture", bank["parent_ids"]))
    pd.testing.assert_frame_equal(table, check, atol=1e-9, rtol=1e-9)
    assert table.error_nonworsening.all() and table.brier_nonworsening.all()
    assert table.parent_groups.eq(20).all()


def test_classwise_failure_not_hidden_by_pooled_improvement():
    before = dict(true_home=np.array([0, 0, 1, 1]))
    for side in ("high", "low"):
        before[f"{side}_error"] = np.array([10.0, 10, 10, 10])
        before[f"{side}_brier"] = np.array([0.25, 0.25, 0.25, 0.25])
    after = deepcopy(before)
    after["high_error"] = np.array([1.0, 1, 11, 11])
    after["low_brier"] = np.array([0.01, 0.01, 0.3, 0.3])
    table = pd.DataFrame(m.class_table(before, after, "Rat1/Open1", "fixture"))
    assert after["high_error"].mean() < before["high_error"].mean()
    assert after["low_brier"].mean() < before["low_brier"].mean()
    assert table.error_nonworsening.sum() == 3
    assert table.brier_nonworsening.sum() == 3


def test_missing_or_nonfinite_truth_and_changed_cells_rejected():
    enc, bank, choice = fixture()
    values, _ = m.score(bank, enc, choice["original"])
    bad = deepcopy(values)
    bad["true_home"][:] = 0
    with pytest.raises(ValueError, match="empty truth class"):
        m.class_table(bad, bad, "Rat1/Open1", "fixture")
    bad = deepcopy(values)
    bad["high_error"][0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        m.class_table(bad, values, "Rat1/Open1", "fixture")
    for field in ("original", "final_pair"):
        bad = deepcopy(choice)
        bad[field]["high"][0] = 11
        with pytest.raises(ValueError):
            m.validate_pair(enc, bad)
    with pytest.raises(ValueError, match="exchange count"):
        m.validate_pair(enc, {**choice, "net_exchanged": 1})


def write_manifest(root, inputs, **metadata):
    record = dict(input_file_sha256=inputs, output_sha256={p.name: m.file_sha256(p) for p in root.iterdir() if p.is_file()}, **metadata)
    (root / "manifest.json").write_text(json.dumps(record))
    (root / "independent_audit.json").write_text(json.dumps(dict(status="pass", manifest_sha256=m.file_sha256(root / "manifest.json"))))


def source_fixture(tmp_path):
    source, reference, selected = [tmp_path / s for s in ("source", "reference", "selected")]
    reference.mkdir()
    selected.mkdir()
    inputs, choices = {}, {}
    for session in m.exact.base.SESSIONS:
        enc, bank, choice = fixture()
        values, _ = m.score(bank, enc, choice["original"])
        bank["early_run_scores"] = np.column_stack([values[f"{s}_home"] for s in ("high", "low")])
        folder = source / session.replace("/", "_")
        folder.mkdir(parents=True)
        np.savez(folder / "encoding.npz", **enc)
        for truth in m.exact.base.TRUTH:
            np.savez(folder / f"{truth}.npz", **bank)
        inputs.update({str(p): m.file_sha256(p) for p in folder.iterdir()})
        choices[session] = choice
    write_manifest(reference, inputs, source_dir=str(source))
    inputs[str(reference / "manifest.json")] = m.file_sha256(reference / "manifest.json")
    (selected / "selection.json").write_text(json.dumps(choices))
    write_manifest(selected, inputs, created_at_utc="2000-01-01T00:00:00+00:00")
    return source, reference, selected


def test_full_artifacts_identity_pass_and_input_tampering(tmp_path):
    source, reference, selected = source_fixture(tmp_path)
    output = tmp_path / "output"
    m.run(selected, reference, reference / "independent_audit.json", output)
    summary = pd.read_csv(output / "summary.csv")
    assert len(summary) == 6 and summary.comparisons_per_metric.eq(16).all()
    assert summary.error_guards_passed.eq(16).all() and summary.brier_guards_passed.eq(16).all()
    assert pd.read_csv(output / "failed_guards.csv").empty
    assert pd.read_csv(output / "gates.csv").passed.all()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["regional_loss_comparisons"] == 192
    assert manifest["paired_truth_observations"] == 4 * 6 * 40
    assert not any(manifest[k] for k in ("replay_scored", "validated_remedy", "external_validation"))
    audit = json.loads((output / "independent_audit.json").read_text())
    assert audit["population_posteriors_reconstructed"] == 4 * 4 * 6 * 40
    assert audit["manifest_sha256"] == m.file_sha256(output / "manifest.json")
    for name, sha in manifest["output_sha256"].items():
        assert m.file_sha256(output / name) == sha
    path = source / "Rat1_Open1" / "run_q4.npz"
    path.write_bytes(path.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="changed source"):
        m.run(selected, reference, reference / "independent_audit.json", tmp_path / "bad")


def test_missing_pair_fails_before_evaluation(tmp_path):
    _, reference, selected = source_fixture(tmp_path)
    choices = json.loads((selected / "selection.json").read_text())
    choices.pop("Rat4/Open2")
    (selected / "selection.json").write_text(json.dumps(choices))
    record = json.loads((selected / "manifest.json").read_text())
    record["output_sha256"]["selection.json"] = m.file_sha256(selected / "selection.json")
    (selected / "manifest.json").write_text(json.dumps(record))
    (selected / "independent_audit.json").write_text(json.dumps(dict(status="pass", manifest_sha256=m.file_sha256(selected / "manifest.json"))))
    with pytest.raises(ValueError, match="missing original pair"):
        m.run(selected, reference, reference / "independent_audit.json", tmp_path / "bad")


def test_worse_population_generates_failed_not_successful_preflight(tmp_path):
    _, reference, selected = source_fixture(tmp_path)
    choices = json.loads((selected / "selection.json").read_text())
    for choice in choices.values():
        choice["final_pair"] = dict(high=choice["original"]["low"], low=choice["original"]["high"])
        choice["net_exchanged"] = 4
    (selected / "selection.json").write_text(json.dumps(choices))
    record = json.loads((selected / "manifest.json").read_text())
    record["output_sha256"]["selection.json"] = m.file_sha256(selected / "selection.json")
    (selected / "manifest.json").write_text(json.dumps(record))
    (selected / "independent_audit.json").write_text(json.dumps(dict(status="pass", manifest_sha256=m.file_sha256(selected / "manifest.json"))))
    output = tmp_path / "output"
    m.run(selected, reference, reference / "independent_audit.json", output)
    gates = pd.read_csv(output / "gates.csv").set_index("gate").passed
    assert not gates.truth_preflight
    assert gates.independent_likelihoods_match
    assert len(pd.read_csv(output / "failed_guards.csv")) == 96
    assert "preflight: FAIL" in (output / "report.md").read_text()
    audit = json.loads((output / "independent_audit.json").read_text())
    assert audit["status"] == "pass" and not audit["truth_preflight"]
