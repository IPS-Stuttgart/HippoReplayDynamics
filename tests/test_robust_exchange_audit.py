import json

import numpy as np
import pandas as pd
import pytest

from scripts import audit_robust_exchange_content as audit
from scripts import robust_exchange_content as producer


def fixture():
    rng = np.random.default_rng(44)
    rates = rng.lognormal(2.5, 1.5, (12, 2))
    positions = np.repeat(np.tile(np.arange(2), 20), 2)
    grid = np.array([[0.0, 0.0], [20.0, 0.0]])
    enc = dict(early_run=rates, full_run=rates, grid_cm=grid, near=np.array([True, False]), high_indices=np.arange(6), low_indices=np.arange(4, 10), cell_ids=np.arange(12))
    bank = dict(counts=rng.poisson(0.02 * rates[:, positions].T), truth_cm=grid[positions], labels=positions == 0, parent_ids=np.repeat(np.arange(40), 2))
    return enc, bank


def test_independent_partition_and_membership_checks():
    _, bank = fixture()
    split = producer.grouped_split(bank, "Rat1/Open1", "run_q3")
    audit.check_partition(bank, split, "Rat1/Open1", "run_q3")
    split["train"][0], split["validation"][0] = split["validation"][0], split["train"][0]
    with pytest.raises(ValueError, match="partition"):
        audit.check_partition(bank, split, "Rat1/Open1", "run_q3")
    original = dict(high=[0, 1, 2], low=[2, 3, 4])
    assert audit.exchanged(original, [0], [3]) == dict(high=[1, 2, 3], low=[0, 2, 4])
    with pytest.raises(ValueError, match="exclusive"):
        audit.exchanged(original, [2], [3])


def test_audit_reconstructs_choices_and_detects_rehashed_false_validation(tmp_path):
    source, reference, output = tmp_path / "source", tmp_path / "reference", tmp_path / "producer"
    reference.mkdir()
    inputs = {}
    for session in producer.exact.base.SESSIONS:
        folder = source / session.replace("/", "_")
        folder.mkdir(parents=True)
        enc, bank = fixture()
        np.savez(folder / "encoding.npz", **enc)
        for s in producer.SOURCES:
            np.savez(folder / f"{s}.npz", **bank)
        inputs.update({str(p): producer.file_sha256(p) for p in folder.iterdir()})
    (reference / "manifest.json").write_text(json.dumps(dict(source_dir=str(source), input_file_sha256=inputs, output_sha256={})))
    reference_audit = reference / "independent_audit.json"
    reference_audit.write_text(json.dumps(dict(status="pass", manifest_sha256=producer.file_sha256(reference / "manifest.json"))))
    producer.run(source, reference, reference_audit, output)
    audit.audit(source, output, tmp_path / "audit")
    record = json.loads((tmp_path / "audit" / "independent_audit.json").read_text())
    assert record["status"] == "pass" and record["validation_risks_reconstructed"] == 96
    assert not record["validated_remedy"]
    assert len(pd.read_csv(tmp_path / "audit" / "partition_support.csv")) == 24
    values = pd.read_csv(output / "internal_validation.csv")
    values.loc[0, "targeted"] += 1
    values.to_csv(output / "internal_validation.csv", index=False)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["output_sha256"]["internal_validation.csv"] = producer.file_sha256(output / "internal_validation.csv")
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        audit.audit(source, output, tmp_path / "corrupt-audit")
