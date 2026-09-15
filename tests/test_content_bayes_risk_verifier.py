import json

import numpy as np
import pandas as pd
import pytest

from scripts import audit_content_bayes_risk as producer
from scripts import verify_content_bayes_risk as verifier
from test_content_bayes_risk import fixture


def test_independent_reconstruction_rejects_rehashed_metric_and_posterior_tampering(tmp_path):
    source, reference, result = tmp_path / "source", tmp_path / "reference", tmp_path / "result"
    reference.mkdir()
    inputs = {}
    for session in producer.SESSIONS:
        folder = source / session.replace("/", "_")
        folder.mkdir(parents=True)
        enc, bank = fixture(session)
        np.savez(folder / "encoding.npz", **enc)
        np.savez(folder / "cal_poisson_gain1.npz", **bank)
        inputs.update({str(p): producer.file_sha256(p) for p in folder.iterdir()})
    manifest = reference / "manifest.json"
    manifest.write_text(json.dumps(dict(source_dir=str(source), input_file_sha256=inputs, output_sha256={})))
    audit = reference / "audit.json"
    audit.write_text(json.dumps(dict(status="pass", manifest_sha256=producer.file_sha256(manifest))))
    producer.run(reference, audit, result)
    record = verifier.audit(result, tmp_path / "verified")
    assert record["status"] == "pass" and record["reconstructed_probability_values"] == 96000
    assert record["maximum_probability_error"] < 1e-10
    meta_path = result / "manifest.json"
    metadata = json.loads(meta_path.read_text())
    for name, column in (("accuracy_costs.csv", "expected_brier_cost_rao_blackwell"), ("Rat1_Open1_probabilities.csv.gz", "high_temperature_2")):
        path = result / name
        original = path.read_bytes()
        frame = pd.read_csv(path)
        frame.loc[0, column] += 0.01
        frame.to_csv(path, index=False)
        metadata["output_sha256"][name] = producer.file_sha256(path)
        meta_path.write_text(json.dumps(metadata))
        with pytest.raises(AssertionError):
            verifier.audit(result, tmp_path / "bad")
        path.write_bytes(original)
        metadata["output_sha256"][name] = producer.file_sha256(path)
        meta_path.write_text(json.dumps(metadata))
