import hashlib
import json

import matplotlib.image as mpimg
import pandas as pd
import pytest

from scripts.report_tanni_multicontext_prediction import PRIMARY, report


def fixture(tmp_path):
    source, audit = tmp_path / "source", tmp_path / "audit"
    source.mkdir()
    audit.mkdir()
    animals = [f"rat{i}" for i in range(5)]
    summary = pd.DataFrame({"metric": PRIMARY, "mean": [1, 3, -2], "ci_low": [0.1, 2, -4], "ci_high": [2, 4, 1], "positive_animals": [4, 5, 3]})
    summary.to_csv(source / "tanni_multicontext_summary.csv", index=False)
    pd.DataFrame({"animal": animals, "phase": "MUA", **{k: [v] * 5 for k, v in zip(PRIMARY, [1, 3, -2], strict=True)}}).to_csv(
        source / "tanni_multicontext_animals.csv", index=False
    )
    pd.DataFrame({"animal": animals, "iid_context_correct": [0.7] * 5, "current_iid_minus_current_global": [1.0] * 5}).to_csv(
        source / "tanni_multicontext_RUN_validation.csv", index=False
    )
    rows = [{"animal": a, "session": c, "context": c, "phase": "RUN", **{f"iid_p_{d}": int(c == d) for d in "ABCD"}} for a in animals for c in "ABCD"]
    pd.DataFrame(rows).to_csv(source / "tanni_multicontext_split_contrasts.csv", index=False)
    manifest = {
        "status": "complete",
        "code_commit": "fixture",
        "git_dirty": False,
        "RUN_windows": 20,
        "events": 10,
        "decisions": {"RUN_context_validation_passed": True, "all_candidate_predictive_contrasts_passed": False},
        "output_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()},
    }
    path = source / "tanni_multicontext_manifest.json"
    path.write_text(json.dumps(manifest))
    verification = {
        "status": "passed",
        "source_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "raw_cached_timestamp_windows_recounted": 30,
        "raw_cached_timestamp_bins_recounted": 300,
        "independent_prediction_rows": 10,
        "maximum_prediction_error": 1e-12,
    }
    (audit / "tanni_multicontext_audit.json").write_text(json.dumps(verification))
    return source, audit


def test_report_preserves_failed_gate_and_does_not_modify_scores(tmp_path):
    source, audit = fixture(tmp_path)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    out = tmp_path / "report"
    report(source, audit, out)
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    text = (out / "tanni_multicontext_report.md").read_text()
    assert "All primary MUA contrasts passed: **False**" in text
    assert "does not establish that remote reactivation is absent" in text
    pixels = mpimg.imread(out / "tanni_multicontext_diagnostic.png")
    assert pixels.std() > 0.05
    assert len(pd.read_csv(out / "tanni_RUN_context_confusion.csv")) == 16


def test_report_refuses_modified_evidence(tmp_path):
    source, audit = fixture(tmp_path)
    (source / "tanni_multicontext_summary.csv").write_text("modified\n")
    with pytest.raises(ValueError, match="changed evidence"):
        report(source, audit, tmp_path / "report")
