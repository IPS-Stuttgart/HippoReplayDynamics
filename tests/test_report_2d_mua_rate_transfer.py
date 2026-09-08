import hashlib
import json

import matplotlib.image as mpimg
import pandas as pd
import pytest

from scripts.report_2d_mua_rate_transfer import PRIMARY, report


def fixture(tmp_path):
    source, audit = tmp_path / "source", tmp_path / "audit"
    source.mkdir()
    audit.mkdir()
    extra = [
        "alpha100__global_adaptation_gain",
        "alpha100__iid_adaptation_gain",
        "alpha100__iid_minus_event_global",
        "alpha100__change_in_real_order_advantage",
        "alpha100__change_in_order_map_interaction",
        "alpha1000__imm_minus_iid",
        "alpha1000__imm_minus_event_global",
        "alpha1000__imm_real_minus_wrong",
    ]
    summary, animals = [], []
    for dataset, n in (("pfeiffer_foster", 4), ("tanni2022", 5)):
        for contrast in PRIMARY + extra:
            effect = -1.0 if contrast.endswith("imm_minus_event_global") else 1.0
            summary.append(
                {
                    "dataset": dataset,
                    "contrast": contrast,
                    "mean": effect,
                    "ci_low": effect - 0.5,
                    "ci_high": effect + 0.5,
                    "positive_animals": n if effect > 0 else 0,
                    "animals": n,
                }
            )
            animals.extend({"dataset": dataset, "animal": str(i), "contrast": contrast, "delta": effect + i * 0.01} for i in range(n))
    pd.DataFrame(summary).to_csv(source / "mua_rate_transfer_summary.csv", index=False)
    pd.DataFrame(animals).to_csv(source / "mua_rate_transfer_by_animal.csv", index=False)
    pd.DataFrame(
        [{"dataset": d, "full_observation_transfer_gate": False, "failed_primary_axes": "alpha100__imm_minus_event_global"} for d in ("pfeiffer_foster", "tanni2022")]
    ).to_csv(source / "mua_rate_transfer_decisions.csv", index=False)
    manifest = {"status": "complete", "code_commit": "fixture", "git_dirty": False, "output_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}}
    path = source / "mua_rate_transfer_manifest.json"
    path.write_text(json.dumps(manifest))
    verification = {
        "status": "pass",
        "input_file_sha256": {"run_manifest": hashlib.sha256(path.read_bytes()).hexdigest()},
        "split_contrasts": 10,
        "event_contrasts": 2,
        "hierarchical_interval_panels": 12,
        "independent_dynamic_predictions": 10,
        "max_prediction_error": 1e-12,
    }
    (audit / "mua_rate_transfer_audit.json").write_text(json.dumps(verification))
    return source, audit


def test_report_preserves_failed_baseline_and_input_files(tmp_path):
    source, audit = fixture(tmp_path)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    out = tmp_path / "report"
    report(source, audit, out)
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    text = (out / "mua_rate_transfer_report.md").read_text()
    assert "passed = **False**" in text
    assert "biological gain" in text
    assert len(pd.read_csv(out / "mua_rate_transfer_primary_table.csv")) == 12
    assert mpimg.imread(out / "mua_rate_transfer_primary.png").std() > 0.05


def test_report_blocks_changed_inputs(tmp_path):
    source, audit = fixture(tmp_path)
    (source / "mua_rate_transfer_summary.csv").write_text("changed\n")
    with pytest.raises(ValueError, match="changed output"):
        report(source, audit, tmp_path / "report")
