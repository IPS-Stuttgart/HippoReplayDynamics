import json

import pandas as pd
import pytest

from scripts.report_position_free_assembly_prediction import PANELS, ROWS, decision, readout, validate_inputs


def fixture():
    return pd.DataFrame(
        [
            {"dataset": d, "phase": phase, "encoding_variant": v, "contrast": c, "n_components": k, "mean": 2.0, "ci_low": 1.0, "ci_high": 3.0, "positive_animals": 4, "animals": 4}
            for d, phase, v, _ in PANELS
            for c, k, _ in ROWS
        ]
    )


def test_readout_requires_every_contrast():
    f = fixture()
    assert len(readout(f)) == 15
    with pytest.raises(ValueError, match="missing"):
        readout(f.iloc[:-1])


def test_bounded_positive_is_not_mechanistic_claim():
    f = fixture()
    mask = f.contrast.eq("assembly_persistent_minus_global")
    f.loc[mask, ["mean", "ci_low", "ci_high", "positive_animals"]] = [-2, -3, -1, 0]
    result = decision(readout(f))
    assert result["bounded_pf_spatial_advantage"]
    assert result["tested_assemblies_underperform_global_pf"]
    assert not result["mechanism_identified"]
    assert not result["external_replication_established"]


def test_failed_primary_cannot_pass():
    f = fixture()
    f.loc[f.n_components.eq(8), "ci_low"] = -1
    assert not decision(readout(f))["bounded_pf_spatial_advantage"]


def test_requires_passing_audit(tmp_path):
    (tmp_path / "assembly_prediction_manifest.json").write_text(json.dumps({"status": "complete"}))
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"status": "fail"}))
    with pytest.raises(ValueError, match="passing"):
        validate_inputs(tmp_path, audit)
