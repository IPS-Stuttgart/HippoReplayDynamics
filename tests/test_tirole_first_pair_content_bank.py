import json

import pytest

from hipporeplayimm.tirole_two_track import file_sha256
from scripts.prepare_tirole_first_pair_content_bank import COHORT, validate_preflight


def fixture(tmp_path):
    folder = tmp_path / "preflight"
    folder.mkdir()
    m = {
        "session": "RAT2_SESS1",
        "status": "complete",
        "git_dirty": False,
        "RUN_gate_passed": True,
        "primary_cohort_promoted": False,
        "scientific_replay_scores_computed": False,
        "scope": "first_two_epochs_before_third_diagnostic",
        "output_sha256": {},
    }
    (folder / "manifest.json").write_text(json.dumps(m))
    v = {
        "status": "pass",
        "RUN_gate_passed": True,
        "primary_cohort_promoted": False,
        "source_manifest_sha256": file_sha256(folder / "manifest.json"),
        "checks": {"map_arrays": 12, "crossvalidation_rows": 10, "source_epochs": 4, "rest_durations": 2},
        "max_map_array_absolute_error": 0,
        "lfp_clock": {"clock_offset_applied_s": 0},
    }
    verification = tmp_path / "verification.json"
    verification.write_text(json.dumps(v))
    return folder, verification, m, v


def test_no_primary_promotion(tmp_path):
    folder, verification, _, _ = fixture(tmp_path)
    m, _ = validate_preflight(folder, verification)
    assert not m["primary_cohort_promoted"]
    assert COHORT != "strict_RUN_pass"


@pytest.mark.parametrize("change", ["failed_RUN", "promoted", "wrong_scope", "changed_source", "incomplete_verification", "offset"])
def test_bad_or_reclassified_preflight_fails(tmp_path, change):
    folder, verification, m, v = fixture(tmp_path)
    if change == "failed_RUN":
        m["RUN_gate_passed"] = False
    elif change == "promoted":
        m["primary_cohort_promoted"] = True
    elif change == "wrong_scope":
        m["scope"] = "all_four_epochs"
    elif change == "changed_source":
        v["source_manifest_sha256"] = "wrong"
    elif change == "incomplete_verification":
        v["checks"]["crossvalidation_rows"] = 0
    else:
        v["lfp_clock"]["clock_offset_applied_s"] = 100
    (folder / "manifest.json").write_text(json.dumps(m))
    if change != "changed_source":
        v["source_manifest_sha256"] = file_sha256(folder / "manifest.json")
    verification.write_text(json.dumps(v))
    with pytest.raises(ValueError):
        validate_preflight(folder, verification)
