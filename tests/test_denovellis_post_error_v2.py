import json
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts._provenance import file_sha256
from scripts.analyze_denovellis_post_error_content import main
from scripts.denovellis_post_error_v2 import (
    FROZEN,
    check_raw_inputs,
    coverage_bound,
    elapsed_thirds,
    full_window_mask,
    restrict_transition,
    run,
    traversal_intervals,
    verify_manifest,
    working_deadline,
)

P = json.loads(FROZEN.read_text())


def transition_fixture():
    visits = [
        {"visit_index": i + 1, "well": w, "history_segment": 0, "arrival_s": 62 + 7 * i, "departure_s": 64 + 7 * i, "pause_end_s": 64 + 7 * i} for i, w in enumerate([1, 2, 1, 3])
    ]
    row = {"eligible": True, "exclusion_reason": "", "error_visit_index": 2, "center_visit_index": 3, "next_visit_index": 4, "next_outcome": "correction"}
    return row, visits


def test_thirds_elapsed_time_not_sample_number():
    np.testing.assert_allclose(elapsed_thirds([0, 0.1, 0.2, 8, 9]), [0, 3, 6, 9])
    np.testing.assert_allclose(elapsed_thirds([100, 101, 160, 190]), [100, 130, 160, 190])


@pytest.mark.parametrize("time", [[], [1], [1, 1], [2, 1], [0, np.nan], [0, np.inf]])
def test_bad_epoch_clocks_rejected(time):
    with pytest.raises(ValueError, match="clock"):
        elapsed_thirds(time)


def test_traversals_wholly_inside_block_and_supported_history():
    _, visits = transition_fixture()
    kept = traversal_intervals(visits, 60, 75)
    assert len(kept) == 1 and kept[0]["start_s"] == 64 and kept[0]["end_s"] == 69
    assert not traversal_intervals(visits, 65, 69)
    visits[1]["history_segment"] = 1
    assert not traversal_intervals(visits, 60, 75)


def test_full_windows_never_bridge_traversals():
    intervals = [{"start_s": 0.0, "end_s": 0.1}, {"start_s": 0.2, "end_s": 0.3}]
    np.testing.assert_array_equal(full_window_mask([0, 0.08, 0.09, 0.19, 0.2], [0.02, 0.1, 0.11, 0.21, 0.22], intervals), [True, True, False, False, True])
    with pytest.raises(ValueError):
        full_window_mask([0], [0], intervals)


def test_entire_transition_inside_final_third():
    row, visits = transition_fixture()
    got = restrict_transition(row, visits, [0, 30, 60, 90])
    assert got["final_third_eligible"] and got["error_departure_s"] == 64
    assert row["eligible"] and "final_third_eligible" not in row


def test_error_arrival_alone_is_insufficient():
    row, visits = transition_fixture()
    visits[0]["departure_s"] = 59
    got = restrict_transition(row, visits, [0, 30, 60, 90])
    assert not got["final_third_eligible"]
    assert got["v2_exclusion_reason"] == "error_to_choice_not_wholly_in_final_third"


def test_no_boundary_movement_for_favorable_outcome():
    row, visits = transition_fixture()
    first = restrict_transition(row, visits, [0, 30, 60, 90])
    row["next_outcome"] = "repeated_error"
    second = restrict_transition(row, visits, [0, 30, 60, 90])
    assert first["final_third_eligible"] == second["final_third_eligible"]


def test_history_gaps_missing_and_unordered_clocks():
    row, visits = transition_fixture()
    visits[-1]["history_segment"] = 1
    assert not restrict_transition(row, visits, [0, 30, 60, 90])["final_third_eligible"]
    visits[-1]["history_segment"] = 0
    visits[2]["arrival_s"] = 60
    with pytest.raises(ValueError, match="ordered"):
        restrict_transition(row, visits, [0, 30, 60, 90])


def test_next_choice_at_epoch_end_not_accepted():
    row, visits = transition_fixture()
    visits[-1]["arrival_s"] = 90
    assert not restrict_transition(row, visits, [0, 30, 60, 90])["final_third_eligible"]


def cohort_fixture():
    return pd.DataFrame(
        [
            {"animal": a, "day": day, "next_outcome": "correction" if trial < 8 else "repeated_error", "final_third_eligible": True, "trial_id": f"{a}-{day}-{trial}"}
            for a in "abcde"
            for day in [1, 2]
            for trial in range(12)
        ]
    )


def test_upper_bound_enforces_days_and_both_outcomes():
    frame = cohort_fixture()
    got, animals, gates = coverage_bound(frame, "abcdef", P)
    assert got.primary_cohort_eligible.sum() == 120 and all(g["passed"] for g in gates)
    assert len(animals) == 6 and not animals.loc[animals.animal == "f", "animal_supported"].item()
    frame.loc[frame.animal == "a", "next_outcome"] = "correction"
    frame.loc[frame.animal == "b", "day"] = 1
    _, animals, gates = coverage_bound(frame, "abcde", P)
    assert animals.animal_supported.sum() == 3
    assert not gates[0]["passed"]


def test_qc_removal_cannot_increase_upper_bound():
    frame = cohort_fixture()
    _, _, optimistic = coverage_bound(frame, "abcde", P)
    rng = np.random.default_rng(123)
    for _ in range(20):
        subset = frame.loc[rng.random(len(frame)) > 0.3]
        _, _, actual = coverage_bound(subset, "abcde", P)
        assert all(a["observed"] <= b["observed"] for a, b in zip(actual, optimistic, strict=True))


def test_no_data_is_not_a_passing_gate():
    _, _, gates = coverage_bound(pd.DataFrame(), "abcde", P)
    assert all(not g["passed"] and g["observed"] == 0 for g in gates)
    with pytest.raises(ValueError, match="Eligibility"):
        coverage_bound(cohort_fixture().assign(final_third_eligible="False"), "abcde", P)


def test_deadline_skips_weekends():
    assert working_deadline(datetime(2026, 10, 1, 8, tzinfo=UTC), 5) == datetime(2026, 10, 8, 8, tzinfo=UTC)


def make_manifest(root, version="2.0", **extra):
    root.mkdir(exist_ok=True)
    table = root / "table.csv"
    table.write_text("status\nnot_scored\n")
    (root / "denovellis_post_error_manifest.json").write_text(
        json.dumps(dict(protocol={**P, "protocol_version": version}, output_sha256={"table.csv": file_sha256(table)}, **extra))
    )


def test_prerequisite_artifacts_are_hashed(tmp_path):
    make_manifest(tmp_path)
    assert verify_manifest(tmp_path, "2.0")["protocol"] == P
    (tmp_path / "table.csv").write_text("changed\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_manifest(tmp_path, "2.0")


def test_context_version_and_raw_input_changes_rejected(tmp_path):
    make_manifest(tmp_path, version="1.0")
    with pytest.raises(ValueError, match="version"):
        verify_manifest(tmp_path, "2.0")
    path = tmp_path / "raw"
    path.write_text("a")
    rows = [{"path": str(path), "sha256": file_sha256(path)}]
    check_raw_inputs(rows)
    path.write_text("b")
    with pytest.raises(ValueError, match="Source input hash"):
        check_raw_inputs(rows)


@pytest.mark.parametrize("stage", ["feasibility", "calibration", "analysis"])
def test_failed_v2_audit_blocks_downstream_without_writes(tmp_path, stage):
    upstream = tmp_path / "upstream"
    make_manifest(upstream, audit_passed=False)
    out = tmp_path / "never"
    with pytest.raises(ValueError, match="coverage audit failed"):
        run(SimpleNamespace(stage=stage, prerequisite_dir=upstream, output_dir=out), P)
    assert not out.exists()


def test_no_empty_calibration_can_authorize_biology(tmp_path):
    make_manifest(tmp_path, audit_passed=True)
    with pytest.raises(ValueError, match="calibration"):
        run(SimpleNamespace(stage="analysis", prerequisite_dir=tmp_path, output_dir=tmp_path / "never"), P)


def test_protocol_mutations_and_v1_audit_rejected(tmp_path):
    with pytest.raises(ValueError, match="frozen"):
        run(SimpleNamespace(), {**P, "min_animals": 1})
    with pytest.raises(SystemExit):
        main(["--stage", "audit", "--output-dir", str(tmp_path / "never")])


def test_plan_constants_are_registered():
    assert P["encoder_policy"] == "causal_within_epoch_equal_elapsed_thirds"
    assert P["primary_predictor"] == "correct_route_rate_minus_mistaken_route_rate"
    assert P["n_order_shuffles"] == P["validation_replicates_per_generator"] == 1000
    assert P["min_animals"] == 5 and P["min_transitions"] == 100
    assert P["min_run_balanced_accuracy"] == 0.8 and P["min_run_arm_recall"] == 0.75
    assert "informative_content_rate" in P["baseline_covariates"]
