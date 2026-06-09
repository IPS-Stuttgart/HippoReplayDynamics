import re
from pathlib import Path


SELECTED_ROW_WORKFLOWS = (
    Path(".github/workflows/model-evidence-event-sharded.yml"),
    Path(".github/workflows/model-evidence-all-sessions.yml"),
)


def test_workflow_dispatch_inputs_stay_within_github_limit():
    workflow_dir = Path(".github/workflows")
    offenders: dict[str, int] = {}
    for workflow in workflow_dir.glob("*.yml"):
        text = workflow.read_text(encoding="utf-8")
        if "workflow_dispatch:" not in text:
            continue
        input_count = _workflow_dispatch_input_count(text)
        if input_count > 25:
            offenders[str(workflow)] = input_count

    assert not offenders


def test_selected_row_model_evidence_workflows_keep_initial_sigma_dispatchable():
    for workflow in SELECTED_ROW_WORKFLOWS:
        text = workflow.read_text(encoding="utf-8")

        assert _workflow_dispatch_input_count(text) <= 25
        assert "state_space_momentum_initial_sigma_cm_sqrt_s:" in text


def test_spike_matched_null_workflow_keeps_lightweight_scope_dispatchable():
    workflow = Path(".github/workflows/spike-matched-event-window-null.yml")
    text = workflow.read_text(encoding="utf-8")

    assert _workflow_dispatch_input_count(text) <= 25
    assert "comparison_scope:" in text
    assert "session_filter:" in text
    assert "scripts/spike_matched_event_window_null.py" in text
    assert "lightweight_matched_null_control_gate_summary.csv" in text


def test_off_swr_trajectory_discovery_workflow_wires_scorer_and_outputs():
    workflow = Path(".github/workflows/off-swr-trajectory-discovery.yml")
    text = workflow.read_text(encoding="utf-8")

    assert _workflow_dispatch_input_count(text) <= 25
    assert "scripts/spike_matched_event_window_null.py score" in text
    assert "speed_metadata_coverage.csv" in text
    assert "animal_speed_mean" in text
    assert "scripts/off_swr_trajectory_discovery.py" in text
    assert "off_swr_trajectory_candidate_events.csv" in text
    assert "off_swr_candidate_clusters.csv" in text
    assert "off_swr_candidate_behavior_lfp_summary.csv" in text
    assert "off_swr_candidate_gate_summary.csv" in text
    assert "off_swr_candidate_table.csv" in text
    assert "off_swr_candidate_cluster_table.csv" in text
    assert "off_swr_candidate_session_summary.csv" in text
    assert "off_swr_candidate_rat_summary.csv" in text
    assert "off_swr_candidate_vs_swr_summary.csv" in text
    assert "off_swr_candidate_vs_swr_window_table.csv" in text
    assert "off_swr_candidate_vs_swr_model_distribution.csv" in text
    assert "off_swr_run_state_stratified_summary.csv" in text
    assert "off_swr_run_state_specificity_summary.csv" in text
    assert "off_swr_nearest_swr_exclusion_summary.csv" in text
    assert "off_swr_nearest_swr_specificity_summary.csv" in text
    assert "off_swr_candidate_tier_threshold_summary.csv" in text
    assert "off_swr_candidate_tier_session_summary.csv" in text
    assert "off_swr_candidate_tier_rat_summary.csv" in text
    assert "off_swr_candidate_tier_nearest_swr_exclusion_summary.csv" in text
    assert "off_swr_high_specificity_candidate_table.csv" in text
    assert "off_swr_promotion_readiness_summary.csv" in text
    assert "off_swr_speed_coverage_summary.csv" in text
    assert "off_swr_candidate_specificity_gate_summary.csv" in text


def _workflow_dispatch_input_count(workflow: str) -> int:
    """Count first-level workflow_dispatch inputs in a GitHub Actions file."""

    match = re.search(
        r"workflow_dispatch:\n\s+inputs:\n(?P<inputs>.*?)(?:\n[a-zA-Z_][^\n]*:|\Z)",
        workflow,
        flags=re.S,
    )
    if match is None:
        return 0
    return len(
        re.findall(
            r"^      [A-Za-z0-9_]+:\s*$",
            match.group("inputs"),
            flags=re.M,
        )
    )
