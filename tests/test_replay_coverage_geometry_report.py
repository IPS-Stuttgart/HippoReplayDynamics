import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_replay_coverage_geometry import compare_tables, direct_support, direct_truth_eligibility
from scripts.report_replay_coverage_geometry import (
    aggregate_responses,
    common_eligible_resolution_contrasts,
    gradient_responses,
    paired_contrasts,
)
from scripts.simulate_replay_coverage_geometry import build_batch, score_batch, summarize_batch
from tests.test_replay_coverage_geometry import args, straight_path


def test_direct_support_has_no_gap_bridge():
    fine = np.ones((200, 2), int)
    fine[10:35] = 0
    actual, _ = direct_support(fine, 20, 5, True)
    assert actual["valid_bins"] < actual["n_decoded_bins"]
    assert actual["all_steps"] < 9


def test_independent_truth_eligibility():
    assert direct_truth_eligibility(straight_path(), 20, 5, "literal_20cm_10frames")
    assert direct_truth_eligibility(straight_path(), 40, 10, "time_scaled_4000cm_s_45ms")
    path = straight_path()
    path["midpoints_cm"][:] = 0
    assert not direct_truth_eligibility(path, 20, 5, "literal_20cm_10frames")


def test_missing_gradients_and_unpaired_contrasts_are_not_zero():
    config = args()
    paths, obs, evaluate, _, bounds = build_batch(config, 0, 0)
    events, _ = score_batch(config, 0, 0, paths, obs, evaluate, bounds)
    summary, gradient = summarize_batch(events)
    paired = gradient_responses(gradient)
    assert paired.gradient_response.isna().all()
    table = aggregate_responses(paired, np.random.default_rng(0), 20)
    assert table.populations_available.eq(0).all()
    contrasts = paired_contrasts(summary, paired, np.random.default_rng(0), 20)
    assert contrasts[contrasts.varied_factor.eq("sigma_cm")].matched_populations.eq(0).all()
    cells = contrasts[contrasts.varied_factor.eq("n_cells") & contrasts.metric.eq("eligible_recovery_fraction")]
    assert cells.matched_populations.eq(1).all()


def test_table_comparison_rejects_duplicate_and_missing():
    import pytest
    frame = pd.DataFrame({"id": [1, 2], "value": [3., 4.]})
    with pytest.raises(AssertionError):
        compare_tables(pd.concat([frame, frame]), frame, ["id"])
    with pytest.raises(AssertionError):
        compare_tables(frame.iloc[:1], frame, ["id"])


def test_common_eligible_denominator_excludes_different_truth_paths():
    config = args()
    paths, obs, evaluate, _, bounds = build_batch(config, 0, 0)
    events, _ = score_batch(config, 0, 0, paths, obs, evaluate, bounds)
    reference = events[events.truth_kind.eq("continuous") & events.gradient.eq(0)].copy()
    reference["truth_geometric_eligible"] = True
    reference["continuity_pass"] = False
    changed = reference.assign(window_ms=40, truth_geometric_eligible=reference.path_id.eq(0), continuity_pass=True)
    pop, summary = common_eligible_resolution_contrasts(pd.concat([reference, changed]), np.random.default_rng(0), 20)
    assert pop.common_eligible_paths.eq(1).all()
    assert pop.paths_matched.eq(2).all()
    assert pop.paired_recovery_difference.eq(1).all()
    assert summary["mean"].eq(1).all()


def test_cli_smoke_and_reconstruction(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "smoke"
    subprocess.run([sys.executable, str(root / "scripts/simulate_replay_coverage_geometry.py"), "--output-dir", str(output),
                    "--populations", "1", "--paths", "1", "--field-ids", "0", "--baseline-only"], check=True, capture_output=True)
    subprocess.run([sys.executable, str(root / "scripts/audit_replay_coverage_geometry.py"), "--input-dir", str(output)], check=True, capture_output=True)
    audit = pd.read_csv(output / "geometry_reconstruction_audit.csv")
    assert audit.metric_rows_recounted.sum() == 192
    subprocess.run([sys.executable, str(root / "scripts/report_replay_coverage_geometry.py"), "--input-dir", str(output),
                    "--output-dir", str(tmp_path / "report"), "--bootstraps", "20"], check=True, capture_output=True)
    assert (tmp_path / "report/geometry_endpoint_summary.csv").exists()
