from __future__ import annotations

import pandas as pd

from hipporeplayimm.recovery_diagnostics import build_recovery_diagnostic_tables


def _row(
    event_index: int,
    model: str,
    log_evidence: float,
    *,
    support: str = "exact_full_grid",
    comparable: bool = True,
    true_triplet_coverage: float | None = None,
    true_path_supported: int | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "status": "success",
        "session": "RatX/OpenY",
        "event_index": event_index,
        "simulation_event_index": event_index,
        "replicate": event_index,
        "true_model": "momentum",
        "expected_model": "sorted-spike-state-space-momentum",
        "model": model,
        "requested_model": model,
        "log_evidence": log_evidence,
        "n_time": 5,
        "n_spikes": 20,
        "evidence_support": support,
        "evidence_comparable": comparable,
        "recovered_expected_model": False,
    }
    if true_triplet_coverage is not None:
        row["candidate_true_bin_coverage"] = 1.0
        row["candidate_true_pair_coverage"] = 1.0
        row["candidate_true_triplet_coverage"] = true_triplet_coverage
    if true_path_supported is not None:
        row["candidate_true_path_fully_supported"] = true_path_supported
        row["candidate_true_path_missing_bins"] = 0 if true_path_supported else 1
    return row


def test_recovery_diagnostics_separates_strict_and_certified_recovery():
    scores = pd.DataFrame(
        [
            _row(0, "sorted-spike-state-space-diffusion", 1.0),
            _row(
                0,
                "sorted-spike-state-space-momentum",
                2.0,
                support="truncated_full_grid",
                comparable=False,
                true_triplet_coverage=1.0,
                true_path_supported=1,
            ),
            _row(1, "sorted-spike-state-space-diffusion", 3.0),
            _row(
                1,
                "sorted-spike-state-space-momentum",
                2.0,
                support="truncated_full_grid",
                comparable=False,
                true_triplet_coverage=0.5,
                true_path_supported=0,
            ),
        ]
    )

    tables = build_recovery_diagnostic_tables(scores)
    events = tables.event_diagnostics.set_index("event_index")

    assert not bool(events.loc[0, "strict_recovered_expected_model"])
    assert bool(events.loc[0, "certified_vs_exact_recovered_expected_model"])
    assert events.loc[0, "failure_mode"] == "strict_gate_excluded_certified_lower_bound"
    assert events.loc[1, "failure_mode"] == "candidate_support_misses_true_path"

    overall = tables.summary[tables.summary["true_model"].eq("overall")].iloc[0]
    assert overall["strict_recovered_events"] == 0
    assert overall["certified_vs_exact_recovered_events"] == 1
    assert overall["strict_recovery_accuracy"] == 0.0
    assert overall["certified_vs_exact_recovery_accuracy"] == 0.5
    assert overall["failure_mode_strict_gate_excluded_certified_lower_bound_events"] == 1
    assert overall["failure_mode_candidate_support_misses_true_path_events"] == 1


def test_recovery_diagnostics_respects_string_false_comparable_flags():
    scores = pd.DataFrame(
        [
            _row(0, "sorted-spike-state-space-diffusion", 0.0),
            _row(
                0,
                "sorted-spike-state-space-momentum",
                100.0,
                support="unknown_noncomparable",
                comparable="False",
            ),
        ]
    )

    tables = build_recovery_diagnostic_tables(scores)
    event = tables.event_diagnostics.iloc[0]
    overall = tables.summary[tables.summary["true_model"].eq("overall")].iloc[0]

    assert event["comparable_scores"] == 1
    assert not bool(event["strict_recovered_expected_model"])
    assert not bool(event["certified_vs_exact_recovered_expected_model"])
    assert event["certified_vs_exact_reason"] == "expected_noncomparable_not_certified"
    assert overall["strict_recovered_events"] == 0
    assert overall["certified_vs_exact_recovered_events"] == 0


def test_recovery_diagnostics_write_outputs(tmp_path):
    scores = pd.DataFrame(
        [
            _row(0, "sorted-spike-state-space-diffusion", 1.0),
            _row(
                0,
                "sorted-spike-state-space-momentum",
                2.0,
                support="truncated_full_grid",
                comparable=False,
                true_triplet_coverage=1.0,
                true_path_supported=1,
            ),
        ]
    )

    tables = build_recovery_diagnostic_tables(scores)
    tables.write(tmp_path)

    assert (tmp_path / "simulation_recovery_diagnostic_event_table.csv").exists()
    assert (tmp_path / "simulation_recovery_diagnostic_summary.csv").exists()
    assert (tmp_path / "simulation_recovery_diagnostic_report.md").exists()
    assert (tmp_path / "simulation_recovery_diagnostic_manifest.json").exists()
