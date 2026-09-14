"""The auditor/reporter must not mistake availability or earlier truth for recovery."""
import numpy as np
import pandas as pd
import pytest

from scripts.audit_edge_support_content import compare_rows, policies, reconstruct, recount, SOURCES
from scripts.measure_edge_support_content import read_event
from scripts.report_edge_support_content import session_summaries, aggregate, gates


def fixture_arrays():
    rng = np.random.default_rng(191)
    grid = np.array([[x, y] for x in (0, 8, 16) for y in (0, 8, 16)], dtype=float)
    rates = rng.uniform(.001, 20, (10, len(grid)))
    base = rng.poisson(.15, (20, 10))
    base[-5:] = 0
    truth = np.column_stack([np.linspace(0, 16, len(base)), np.full(len(base), 8)])
    return dict(counts=base, truth_base_cm=truth, event_ids=np.array([22]), offsets=np.array([0, len(base)]),
                starts_s=np.array([120.123]), rates_hz=rates, grid_cm=grid, cell_ids=np.arange(10))


def test_dense_independent_reconstruction_and_tampering():
    data = fixture_arrays()
    groups = [np.arange(5), np.arange(5, 10)]
    expected = reconstruct(data, groups)
    actual = pd.DataFrame(read_event(data["counts"], data["rates_hz"], data["grid_cm"], groups,
                                    data["starts_s"][0], 22, data["truth_base_cm"]))
    compare_rows(actual, expected)
    actual.loc[0, "b_original_truth_error_cm"] += 1
    with pytest.raises(AssertionError):
        compare_rows(actual, expected)


def test_abstention_and_original_clock_not_silently_replaced():
    data = fixture_arrays()
    data["counts"][:] = 0
    groups = [np.arange(5), np.arange(5, 10)]
    frame = reconstruct(data, groups)
    assert frame.status.eq("abstain").sum() == 3
    assert frame.loc[frame.status.eq("abstain"), "b_original_truth_error_cm"].isna().all()
    assert frame.raw_b_original_truth_error_cm.notna().all()
    assert frame.raw_end_s.nunique() == 1
    assert (frame.raw_window_index == 16).all()
    with pytest.raises(ValueError):
        compare_rows(frame.iloc[1:], frame)
    with pytest.raises(ValueError):
        compare_rows(pd.concat([frame, frame.iloc[:1]]), frame)


def test_a_only_timing_not_affected_by_b():
    data = fixture_arrays()
    groups = [np.arange(5), np.arange(5, 10)]
    _, chosen = policies(data["counts"], groups)
    changed = data["counts"].copy()
    changed[:, 5:] = 100
    _, other = policies(changed, groups)
    assert other[2] == chosen[2]


def test_recount_exact_half_open_boundaries():
    spikes = np.array([[0, 1], [.005, 1], [.010, 1], [.005, 2]])
    expected = np.array([[1, 0], [1, 1]])
    np.testing.assert_array_equal(recount(spikes, [1, 2], [0, .005], [.005, .010]), expected)


def report_fixture():
    data = fixture_arrays()
    rows = []
    for rat in range(4):
        for source in SOURCES:
            for split in range(3):
                rebuilt = reconstruct(data, [np.arange(5), np.arange(5, 10)])
                rebuilt["dataset"], rebuilt["animal"], rebuilt["session"] = "test", f"rat{rat}", f"session{rat}"
                rebuilt["source"], rebuilt["split"] = source, split
                for metric in ("separation_cm", "regional_tv", "a_entropy", "b_entropy",
                               "a_original_truth_error_cm", "b_original_truth_error_cm"):
                    rebuilt[f"raw_{metric}"] = 1.
                    rebuilt[metric] = .5
                if source == "sim_late_jump":
                    rebuilt["b_original_truth_error_cm"] = 2.
                    rebuilt["b_selected_truth_error_cm"] = 0.
                rows.append(rebuilt)
    return pd.concat(rows, ignore_index=True)


def test_late_jump_original_error_blocks_remedy_even_with_earlier_accuracy():
    by_session, _ = session_summaries(report_fixture())
    animals, summary = aggregate(by_session)
    result = gates(summary, animals, audit_passed=True, cohort_complete=True)
    assert not result.loc[result.gate.eq("original_endpoint_remedy"), "passed"].any()
    assert not result.loc[result.gate.eq("all_known_sources_original_time_error_not_worse"), "passed"].any()
    result = gates(summary.loc[summary.source.ne("sim_late_jump")], animals, True, True)
    assert not result.loc[result.gate.eq("original_endpoint_remedy"), "passed"].any()


def test_missing_audit_and_empty_readouts_fail():
    with pytest.raises(ValueError):
        session_summaries(pd.DataFrame())
    frame = report_fixture()
    with pytest.raises(ValueError):
        session_summaries(pd.concat([frame, frame.iloc[:1]]))
    by_session, _ = session_summaries(frame)
    animals, summary = aggregate(by_session)
    result = gates(summary, animals, False, True)
    assert not result.loc[result.gate.eq("independent_audit_passed"), "passed"].any()


def test_same_retained_raw_baseline_and_rat_weighting():
    frame = report_fixture().query("source == 'real' and split == 0").copy()
    extra = frame.loc[frame.animal.eq("rat0")].copy()
    extra.event_index = 23
    extra["raw_separation_cm"] = 99.
    extra.status = "abstain"
    for key in ("separation_cm", "a_entropy", "b_entropy"):
        extra[key] = np.nan
    session, _ = session_summaries(pd.concat([frame, extra], ignore_index=True))
    row = session.query("animal == 'rat0' and metric == 'separation_cm' and policy == 'a_supported_edge'").iloc[0]
    assert row.raw_same_retained == 1
    assert row.raw_all_events == 50
    animals, summary = aggregate(session)
    result = summary.query("metric == 'separation_cm' and policy == 'a_supported_edge'").iloc[0]
    assert result.availability == .875  # (.5 + 1 + 1 + 1)/4, not 4/5
    assert result.selected_minus_raw == -.5
