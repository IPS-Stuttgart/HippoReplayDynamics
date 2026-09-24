import numpy as np
import pytest

from scripts import audit_kleinman_ripple_run_opportunities as audit


def test_interval_union_intersection_and_half_open_counts():
    union = audit.merge_intervals([[2, 4], [0, 2], [3, 5], [7, 9]])
    np.testing.assert_array_equal(union, [[0, 5], [7, 9]])
    result = audit.intersect_intervals(union, [[1, 3], [4, 8]])
    np.testing.assert_array_equal(result, [[1, 3], [4, 5], [7, 8]])
    assert audit.duration(result) == 4
    assert audit.counts_in_intervals([np.arange(10)], result).tolist() == [4]
    assert audit.counts_in_intervals([np.arange(10)], []).tolist() == [0]


@pytest.mark.parametrize("bad", [[[1, 1]], [[2, 1]], [[0, np.nan]]])
def test_invalid_intervals_rejected(bad):
    with pytest.raises(ValueError, match="invalid_intervals"):
        audit.merge_intervals(bad)


def runs():
    return [{"traversal": i, "start_s": 10 * i + 2, "end_s": 10 * i + 10, "epoch": 1 if i < 14 else 2, "direction": (i + 1) % 2, "fold": 0} for i in range(28)]


def test_chronology_has_no_baseline_or_future_reference_leakage():
    pairs = audit.opportunity_pairs(runs())
    assert len(pairs) == 24
    ready = [p for p in pairs if p["status"] == "audited"]
    assert len(ready) == 12
    for p in ready:
        ref = p["reference"]
        assert len(ref) == 3
        assert max(r["end_s"] for r in ref) < p["baseline"]["start_s"]
        assert p["baseline"]["end_s"] < p["target"]["start_s"]
        assert len({r["epoch"] for r in [*ref, p["baseline"], p["target"]]}) == 1
        assert len({r["direction"] for r in [*ref, p["baseline"], p["target"]]}) == 1
    assert {p["epoch"] for p in pairs if p["status"] != "audited"} == {1, 2}


def test_overlapping_runs_rejected():
    r = runs()
    r[2]["start_s"] = r[0]["end_s"] - 0.1
    with pytest.raises(ValueError, match="overlapping"):
        audit.opportunity_pairs(r)


def fixture_source():
    t = np.arange(3841) / 20
    visit_index = np.minimum((t // 10).astype(int), 19)
    phase = t - visit_index * 10
    side = visit_index % 2
    x = np.where(phase <= 2, side * 200, np.where(side == 0, (phase - 2) * 25, 200 - (phase - 2) * 25))
    x = np.clip(x, 0, 200)
    speed = np.where((phase > 2) & (phase < 10), 25, 0)
    left, right = [], []
    for i in range(20):
        a, b = 10 * i, 10 * i + 2
        (left if i % 2 == 0 else right).append([round(a * 20) + 2, round(b * 20) + 2])
    info = {
        "position": np.r_[x[0], x],
        "velocity": np.column_stack([t, speed]),
        "reward_ends": [5, 195],
        "left_visit": left,
        "right_visit": right,
        "epoch_change": np.empty((0, 2)),
    }
    trains = [np.arange(0.01 + i * 0.01, 192, 0.2) for i in range(6)]
    spike = np.concatenate([np.column_stack([s, np.full(len(s), i + 1), np.ones(len(s))]) for i, s in enumerate(trains)])
    return info, spike


def patch_source(monkeypatch, info, spike, ripples):
    payload = {"session_info.mat": {"session_info": info}, "spike_data.mat": {"spike_data": spike}, "ripple_events.mat": {"ripple_events": np.asarray(ripples).reshape(-1, 4)}}
    monkeypatch.setattr(audit, "loadmat", lambda path, **kwargs: payload[path.name])


def test_zero_ripple_opportunities_not_removed(monkeypatch, tmp_path):
    info, spike = fixture_source()
    patch_source(monkeypatch, info, spike, [])
    summary, rows = audit.session_audit(tmp_path)
    assert summary["n_opportunities"] > summary["n_prior_reference_available"] > 0
    assert summary["n_reference_min_units_without_ripple"] > 0
    assert summary["n_reference_min_units_with_ripple"] == 0
    assert all(r["eligible_ripple_s"] == 0 for r in rows)
    ready = [r for r in rows if r["status"] == "audited"]
    assert all(r["n_encoding_units"] == 6 for r in ready)
    assert all(r["baseline_encoding_spikes"] > 0 and r["target_encoding_spikes"] > 0 for r in ready)
    assert all(r["reference_end_s"] < r["baseline_start_s"] for r in ready)


def test_eligible_ripple_segments_do_not_double_count(monkeypatch, tmp_path):
    info, spike = fixture_source()
    patch_source(monkeypatch, info, spike, [[80.5, 81.2, 80.7, 0], [81.0, 81.5, 81.1, 0]])
    _, rows = audit.session_audit(tmp_path)
    intersecting = [r for r in rows if r["eligible_ripple_s"] > 0]
    assert intersecting
    assert all(r["eligible_ripple_s"] == pytest.approx(1) for r in intersecting)
    assert all(r["n_native_ripples_intersecting"] == 2 for r in intersecting)
    for r in rows:
        assert r["eligible_immobile_s"] == pytest.approx(r["eligible_ripple_s"] + r["eligible_background_s"])


def test_future_only_cell_cannot_enter_reference_encoding(monkeypatch, tmp_path):
    info, spike = fixture_source()
    patch_source(monkeypatch, info, spike, [])
    _, original = audit.session_audit(tmp_path)
    p = next(r for r in original if r["status"] == "audited")
    s = np.linspace(p["target_start_s"] + 0.2, p["target_end_s"] - 0.2, 200)
    changed = np.vstack([spike, np.column_stack([s, np.full(len(s), 99), np.ones(len(s))])])
    patch_source(monkeypatch, info, changed, [])
    _, rerun = audit.session_audit(tmp_path)
    q = next(r for r in rerun if r["opportunity_id"] == p["opportunity_id"])
    assert q["n_units_raw"] == p["n_units_raw"] + 1
    for column in ("n_encoding_units", "reference_encoding_spikes", "baseline_encoding_spikes", "target_encoding_spikes"):
        assert q[column] == p[column]


def test_independent_chronology_matches_source_fixture():
    from scripts.validate_kleinman_run_decoder import align_behavior, make_traversals
    from scripts.verify_kleinman_ripple_run_opportunities import expected_pairs, raw_runs

    info, _ = fixture_source()
    _, _, _, native_runs = raw_runs(info)
    _, _, _, _, visits, epochs = align_behavior(info)
    implemented = audit.opportunity_pairs(make_traversals(visits, epochs))
    assert {r["opportunity_id"] for r in implemented} == set(expected_pairs(native_runs))


def test_independent_raw_counts_and_reference_maps(monkeypatch, tmp_path):
    import pandas as pd

    from scripts import verify_kleinman_ripple_run_opportunities as check

    info, spike = fixture_source()
    patch_source(monkeypatch, info, spike, [[80.5, 81.2, 80.7, 0], [81.0, 81.5, 81.1, 0]])
    monkeypatch.setattr(check, "loadmat", audit.loadmat)
    _, rows = audit.session_audit(tmp_path)
    row = pd.Series(next(r for r in rows if r["status"] == "audited" and r["eligible_ripple_s"] > 0))
    check.raw_count_check(tmp_path, row)
    row["ripple_encoding_spikes"] += 1
    with pytest.raises(AssertionError):
        check.raw_count_check(tmp_path, row)


def test_independent_full_run_generating_bank(monkeypatch, tmp_path):
    import pandas as pd

    from scripts import calibrate_kleinman_replay_content as maps
    from scripts import calibrate_kleinman_spatial_expression as producer
    from scripts import verify_kleinman_spatial_expression as checker

    info, spike = fixture_source()
    patch_source(monkeypatch, info, spike, [])
    for module in (maps, producer, checker):
        monkeypatch.setattr(module, "loadmat", audit.loadmat)
    monkeypatch.setattr(producer, "full_map", maps.full_map)
    _, rows = audit.session_audit(tmp_path)
    row = pd.Series(next(r for r in rows if r["status"] == "audited"))
    bank = producer.build_bank(tmp_path, row)
    independent = checker.reference_bank(tmp_path, row.opportunity_id, row.direction)
    for key in bank:
        np.testing.assert_allclose(bank[key], independent[key], atol=1e-10, rtol=1e-10)


def test_recruitment_predictor_does_not_use_future_run_spikes(monkeypatch, tmp_path):
    import pandas as pd

    from scripts import calibrate_kleinman_conditional_coupling as coupling

    info, spike = fixture_source()
    patch_source(monkeypatch, info, spike, [[80.5, 81.2, 80.7, 0], [81.0, 81.5, 81.1, 0]])
    monkeypatch.setattr(coupling, "loadmat", audit.loadmat)
    _, rows = audit.session_audit(tmp_path)
    row = pd.Series(next(r for r in rows if r["status"] == "audited" and r["eligible_ripple_s"] > 0))
    keys = np.column_stack([np.ones(6, dtype=int), np.arange(1, 7)])
    original = coupling.recruitment(tmp_path, row, keys)
    assert original["ripple_counts"].sum() == row.ripple_encoding_spikes
    assert original["background_counts"].sum() == row.background_encoding_spikes

    from scripts import verify_kleinman_conditional_coupling as checker

    monkeypatch.setattr(checker, "loadmat", audit.loadmat)
    table = pd.DataFrame(
        {
            "unit_id": ["_".join(map(str, k)) for k in keys],
            "ripple_spikes": original["ripple_counts"],
            "background_spikes": original["background_counts"],
            "ripple_s": original["ripple_seconds"],
            "background_s": original["background_seconds"],
            "log_rate_enrichment": original["predictor"],
        }
    )
    eligible, ripples = checker.raw_predictors(tmp_path, row, table)
    np.testing.assert_allclose(eligible, original["eligible_intervals"])
    np.testing.assert_allclose(ripples, original["ripple_intervals"])
    times = np.linspace(row.target_start_s + 0.2, row.target_end_s - 0.2, 200)
    changed = np.vstack([spike, np.column_stack([times, np.ones(200), np.ones(200)])])
    patch_source(monkeypatch, info, changed, [[80.5, 81.2, 80.7, 0], [81.0, 81.5, 81.1, 0]])
    monkeypatch.setattr(coupling, "loadmat", audit.loadmat)
    repeat = coupling.recruitment(tmp_path, row, keys)
    for key in ("ripple_counts", "background_counts", "predictor"):
        np.testing.assert_array_equal(original[key], repeat[key])


def test_native_pilot_keeps_reference_selection_before_future_outcomes(monkeypatch, tmp_path):
    import pandas as pd

    from scripts import calibrate_kleinman_conditional_coupling as coupling
    from scripts import score_kleinman_native_coupling_pilot as pilot

    info, spike = fixture_source()
    events = [[80.5, 81.2, 80.7, 0], [81.0, 81.5, 81.1, 0]]
    patch_source(monkeypatch, info, spike, events)
    monkeypatch.setattr(pilot, "loadmat", audit.loadmat)
    monkeypatch.setattr(coupling, "loadmat", audit.loadmat)
    monkeypatch.setattr(pilot, "recruitment", coupling.recruitment)
    _, rows = audit.session_audit(tmp_path)
    row = pd.Series(next(r for r in rows if r["status"] == "audited" and r["eligible_ripple_s"] > 0))
    row = next(pd.DataFrame([row]).itertuples(index=False))
    cells, summary = pilot.score_anchor(tmp_path, row)
    assert summary["n_reference_units"] == 6 and len(cells) == 6
    from scripts import verify_kleinman_conditional_coupling as conditional_check
    from scripts import verify_kleinman_native_coupling_pilot as native_check

    monkeypatch.setattr(native_check, "loadmat", audit.loadmat)
    monkeypatch.setattr(conditional_check, "loadmat", audit.loadmat)
    monkeypatch.setattr(native_check, "raw_predictors", conditional_check.raw_predictors)
    checked, discrepancy = native_check.check_anchor(tmp_path, row, cells, pd.Series(summary))
    assert checked == 6 and discrepancy < 1e-5
    times = np.linspace(row.target_start_s + 0.2, row.target_end_s - 0.2, 200)
    changed = np.vstack([spike, np.column_stack([times, np.full(len(times), 99), np.ones(len(times))])])
    patch_source(monkeypatch, info, changed, events)
    monkeypatch.setattr(pilot, "loadmat", audit.loadmat)
    monkeypatch.setattr(coupling, "loadmat", audit.loadmat)
    new, new_summary = pilot.score_anchor(tmp_path, row)
    assert new_summary["n_reference_units"] == 6 and len(new) == 7
    assert not new.set_index("unit_id").loc["1_99", "reference_included"]
    for key in ("reference_spikes", "reference_peak_hz", "log_rate_enrichment"):
        np.testing.assert_allclose(cells[key], new.loc[new.unit_id.isin(cells.unit_id), key], equal_nan=True)
