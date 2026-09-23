"""Synthetic-only tests; no privately supplied annotations or raw NWB required."""

import numpy as np
import pandas as pd
import pytest

from scripts import dandi000978_verified_units as mapping
from scripts import validate_dandi000978_run_readouts as run


def table(animal="ZT2", ids=(7, 0), refs=(1, 2), crosswalk=None):
    if crosswalk is None:
        crosswalk = pd.DataFrame({"Cell": [0, 7], "Tetrode": [2, 1], "tetCellDandiIdx3": [3, 4]})
    filename = next(iter(mapping.ZT2_CROSSWALKS)) if animal == "ZT2" else "sub-JDS-SingleDay-JS14_behavior+ecephys.nwb"
    return mapping.verified_unit_table(animal, filename, ids, refs, ["CA1", "CA1", "PFC", "PFC"], ["Tetrode1", "Tetrode1", "Tetrode2", "Tetrode2"], crosswalk)


def test_mapping_uses_ids_not_rows_and_preserves_source_scope():
    out = table()
    assert list(out.region) == ["CA1", "PFC"]
    assert list(out.stable_unit_key) == ["ZT2:t1:c4", "ZT2:t2:c3"]
    assert out.cross_file_source_identity_verified.all()
    other = table(animal="JS14")
    assert not other.cross_file_source_identity_verified.any()
    assert "not_explicit_crosswalk" in other.mapping_basis.iloc[0]
    with pytest.raises(ValueError, match="not source-confirmed"):
        table(animal="JS15")


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_tetrode", "noninteger", "duplicate_cluster"])
def test_bad_crosswalks_fail_closed(change):
    c = pd.DataFrame({"Cell": [0.0, 7.0], "Tetrode": [2.0, 1.0], "tetCellDandiIdx3": [3.0, 4.0]})
    if change == "missing":
        c = c.iloc[:1]
    elif change == "duplicate":
        c.loc[1, "Cell"] = 0
    elif change == "wrong_tetrode":
        c.loc[0, "Tetrode"] = 1
    elif change == "noninteger":
        c.loc[0, "Cell"] = 0.5
    else:
        c.loc[1, ["Tetrode", "tetCellDandiIdx3"]] = c.loc[0, ["Tetrode", "tetCellDandiIdx3"]]
    with pytest.raises(ValueError):
        table(crosswalk=c)


def test_conflicting_anatomy_and_unpinned_archive_rejected(tmp_path):
    with pytest.raises(ValueError, match="Conflicting"):
        mapping.verified_unit_table("JS14", "sub-JDS-SingleDay-JS14_behavior+ecephys.nwb", [0], [1], ["CA1", "PFC"], ["Tetrode1", "Tetrode1"])
    path = tmp_path / "source.zip"
    path.write_bytes(b"not the original private source")
    with pytest.raises(ValueError, match="pinned"):
        mapping.read_author_crosswalks(path)


def synthetic():
    rows, counts = [], []
    for epoch in range(3):
        for trial in range(12):
            route = trial % 4
            for step in range(20):
                row = {
                    "epoch": epoch,
                    "trial_id": epoch * 12 + trial,
                    "route": route,
                    "x_cm": step * 4 + 2,
                    "y_cm": route * 24 + 4,
                    "start_s": (epoch * 240 + trial * 20 + step) * 0.25,
                    "stop_s": (epoch * 240 + trial * 20 + step + 1) * 0.25,
                    "speed_cm_s": 16.0,
                }
                rows.append(row)
                c = np.zeros(24, int)
                c[route * 4 : route * 4 + 4] = 2
                c[16 + step // 3] = 4
                counts.append(c)
    return pd.DataFrame(rows), np.array(counts)


def test_whole_epoch_split_and_train_only_maps():
    bins, counts = synthetic()
    train, test = run.epoch_masks(bins, 1)
    assert set(bins.loc[train, "trial_id"]).isdisjoint(bins.loc[test, "trial_id"])
    xy = bins[["x_cm", "y_cm"]].to_numpy(copy=True)
    first = run.fit_spatial(counts, xy, train)
    counts[test] += 10000
    xy[test] += 10000
    second = run.fit_spatial(counts, xy, train)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
    # A held-out-only unit must never enter the encoding model.
    assert not first[1][-1]
    broken = bins.copy()
    broken.loc[broken.epoch == 1, "trial_id"] = 0
    with pytest.raises(ValueError, match="crosses"):
        run.epoch_masks(broken, 1)


def test_composition_cannot_read_rate_only_differences():
    counts = np.array([[1, 2, 3], [5, 10, 15]])
    rates = np.array([[1, 2, 3], [10, 20, 30]])
    p = run.posterior(counts, rates, np.ones(2), "composition")
    np.testing.assert_allclose(p, 0.5)
    assert not np.allclose(run.posterior(counts, rates, np.ones(2), "poisson"), 0.5)
    zero = run.posterior(np.zeros((1, 3)), rates, np.ones(1), "composition")
    np.testing.assert_allclose(zero, 0.5)


def test_recovery_vs_count_only_and_seeded_whole_trial_nulls(monkeypatch):
    monkeypatch.setitem(run.PARAMETERS, "shuffles", 19)
    bins, counts = synthetic()
    rows, predictions, nulls = run.route_scores(bins, counts, 1, ("synthetic", "CA1", 1))
    scores = {r["readout"]: r for r in rows}
    assert scores["composition"]["balanced_accuracy"] == 1
    assert scores["composition"]["above_null_p95"]
    assert scores["count_only"]["balanced_accuracy"] == 0.25
    assert set(predictions.epoch) == {1}
    assert len(predictions) == 12 * 3
    again = run.route_scores(bins, counts, 1, ("synthetic", "CA1", 1))[2]
    pd.testing.assert_frame_equal(nulls, again)


def test_training_label_permutations_respect_epochs():
    labels = np.array([0, 0, 1, 2, 3, 3])
    epochs = np.array([0, 0, 0, 1, 1, 1])
    result = run.permute_training_routes(labels, epochs, np.random.default_rng(10))
    for ep in (0, 1):
        assert sorted(result[epochs == ep]) == sorted(labels[epochs == ep])


def test_missing_test_route_is_not_a_pass():
    bins, counts = synthetic()
    keep = ~((bins.epoch == 1) & (bins.route == 3))
    with pytest.raises(ValueError, match="lacks"):
        run.route_scores(bins.loc[keep].reset_index(drop=True), counts[keep], 1, ("missing",))


def test_spatial_decode_and_null_are_trial_weighted(monkeypatch):
    monkeypatch.setitem(run.PARAMETERS, "shuffles", 9)
    bins, counts = synthetic()
    train, test = run.epoch_masks(bins, 1)
    rows, predictions, audit = run.spatial_scores(bins, counts, train, test, ("synthetic",))
    assert len(rows) == 2 and set(predictions.epoch) == {1}
    assert all(r["gain_over_null_cm"] > 0 for r in rows)
    assert np.isfinite(audit["posterior_composition"]).all()
    assert set(audit["train_indices"]).isdisjoint(audit["test_indices"])
    assert run.trial_weighted_mean([0, 0, 0, 10], [0, 0, 0, 1]) == 5


def test_bins_do_not_cross_trial_boundaries_and_filter_behavior_only():
    clock = np.arange(0, 2, 1 / 30)
    data = np.column_stack([clock * 20, np.zeros(len(clock)), np.full(len(clock), 20)])
    arrays = {"clock": clock, "position_and_speed": data, "spike_ends": np.array([4, 4]), "spikes": np.array([0.0, 0.25, 1.0, 1.75])}
    trials = pd.DataFrame(
        {
            "id": [0, 1],
            "epoch_index": [0, 1],
            "start_time": [0.0, 1.0],
            "stop_time": [1.0, 2.0],
            "start_well": [1, 2],
            "end_well": [2, 1],
            "correct": [1, 0],
            "trajectory_type": [0, 1],
        }
    )
    bins, counts, audit = run.prepare_bins(arrays, trials)
    assert len(bins) == 8 and counts.sum() == 4
    assert counts[:, 1].sum() == 0  # silent unit is not a reason to exclude behavior bins
    assert list(audit.eligible_bins) == [4, 4]
    assert not audit.native_direction_disagrees_with_endpoint_rule.any()
    assert counts[1, 0] == 1  # the .25 spike belongs only to the second half-open bin
    assert bins.loc[bins.trial_id == 0, "stop_s"].max() == 1


def test_summary_keeps_failed_folds_and_pending_animals(tmp_path):
    rows = [{"animal": "JS14", "file": "synthetic", "region": "PFC", "task": "route", "readout": "composition", "status": "failed", "failure_reason": "no routes"}]
    assert not run.summarize(tmp_path, rows, [{"animal": "JS14", "status": "failed"}])
    gates = pd.read_csv(tmp_path / "gate_summary.csv").set_index("gate").passed
    assert not gates.all_expected_readouts_scored
    readiness = pd.read_csv(tmp_path / "readiness_by_file_region.csv")
    assert readiness.expected_route_folds.iloc[0] == 1
    assert not readiness.RUN_route_content_supported.iloc[0]
