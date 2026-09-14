import numpy as np
import pandas as pd
import pytest

from scripts.prepare_autopi_three_population_inputs import (
    detect_rest_candidates,
    extract_session,
    foraging_rest_pair,
    select_ca1_units,
)


def metadata():
    info = pd.DataFrame(dict(cluster_id=[0, 7, 99], group=["good", "good", "noise"], sh=[0, 1, 0]))
    groups = info[["cluster_id", "group"]].copy()
    shanks = pd.DataFrame(dict(id=["r-1_0", "r-1_7"], shank=[1, 2], layer=["pyr", "dn"]))
    return info, groups, shanks


def test_native_epoch_no_later_replacement():
    names = ["circ80", "autopi", "circ80", "rest"]
    intervals = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    with pytest.raises(ValueError, match="immediately"):
        foraging_rest_pair(names, intervals)
    i, run, rest = foraging_rest_pair(["circ80", "rest"], intervals[:2])
    assert i == 0
    np.testing.assert_equal(run, [0, 1])
    np.testing.assert_equal(rest, [1, 2])


def test_epoch_overlap_rejected():
    with pytest.raises(ValueError, match="clock"):
        foraging_rest_pair(["circ80", "rest"], [[0, 2], [1, 3]])


def test_author_good_and_ca1_not_kslabel_or_layer():
    info, groups, shanks = metadata()
    info["KSLabel"] = "mua"
    units = select_ca1_units(info, groups, shanks, ["CA1", "MEC"], "r-1")
    assert units.cluster_id.tolist() == [0, 7]
    assert units.included_ca1.tolist() == [True, False]
    assert units.cell_type.str.startswith("unknown").all()


def test_inconsistent_curation_excluded_and_explicit_electrode_used():
    info, groups, shanks = metadata()
    groups.loc[0, "group"] = "noise"
    units = select_ca1_units(info, groups, shanks, ["ca1", "ca1"], "r-1")
    assert units.included_ca1.tolist() == [False, True]
    info, groups, shanks = metadata()
    info.loc[0, "sh"] = 1
    units = select_ca1_units(info, groups, shanks, ["ca1", "ca1"], "r-1")
    assert units.included_ca1.tolist() == [True, True]


def test_missing_legacy_entry_and_unmapped_region_are_explicit():
    info, groups, shanks = metadata()
    info = info.rename(columns={"cluster_id": "id"})
    groups = groups.loc[groups.cluster_id.ne(0)]
    units = select_ca1_units(info, groups, shanks, ["ca1"], "r-1")
    assert units.included_ca1.tolist() == [True, False]
    assert units.curation_status.iloc[0] == "missing_legacy_group_entry"
    assert units.brain_region.iloc[1] == "unknown_unmapped_exclude"


def test_mua_fixed_clock_support_and_no_boundary_event():
    times = np.r_[np.linspace(12, 12.08, 40), np.linspace(15, 15.08, 40)]
    spikes = np.column_stack([times, np.tile([0, 7, 20, 24], 20)])
    events, summary = detect_rest_candidates(spikes, [0, 7, 20, 24], [10, 20])
    assert len(events) == 2
    assert (events.n_spikes == 40).all() and (events.n_active_units == 4).all()
    assert (events.end_s-events.start_s >= .05).all()
    assert ((events.start_s >= 10) & (events.end_s < 20)).all()
    assert summary["n_rest_spikes"] == 80
    spikes[:, 1] = 0
    assert detect_rest_candidates(spikes, [0, 7, 20, 24], [10, 20])[0].empty
    assert detect_rest_candidates(np.empty((0, 2)), [0, 7, 20], [10, 20])[0].empty


def test_native_fixture_seconds_cm_ids_preserved(tmp_path):
    folder = tmp_path/"r"/"r-1"
    folder.mkdir(parents=True)
    info, groups, shanks = metadata()
    info.to_csv(folder/"cluster_info.tsv", sep="\t", index=False)
    groups.to_csv(folder/"cluster_group.tsv", sep="\t", index=False)
    shanks.to_csv(folder/"r-1.shank_neuron.csv", index=False)
    (folder/"r-1.desen").write_text("circ80\nrest\n")
    (folder/"r-1.desel").write_text("ca1\nca1\n")
    (folder/"r-1.sampling_rate_dat").write_text("20000\n")
    np.save(folder/"sessionIntervals.npy", [[0, 10], [10, 20]])
    np.save(folder/"spike_times.npy", np.array([10000, 20000, 220000], dtype=np.uint64)[:, None])
    np.save(folder/"spike_clusters.npy", [0, 99, 7])
    np.save(folder/"r-1.pose.npy", [[0, 1, 2], [1, 3, 4], [11, 500, 600]])
    result = extract_session(folder, tmp_path/"out")
    data = np.load(result["artifact_path"], allow_pickle=False)
    np.testing.assert_equal(data["cell_ids"], [0, 7])
    np.testing.assert_equal(data["spikes"], [[.5, 0], [11, 7]])
    np.testing.assert_equal(data["position"], [[0, 1, 2], [1, 3, 4]])
    assert result["candidates"] == 0
