import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax

from hipporeplayimm.detector_decoder_cross import (
    count_windows,
    detect,
    endpoint_windows,
    event_matches,
    factorial,
    make_populations,
    posterior_readout,
    simulate,
    tile_ids,
    truth_mass,
)
from hipporeplayimm.regional_blind_bank import Geometry
from scripts.measure_edge_support_content import occupied_graph


def test_tiles_geometry_and_edges():
    grid = np.array([[0, 0], [15, 15], [30, 30], [10, 10]])
    np.testing.assert_array_equal(tile_ids(grid, [[0, 0], [30, 30]]), [0, 4, 8, 4])


def test_populations_use_only_run_maps_and_preserve_tetrodes():
    ids = np.arange(24)
    rates = np.ones((24, 9))
    rates[:, 4] = np.arange(1, 25)
    tts = ids // 4
    args = (ids, rates, np.arange(9) == 4, tts)
    a, coverage, missing = make_populations(*args, np.random.default_rng(12))
    b, _, _ = make_populations(*args, np.random.default_rng(12))
    assert a == b and not missing and len(a) == 11
    assert coverage[a[1]["indices"]].max() <= coverage[a[2]["indices"]].min()
    assert set(a[1]["indices"]).isdisjoint(a[2]["indices"])
    for p in a:
        if p["family"] == "whole_tetrode":
            ix = np.array(p["indices"])
            assert set(ix) == set(np.flatnonzero(np.isin(tts, tts[ix])))


def test_missing_tetrodes_are_not_invented():
    pops, _, missing = make_populations(np.arange(20), np.ones((20, 9)), np.arange(9) == 4, np.zeros(20), np.random.default_rng(1))
    assert missing and not any(p["family"] == "whole_tetrode" for p in pops)


def test_half_open_counts_and_silent_windows():
    spikes = np.array([[0, 3], [0.02, 3], [0.025, 4], [0.04, 4]])
    np.testing.assert_array_equal(count_windows(spikes, [3, 4], [[0, 0.02], [0.02, 0.04], [0.1, 0.12]]), [[1, 0], [1, 1], [0, 0]])


def test_endpoint_does_not_shift_for_spike_support():
    w = endpoint_windows([{"event_start_s": 10.0, "event_end_s": 10.103}])
    np.testing.assert_allclose(w, [[10.08, 10.10]])


def test_poisson_likelihood_against_direct_sum_and_silence():
    r = np.array([[1.0, 2.0, 5.0], [5.0, 1.0, 2.0]])
    n = np.array([[2, 1], [0, 0]])
    result = posterior_readout(n, r, [0, 4, 8], "poisson")
    exact = np.array([softmax(np.sum(x[:, None] * np.log(r) - 0.02 * r, axis=0)) for x in n])
    np.testing.assert_allclose(result["regional"][:, [0, 4, 8]], exact)
    assert not np.allclose(exact[1], 1 / 3)


def test_conditional_common_gain_invariance_and_zero_uniform():
    r = np.array([[1.0, 2.0, 5.0], [5.0, 1.0, 2.0]])
    n = np.array([[2, 1], [0, 0]])
    a = posterior_readout(n, r, [0, 4, 8], "conditional_multinomial")
    b = posterior_readout(n, 9 * r, [0, 4, 8], "conditional_multinomial")
    np.testing.assert_allclose(a["regional"], b["regional"])
    np.testing.assert_allclose(a["regional"][1, [0, 4, 8]], 1 / 3)


def test_exact_truth_arbitrary_edges_and_crossing():
    t = truth_mass([0, 4, 4, 8], [[0.0025, 0.0175], [0, 0.02]])
    np.testing.assert_allclose(t[:, 4], [2 / 3, 0.5])
    np.testing.assert_allclose(t.sum(axis=1), 1.0)
    with pytest.raises(ValueError):
        truth_mass([4], [[0, 0.02]])


def test_factorial_identity_and_nonvacuous_empty():
    f = factorial(0.1, 0.15, 0.2, 0.3)
    assert f["detection"] == pytest.approx(0.05)
    assert f["decoding"] == pytest.approx(0.1)
    assert f["interaction"] == pytest.approx(0.05)
    assert f["total"] == pytest.approx(sum(f[k] for k in ("detection", "decoding", "interaction")))
    assert factorial(0.1, np.nan, 0.2, np.nan)["status"] == "incomplete"


def test_matches_do_not_hide_split_events_or_unmatched():
    ix, _ = event_matches([[0, 0.2]], [[0, 0.1], [0.11, 0.15], [0.3, 0.4]])
    np.testing.assert_array_equal(ix, [0, 0, -1])


def test_detector_gets_only_its_own_cells_and_unique_ids():
    class Detector:
        def detect_high_mua_in_interval(self, spikes, times, speeds, start, end, **kwargs):
            assert set(spikes[:, 1]) == {1}
            assert kwargs["minimum_active_cells"] == 1
            return [{"event_start_s": start + 0.1, "event_end_s": start + 0.2}] * 2

    events = detect(Detector(), np.array([[0.12, 1], [0.13, 2]]), np.array([1]), [0, 4], [0, 0], [[0, 1], [2, 3]])
    assert [e["event_id"] for e in events] == list(range(4))


def test_region_blind_simulation_reproducible_sorted_and_no_fixed_totals():
    grid = np.array([[0.0, 0.0], [8.0, 0.0], [0.0, 8.0], [8.0, 8.0]])
    geo = Geometry.from_grid(grid, occupied_graph(grid))
    rates = np.array([[1.0, 2.0, 4.0, 8.0], [8.0, 4.0, 2.0, 1.0]])
    for kind in ("stationary", "moving"):
        a = simulate(geo, rates, np.array([11, 12]), kind, 3, 6, np.random.default_rng(4))
        b = simulate(geo, rates, np.array([11, 12]), kind, 3, 6, np.random.default_rng(4))
        for x, y in zip(a, b, strict=True):
            np.testing.assert_array_equal(x, y)
        assert a[1].shape == (1200, 2) and a[2].shape == (3, 2)
        assert np.all(np.diff(a[0][:, 0]) >= 0) and a[0][:, 0].max() < 6


def test_source_cross_uses_identical_windows_per_detector(tmp_path):
    from scripts.measure_detector_decoder_cross import source_measure

    class Detector:
        def detect_high_mua_in_interval(self, spikes, times, speeds, start, end, **kwargs):
            edge = 0.1 if len(np.unique(spikes[:, 1])) == 2 else 0.105
            return [{"event_start_s": 0.0, "event_end_s": edge, "event_peak_s": 0.05}]

    pops = [{"name": "full", "family": "full", "side": "full", "repeat": 0, "indices": [0, 1]}, {"name": "subset", "family": "random", "side": "a", "repeat": 0, "indices": [0]}]
    spikes = np.array([[0.09, 1], [0.095, 2]])
    r = np.array([[1.0, 9.0], [9.0, 1.0]])
    source_measure(
        tmp_path / "source",
        spikes,
        None,
        np.array([1, 2]),
        r,
        [0, 4],
        pops,
        Detector(),
        [0, 1],
        [0, 0],
        [[0, 1]],
        {"dataset": "test", "animal": "a", "session": "s", "source": "real", "generator": "real", "peak_gain": 0, "replicate": 0},
    )
    rows = pd.read_csv(tmp_path / "source/window_readouts.csv.gz")
    assert rows.groupby("detector").start_s.nunique().eq(1).all()
    contrast = pd.read_csv(tmp_path / "source/factorial_contrasts.csv")
    np.testing.assert_allclose(contrast.total, contrast.detection + contrast.decoding + contrast.interaction, atol=1e-14)
    assert rows.filter(regex="^truth_").isna().all().all()
