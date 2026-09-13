import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("goal_content", ROOT / "scripts/analyze_pf_recording_goal_content.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
audit_spec = importlib.util.spec_from_file_location("goal_audit", ROOT / "scripts/audit_pf_recording_goal_content.py")
audit = importlib.util.module_from_spec(audit_spec)
audit_spec.loader.exec_module(audit)


def test_flat_poisson_matches_dense_and_cell_order_invariant():
    rng = np.random.default_rng(15)
    c, r = rng.poisson(1, (20, 7)), rng.uniform(.001, 15, (7, 12))
    expected = softmax(c @ np.log(r) - .02*r.sum(axis=0), axis=1)
    assert np.allclose(m.decode(c, r), expected)
    order = rng.permutation(7)
    assert np.allclose(m.decode(c[:, order], r[order]), expected)
    assert np.allclose(audit.posterior(c[-1], r), expected[-1])


def test_zero_spikes_not_flat_when_rates_differ():
    p = m.decode(np.zeros((1, 2)), np.array([[1., 100.], [1., 100.]]))[0]
    assert p[0] > p[1]
    assert p.sum() == pytest.approx(1)


@pytest.mark.parametrize("bad", [np.array([[np.nan, 1]]), np.array([[-1, 1]]), np.array([[.1, 1]])])
def test_invalid_counts(bad):
    with pytest.raises(ValueError):
        m.decode(bad, np.ones((2, 5)))


def test_home_inference_uses_next_fill_and_is_not_position_origin():
    seq = np.column_stack([np.arange(13)*10., [15, 1, 15, 2, 15, 3, 15, 4, 15, 5, 15, 6, 15]])
    pos = []
    for i in range(12):
        for t in [seq[i+1, 0]-.8, seq[i+1, 0]-.5, seq[i+1, 0]-.2]:
            xy = [100, 75] if seq[i, 1] == 15 else [10, 20]
            pos.append([t, *xy])
    meta, visits = m.infer_home(seq, np.array(pos))
    assert meta["home_id"] == 15
    assert meta["home_x_cm"] == 100
    assert meta["home_y_cm"] == 75
    assert len(visits) == 6


@pytest.mark.parametrize("labels", [[1, 2, 3, 1, 2], [1, 1, 1, 1, 1], [1, 2, 1, 2, 1]])
def test_ambiguous_or_missing_home_fails(labels):
    seq = np.column_stack([np.arange(len(labels)), labels])
    with pytest.raises(ValueError):
        m.infer_home(seq, np.zeros((100, 3)))


def test_longest_earliest_and_support_edges():
    grid = np.array([[0, 0], [8, 0], [16, 0], [80, 0], [88, 0], [96, 0]])
    path = np.array([0, 1, 2, 3, 4, 5])
    counts = np.full((6, 2), 1)
    assert m.longest_segment(path, grid, counts).tolist() == [0, 1, 2]
    counts[0] = 0
    assert m.longest_segment(path, grid, counts).tolist() == [3, 4, 5]
    assert not len(m.longest_segment(path, grid, np.zeros((6, 2))))


def test_twenty_cm_jump_not_continuous():
    grid = np.array([[0., 0], [20, 0], [28, 0]])
    assert m.longest_segment(np.arange(3), grid, np.ones((3, 2))).tolist() == [1, 2]


def test_independent_segment_anchor_reconstruction():
    rng = np.random.default_rng(772)
    grid = rng.uniform(0, 200, (100, 2))
    for _ in range(50):
        path, counts = rng.integers(0, 100, 25), rng.poisson(.1, (25, 10))
        segment = m.longest_segment(path, grid, counts)
        expected = int(segment[-1]) if len(segment) else None
        assert audit.segment_endpoint(path, counts, grid) == expected


def test_decomposition_is_exact_and_selection_is_separate():
    f = pd.DataFrame(dict(accepted_full=[True, True, False], accepted_half=[False, True, True],
        trajectory_home_mass_full=[.2, .4, .8], trajectory_home_mass_half=[.3, .5, .9],
        own_home_mass_half=[.7, .6, 1.]))
    d = m.decompose(f)
    assert d["decoding"] == pytest.approx(.1)
    assert d["composition"] == pytest.approx(.3)
    assert d["timing"] == pytest.approx(.1)
    assert d["total"] == pytest.approx(.5)
    assert d["total"] == pytest.approx(d["decoding"]+d["composition"]+d["timing"])


def test_no_selection_group_is_not_zero_effect():
    f = pd.DataFrame(dict(accepted_full=[True], accepted_half=[False]))
    assert np.isnan(m.decompose(f)["total"])


def test_home_radius_uses_physical_coordinates():
    grid = np.array([[0., 0], [20., 0], [40., 0]])
    feat = m.home_features(np.array([.2, .5, .3]), grid, np.array([0, 0]))
    assert feat["home_mass_r20"] == pytest.approx(.7)
    assert feat["home_map"]


def test_simulation_freezes_events_and_recovers_identical_subset():
    x, y = np.meshgrid(np.arange(0, 201, 8), np.arange(0, 201, 8))
    grid = np.column_stack([x.ravel(), y.ravel()])
    r = np.random.default_rng(9).uniform(.1, 15, (10, len(grid)))
    specs = [dict(indices=list(range(10)), cell_fraction=1., population_replicate=0),
             dict(indices=list(range(10)), cell_fraction=.5, population_replicate=0)]
    a = m.simulate(r, grid, np.array([96, 96]), specs, 1234, n=4)
    b = m.simulate(r, grid, np.array([96, 96]), specs, 1234, n=4)
    pd.testing.assert_frame_equal(a, b)
    f, h = a[a.cell_fraction.eq(1)], a[a.cell_fraction.eq(.5)]
    assert np.array_equal(f.endpoint_error_cm, h.endpoint_error_cm)
    assert f.true_home.sum() == 2
    assert np.array_equal(f.home_mass_r20, h.home_mass_r20)


def test_equal_rat_summary_not_event_count_weighted():
    events = []
    for animal, n, shift in [("Rat1", 10, 10), ("Rat2", 1, 30)]:
        for i in range(n):
            events.append(dict(animal=animal, session=animal+"/Open1", population_replicate=0,
                endpoint_status="available", accepted_full=True, accepted_half=True,
                endpoint_mean_shift_cm=shift, endpoint_map_shift_cm=shift, endpoint_posterior_tv=.2,
                path_mean_shift_cm=shift, home_map_disagreement=False,
                home_mass_r10_full=.2, home_mass_r10_half=.2,
                home_mass_r20_full=.2, home_mass_r20_half=.2,
                home_mass_r30_full=.2, home_mass_r30_half=.2,
                trajectory_home_mass_full=.2, trajectory_home_mass_half=.2, own_home_mass_half=.2))
    sim = pd.DataFrame([dict(animal="Rat1", session="Rat1/Open1", simulation_event=0,
        cell_fraction=f, population_replicate=0, true_home=True, home_map=True, endpoint_error_cm=1)
        for f in [1., .5]])
    _, _, _, summary = m.summarize(pd.DataFrame(events), sim, bootstraps=20)
    s = summary[summary.cohort.eq("all_fixed_candidates") & summary.metric.eq("endpoint_mean_shift_cm")]
    assert s.equal_rat_mean.iloc[0] == 20
