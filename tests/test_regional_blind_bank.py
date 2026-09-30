import inspect

import numpy as np
import pytest

from hipporeplayimm.regional_blind_bank import BlindPath, Geometry, allocate_identities, circle_interval, path_content, sample_conditional_event
from scripts.measure_edge_support_content import occupied_graph


@pytest.fixture
def geometry():
    grid = np.array([(x, y) for x in np.arange(0, 81, 8) for y in np.arange(0, 81, 8)], float)
    return Geometry.from_grid(grid, occupied_graph(grid))


def path(geometry, kind="jumping", seed=1):
    return BlindPath(geometry, kind, 500., [12., 20., 35.], [.01, .02, .06], 1., np.random.default_rng(seed))


def test_generator_has_no_region_or_cell_tuning_input():
    params = set(inspect.signature(BlindPath).parameters)
    assert params == {"geometry", "kind", "speed", "jump_lengths", "intervals", "scale", "rng"}


@pytest.mark.parametrize("kind", ["stationary", "moving", "jumping"])
def test_relabeling_region_does_not_change_path(geometry, kind):
    p = path(geometry, kind).extend(.2)
    ages, nodes = list(p.ages), list(p.nodes)
    path_content(p, np.array([10., 10.]), 20., .1)
    path_content(p, np.array([70., 70.]), 20., .1)
    assert p.ages == ages and p.nodes == nodes
    again = path(geometry, kind).extend(.2)
    np.testing.assert_array_equal(again.ages, p.ages)
    np.testing.assert_array_equal(again.nodes, p.nodes)


def test_exact_disc_intersection():
    np.testing.assert_allclose(circle_interval([-2, 0], [2, 0], [0, 0], 1), [.25, .75])
    assert circle_interval([-2, 2], [2, 2], [0, 0], 1) is None
    assert circle_interval([0, 0], [0, 0], [0, 0], 1) == (0., 1.)


def test_moving_dwell_not_sampled_endpoint_label(geometry):
    p = path(geometry, "moving")
    a = int(np.flatnonzero(np.all(geometry.grid == [0, 40], axis=1))[0])
    b = int(np.flatnonzero(np.all(geometry.grid == [80, 40], axis=1))[0])
    p.ages, p.nodes = [0., .1], [a, b]
    c = path_content(p, np.array([40., 40.]), 20., .1)
    assert c["label"] and not c["endpoint_home"]
    assert abs(c["dwell_s"]-.05) < 1e-12
    assert abs(c["occupancy_fraction"]-.5) < 1e-12


def test_cumulative_dwell_and_longest_visit_are_distinct(geometry):
    p = path(geometry)
    h = int(np.flatnonzero(np.all(geometry.grid == [40, 40], axis=1))[0])
    o = int(np.flatnonzero(np.all(geometry.grid == [0, 0], axis=1))[0])
    p.ages, p.nodes = [0., .01, .02, .03, .1], [h, o, h, o, o]
    c = path_content(p, np.array([40., 40.]), 20., .04)
    assert c["label"]
    assert c["dwell_s"] == pytest.approx(.02)
    assert c["longest_visit_s"] == pytest.approx(.01)


def test_twenty_ms_positive_requires_full_occupancy(geometry):
    p = path(geometry, "stationary").extend(.1)
    c = path_content(p, geometry.grid[p.nodes[0]], 20., .02)
    assert c["label"] and c["occupancy_fraction"] == 1.


def test_prefix_extension_does_not_change_terminal_geometry(geometry):
    p = path(geometry).extend(.1)
    c = path_content(p, np.array([40., 40.]), 20., .1)
    p.extend(.7)
    again = path_content(p, np.array([40., 40.]), 20., .1)
    assert c["dwell_s"] == again["dwell_s"]
    np.testing.assert_array_equal(c["jump_crossings"], again["jump_crossings"])


def test_jump_phase_not_fixed_at_five_ms(geometry):
    phases = [path(geometry, seed=k).extend(.1).ages[1] for k in range(100)]
    assert len(np.unique(phases)) == 100
    assert min(phases) < .005 < max(phases)


def test_spike_identity_allocation_keeps_exact_number(geometry):
    p = path(geometry, "moving").extend(.1)
    rates = np.ones((10, len(geometry.grid)))
    result = allocate_identities(p, np.linspace(0, .1, 23), rates, np.random.default_rng(1))
    assert len(result) == 23
    assert ((result >= 0) & (result < 10)).all()


def test_impossible_label_fails_without_home_aware_generator(geometry):
    with pytest.raises(RuntimeError, match="conditional sampling failed"):
        sample_conditional_event(geometry, "stationary", 500., [20.], [.03], 1., np.array([1000., 1000.]),
            .1, True, {"start": 0., "end": .2, "times": np.array([.1])},
            np.ones((10, len(geometry.grid))), np.random.default_rng(1), max_attempts=5)
