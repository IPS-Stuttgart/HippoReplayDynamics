from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from scripts.audit_denovellis_run_readout import (
    choose_matching, decode_metrics, deduplicate_marks, fit_variant,
    gapped_coordinate, matching_context, parity_check, sample_windows,
)
from scripts.denovellis_post_error_neural import fit_encoding, make_graph


def fixture():
    coords = np.array([[0, 0, 0, 30], [0, 30, -30, 30], [-30, 30, -30, 0],
                       [0, 30, 30, 30], [30, 30, 30, 0]], float)
    wells = np.array([[0, 0], [-30, 0], [30, 0]], float)
    graph = make_graph(coords, wells, 1, (2, 3))
    t = np.arange(0, 8, .04)
    xy = graph.xy[np.resize(np.arange(len(graph.xy)), len(t))]
    speed = np.ones(len(t))*10
    marks = {1: (t+.005, np.ones((len(t), 4))*40), 2: (t+.01, np.ones((len(t), 4))*80)}
    protocol = json.loads((Path(__file__).resolve().parents[1]/"docs/denovellis_run_readout_audit_protocol.json").read_text())
    return graph, t, xy, speed, marks, protocol


def equation(intensity, ground, time_bin_size=1):
    return np.log(intensity+np.spacing(1))-(ground+np.spacing(1))*time_bin_size


def test_frozen_variant_is_identical_to_existing_fit():
    graph, t, xy, speed, marks, p = fixture()
    a = fit_encoding(graph, t, xy, speed, marks, start=0, end=5)
    b, _ = fit_variant(graph, t, xy, speed, marks, 0, 5, "frozen_geodesic", p)
    np.testing.assert_allclose(a.occupancy, b.occupancy)
    for tet in a.features:
        np.testing.assert_allclose(a.features[tet], b.features[tet])
        np.testing.assert_allclose(a.spatial[tet], b.spatial[tet])
        np.testing.assert_allclose(a.rate[tet], b.rate[tet])


def test_parity_including_silent_and_noncontiguous_windows():
    graph, t, xy, speed, marks, p = fixture()
    encoding, _ = fit_variant(graph, t, xy, speed, marks, 0, 5, "frozen_geodesic", p)
    starts = np.array([5., 5.02, 5.20, 6.02])
    assert parity_check(encoding, starts, starts+.02, marks, equation) < 1e-10
    _, counts, _ = encoding.likelihood(starts, starts+.02, marks)
    assert (counts == 0).any()


def test_windows_bounded_and_chronologically_heldout():
    graph, t, xy, speed, _, _ = fixture()
    a = sample_windows(graph, t, xy, speed, 5., t[-1], .02, 7)
    b = sample_windows(graph, t, xy, speed, 5., t[-1], .02, 7)
    np.testing.assert_array_equal(a[0], b[0])
    assert a[0].min() >= 5
    assert a[1].max() <= t[-1]+1e-8
    assert np.all(np.diff(a[0]) > 0)
    assert all(np.sum(a[2] == arm) == 7 for arm in (0, 1))


def test_frame_fit_never_reads_heldout_marks():
    graph, t, xy, speed, marks, p = fixture()
    changed = {tet: (times, features.copy()) for tet, (times, features) in marks.items()}
    for times, features in changed.values():
        features[times >= 5] += 10000
    a, _ = fit_variant(graph, t, xy, speed, marks, 0, 5, "frame_geodesic", p)
    b, _ = fit_variant(graph, t, xy, speed, changed, 0, 5, "frame_geodesic", p)
    for tet in a.features:
        np.testing.assert_allclose(a.features[tet], b.features[tet])
        np.testing.assert_allclose(a.spatial[tet], b.spatial[tet])


def test_same_timestamp_dedup_keeps_first():
    marks = {1: (np.array([0., 1., 1., 2.]), np.arange(16).reshape(4, 4))}
    cleaned, removed = deduplicate_marks(marks)
    assert removed == 1
    np.testing.assert_array_equal(cleaned[1][0], [0, 1, 2])
    np.testing.assert_array_equal(cleaned[1][1], marks[1][1][[0, 1, 3]])


def test_gapped_linear_arms_do_not_share_coordinates():
    graph, *_ = fixture()
    coordinate = gapped_coordinate(graph, 15)
    a, b = [coordinate[m] for m in graph.unique_masks]
    assert b.min() > a.max()
    assert np.all(np.isfinite(coordinate))


def test_latest_match_is_not_latest_run_or_best_qc():
    rows = [{"source_epoch": 2, "context_matches": True, "qc": .1},
            {"source_epoch": 4, "context_matches": False, "qc": 1.}]
    assert choose_matching(rows) == 2
    rows.append({"source_epoch": 3, "context_matches": True, "qc": 0.})
    assert choose_matching(rows) == 3
    assert choose_matching([]) is None


def test_context_missing_and_geometry_mismatch_are_explicit():
    graph, *_ = fixture()
    wells = np.array([[0, 0], [-30, 0], [30, 0]])
    value = [None, None, None, wells, graph.coordinates, 1, (2, 3), None, SimpleNamespace(environment="W")]
    assert matching_context(value, value) == (True, "match")
    changed = list(value)
    changed[8] = SimpleNamespace(environment="")
    assert matching_context(value, changed)[1] == "missing_context"
    changed = list(value)
    changed[4] = graph.coordinates+100
    assert matching_context(value, changed)[1] == "segment_geometry_mismatch"
    assert matching_context(value, changed, check_segments=False)[0]


def test_silent_bins_retained_and_no_spurious_perfect_accuracy():
    graph, t, xy, speed, marks, p = fixture()
    encoding, _ = fit_variant(graph, t, xy, speed, marks, 0, 5, "frame_geodesic", p)
    silent = {tet: (np.array([]), np.empty((0, 4))) for tet in marks}
    _, metric, confusion, *_ = decode_metrics(encoding, np.array([5., 5.1]), np.array([5.02, 5.12]), np.array([0, 1]), silent)
    assert metric["zero_spike_windows"] == 2
    assert metric["balanced_accuracy"] == .5
    assert confusion.sum() == 2


def test_unknown_variant_rejected():
    graph, t, xy, speed, marks, p = fixture()
    with pytest.raises(ValueError, match="Unknown"):
        fit_variant(graph, t, xy, speed, marks, 0, 5, "chosen_after_results", p)
