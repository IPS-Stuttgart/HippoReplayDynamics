import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from scripts import audit_kleinman_nonoverlapping_coupling_design as audit


def runs(n=24):
    return [{"traversal": i, "start_s": 10 * i + 2.0, "end_s": 10 * i + 10.0, "epoch": 1, "direction": i % 2} for i in range(n)]


def test_six_runs_nonoverlap_across_directions_and_determinism():
    rows = audit.candidate_packets(runs())
    selected = sorted((r for r in rows if r["selected"]), key=lambda r: r["span_start_s"])
    assert len(rows) == 14
    assert len(selected) == 2
    assert {r["direction"] for r in selected} == {0, 1}
    assert all(a["span_end_s"] <= b["span_start_s"] for a, b in itertools.pairwise(selected))
    assert rows == audit.candidate_packets(list(reversed(runs())))
    for p in rows:
        chain = p["reference"] + [p["past"], p["baseline"], p["target"]]
        assert len(chain) == 6
        assert all(a["end_s"] < b["start_s"] for a, b in itertools.pairwise(chain))
        assert len({r["direction"] for r in chain}) == 1


def test_earliest_finish_has_maximal_count_on_small_fixture():
    rows = audit.candidate_packets(runs(20))
    best = 0
    for keep in itertools.product((False, True), repeat=len(rows)):
        chosen = sorted([r for r, k in zip(rows, keep, strict=True) if k], key=lambda r: r["span_start_s"])
        if all(a["span_end_s"] <= b["span_start_s"] for a, b in itertools.pairwise(chosen)):
            best = max(best, len(chosen))
    assert sum(r["selected"] for r in rows) == best


def test_epochs_never_cross():
    rs = runs(24)
    for r in rs:
        r["epoch"] = r["traversal"] // 12 + 1
    rows = audit.candidate_packets(rs)
    assert len(rows) == 4
    assert sum(p["selected"] for p in rows) == 2
    for p in rows:
        assert len({r["epoch"] for r in p["reference"] + [p["past"], p["baseline"], p["target"]]}) == 1


@pytest.mark.parametrize("kind", ["overlap", "duplicate", "nan"])
def test_invalid_runs_rejected(kind):
    rs = runs()
    if kind == "overlap":
        rs[1]["start_s"] = rs[0]["end_s"] - 0.1
    elif kind == "duplicate":
        rs[1]["traversal"] = rs[0]["traversal"]
    else:
        rs[1]["end_s"] = np.nan
    with pytest.raises(ValueError):
        audit.candidate_packets(rs)


def fixture():
    t = np.arange(3841) / 20
    lap = np.minimum((t // 10).astype(int), 19)
    phase, side = t - lap * 10, lap % 2
    x = np.clip(np.where(phase <= 2, side * 200, np.where(side == 0, (phase - 2) * 25, 200 - (phase - 2) * 25)), 0, 200)
    speed = np.where((phase > 2) & (phase < 10), 25, 0)
    visits = [[round(10 * i * 20) + 2, round((10 * i + 2) * 20) + 2] for i in range(20)]
    info = {
        "position": np.r_[x[0], x],
        "velocity": np.column_stack([t, speed]),
        "reward_ends": [5, 195],
        "left_visit": visits[::2],
        "right_visit": visits[1::2],
        "epoch_change": np.empty((0, 2)),
        "drug": 0,
        "novel": 1,
    }
    s = np.arange(0.01, 192, 0.2)
    spikes = np.concatenate([np.column_stack([s + i * 0.01, np.full(len(s), i + 1), np.ones(len(s))]) for i in range(6)])
    return info, spikes


def patch(monkeypatch, info, spikes, ripples):
    d = {"session_info": info, "spike_data": spikes, "ripple_events": np.asarray(ripples).reshape(-1, 4)}
    monkeypatch.setattr(audit, "loadmat", lambda p, **kw: {p.stem: d[p.stem]})


def test_missing_ripple_is_visible_not_replaced(monkeypatch, tmp_path):
    info, spike = fixture()
    patch(monkeypatch, info, spike, [])
    rows = audit.packet_coverage(tmp_path)
    selected = [r for r in rows if r["selected"]]
    assert selected
    assert all(r["n_reference_units"] == 6 for r in selected)
    assert all("no_ripple_exposure" in r["coverage_status"] for r in selected)
    assert not any(r["availability_descriptor"] for r in rows)
    patch(monkeypatch, info, spike, [[90.5, 91.5, 91.0, 0]])
    changed = audit.packet_coverage(tmp_path)
    assert [(r["packet_id"], r["selected"]) for r in rows] == [(r["packet_id"], r["selected"]) for r in changed]
    assert any(r.get("eligible_ripple_s", 0) > 0 for r in changed)


def test_future_spikes_cannot_select_cells_or_change_predictor(monkeypatch, tmp_path):
    info, spike = fixture()
    patch(monkeypatch, info, spike, [[90.5, 91.5, 91.0, 0]])
    rows = audit.packet_coverage(tmp_path)
    first = next(r for r in rows if r["selected"])
    s = np.linspace(first["target_start_s"] + 0.1, first["target_end_s"] - 0.1, 500)
    extra = np.column_stack([s, np.full(len(s), 99), np.ones(len(s))])
    patch(monkeypatch, info, np.vstack([spike, extra]), [[90.5, 91.5, 91.0, 0]])
    other = next(r for r in audit.packet_coverage(tmp_path) if r["packet_id"] == first["packet_id"])
    for key in ("n_reference_units", "reference_spikes", "ripple_reference_spikes", "background_reference_spikes", "variable_recruitment", "selected"):
        assert first[key] == other[key]


def test_metadata_does_not_infer_track_from_geometry_or_novelty(tmp_path):
    p = tmp_path / "Experiment_1" / "Exp_3" / "20190609_run2" / "session_info.mat"
    p.parent.mkdir(parents=True)
    info, _ = fixture()
    savemat(p, {"session_info": info})
    row = audit.metadata(p)
    assert row["session_ordinal"] == 2
    assert row["date"] == "20190609"
    assert row["track_candidate_fields"] == "[]"
    assert row["authoritative_track_id_available"] is False


def test_zero_condition_cells_retained():
    sf = pd.DataFrame([{"animal": "A", "drug": 0, "novel": 1, "run_pass": True, "status": "audited"}])
    pf = pd.DataFrame(columns=["animal", "session", "drug", "novel", "selected", "availability_descriptor", "minimum_units_descriptor", "direction", "date"])
    result = audit.coverage_table(sf, pf)
    assert len(result) == 4
    assert result.available_packets.sum() == 0
    assert result.source_sessions.sum() == 1
