from types import SimpleNamespace

import numpy as np
import pytest

from scripts import audit_kleinman_nonoverlapping_coupling_design as audit
from scripts import calibrate_kleinman_conditional_coupling as coupling
from scripts import score_kleinman_native_joint_temporal_pilot as native


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


def patched(monkeypatch, extra=None):
    info, spikes = fixture()
    spikes = np.vstack([spikes, [[90.7, 1.0, 1.0], [90.8, 1.0, 1.0]]])
    if extra is not None:
        spikes = np.vstack([spikes, extra])
    payload = {"session_info": info, "spike_data": spikes, "ripple_events": np.array([[90.5, 91.5, 91.0, 0]])}
    monkeypatch.setattr(native, "recruitment", coupling.recruitment)
    for module in (audit, native, coupling):
        monkeypatch.setattr(module, "loadmat", lambda p, **kwargs: {p.stem: payload[p.stem]})


def test_native_three_readout_schema_and_counts(monkeypatch, tmp_path):
    patched(monkeypatch)
    row = SimpleNamespace(**next(r for r in audit.packet_coverage(tmp_path) if r["selected"]))
    cells, score = native.score_packet(tmp_path, row)
    assert len(cells) == score["n_reference_cells"] == 6
    assert cells.reference_included.all()
    assert score["status"] == "paired_information"
    assert cells.past_spikes.sum() == row.past_spikes
    assert cells.baseline_spikes.sum() == row.baseline_spikes
    assert cells.target_spikes.sum() == row.target_spikes


def test_native_future_only_unit_not_selected(monkeypatch, tmp_path):
    patched(monkeypatch)
    row = SimpleNamespace(**next(r for r in audit.packet_coverage(tmp_path) if r["selected"]))
    _, first = native.score_packet(tmp_path, row)
    times = np.linspace(row.target_start_s + 0.1, row.target_end_s - 0.1, 500)
    extra = np.column_stack([times, np.full(len(times), 99), np.ones(len(times))])
    patched(monkeypatch, extra)
    cells, second = native.score_packet(tmp_path, row)
    assert len(cells) == 7 and not cells.iloc[-1].reference_included
    for k in ("preceding_score", "preceding_information", "prospective_score", "prospective_information"):
        assert first[k] == pytest.approx(second[k], abs=1e-12)


def test_native_shared_support_matches_calibrated_pair(monkeypatch, tmp_path):
    from scripts.calibrate_kleinman_joint_temporal_specificity import paired_scores

    patched(monkeypatch)
    row = SimpleNamespace(**next(r for r in audit.packet_coverage(tmp_path) if r["selected"]))
    _, occ, hist, rec = native.packet_data(tmp_path, row)
    _, actual = native.score_packet(tmp_path, row)
    models = [native.estimate_reference(c, occ[0])[0] for c in hist[0]]
    x = rec["predictor"]
    x = (x - x.mean()) / x.std()
    check = paired_scores(hist[1:].transpose(1, 0, 2), occ[1:], models, x)
    for key in ("preceding_score", "preceding_information", "prospective_score", "prospective_information"):
        assert actual[key] == pytest.approx(check[key], abs=1e-12)


def test_constant_recruitment_is_missing_information(monkeypatch, tmp_path):
    patched(monkeypatch)
    row = SimpleNamespace(**next(r for r in audit.packet_coverage(tmp_path) if r["selected"]))
    original = native.recruitment

    def constant(*args):
        rec = original(*args)
        rec["predictor"] = np.zeros_like(rec["predictor"])
        return rec

    monkeypatch.setattr(native, "recruitment", constant)
    _, result = native.score_packet(tmp_path, row)
    assert result["status"] == "missing_paired_information"
    assert result["preceding_information"] == result["prospective_information"] == 0


def test_independent_histograms_and_reference_fit(monkeypatch, tmp_path):
    from scripts import verify_kleinman_native_joint_temporal_pilot as checker

    patched(monkeypatch)
    monkeypatch.setattr(checker, "loadmat", native.loadmat)
    monkeypatch.setattr(checker.producer, "recruitment", native.recruitment)
    row = SimpleNamespace(**next(r for r in audit.packet_coverage(tmp_path) if r["selected"]))
    keys, occ, hist, _ = native.packet_data(tmp_path, row)
    other_keys, other_occ, other_hist, _ = checker.independent_data(tmp_path, row)
    np.testing.assert_array_equal(keys, other_keys)
    np.testing.assert_array_equal(hist, other_hist)
    np.testing.assert_allclose(occ, other_occ, atol=1e-12)
    for h in hist[0]:
        model, keep = native.estimate_reference(h, occ[0])
        independent, included = checker.independent_reference(h, occ[0])
        np.testing.assert_allclose(model, independent, atol=1e-10)
        assert keep == included
