import numpy as np

from scripts import calibrate_tirole_selection_transport as module


def test_fixed_subsets_and_independent_evaluation_do_not_change_with_truth_or_draw(monkeypatch):
    calls = []

    def classifier(counts, rates, valid, centers, shifts, permutations):
        calls.append((counts.copy(), rates.copy()))
        return {"sequence_accepted": False, "sequence_eligible": False, "inferred_track": -1}

    monkeypatch.setattr(module, "classify_sequence", classifier)
    monkeypatch.setattr(module, "content_readout", lambda *a: {"track2_probability": 0.5, "z_log_odds": 0.0})
    info = {"session": "fixture", "animal": "RAT"}
    parts = {"splits": [{"inference": list(range(10)), "evaluation": [10, 11]}]}
    event = {"event_id": 3, "epoch": "POST", "primary_ripple_candidate": True}
    original = np.full((10, 12), 2, int)
    rates = np.broadcast_to(np.arange(1.0, 13.0)[None, :, None], (2, 12, 20)).copy()
    maps = {"rates": rates, "valid_bins": np.ones((2, 20), bool), "bin_centers_cm": np.arange(20) * 10.0}
    scores, content = module.simulate_anchor(info, parts, event, original, maps, n_draws=2, n_repeats=2)
    assert len(scores) == 24 and len(content) == 4
    assert len(calls) == 24
    assert content.n_evaluation_spikes.eq(40).all()
    assert scores[scores.fraction == 1].n_inference_spikes.eq(200).all()
    assert len(scores[scores.repeat == -1]) == 8
    # The same rate columns identify the same fixed half subset in every copy.
    for repeat in range(2):
        ids = [tuple(r[0, :, 0]) for (_, r), (_, row) in zip(calls, scores.iterrows(), strict=True) if row["repeat"] == repeat]
        assert len(set(ids)) == 1
        assert max(ids[0]) <= 10
    assert scores.groupby(["split", "draw", "truth_track"]).true_path_span_cm.nunique().eq(1).all()


def test_evaluation_overlap_rejected_before_scoring():
    info = {"session": "fixture", "animal": "RAT"}
    parts = {"splits": [{"inference": [0, 1], "evaluation": [1, 2]}]}
    event = {"event_id": 0, "epoch": "POST", "primary_ripple_candidate": True}
    maps = {"rates": np.ones((2, 3, 20)), "valid_bins": np.ones((2, 20), bool), "bin_centers_cm": np.arange(20)}
    import pytest

    with pytest.raises(ValueError, match="overlap"):
        module.simulate_anchor(info, parts, event, np.ones((10, 3), int), maps, n_draws=1)
