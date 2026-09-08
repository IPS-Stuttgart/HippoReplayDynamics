import numpy as np
import pandas as pd
import pytest

from scripts.audit_position_free_assembly_prediction import aggregate, bin_calibration, contrasts, gates


def test_pooling_preserves_counts_and_partial_bin():
    counts = np.arange(33).reshape(11, 3)
    result = bin_calibration(counts, np.arange(12) * 0.004)
    assert result.shape == (3, 3)
    assert np.array_equal(result[-1], counts[-1])
    assert np.array_equal(result.sum(axis=0), counts.sum(axis=0))
    assert np.array_equal(bin_calibration(counts, np.arange(12) * 0.02), counts)
    with pytest.raises(ValueError):
        bin_calibration(counts, np.arange(12) * 0.1)


def test_empty_gates_fail():
    assert not gates(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())


def test_paired_event_and_animal_weighting():
    records, spatial = [], []
    for rat in ("a", "b", "c", "d"):
        for split in range(5):
            common = {"dataset": "PF", "phase": "RUN", "session": rat, "rat": rat, "event_id": 1, "split": split}
            records.append({**common, "n_components": 3, "n_heldout_spikes": 2, "score_persistent": -3.0, "score_static": -4.0, "score_independent": -5.0, "score_global": -6.0})
            for model, value in (("first_order_imm", -1), ("diffusion", -2), ("iid_position", -3), ("static_location", -4)):
                spatial.append({**common, "encoding_variant": "pooled", "map": "real", "model": model, "conditional_heldout_log_score": value})
    paired, events = contrasts(pd.DataFrame(records), pd.DataFrame(spatial))
    assert len(events) == 76
    selected = events[events.contrast.eq("spatial_first_order_imm_minus_assembly_persistent")]
    assert selected.delta.eq(2).all() and selected.delta_per_heldout_spike.eq(1).all()
    summary, animals, _ = aggregate(events)
    assert summary[summary.contrast.eq("spatial_first_order_imm_minus_assembly_persistent")]["mean"].iloc[0] == 2
    assert len(animals) == 76 and len(paired) == 380
