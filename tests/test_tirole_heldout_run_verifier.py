import numpy as np
import pandas as pd

from scripts.report_tirole_heldout_run_selection import block_weights
from scripts.verify_tirole_heldout_run_selection import reference_cube, reference_metrics, reference_weights


def test_tensor_verifier_detects_true_label_shift_and_no_vacuous_fraction():
    rows = []
    content = []
    for eid, track in enumerate([1, 2]):
        for split in range(5):
            content.append({"window_id": eid, "split": split, "truth_track": track, "evaluation_true_probability": 0.8, "evaluation_true_z": 1.2, "evaluation_correct": 1.0})
            for repeat in range(-1, 5):
                rows.append(
                    {
                        "window_id": eid,
                        "split": split,
                        "repeat": repeat,
                        "order": "original",
                        "likelihood": "poisson",
                        "sequence_accepted": repeat == -1 or track == 2,
                        "sequence_eligible": True,
                        "inferred_track": track,
                    }
                )
    cube, _ = reference_cube(pd.DataFrame(rows), pd.DataFrame(content), [0, 1], "original", "poisson")
    result = reference_metrics(cube, np.array([1, 2]), np.ones((1, 2)))
    assert result["half_minus_full_true_track2_fraction"][0] == 0.5
    assert result["lost_evaluation_true_z"][0] == 1.2
    assert result["lost_sequence_mass"][0] == 0.5
    assert result["lost_support_mass"][0] == 0
    assert np.isnan(result["gained_true_track2_fraction"][0])


def test_independent_block_weights_match_frozen_protocol():
    rows = []
    for track in [1, 2]:
        for fold in range(5):
            for block, n in [(fold + track * 100, 1), (fold + 5 + track * 100, 3)]:
                for _ in range(n):
                    rows.append({"window_id": len(rows), "truth_track": track, "fold": fold, "time_block": block})
    w = pd.DataFrame(rows)
    actual, _, ok = block_weights(w, 12)
    expected, adequate = reference_weights(w, np.random.default_rng(12))
    assert ok and adequate
    np.testing.assert_allclose(actual, expected)
