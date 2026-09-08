import numpy as np
import pandas as pd

from scripts.audit_2d_predictive_order_map import contrasts
from scripts.verify_2d_predictive_order_map import reconstruct_contrasts


def fixture_scores():
    original, shuffled = [], []
    for split in range(5):
        for map_name in ("real", "population_code_permuted"):
            common = {
                "dataset": "fixture",
                "animal": "animal",
                "session": "session",
                "event_id": 31,
                "split": split,
                "map": map_name,
                "n_heldout_spikes": 10,
                "posterior_unchanged": True,
                "heldout_used_for_inference": False,
                "score_iid_position": -20.0,
                "score_static_location": -21.0,
            }
            original.append(common | {"score_first_order_imm": -10.0 if map_name == "real" else -12.0, "score_diffusion": -11.0})
            for shuffle in range(3):
                shuffled.append(common | {"shuffle": shuffle, "score_first_order_imm": -16.0, "score_diffusion": -15.0})
    return pd.DataFrame(original), pd.DataFrame(shuffled)


def test_independent_factorial_reconstruction_with_zero_counts():
    original, shuffled = fixture_scores()
    original.loc[original.split.eq(0), "n_heldout_spikes"] = 0
    shuffled.loc[shuffled.split.eq(0), "n_heldout_spikes"] = 0
    for model in ("diffusion", "first_order_imm", "iid_position", "static_location"):
        original.loc[original.split.eq(0), "score_" + model] = 0
        shuffled.loc[shuffled.split.eq(0), "score_" + model] = 0
    a, ae = contrasts(shuffled, original, k=3)
    b, be = reconstruct_contrasts(shuffled, original)
    for left, right in ((a, b), (ae, be)):
        key = ["contrast"] + (["split"] if "split" in left else [])
        np.testing.assert_allclose(left.sort_values(key).delta, right.sort_values(key).delta)
        np.testing.assert_allclose(left.sort_values(key).delta_per_heldout_spike, right.sort_values(key).delta_per_heldout_spike, equal_nan=True)
