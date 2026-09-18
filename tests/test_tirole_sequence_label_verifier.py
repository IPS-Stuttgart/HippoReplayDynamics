import numpy as np
import pandas as pd

from scripts.audit_tirole_sequence_labels import confusion, contributions
from scripts.verify_tirole_sequence_labels import GROUPS, METRICS, reference_contributions


def test_independent_tensor_covers_label_switches_missing_content_and_empty_groups():
    rng = np.random.default_rng(32)
    records = []
    for eid in range(4):
        for split in range(5):
            b = rng.random() if eid else np.nan
            for repeat in range(-1, 5):
                records.append(
                    {
                        "event_id": eid,
                        "split": split,
                        "repeat": repeat,
                        "start_s": eid * 60.0,
                        "truth_track": 1 if eid % 2 else 2,
                        "fold": 0,
                        "time_block": eid,
                        "sequence_accepted": bool(rng.random() < 0.7) if eid != 3 else False,
                        "sequence_eligible": True,
                        "inferred_track": int(rng.integers(1, 3)),
                        "A_poisson_track2_mass": rng.random(),
                        "A_conditional_track2_mass": 0.5 if eid == 2 else rng.random(),
                        "B_track2_mass": b,
                        "B_track2_z": b - 0.5,
                    }
                )
    rows = pd.DataFrame(records)
    cube, reference = reference_contributions(rows)
    a = contributions(rows)
    ix = pd.MultiIndex.from_product([range(4), GROUPS], names=["event_id", "group"])
    cols = [k + s for k in METRICS for s in ["_num", "_den"]]
    np.testing.assert_allclose(cube.reshape(-1, len(cols)), a.set_index(ix.names).reindex(ix)[cols])
    keys = ["group", "reference", "sequence_label", "reference_label"]
    actual = confusion(rows).set_index(keys)
    expected = reference.set_index(keys)
    np.testing.assert_allclose(actual.sort_index().to_numpy(), expected.sort_index().to_numpy(), equal_nan=True)
