import numpy as np
import pandas as pd

from scripts.audit_kleinman_extent_cohort import cohort_tables, new_bank, session_eligible
from scripts.calibrate_kleinman_integrated_extent import replay_counts


def test_both_ends_all_screens_required_and_no_duplicate_rescue():
    gates = pd.DataFrame([{"side": s, "screen": c, "passed": True} for s in [0, 1] for c in ["matched_timing", "unseen_timing", "duration_only"]])
    assert session_eligible(gates)
    assert not session_eligible(gates.iloc[:3])
    assert not session_eligible(pd.concat([gates.iloc[:3], gates.iloc[:3]]))
    gates.loc[5, "passed"] = False
    assert not session_eligible(gates)


def test_missing_animal_or_condition_cannot_be_hidden_by_many_sessions():
    f = pd.DataFrame(
        [
            {
                "animal": str(i % 6),
                "session": str(i),
                "drug": (i // 6) % 2,
                "novel": (i // 12) % 2,
                "run_pass": i < 127,
                "attempted": i < 127,
                "extent_eligible": i < 127,
                "status": "scored",
            }
            for i in range(135)
        ]
    )
    animals, conditions, gates = cohort_tables(f)
    assert len(animals) == 6 and len(conditions) == 24 and gates.passed.all()
    f.loc[f.animal == "5", "extent_eligible"] = False
    _, _, gates = cohort_tables(f)
    assert not gates.loc[gates.gate == "reward_content_all_six_animals_retained", "passed"].item()
    assert not gates.loc[gates.gate == "drug_context_all_24_cells_retained", "passed"].item()
    f.loc[0, "status"] = "technical_failure"
    _, _, gates = cohort_tables(f)
    assert not gates.loc[gates.gate == "no_technical_failures", "passed"].item()


def test_new_session_bank_has_unique_ids_and_original_reconstructible_counts():
    x = np.arange(1, 100, 2)
    fields = 0.02 + 40 * np.exp(-(((x[:, None] - np.linspace(5, 95, 8)) / 9) ** 2))
    model = {"rates": np.block([[fields, fields * 0.1], [fields * 0.1, fields]]), "edges": np.arange(0, 102, 2), "ends": np.array([10.0, 90.0]), "support": np.ones(100, bool)}
    source, counts = new_bank(model, "A", "s", 17280)
    assert len(source) == 2880 and source.event_index.nunique() == 2880
    assert source.event_index.min() == 17280 and source.event_index.max() == 20159
    assert source.groupby(["side", "profile", "extent", "duration_s", "expected_spikes"]).size().eq(16).all()
    for row in source.sample(12, random_state=88).itertuples():
        np.testing.assert_array_equal(counts[row.event_index], replay_counts(model, row))
