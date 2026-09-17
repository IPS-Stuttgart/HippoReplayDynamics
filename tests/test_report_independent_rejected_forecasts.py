"""The reporter cannot replace a failed primary with a favorable sensitivity."""

import pandas as pd
from scripts.report_independent_rejected_forecasts import evidence_role, primary_rows


def test_primary_and_all_adequacy_controls_required():
    x = pd.DataFrame([dict(contrast=c, animals=4, positive_animals=4, ci_low=0.01) for c in ["dynamic_minus_matched_own", "dynamic_minus_global", "dynamic_minus_no_history"]])
    assert evidence_role(x, 4)
    assert not evidence_role(x.iloc[:2], 4)
    assert not evidence_role(x.iloc[:0], 4)
    x.loc[0, "ci_low"] = -0.01
    assert not evidence_role(x, 4)


def test_report_does_not_substitute_a_favorable_group_or_horizon():
    x = pd.DataFrame(
        [
            dict(horizon=h, model="learned_hmm", group=g, level=level, metric=m)
            for h in [1, 2, 4]
            for g in ["rejected_with_opportunity", "geometric_pass"]
            for level in ["full", "half"]
            for m in ["delta", "delta_per_spike"]
        ]
    )
    y = primary_rows(x)
    assert len(y) == 1
    assert y.iloc[0].horizon == 2
    assert y.iloc[0].level == "full"
    assert y.iloc[0].metric == "delta_per_spike"
