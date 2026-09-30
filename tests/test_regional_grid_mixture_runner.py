import pandas as pd
import pytest

from scripts.test_regional_grid_mixture import PENALTIES, choose_penalty


def cv_rows():
    return pd.DataFrame(
        [
            {
                "session": f"Rat{p % 4}/Session{p}",
                "animal": f"Rat{p % 4}",
                "population": f"population{p}",
                "fold": fold,
                "penalty": penalty,
                "score": -abs(penalty - 0.01) - fold * 0.01 - p * 0.001,
                "converged": True,
            }
            for p in range(22)
            for penalty in PENALTIES
            for fold in range(3)
        ]
    )


def test_penalty_choice_uses_only_predictive_scores():
    penalty, _ = choose_penalty(cv_rows())
    assert penalty == 0.01


def test_duplicate_and_unconverged_cv_not_silently_dropped():
    frame = cv_rows()
    frame.iloc[0] = frame.iloc[1]
    with pytest.raises(ValueError):
        choose_penalty(frame)
    frame = cv_rows()
    frame.loc[0, "converged"] = False
    with pytest.raises(ValueError):
        choose_penalty(frame)


def test_ties_choose_frozen_stronger_penalty():
    frame = cv_rows()
    frame["score"] = 0.0
    assert choose_penalty(frame)[0] == 0.1
