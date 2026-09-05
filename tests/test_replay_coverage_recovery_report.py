import numpy as np
import pandas as pd
import pytest

from scripts.report_replay_coverage_recovery import paired_effect


def test_pair_before_equal_animal_weighting():
    table = pd.DataFrame([
        {"animal": animal, "session": session, "fraction": fraction, "value": value}
        for animal, session, low, high in [("a", "a1", 2, 1), ("a", "a2", 2, 1), ("b", "b1", 4, 1)]
        for fraction, value in [(.5, low), (1., high)]
    ])
    result = paired_effect(table, "fraction", .5, 1., "value", np.random.default_rng(1), 100)
    assert result["effect"] == 2
    assert result["paired_sessions"] == 3 and result["animals"] == 2
    with pytest.raises(ValueError):
        paired_effect(pd.concat([table, table]), "fraction", .5, 1., "value", np.random.default_rng(1))


def test_missing_pair_does_not_produce_a_zero_effect():
    table = pd.DataFrame({"animal": ["a"], "session": ["s"], "fraction": [.5], "value": [1.]})
    result = paired_effect(table, "fraction", .5, 1., "value", np.random.default_rng(1))
    assert result["animals"] == 0 and np.isnan(result["effect"])
