from itertools import product

import numpy as np
import pandas as pd
import pytest

from scripts.report_fresh_path_clock_recovery import DATASETS, TEACHERS, comparison_table


def tables():
    rows = []
    for dataset, stream, fraction in product(DATASETS, range(2), (0.25, 0.5, 0.75)):
        rows.append(
            {
                "dataset": dataset,
                "teacher": TEACHERS[stream],
                "scenario": fraction,
                "support": "exhaustive",
                "mean_estimate": fraction,
                "bias": 0.0,
                "rmse": 0.1,
                "median_interval_width": 0.3,
                "covered_fraction": 0.94,
                "correct_direction_fraction": 0.5,
            }
        )
    fresh = pd.DataFrame(rows)
    old = fresh.copy()
    old["teacher"] = old.teacher.map(dict(zip(TEACHERS, ("original_bank_0", "original_bank_1"), strict=True)))
    old["mean_estimate"] += 0.1
    old["bias"] += 0.1
    return fresh, old


def test_comparison_is_complete_and_has_correct_bias_direction():
    fresh, old = tables()
    result = comparison_table(fresh, old)
    assert len(result) == 12
    np.testing.assert_allclose(result.absolute_bias_reduction, 0.1)
    np.testing.assert_allclose(result.mean_estimate_library - result.mean_estimate_fresh, 0.1)
    result2 = comparison_table(fresh.sample(frac=1, random_state=3), old.sample(frac=1, random_state=1))
    pd.testing.assert_frame_equal(
        result.sort_values(["dataset", "stream", "scenario"]).reset_index(drop=True), result2.sort_values(["dataset", "stream", "scenario"]).reset_index(drop=True)
    )


def test_missing_duplicate_and_nonfinite_report_inputs_fail():
    fresh, old = tables()
    for bad in (fresh.iloc[:-1], pd.concat([fresh.iloc[:-1], fresh.iloc[:1]], ignore_index=True)):
        with pytest.raises(ValueError):
            comparison_table(bad, old)
    fresh.loc[0, "mean_estimate"] = np.nan
    with pytest.raises(ValueError):
        comparison_table(fresh, old)
