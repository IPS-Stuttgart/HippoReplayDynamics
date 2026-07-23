import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.ground_truth import _parse_cell_ids, _unique_int_from_column


_LONGDOUBLE_WIDER_THAN_FLOAT64 = np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant


def test_unique_int_from_column_accepts_integer_valued_metadata():
    frame = pd.DataFrame({"benchmark_random_seed": ["7", "7.0"]})

    assert _unique_int_from_column(frame, "benchmark_random_seed", 1) == 7


def test_unique_int_from_column_rejects_fractional_metadata():
    frame = pd.DataFrame({"benchmark_random_seed": ["7.5"]})

    with pytest.raises(ValueError, match="benchmark_random_seed"):
        _unique_int_from_column(frame, "benchmark_random_seed", 1)


def test_unique_int_from_column_rejects_near_integral_metadata():
    frame = pd.DataFrame({"benchmark_random_seed": ["7.0000000005"]})

    with pytest.raises(ValueError, match="benchmark_random_seed"):
        _unique_int_from_column(frame, "benchmark_random_seed", 1)


def test_unique_int_from_column_rejects_boolean_metadata():
    frame = pd.DataFrame({"benchmark_random_seed": [True]})

    with pytest.raises(ValueError, match="benchmark_random_seed"):
        _unique_int_from_column(frame, "benchmark_random_seed", 1)


@pytest.mark.skipif(not _LONGDOUBLE_WIDER_THAN_FLOAT64, reason="longdouble has no extra integer precision")
def test_unique_int_from_column_preserves_extended_precision_integer():
    exact_seed = 9_007_199_254_740_993
    frame = pd.DataFrame(
        {"benchmark_random_seed": pd.Series([np.longdouble(str(exact_seed))], dtype=object)}
    )

    assert _unique_int_from_column(frame, "benchmark_random_seed", 1) == exact_seed


@pytest.mark.skipif(not _LONGDOUBLE_WIDER_THAN_FLOAT64, reason="longdouble has no extra integer precision")
def test_unique_int_from_column_rejects_extended_precision_fraction():
    frame = pd.DataFrame(
        {
            "benchmark_random_seed": pd.Series(
                [np.longdouble("9007199254740992.5")],
                dtype=object,
            )
        }
    )

    with pytest.raises(ValueError, match="benchmark_random_seed"):
        _unique_int_from_column(frame, "benchmark_random_seed", 1)


def test_parse_cell_ids_rejects_near_integral_metadata():
    with pytest.raises(ValueError, match="score-table cell IDs"):
        _parse_cell_ids("[1.0000000005 2]")


def test_integer_metadata_patch_refreshes_restored_unique_int_helper(monkeypatch):
    import hipporeplayimm.ground_truth as gt
    from hipporeplayimm.ground_truth_integer_metadata import _PATCHED_FLAG, apply_ground_truth_integer_metadata_patch

    def lossy_unique_int_from_column(frame, column, default):
        values = [int(float(value)) for value in gt._iter_present_column_values(frame, (column,))]
        return int(default) if not values else values[0]

    monkeypatch.setattr(gt, "_unique_int_from_column", lossy_unique_int_from_column)
    monkeypatch.setattr(gt, _PATCHED_FLAG, True, raising=False)

    apply_ground_truth_integer_metadata_patch()

    frame = pd.DataFrame({"benchmark_random_seed": ["7.5"]})
    with pytest.raises(ValueError, match="benchmark_random_seed"):
        gt._unique_int_from_column(frame, "benchmark_random_seed", 1)


def test_integer_metadata_patch_refreshes_stale_cell_id_flag(monkeypatch):
    import hipporeplayimm.ground_truth as gt
    from hipporeplayimm.ground_truth_integer_metadata import _CELL_ID_PATCHED_FLAG, apply_ground_truth_integer_metadata_patch

    def lossy_parse_cell_ids(_value):
        return pd.Series([0]).to_numpy(dtype=int)

    monkeypatch.setattr(gt, "_parse_cell_ids", lossy_parse_cell_ids)
    monkeypatch.setattr(gt, _CELL_ID_PATCHED_FLAG, True, raising=False)

    apply_ground_truth_integer_metadata_patch()

    with pytest.raises(ValueError, match="score-table cell IDs"):
        gt._parse_cell_ids("[1.5 2]")
