import pandas as pd
import pytest

from hipporeplayimm.ground_truth import _parse_cell_ids, _unique_int_from_column
from hipporeplayimm.ground_truth_integer_metadata import _read_ground_truth_score_csv


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


def test_parse_cell_ids_rejects_near_integral_metadata():
    with pytest.raises(ValueError, match="score-table cell IDs"):
        _parse_cell_ids("[1.0000000005 2]")


def test_ground_truth_score_csv_preserves_decimal_form_integer_identifiers(tmp_path):
    scores = tmp_path / "scores.csv"
    scores.write_text(
        "session,event_index,benchmark_random_seed,benchmark_cell_split_seed,"
        "benchmark_cell_split_index,model\n"
        "Rat1/Open1,9007199254740993.0,9007199254740995.0,"
        "9007199254740997.0,9007199254740999.0,state-space-imm\n"
        "Rat1/Open1,9007199254741001.0,9007199254741003.0,"
        "9007199254741005.0,,state-space-imm\n",
        encoding="utf-8",
    )

    frame = _read_ground_truth_score_csv(scores)

    assert frame.loc[0, "event_index"] == 9007199254740993
    assert frame.loc[0, "benchmark_random_seed"] == 9007199254740995
    assert frame.loc[0, "benchmark_cell_split_seed"] == 9007199254740997
    assert frame.loc[0, "benchmark_cell_split_index"] == 9007199254740999
    assert frame.loc[1, "event_index"] == 9007199254741001
    assert pd.isna(frame.loc[1, "benchmark_cell_split_index"])


def test_ground_truth_compare_patch_reads_integer_identity_columns_exactly(monkeypatch, tmp_path):
    import hipporeplayimm.ground_truth as gt
    from hipporeplayimm.ground_truth_integer_metadata import apply_ground_truth_integer_metadata_patch

    scores = tmp_path / "scores.csv"
    scores.write_text(
        "session,event_index,benchmark_random_seed,model\n"
        "Rat1/Open1,9007199254740993.0,9007199254740995.0,state-space-imm\n",
        encoding="utf-8",
    )
    captured = {}

    def base_compare(_root, score_rows, *args, **kwargs):
        captured["scores"] = score_rows
        return score_rows

    monkeypatch.setattr(gt, "compare_scores_to_ground_truth", base_compare)

    apply_ground_truth_integer_metadata_patch()
    result = gt.compare_scores_to_ground_truth("unused", scores)

    assert isinstance(captured["scores"], pd.DataFrame)
    assert captured["scores"].loc[0, "event_index"] == 9007199254740993
    assert captured["scores"].loc[0, "benchmark_random_seed"] == 9007199254740995
    assert result.equals(captured["scores"])


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
