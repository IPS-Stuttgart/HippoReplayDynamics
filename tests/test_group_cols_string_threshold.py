from hipporeplayimm.advanced_result_empty_threshold_patch import _normalize_group_cols


def test_string_group_cols_stays_single_column():
    assert _normalize_group_cols("session", None) == ("session",)
