from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "hipporeplayimm" / "cell_split_hashable_grouping.py"


def _load_grouping_module():
    spec = importlib.util.spec_from_file_location(
        "cell_split_hashable_grouping_nested_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_metadata_group_key_normalizes_nested_numpy_scalars_and_missing_values():
    grouping = _load_grouping_module()

    native = [
        7,
        2.5,
        pd.NA,
        {"cell": 3, "subset": (1, 2)},
    ]
    numpy_backed = [
        np.int64(7),
        np.float64(2.5),
        np.nan,
        {"subset": (np.int64(1), np.int64(2)), "cell": np.int64(3)},
    ]

    assert grouping._metadata_group_key(native) == grouping._metadata_group_key(
        numpy_backed
    )


def test_metadata_group_key_keeps_nested_sequence_order_distinct():
    grouping = _load_grouping_module()

    assert grouping._metadata_group_key([1, 2]) != grouping._metadata_group_key([2, 1])
