from pathlib import Path
import importlib.util

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "hipporeplayimm" / "cell_split_hashable_grouping.py"


def _load_grouping_module():
    spec = importlib.util.spec_from_file_location("cell_split_hashable_grouping_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_metadata_group_key_preserves_array_shape():
    grouping = _load_grouping_module()

    matrix_key = grouping._metadata_group_key(np.array([[1, 2], [3, 4]]))
    flat_key = grouping._metadata_group_key(np.array([1, 2, 3, 4]))

    assert matrix_key != flat_key
