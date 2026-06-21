import numpy as np
import pytest

from hipporeplayimm.ground_truth import _parse_cell_ids


def test_parse_cell_ids_accepts_integer_valued_metadata():
    np.testing.assert_array_equal(_parse_cell_ids("1 2.0 3"), np.array([1, 2, 3]))
    np.testing.assert_array_equal(_parse_cell_ids([1, 2.0, np.int64(3)]), np.array([1, 2, 3]))


def test_parse_cell_ids_rejects_fractional_metadata():
    fractional_cell = 2 + 0.5
    with pytest.raises(ValueError, match="cell ID metadata"):
        _parse_cell_ids([1, fractional_cell])
