import numpy as np
import pandas as pd

from hipporeplayimm import ground_truth as gt


def test_explicit_cell_id_metadata_order_is_canonicalized_within_group():
    rows = pd.DataFrame(
        {
            "train_cell_ids": ["3 1 2", "2 3 1"],
            "test_cell_ids": [np.array([5, 4]), [4, 5]],
        }
    )

    train_ids = gt._cell_ids_from_score_column(rows, "train_cell_ids")
    test_ids = gt._cell_ids_from_score_column(rows, "test_cell_ids")

    np.testing.assert_array_equal(train_ids, np.array([1, 2, 3]))
    np.testing.assert_array_equal(test_ids, np.array([4, 5]))
