from __future__ import annotations

import pandas as pd

from hipporeplayimm.simulation_best_row_flags import _status_success_mask


def test_simulation_best_row_status_mask_accepts_legacy_na_aliases():
    frame = pd.DataFrame(
        {
            "status": ["success", "", "nan", "NA", "n/a", "failed"],
        }
    )

    assert _status_success_mask(frame).tolist() == [True, True, True, True, True, False]
