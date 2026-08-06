import warnings

import numpy as np
import pandas as pd

from hipporeplayimm.shuffle_controls import add_shuffle_p_values


def test_add_shuffle_p_values_ignores_complex_evidence_without_casting() -> None:
    nested_complex = np.empty((), dtype=object)
    nested_complex[()] = np.complex128(1.0 + 100.0j)

    real_scores = pd.DataFrame(
        {
            "session": ["s"],
            "event_index": [0],
            "model": ["m"],
            "log_evidence": [2.0],
        }
    )
    control_scores = pd.DataFrame(
        {
            "session": ["s", "s", "s"],
            "event_index": [0, 0, 0],
            "model": ["m", "m", "m"],
            "log_evidence": [1.0 + 100.0j, nested_complex, 3.0],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = add_shuffle_p_values(real_scores, control_scores)

    row = result.iloc[0]
    assert row["shuffle_p_value"] == 1.0
    assert row["shuffle_log_evidence_median"] == 3.0
    assert row["shuffle_log_evidence_mean"] == 3.0
    assert row["shuffle_count"] == 1
