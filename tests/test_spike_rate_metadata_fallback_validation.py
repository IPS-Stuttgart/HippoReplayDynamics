import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.spike_rate_metadata import _unique_float_from_columns


@pytest.mark.parametrize("fallback", [np.nan, np.inf, -np.inf, True, False, "invalid"])
def test_unique_float_rejects_invalid_missing_metadata_fallback(fallback):
    frame = pd.DataFrame()

    with pytest.raises(ValueError, match="metadata fallback"):
        _unique_float_from_columns(frame, ("missing_column",), fallback)


def test_unique_float_accepts_finite_numeric_missing_metadata_fallback():
    frame = pd.DataFrame({"metadata": [None, "nan", "<NA>"]})

    value = _unique_float_from_columns(frame, ("metadata",), np.float64(1.25))

    assert value == pytest.approx(1.25)
