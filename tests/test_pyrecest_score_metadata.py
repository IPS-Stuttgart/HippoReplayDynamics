from __future__ import annotations

import pandas as pd
import pytest

from hipporeplayimm.pyrecest_score_metadata import pyrecest_config_kwargs_for_scores


def test_pyrecest_float_metadata_rejects_near_conflicts() -> None:
    scores = pd.DataFrame({"pyrecest_alpha": [0.8, 0.800001]})

    with pytest.raises(ValueError, match="pyrecest_alpha.*multiple values"):
        pyrecest_config_kwargs_for_scores(scores)


def test_pyrecest_float_metadata_accepts_roundoff_equivalent_values() -> None:
    scores = pd.DataFrame({"pyrecest_alpha": [0.8, 0.8 + 5e-13]})

    kwargs = pyrecest_config_kwargs_for_scores(scores)

    assert kwargs["pyrecest_alpha"] == pytest.approx(0.8)
