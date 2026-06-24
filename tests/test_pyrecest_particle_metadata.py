from __future__ import annotations

import pandas as pd
import pytest

from hipporeplayimm.pyrecest_score_metadata import pyrecest_config_kwargs_for_scores


def test_pyrecest_metadata_rejects_nonpositive_particle_count_metadata() -> None:
    for value in ("0", "-5"):
        scores = pd.DataFrame({"pyrecest_particles": [value]})

        with pytest.raises(ValueError, match="pyrecest_particles.*must be positive"):
            pyrecest_config_kwargs_for_scores(scores)
