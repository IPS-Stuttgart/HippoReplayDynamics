from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.state_space import _mass_retaining_candidate_indices


def test_mass_retaining_candidate_support_rejects_boolean_threshold() -> None:
    log_emission = np.log(np.array([0.60, 0.25, 0.10, 0.05]))

    for threshold in (True, np.bool_(False), np.array(True, dtype=object)):
        with pytest.raises(TypeError, match="mass_threshold.*not boolean"):
            _mass_retaining_candidate_indices(
                log_emission,
                mass_threshold=threshold,
                top_k=1,
                min_k=0,
                max_k=0,
            )
