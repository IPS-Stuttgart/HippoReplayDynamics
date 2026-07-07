from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.kd_reference import random_effects_model_probabilities


@pytest.mark.parametrize("prior", [True, np.bool_(True), np.array(True, dtype=object)])
def test_random_effects_rejects_boolean_prior(prior: object) -> None:
    with pytest.raises(TypeError, match="prior.*boolean"):
        random_effects_model_probabilities(
            np.zeros((2, 2), dtype=float),
            ["random", "imm"],
            prior=prior,
            n_iterations=3,
            burnin=1,
        )
