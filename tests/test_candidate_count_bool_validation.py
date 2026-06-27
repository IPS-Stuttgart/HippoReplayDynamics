from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.state_space_utils import (
    _mass_retaining_candidate_indices,
    _top_candidate_indices,
)


@pytest.mark.parametrize("value", [True, np.bool_(False), np.asarray(True, dtype=object)])
def test_top_candidate_indices_rejects_boolean_top_k(value: object) -> None:
    with pytest.raises(TypeError, match="top_k.*not boolean"):
        _top_candidate_indices(np.array([0.0, -1.0, -2.0], dtype=float), value)


@pytest.mark.parametrize("name", ["top_k", "min_k", "max_k"])
@pytest.mark.parametrize("value", [True, np.bool_(False), np.asarray(True, dtype=object)])
def test_mass_retaining_candidate_indices_rejects_boolean_counts(name: str, value: object) -> None:
    kwargs: dict[str, object] = {"mass_threshold": 0.8, name: value}
    with pytest.raises(TypeError, match=f"{name}.*not boolean"):
        _mass_retaining_candidate_indices(
            np.log(np.array([0.7, 0.2, 0.1], dtype=float)),
            **kwargs,
        )
