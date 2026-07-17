from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import hipporeplayimm.simulation_recovery as recovery


def test_emissions_from_counts_normalizes_arbitrary_precision_overflow() -> None:
    encoding = SimpleNamespace(n_cells=1)

    with pytest.raises(ValueError, match="counts must contain numeric values"):
        recovery.emissions_from_counts(
            encoding,
            np.array([[10**400]], dtype=object),
            dt=0.003,
        )
