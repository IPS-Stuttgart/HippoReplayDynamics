import numpy as np
import pytest

import hipporeplayimm  # noqa: F401  # import applies runtime patches
from hipporeplayimm.duration_occupancy import (
    _candidate_selection_emissions,
    _uniform_probabilities,
)
from hipporeplayimm.encoding import LogEmissionTensor


def _emissions_with_metadata() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.array(
            [[0.0, -1.0, -2.0], [-3.0, -4.0, -5.0]],
            dtype=float,
        ),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 0.1], dtype=float),
        dt=0.1,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
        metadata={"source": "original"},
    )


def test_candidate_selection_emissions_copies_metadata_for_masked_support():
    emissions = _emissions_with_metadata()

    restricted = _candidate_selection_emissions(
        emissions,
        np.array([True, False, True], dtype=bool),
    )

    assert restricted is not emissions
    assert restricted.metadata == {"source": "original"}
    assert restricted.metadata is not emissions.metadata
    restricted.metadata["derived"] = True
    assert emissions.metadata == {"source": "original"}
    np.testing.assert_allclose(
        restricted.log_likelihood[:, [0, 2]],
        emissions.log_likelihood[:, [0, 2]],
    )
    assert np.all(np.isneginf(restricted.log_likelihood[:, 1]))


def test_candidate_selection_emissions_reuses_input_when_no_masking_needed():
    emissions = _emissions_with_metadata()

    assert _candidate_selection_emissions(emissions, None) is emissions
    assert _candidate_selection_emissions(
        emissions,
        np.array([True, True, True], dtype=bool),
    ) is emissions


def test_uniform_probabilities_rejects_nonpositive_bin_count():
    with pytest.raises(ValueError, match="n_bins must be positive"):
        _uniform_probabilities(0)
