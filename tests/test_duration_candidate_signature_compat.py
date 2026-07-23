from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from hipporeplayimm import duration_occupancy


@dataclass
class _Emissions:
    log_likelihood: np.ndarray
    metadata: dict[str, object]

    @property
    def n_time(self) -> int:
        return int(self.log_likelihood.shape[0])

    @property
    def n_bins(self) -> int:
        return int(self.log_likelihood.shape[1])


class _StateSpaceHelpers:
    @staticmethod
    def _validate_candidate_indices(candidates, n_time, n_bins):
        assert len(candidates) == n_time
        resolved = [np.asarray(candidate, dtype=int) for candidate in candidates]
        assert all(np.all((candidate >= 0) & (candidate < n_bins)) for candidate in resolved)
        return resolved

    @staticmethod
    def _restrict_candidates_to_valid_bins(candidates, log_likelihood, valid_bin_mask):
        if valid_bin_mask is None:
            return candidates
        mask = np.asarray(valid_bin_mask, dtype=bool)
        return [candidate[mask[candidate]] for candidate in candidates]


def _inputs():
    emissions = _Emissions(
        log_likelihood=np.zeros((2, 3), dtype=float),
        metadata={"source": "caller"},
    )
    bin_centers = np.zeros((3, 2), dtype=float)
    valid_bin_mask = np.array([True, False, True])
    return emissions, bin_centers, valid_bin_mask


def test_duration_candidates_accept_legacy_two_argument_generator() -> None:
    emissions, bin_centers, valid_bin_mask = _inputs()

    class LegacyModel:
        def candidate_indices(self, candidate_emissions, candidate_bin_centers):
            assert candidate_bin_centers is bin_centers
            assert np.isneginf(candidate_emissions.log_likelihood[:, 1]).all()
            return [np.arange(3), np.arange(3)]

    candidates = duration_occupancy._duration_candidates(
        _StateSpaceHelpers,
        LegacyModel(),
        emissions,
        bin_centers,
        None,
        valid_bin_mask,
    )

    assert [candidate.tolist() for candidate in candidates] == [[0, 2], [0, 2]]


def test_duration_candidates_pass_mask_to_modern_generator() -> None:
    emissions, bin_centers, valid_bin_mask = _inputs()

    class ModernModel:
        received_mask = None

        def candidate_indices(
            self,
            candidate_emissions,
            candidate_bin_centers,
            *,
            valid_bin_mask=None,
        ):
            self.received_mask = valid_bin_mask
            return [np.arange(3), np.arange(3)]

    model = ModernModel()
    duration_occupancy._duration_candidates(
        _StateSpaceHelpers,
        model,
        emissions,
        bin_centers,
        None,
        valid_bin_mask,
    )

    np.testing.assert_array_equal(model.received_mask, valid_bin_mask)


def test_duration_candidates_pass_positional_only_mask() -> None:
    emissions, bin_centers, valid_bin_mask = _inputs()

    class PositionalOnlyMaskModel:
        received_mask = None

        def candidate_indices(
            self,
            candidate_emissions,
            candidate_bin_centers,
            valid_bin_mask,
            /,
        ):
            assert candidate_bin_centers is bin_centers
            self.received_mask = valid_bin_mask
            return [np.arange(3), np.arange(3)]

    model = PositionalOnlyMaskModel()
    candidates = duration_occupancy._duration_candidates(
        _StateSpaceHelpers,
        model,
        emissions,
        bin_centers,
        None,
        valid_bin_mask,
    )

    np.testing.assert_array_equal(model.received_mask, valid_bin_mask)
    assert [candidate.tolist() for candidate in candidates] == [[0, 2], [0, 2]]


def test_duration_candidates_do_not_swallow_implementation_type_error() -> None:
    emissions, bin_centers, valid_bin_mask = _inputs()

    class BrokenModel:
        def candidate_indices(self, candidate_emissions, candidate_bin_centers):
            raise TypeError("internal candidate bug")

    with pytest.raises(TypeError, match="internal candidate bug"):
        duration_occupancy._duration_candidates(
            _StateSpaceHelpers,
            BrokenModel(),
            emissions,
            bin_centers,
            None,
            valid_bin_mask,
        )
