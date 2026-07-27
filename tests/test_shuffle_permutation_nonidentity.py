from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm.shuffle_spike_time_order as shuffle_patch
from hipporeplayimm.encoding import EncodingConfig, EncodingModel
from hipporeplayimm.shuffle_controls import shuffled_encoding


class _DuplicateNoopThenChangeGenerator:
    """Yield a nonidentity value-no-op draw before a genuinely changed draw."""

    def __init__(self) -> None:
        self._permutations = iter(
            (
                np.array([1, 0, 2], dtype=int),
                np.array([2, 1, 0], dtype=int),
            )
        )

    def permutation(self, size: int) -> np.ndarray:
        assert size == 3
        return next(self._permutations).copy()


def _two_cell_two_bin_encoding() -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([0.0, 1.0, 2.0], dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        bin_centers=np.array([[0.5, 0.5], [1.5, 0.5]], dtype=float),
        rates_hz=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        occupancy_s=np.ones(2, dtype=float),
        cell_ids=np.array([10, 11], dtype=int),
        config=EncodingConfig(),
    )


def _duplicate_backed_encoding(rates_hz: np.ndarray) -> EncodingModel:
    n_bins = rates_hz.shape[1]
    return EncodingModel(
        x_edges=np.arange(n_bins + 1, dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        bin_centers=np.column_stack(
            (np.arange(n_bins, dtype=float) + 0.5, np.full(n_bins, 0.5))
        ),
        rates_hz=np.asarray(rates_hz, dtype=float),
        occupancy_s=np.ones(n_bins, dtype=float),
        cell_ids=np.arange(10, 10 + rates_hz.shape[0], dtype=int),
        config=EncodingConfig(),
    )


def test_cell_permutation_rejects_identity_draw() -> None:
    encoding = _two_cell_two_bin_encoding()

    control = shuffled_encoding(
        encoding,
        mode="cell-permutation",
        random_seed=0,
    )

    np.testing.assert_array_equal(
        control.rates_hz,
        encoding.rates_hz[[1, 0]],
    )


def test_spatial_permutation_rejects_identity_draw() -> None:
    encoding = _two_cell_two_bin_encoding()

    control = shuffled_encoding(
        encoding,
        mode="spatial-permutation",
        random_seed=0,
    )

    np.testing.assert_array_equal(
        control.rates_hz,
        encoding.rates_hz[:, [1, 0]],
    )


def test_independent_spatial_permutation_rejects_identity_per_cell() -> None:
    encoding = _two_cell_two_bin_encoding()

    control = shuffled_encoding(
        encoding,
        mode="independent-spatial-permutation",
        random_seed=0,
    )

    np.testing.assert_array_equal(
        control.rates_hz,
        encoding.rates_hz[:, [1, 0]],
    )


def test_cell_permutation_retries_duplicate_row_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoding = _duplicate_backed_encoding(
        np.array(
            [
                [1.0, 2.0],
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        )
    )
    monkeypatch.setattr(
        shuffle_patch.np.random,
        "default_rng",
        lambda _seed: _DuplicateNoopThenChangeGenerator(),
    )

    control = shuffled_encoding(encoding, mode="cell-permutation", random_seed=0)

    assert not np.array_equal(control.rates_hz, encoding.rates_hz)
    np.testing.assert_array_equal(
        control.rates_hz,
        encoding.rates_hz[[2, 1, 0]],
    )


def test_spatial_permutation_retries_duplicate_column_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoding = _duplicate_backed_encoding(
        np.array(
            [
                [1.0, 1.0, 2.0],
                [5.0, 5.0, 6.0],
            ]
        )
    )
    monkeypatch.setattr(
        shuffle_patch.np.random,
        "default_rng",
        lambda _seed: _DuplicateNoopThenChangeGenerator(),
    )

    control = shuffled_encoding(encoding, mode="spatial-permutation", random_seed=0)

    assert not np.array_equal(control.rates_hz, encoding.rates_hz)
    np.testing.assert_array_equal(
        control.rates_hz,
        encoding.rates_hz[:, [2, 1, 0]],
    )


def test_independent_spatial_permutation_retries_duplicate_value_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoding = _duplicate_backed_encoding(
        np.array(
            [
                [1.0, 1.0, 2.0],
                [5.0, 5.0, 5.0],
            ]
        )
    )
    monkeypatch.setattr(
        shuffle_patch.np.random,
        "default_rng",
        lambda _seed: _DuplicateNoopThenChangeGenerator(),
    )

    control = shuffled_encoding(
        encoding,
        mode="independent-spatial-permutation",
        random_seed=0,
    )

    np.testing.assert_array_equal(control.rates_hz[0], np.array([2.0, 1.0, 1.0]))
    np.testing.assert_array_equal(control.rates_hz[1], encoding.rates_hz[1])


def test_permutation_modes_preserve_unavoidable_singletons() -> None:
    encoding = EncodingModel(
        x_edges=np.array([0.0, 1.0], dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        bin_centers=np.array([[0.5, 0.5]], dtype=float),
        rates_hz=np.array([[2.5]], dtype=float),
        occupancy_s=np.ones(1, dtype=float),
        cell_ids=np.array([10], dtype=int),
        config=EncodingConfig(),
    )

    for mode in (
        "cell-permutation",
        "spatial-permutation",
        "independent-spatial-permutation",
    ):
        control = shuffled_encoding(
            encoding,
            mode=mode,
            random_seed=0,
        )
        np.testing.assert_array_equal(control.rates_hz, encoding.rates_hz)
