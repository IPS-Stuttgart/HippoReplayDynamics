from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipporeplayimm.encoding import EncodingConfig, EncodingModel
from hipporeplayimm.shuffle_controls import ShuffleControlConfig, score_shuffle_controls, shuffled_encoding


def _two_bin_encoding() -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([0.0, 1.0, 2.0], dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        bin_centers=np.array([[0.5, 0.5], [1.5, 0.5]], dtype=float),
        rates_hz=np.ones((1, 2), dtype=float),
        occupancy_s=np.ones(2, dtype=float),
        cell_ids=np.array([10], dtype=int),
        config=EncodingConfig(),
    )


@pytest.mark.parametrize(
    "random_seed",
    [
        np.asarray(True, dtype=object),
        np.asarray([1]),
        np.asarray([True]),
    ],
)
def test_shuffled_encoding_rejects_wrapped_or_array_random_seed(random_seed: object) -> None:
    with pytest.raises(ValueError, match="random_seed"):
        shuffled_encoding(_two_bin_encoding(), random_seed=random_seed)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("config_kwargs", "message"),
    [
        ({"n_shuffles": np.asarray(True, dtype=object)}, "n_shuffles"),
        ({"n_shuffles": np.asarray([1])}, "n_shuffles"),
        ({"n_shuffles": np.asarray([True])}, "n_shuffles"),
        ({"random_seed": np.asarray(True, dtype=object)}, "random_seed"),
        ({"random_seed": np.asarray([1])}, "random_seed"),
        ({"random_seed": np.asarray([True])}, "random_seed"),
    ],
)
def test_score_shuffle_controls_rejects_wrapped_or_array_integer_config(
    config_kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        score_shuffle_controls(
            SimpleNamespace(session_id="Rat1/Open1"),
            _two_bin_encoding(),
            [],
            {},
            control_config=ShuffleControlConfig(**config_kwargs),
        )


@pytest.mark.parametrize(
    "event_indices",
    [
        [np.asarray(True, dtype=object)],
        [np.asarray([1])],
        [np.asarray([True])],
    ],
)
def test_score_shuffle_controls_rejects_wrapped_or_array_event_indices(event_indices: list[object]) -> None:
    with pytest.raises(ValueError, match="event_indices"):
        score_shuffle_controls(
            SimpleNamespace(session_id="Rat1/Open1"),
            _two_bin_encoding(),
            event_indices,
            {},
            control_config=ShuffleControlConfig(n_shuffles=0),
        )
