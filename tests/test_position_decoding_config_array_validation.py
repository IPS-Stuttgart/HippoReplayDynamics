from decimal import Decimal

import numpy as np
import pytest

from hipporeplayimm.position_decoding_config_validation import _validated_position_decoding_config
from hipporeplayimm.position_validation import PositionDecodingConfig


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"decode_bin_s": np.array([1.0])}, "decode_bin_s"),
        ({"n_folds": np.array([3])}, "n_folds"),
        ({"random_seed": np.array([[7]])}, "random_seed"),
        ({"max_windows_per_session": np.array([9])}, "max_windows_per_session"),
        ({"min_spikes_per_window": np.array([0])}, "min_spikes_per_window"),
    ],
)
def test_position_decoding_config_rejects_array_shaped_scalars(kwargs, message):
    config = PositionDecodingConfig(**kwargs)

    with pytest.raises(ValueError, match=message):
        _validated_position_decoding_config(config)


def test_position_decoding_config_accepts_zero_dimensional_numpy_scalars():
    config = PositionDecodingConfig(
        decode_bin_s=np.array(1.0),
        n_folds=np.array(3.0),
        random_seed=np.array(7.0),
        max_windows_per_session=np.array(9.0),
        min_spikes_per_window=np.array(0.0),
    )

    normalized = _validated_position_decoding_config(config)

    assert normalized.decode_bin_s == 1.0
    assert normalized.n_folds == 3
    assert normalized.random_seed == 7
    assert normalized.max_windows_per_session == 9
    assert normalized.min_spikes_per_window == 0


@pytest.mark.parametrize(
    "random_seed",
    [
        2**53 + 1,
        np.int64(2**53 + 1),
        np.array(2**53 + 1, dtype=np.int64),
        Decimal(str(2**53 + 1)),
        str(2**53 + 1),
        f"{2**53 + 1}.0",
    ],
)
def test_position_decoding_config_preserves_exact_large_random_seed(random_seed):
    normalized = _validated_position_decoding_config(PositionDecodingConfig(random_seed=random_seed))

    assert normalized.random_seed == 2**53 + 1
    assert isinstance(normalized.random_seed, int)
