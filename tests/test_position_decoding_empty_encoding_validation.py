from types import SimpleNamespace

import numpy as np
import pytest

from hipporeplayimm.encoding import EncodingConfig
from hipporeplayimm.position_validation import PositionDecodingConfig, validate_session_position_decoding


def _empty_position_session() -> SimpleNamespace:
    return SimpleNamespace(
        position=np.empty((0, 4), dtype=float),
        spikes=np.empty((0, 2), dtype=float),
        excitatory_neurons=np.empty(0, dtype=int),
        inhibitory_neurons=np.empty(0, dtype=int),
        run_times=np.empty((0, 2), dtype=float),
    )


@pytest.mark.parametrize(
    ("encoding", "message"),
    [
        (EncodingConfig(bin_size_cm=0.0), "bin_size_cm"),
        (EncodingConfig(smoothing_sigma_bins=-1.0), "smoothing_sigma_bins"),
        (EncodingConfig(min_speed_cm_s=float("nan")), "min_speed_cm_s"),
    ],
)
def test_empty_position_session_still_validates_encoding_config(
    encoding: EncodingConfig,
    message: str,
) -> None:
    config = PositionDecodingConfig(encoding=encoding)

    with pytest.raises(ValueError, match=message):
        validate_session_position_decoding(_empty_position_session(), config)
