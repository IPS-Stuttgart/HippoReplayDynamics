import numpy as np
import pytest

from hipporeplayimm.position_decoding_config_validation import _validated_train_frame_mask


@pytest.mark.parametrize(
    "bad_mask",
    [
        np.array(["1.0", "0.0", "1.0"]),
        np.array([b"1", b"0", b"1"]),
        np.array([1.0 + 0.0j, 0.0 + 0.0j, 1.0 + 1.0j]),
        np.array([1.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j], dtype=object),
    ],
)
def test_position_train_frame_mask_rejects_textual_and_complex_values(bad_mask):
    with pytest.raises(ValueError, match="train_frame_mask"):
        _validated_train_frame_mask(bad_mask, 3)
