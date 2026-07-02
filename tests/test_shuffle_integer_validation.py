from __future__ import annotations

import numpy as np

from hipporeplayimm.shuffle_controls import _nonnegative_integer_value


def test_shuffle_integer_validation_preserves_large_uint64_values() -> None:
    value = np.uint64(2**63 + 123)

    assert _nonnegative_integer_value("random_seed", value) == int(value)
