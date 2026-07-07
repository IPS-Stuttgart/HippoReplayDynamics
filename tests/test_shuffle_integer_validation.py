from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.shuffle_controls import _nonnegative_integer_value


@pytest.mark.parametrize(
    "value",
    ["4", b"4", np.str_("4"), np.bytes_(b"4"), np.asarray("4"), np.asarray(b"4")],
)
def test_shuffle_integer_validation_rejects_string_values(value: object) -> None:
    with pytest.raises(ValueError, match="random_seed.*string"):
        _nonnegative_integer_value("random_seed", value)


def test_shuffle_integer_validation_preserves_large_uint64_values() -> None:
    value = np.uint64(2**63 + 123)

    assert _nonnegative_integer_value("random_seed", value) == int(value)
