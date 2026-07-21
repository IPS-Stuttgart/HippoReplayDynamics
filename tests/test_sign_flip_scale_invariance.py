from __future__ import annotations

import pytest

from hipporeplayimm.sign_flip_report import paired_sign_flip_test


def test_exact_sign_flip_p_value_is_invariant_to_small_rescaling() -> None:
    reference = paired_sign_flip_test([1.0, 1.0])
    rescaled = paired_sign_flip_test([1e-100, 1e-100])

    assert reference.p_value == pytest.approx(0.5)
    assert rescaled.p_value == pytest.approx(reference.p_value)
