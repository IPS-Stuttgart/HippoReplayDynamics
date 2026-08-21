from decimal import Decimal
import sys
from types import ModuleType

import numpy as np
import pytest

from hipporeplayimm.state_space_bin_count_validation import (
    _integer_count,
    _positive_bin_count,
    apply_state_space_bin_count_validation_patch,
)
from hipporeplayimm.state_space_utils import _top_candidate_indices


@pytest.mark.parametrize(
    "exact_count",
    [
        2**53 + 1,
        np.uint64(2**53 + 1),
        Decimal("9007199254740993"),
        "9007199254740993",
    ],
)
def test_state_space_count_validation_preserves_exact_large_integers(exact_count):
    assert _integer_count("top_k", exact_count) == 2**53 + 1
    assert _positive_bin_count(exact_count) == 2**53 + 1


@pytest.mark.parametrize(
    "fractional_count",
    [
        Decimal("9007199254740992.5"),
        "9007199254740992.5",
    ],
)
def test_state_space_count_validation_rejects_fractional_large_decimals(fractional_count):
    with pytest.raises(TypeError, match="top_k must be an integer"):
        _integer_count("top_k", fractional_count)
    with pytest.raises(TypeError, match="top_k must be an integer"):
        _top_candidate_indices(np.array([0.0, 1.0]), fractional_count)
    with pytest.raises(ValueError, match="n_bins must be a positive integer"):
        _positive_bin_count(fractional_count)


def test_state_space_count_validation_rejects_fractional_extended_precision_float():
    fractional_count = np.longdouble("9007199254740992.5")
    if fractional_count.is_integer():
        pytest.skip("platform longdouble does not exceed binary64 precision")

    with pytest.raises(TypeError, match="top_k must be an integer"):
        _integer_count("top_k", fractional_count)


def test_state_space_alias_sync_is_limited_to_package_namespace(monkeypatch) -> None:
    external = ModuleType("hipporeplayimm_extension")
    package_probe = ModuleType("hipporeplayimm._state_space_alias_probe")

    def permissive_mask(valid_bin_mask, n_bins):
        return valid_bin_mask

    external._coerce_valid_bin_mask = permissive_mask
    package_probe._coerce_valid_bin_mask = permissive_mask
    monkeypatch.setitem(sys.modules, external.__name__, external)
    monkeypatch.setitem(sys.modules, package_probe.__name__, package_probe)

    apply_state_space_bin_count_validation_patch()

    assert external._coerce_valid_bin_mask is permissive_mask
    assert package_probe._coerce_valid_bin_mask is not permissive_mask
