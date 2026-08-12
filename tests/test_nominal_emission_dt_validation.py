from types import SimpleNamespace

import numpy as np
import pytest

from hipporeplayimm.nominal_emission_dt import (
    _configured_builder,
    _kd_builder,
    _positive_finite_dt,
)


def _emissions():
    return SimpleNamespace(dt=0.5)


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(False),
        np.array(True),
        np.array(np.array(True, dtype=object), dtype=object),
        0.0,
        -0.02,
        np.nan,
        np.inf,
        [0.02],
    ],
)
def test_positive_finite_dt_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        _positive_finite_dt(value, name="dt")


def test_positive_finite_dt_accepts_nested_real_scalar():
    value = np.array(np.array(0.02, dtype=object), dtype=object)

    assert _positive_finite_dt(value, name="dt") == pytest.approx(0.02)


def test_configured_builder_rejects_boolean_nominal_dt_after_builder_returns():
    config = SimpleNamespace(time_bin_s=True)
    wrapped = _configured_builder(lambda *args, **kwargs: _emissions(), default_config_factory=lambda: config)

    with pytest.raises(ValueError, match="not boolean"):
        wrapped(None, None, None, config)


def test_kd_builder_rejects_nonfinite_nominal_dt_after_builder_returns():
    wrapped = _kd_builder(lambda *args, **kwargs: _emissions())

    with pytest.raises(ValueError, match="finite and positive"):
        wrapped(None, None, None, np.inf)


def test_wrappers_store_canonical_float_nominal_dt():
    config = SimpleNamespace(time_bin_s=np.float32(0.02))
    configured = _configured_builder(lambda *args, **kwargs: _emissions(), default_config_factory=lambda: config)
    kd = _kd_builder(lambda *args, **kwargs: _emissions())

    configured_result = configured(None, None, None, config)
    kd_result = kd(None, None, None, np.float32(0.03))

    assert type(configured_result.dt) is float
    assert configured_result.dt == pytest.approx(0.02)
    assert type(kd_result.dt) is float
    assert kd_result.dt == pytest.approx(0.03)
