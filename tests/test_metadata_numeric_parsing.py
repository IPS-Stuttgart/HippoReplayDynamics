from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.pyrecest_score_metadata import (
    PYRECEST_DEFAULTS,
    pyrecest_config_kwargs_for_scores,
)
from hipporeplayimm.score_metadata import (
    _optional_float_from_columns,
    _unique_float_from_columns,
    _unique_int_from_columns,
)
from hipporeplayimm.spike_rate_metadata import (
    _unique_float_from_columns as _spike_rate_unique_float_from_columns,
)


def test_score_metadata_ignores_missing_numeric_sentinel_values() -> None:
    scores = pd.DataFrame(
        {
            "encoding_bin_size_cm": ["", "nan", None],
            "emission_time_bin_s": ["null", np.nan, "<NA>"],
            "state_space_momentum_candidate_mass_threshold": ["none", pd.NA, ""],
        }
    )

    assert _unique_float_from_columns(scores, ("encoding_bin_size_cm",), default=4.0) == pytest.approx(4.0)
    assert _unique_float_from_columns(scores, ("emission_time_bin_s",), default=0.02) == pytest.approx(0.02)
    assert _optional_float_from_columns(
        scores,
        ("state_space_momentum_candidate_mass_threshold",),
        default=None,
    ) is None


def test_score_metadata_rejects_fractional_integer_metadata() -> None:
    scores = pd.DataFrame({"state_space_momentum_candidate_top_k": ["128.5"]})

    with pytest.raises(ValueError, match="must be an integer"):
        _unique_int_from_columns(
            scores,
            ("state_space_momentum_candidate_top_k",),
            default=128,
        )


def test_score_metadata_rejects_nonfinite_numeric_metadata() -> None:
    scores = pd.DataFrame({"state_space_diffusion_sigma_cm_sqrt_s": ["inf"]})

    with pytest.raises(ValueError, match="must be finite"):
        _unique_float_from_columns(
            scores,
            ("state_space_diffusion_sigma_cm_sqrt_s",),
            default=85.0,
        )


def test_pyrecest_metadata_ignores_missing_numeric_sentinel_values() -> None:
    scores = pd.DataFrame(
        {
            "pyrecest_particles": ["", "nan", None],
            "diagnostic_pyrecest_alpha": ["none", pd.NA, ""],
            "pyrecest_beta": [np.nan, "null", "<NA>"],
        }
    )

    kwargs = pyrecest_config_kwargs_for_scores(scores)

    assert kwargs["pyrecest_particles"] == PYRECEST_DEFAULTS["pyrecest_particles"]
    assert kwargs["pyrecest_alpha"] == pytest.approx(PYRECEST_DEFAULTS["pyrecest_alpha"])
    assert kwargs["pyrecest_beta"] == pytest.approx(PYRECEST_DEFAULTS["pyrecest_beta"])


def test_pyrecest_metadata_rejects_fractional_integer_metadata() -> None:
    scores = pd.DataFrame({"pyrecest_particles": ["512.5"]})

    with pytest.raises(ValueError, match="must be an integer"):
        pyrecest_config_kwargs_for_scores(scores)


def test_pyrecest_metadata_rejects_nonfinite_numeric_metadata() -> None:
    scores = pd.DataFrame({"pyrecest_alpha": ["inf"]})

    with pytest.raises(ValueError, match="must be finite"):
        pyrecest_config_kwargs_for_scores(scores)


def test_pyrecest_metadata_rejects_boolean_numeric_metadata() -> None:
    for column, value in (("pyrecest_alpha", True), ("pyrecest_particles", np.bool_(False))):
        scores = pd.DataFrame({column: [value]})

        with pytest.raises(ValueError, match=f"{column}.*finite numeric"):
            pyrecest_config_kwargs_for_scores(scores)


def test_spike_rate_metadata_rejects_nonscalar_numeric_metadata() -> None:
    scores = pd.DataFrame(
        {
            "emission_spike_rate_scale": pd.Series(
                [np.array([1.25])],
                dtype=object,
            )
        }
    )

    with pytest.raises(ValueError, match="emission_spike_rate_scale.*scalar numeric"):
        _spike_rate_unique_float_from_columns(
            scores,
            ("emission_spike_rate_scale",),
            default=1.0,
        )


def test_spike_rate_metadata_accepts_zero_dimensional_numeric_array() -> None:
    scores = pd.DataFrame(
        {
            "emission_spike_rate_scale": pd.Series(
                [np.array(1.25)],
                dtype=object,
            )
        }
    )

    assert _spike_rate_unique_float_from_columns(
        scores,
        ("emission_spike_rate_scale",),
        default=1.0,
    ) == pytest.approx(1.25)


def test_spike_rate_metadata_accepts_equivalent_float32_and_float64_aliases() -> None:
    scores = pd.DataFrame(
        {
            "emission_spike_rate_scale": pd.Series([np.float32(0.1)], dtype=object),
            "spike_rate_scale": pd.Series([0.1], dtype=object),
        }
    )

    assert _spike_rate_unique_float_from_columns(
        scores,
        ("emission_spike_rate_scale", "spike_rate_scale"),
        default=1.0,
    ) == pytest.approx(0.1)
