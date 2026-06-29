from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from hipporeplayimm.pyrecest_score_metadata import (
    pyrecest_config_kwargs_for_scores,
    pyrecest_metadata_for_config,
)


def test_pyrecest_metadata_rejects_nonpositive_particle_count_metadata() -> None:
    for value in ("0", "-5"):
        scores = pd.DataFrame({"pyrecest_particles": [value]})

        with pytest.raises(ValueError, match="pyrecest_particles.*must be positive"):
            pyrecest_config_kwargs_for_scores(scores)


def test_pyrecest_score_defaults_reject_nonpositive_particle_count() -> None:
    for value in ("0", -5):
        with pytest.raises(ValueError, match="pyrecest_particles.*must be positive"):
            pyrecest_config_kwargs_for_scores(pd.DataFrame(), defaults={"pyrecest_particles": value})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("pyrecest_particles", True),
        ("pyrecest_alpha", False),
    ],
)
def test_pyrecest_config_metadata_rejects_boolean_values(name: str, value: object) -> None:
    config = SimpleNamespace(**{name: value})

    with pytest.raises(ValueError, match=f"{name}.*finite numeric metadata"):
        pyrecest_metadata_for_config(config)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("pyrecest_particles", True),
        ("pyrecest_alpha", False),
    ],
)
def test_pyrecest_score_defaults_reject_boolean_values(name: str, value: object) -> None:
    with pytest.raises(ValueError, match=f"{name}.*finite numeric metadata"):
        pyrecest_config_kwargs_for_scores(pd.DataFrame(), defaults={name: value})


def test_pyrecest_config_metadata_accepts_numeric_strings() -> None:
    config = SimpleNamespace(pyrecest_particles="64", pyrecest_alpha="0.5")

    metadata = pyrecest_metadata_for_config(config)

    assert metadata["pyrecest_particles"] == 64
    assert metadata["pyrecest_alpha"] == 0.5


def test_pyrecest_score_defaults_accept_numeric_strings() -> None:
    metadata = pyrecest_config_kwargs_for_scores(
        pd.DataFrame(),
        defaults={"pyrecest_particles": "64", "pyrecest_alpha": "0.5"},
    )

    assert metadata["pyrecest_particles"] == 64
    assert metadata["pyrecest_alpha"] == 0.5
