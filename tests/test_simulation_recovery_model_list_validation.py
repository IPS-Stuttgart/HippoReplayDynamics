from __future__ import annotations

import pytest

from hipporeplayimm.simulation_recovery import parse_model_list


def test_model_list_parser_keeps_valid_comma_and_whitespace_separators() -> None:
    assert parse_model_list("stationary, diffusion momentum") == (
        "stationary",
        "diffusion",
        "momentum",
    )


@pytest.mark.parametrize(
    "spec",
    [
        "stationary,,momentum",
        "stationary,",
        ",momentum",
        "stationary,   ,momentum",
    ],
)
def test_model_list_parser_rejects_empty_comma_entries(spec: str) -> None:
    with pytest.raises(ValueError, match="empty comma-separated entries"):
        parse_model_list(spec)


@pytest.mark.parametrize(
    "spec",
    [
        ("stationary", "", "momentum"),
        ("stationary", "   ", "momentum"),
    ],
)
def test_model_list_parser_rejects_empty_iterable_entries(spec: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="empty entries"):
        parse_model_list(spec)
