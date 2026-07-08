import pandas as pd
import pytest

from scripts.repeated_cell_split_benchmark import _aggregate_summary, _parse_models, _parse_seeds


def test_aggregate_summary_counts_split_seeds_per_model():
    rows = pd.DataFrame(
        {
            "model": ["stationary", "stationary", "momentum"],
            "split_seed": [1, 2, 1],
            "heldout_log_likelihood": [-10.0, -11.0, -9.0],
            "delta_vs_best_static": [0.0, 0.0, 2.0],
        }
    )

    summary = _aggregate_summary(rows)
    split_counts = dict(zip(summary["model"], summary["split_seeds"]))

    assert split_counts == {"stationary": 2, "momentum": 1}


def test_parse_models_accepts_comma_and_whitespace_separators() -> None:
    assert _parse_models("momentum, imm") == ("momentum", "imm")
    assert _parse_models("momentum imm") == ("momentum", "imm")
    assert _parse_models("stationary, diffusion momentum") == (
        "stationary",
        "diffusion",
        "momentum",
    )


@pytest.mark.parametrize("value", ["", "   ", "momentum,,imm", "momentum,", ",imm"])
def test_parse_models_rejects_empty_model_entries(value: str) -> None:
    with pytest.raises(ValueError):
        _parse_models(value)


def test_parse_seeds_accepts_comma_separated_nonnegative_integers() -> None:
    assert _parse_seeds("1, 2,003") == (1, 2, 3)


@pytest.mark.parametrize("value", ["", "   ", "1,,2", "1,", ",2"])
def test_parse_seeds_rejects_empty_seed_entries(value: str) -> None:
    with pytest.raises(ValueError):
        _parse_seeds(value)


@pytest.mark.parametrize("value", ["1.5", "nan", "abc", "-1"])
def test_parse_seeds_rejects_invalid_seed_values(value: str) -> None:
    with pytest.raises(ValueError):
        _parse_seeds(value)
