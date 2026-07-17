from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.result_improvement_extensions import add_model_averaged_endpoint_columns


def _extended_precision_seed() -> tuple[np.longdouble, int]:
    exact = 2**53 + 1
    seed = np.longdouble(str(exact))
    if int(seed) != exact:
        pytest.skip("platform longdouble does not exceed binary64 integer precision")
    return seed, exact


def _row(
    seed: object,
    model: str,
    probability: float,
    endpoint_x: float,
) -> dict[str, object]:
    return {
        "session": "Rat1/Open1",
        "event_index": 3,
        "benchmark_random_seed": seed,
        "model": model,
        "log_evidence": 0.0,
        "model_probability": probability,
        "evidence_comparable": True,
        "diagnostic_decoded_endpoint_x": endpoint_x,
        "diagnostic_decoded_endpoint_y": 0.0,
    }


def test_model_averaged_endpoints_keep_adjacent_extended_precision_seeds_separate() -> None:
    second_seed, exact_second = _extended_precision_seed()
    first_seed = np.longdouble(str(exact_second - 1))
    frame = pd.DataFrame(
        [
            _row(first_seed, "a", 0.25, 0.0),
            _row(first_seed, "b", 0.75, 4.0),
            _row(second_seed, "a", 0.50, 100.0),
            _row(second_seed, "b", 0.50, 200.0),
        ]
    )
    frame["benchmark_random_seed"] = pd.Series(
        [first_seed, first_seed, second_seed, second_seed],
        dtype=object,
    )

    out = add_model_averaged_endpoint_columns(frame)

    np.testing.assert_allclose(out.iloc[:2]["model_averaged_endpoint_x"], 3.0)
    np.testing.assert_allclose(out.iloc[2:]["model_averaged_endpoint_x"], 150.0)
    assert out["model_averaged_endpoint_models"].tolist() == [2, 2, 2, 2]


def test_model_averaged_endpoints_match_equal_integer_and_extended_precision_seeds() -> None:
    extended_seed, exact_seed = _extended_precision_seed()
    frame = pd.DataFrame(
        [
            _row(extended_seed, "a", 0.25, 0.0),
            _row(exact_seed, "b", 0.75, 4.0),
        ]
    )
    frame["benchmark_random_seed"] = pd.Series(
        [extended_seed, exact_seed],
        dtype=object,
    )

    out = add_model_averaged_endpoint_columns(frame)

    np.testing.assert_allclose(out["model_averaged_endpoint_x"], 3.0)
    assert out["model_averaged_endpoint_models"].tolist() == [2, 2]
