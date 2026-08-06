import warnings

import numpy as np
import pandas as pd

from hipporeplayimm import benchmarks


def _nested_scalar(value: object) -> np.ndarray:
    wrapped = np.empty((), dtype=object)
    wrapped[()] = np.asarray(value)
    return wrapped


def test_bootstrap_delta_ci_filters_boolean_and_complex_target_metrics() -> None:
    rows = pd.DataFrame(
        {
            "model": ["imm", "imm", "imm", "imm", "imm", "diffusion"],
            "delta_vs_best_static": [
                True,
                np.complex128(1.0 + 4.0j),
                np.complex128(2.0 + 0.0j),
                _nested_scalar(np.complex128(4.0 + 2.0j)),
                "3.0",
                np.complex128(9.0 + 2.0j),
            ],
        },
        dtype=object,
    )
    reference = pd.DataFrame(
        {
            "model": ["imm", "diffusion"],
            "delta_vs_best_static": [3.0, np.complex128(9.0 + 2.0j)],
        },
        dtype=object,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = benchmarks.bootstrap_delta_ci(
            rows,
            model="imm",
            n_bootstrap=128,
            random_seed=4,
        )
        expected = benchmarks.bootstrap_delta_ci(
            reference,
            model="imm",
            n_bootstrap=128,
            random_seed=4,
        )

    assert actual == expected == (3.0, 3.0)
