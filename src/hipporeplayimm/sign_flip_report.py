"""Reproducible paired sign-flip inference for replay score tables.

Run this module directly to summarize one or more model rows from an existing
score CSV::

    python -m hipporeplayimm.sign_flip_report \
        --scores results/event_scores.csv \
        --output results/sign_flip_summary.csv

Small samples are evaluated exactly. Larger samples use a chunked Monte Carlo
randomization test with the standard plus-one correction.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignFlipResult:
    """Result and provenance for one two-sided paired sign-flip test."""

    observed_mean: float
    n_observations: int
    n_nonzero: int
    p_value: float
    method: str
    permutations_evaluated: int
    random_seed: int | None


def paired_sign_flip_test(
    values: Sequence[float] | np.ndarray,
    *,
    max_exact_n: int = 20,
    n_permutations: int = 10_000,
    random_seed: int = 1,
    chunk_size: int = 65_536,
) -> SignFlipResult:
    """Run a two-sided paired sign-flip test with exact small-sample inference.

    Zero deltas are retained in the reported sample size and observed mean but
    removed from the sign enumeration because flipping their signs cannot alter
    the test statistic. This preserves the exact p-value while reducing work.
    """

    max_exact_n = _nonnegative_integer(max_exact_n, "max_exact_n")
    n_permutations = _positive_integer(n_permutations, "n_permutations")
    random_seed = _nonnegative_integer(random_seed, "random_seed")
    chunk_size = _positive_integer(chunk_size, "chunk_size")

    array = _finite_values(values)
    if array.size == 0:
        raise ValueError("values must contain at least one finite observation")

    scale = float(np.max(np.abs(array)))
    if scale == 0.0:
        observed_mean = 0.0
    else:
        observed_mean = float(scale * np.mean(array / scale))
        if not np.isfinite(observed_mean):
            raise ValueError("mean of values exceeds floating-point range")
    nonzero = array[array != 0.0]
    n_nonzero = int(nonzero.size)

    if n_nonzero == 0:
        return SignFlipResult(
            observed_mean=observed_mean,
            n_observations=int(array.size),
            n_nonzero=0,
            p_value=1.0,
            method="exact",
            permutations_evaluated=1,
            random_seed=None,
        )

    # Sign-flip p-values are invariant under positive rescaling. Normalizing by
    # the largest magnitude prevents finite inputs near the floating-point limit
    # from overflowing the observed and permuted sums.
    statistic_values = nonzero / scale
    observed_abs_sum = abs(float(np.sum(statistic_values, dtype=float)))
    threshold = _comparison_threshold(observed_abs_sum, statistic_values)
    if n_nonzero <= max_exact_n:
        total = 1 << n_nonzero
        extreme = _count_exact_extremes(
            statistic_values,
            threshold=threshold,
            total=total,
            chunk_size=chunk_size,
        )
        p_value = float(extreme / total)
        method = "exact"
        permutations_evaluated = total
        seed_used: int | None = None
    else:
        extreme = _count_monte_carlo_extremes(
            statistic_values,
            threshold=threshold,
            n_permutations=n_permutations,
            random_seed=random_seed,
            chunk_size=chunk_size,
        )
        p_value = float((extreme + 1) / (n_permutations + 1))
        method = "monte_carlo"
        permutations_evaluated = n_permutations
        seed_used = random_seed

    return SignFlipResult(
        observed_mean=observed_mean,
        n_observations=int(array.size),
        n_nonzero=n_nonzero,
        p_value=p_value,
        method=method,
        permutations_evaluated=permutations_evaluated,
        random_seed=seed_used,
    )


def score_table_sign_flip_summary(
    frame: pd.DataFrame,
    *,
    model_column: str = "model",
    value_column: str = "delta_vs_best_static",
    models: Iterable[str] | None = None,
    max_exact_n: int = 20,
    n_permutations: int = 10_000,
    random_seed: int = 1,
    chunk_size: int = 65_536,
) -> pd.DataFrame:
    """Return one adaptive sign-flip result per selected model in a score table."""

    missing = [column for column in (model_column, value_column) if column not in frame]
    if missing:
        raise KeyError(f"required columns missing from score table: {missing}")

    selected = frame.copy()
    requested_models = _normalize_models(models)
    if requested_models is not None:
        selected = selected.loc[selected[model_column].astype(str).isin(requested_models)]

    rows: list[dict[str, object]] = []
    grouped = selected.groupby(model_column, sort=False, dropna=False)
    for group_index, (model, group) in enumerate(grouped):
        values = _numeric_series(group[value_column], value_column)
        if values.size == 0:
            rows.append(
                {
                    model_column: model,
                    "value_column": value_column,
                    "observed_mean": np.nan,
                    "n_observations": 0,
                    "n_nonzero": 0,
                    "p_value": np.nan,
                    "method": "no_data",
                    "permutations_evaluated": 0,
                    "random_seed": pd.NA,
                }
            )
            continue

        result = paired_sign_flip_test(
            values,
            max_exact_n=max_exact_n,
            n_permutations=n_permutations,
            random_seed=random_seed + group_index,
            chunk_size=chunk_size,
        )
        row = {model_column: model, "value_column": value_column, **asdict(result)}
        rows.append(row)

    columns = [
        model_column,
        "value_column",
        "observed_mean",
        "n_observations",
        "n_nonzero",
        "p_value",
        "method",
        "permutations_evaluated",
        "random_seed",
    ]
    return pd.DataFrame(rows, columns=columns)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m hipporeplayimm.sign_flip_report``."""

    parser = argparse.ArgumentParser(
        description=(
            "Compute two-sided paired sign-flip tests from a replay score CSV. "
            "Samples up to --max-exact-n nonzero deltas are enumerated exactly."
        )
    )
    parser.add_argument("--scores", required=True, help="Input score CSV.")
    parser.add_argument("--output", required=True, help="Output summary CSV.")
    parser.add_argument("--model-column", default="model")
    parser.add_argument("--value-column", default="delta_vs_best_static")
    parser.add_argument(
        "--models",
        help="Optional comma-separated model names. By default every model is tested.",
    )
    parser.add_argument("--max-exact-n", type=int, default=20)
    parser.add_argument("--n-permutations", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=65_536)
    args = parser.parse_args(argv)

    frame = pd.read_csv(args.scores)
    models = None if args.models is None else args.models.split(",")
    summary = score_table_sign_flip_summary(
        frame,
        model_column=args.model_column,
        value_column=args.value_column,
        models=models,
        max_exact_n=args.max_exact_n,
        n_permutations=args.n_permutations,
        random_seed=args.random_seed,
        chunk_size=args.chunk_size,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(summary.to_string(index=False))
    return 0


def _count_exact_extremes(
    values: np.ndarray,
    *,
    threshold: float,
    total: int,
    chunk_size: int,
) -> int:
    bit_positions = np.arange(values.size, dtype=np.uint64)
    extreme = 0
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        masks = np.arange(start, stop, dtype=np.uint64)[:, None]
        bits = ((masks >> bit_positions) & np.uint64(1)).astype(np.int8)
        signs = 1.0 - 2.0 * bits
        signed_sums = signs @ values
        extreme += int(np.count_nonzero(np.abs(signed_sums) >= threshold))
    return extreme


def _count_monte_carlo_extremes(
    values: np.ndarray,
    *,
    threshold: float,
    n_permutations: int,
    random_seed: int,
    chunk_size: int,
) -> int:
    rng = np.random.default_rng(random_seed)
    extreme = 0
    remaining = n_permutations
    while remaining:
        current = min(chunk_size, remaining)
        bits = rng.integers(0, 2, size=(current, values.size), dtype=np.int8)
        signs = 1.0 - 2.0 * bits
        signed_sums = signs @ values
        extreme += int(np.count_nonzero(np.abs(signed_sums) >= threshold))
        remaining -= current
    return extreme


def _comparison_threshold(observed_abs_sum: float, values: np.ndarray) -> float:
    scale = float(np.sum(np.abs(values), dtype=float))
    tolerance = 8.0 * np.finfo(float).eps * scale
    return max(0.0, observed_abs_sum - tolerance)


def _finite_values(values: Sequence[float] | np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=object)
    if any(isinstance(value, (bool, np.bool_)) for value in raw.flat):
        raise ValueError("values must be numeric deltas, not booleans")

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    try:
        numeric = array.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("values must contain only numeric deltas") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError("values must contain only finite numeric deltas")
    return np.asarray(numeric, dtype=float)


def _numeric_series(series: pd.Series, name: str) -> np.ndarray:
    boolean = series.map(lambda value: isinstance(value, (bool, np.bool_)))
    if bool(boolean.any()):
        raise ValueError(f"{name} contains boolean values")

    numeric = pd.to_numeric(series, errors="coerce")
    missing = series.isna()
    invalid = ~missing & numeric.isna()
    if bool(invalid.any()):
        examples = series.loc[invalid].astype(str).head(3).tolist()
        raise ValueError(f"{name} contains populated nonnumeric values: {examples}")
    finite = numeric.loc[~numeric.isna()].to_numpy(dtype=float)
    if finite.size and not np.all(np.isfinite(finite)):
        raise ValueError(f"{name} contains non-finite values")
    return finite


def _normalize_models(models: Iterable[str] | None) -> set[str] | None:
    if models is None:
        return None
    normalized = {str(model).strip() for model in models if str(model).strip()}
    if not normalized:
        raise ValueError("models must contain at least one non-empty model name")
    return normalized


def _positive_integer(value: object, name: str) -> int:
    integer = _nonnegative_integer(value, name)
    if integer == 0:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    try:
        is_exact = value == integer
    except Exception as exc:  # pragma: no cover - defensive for exotic scalars
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if not bool(is_exact) or integer < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return integer


if __name__ == "__main__":
    raise SystemExit(main())
