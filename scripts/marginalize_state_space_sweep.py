#!/usr/bin/env python3
"""Marginalize selected sorted-spike state-space parameter sweeps.

The state-space evidence sweep scores point-estimate configurations. This
script turns those grid scores into Bayesian model-evidence rows by integrating
over the selected parameter grid, mirroring the KD-style workflow while keeping
the sorted-spike state-space implementation separate from the KD reference
implementation. Observation hyperparameters can be folded into the same grid so
rates/binning/smoothing choices are marginalized rather than selected by a
single point estimate.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from benchmark_model_evidence import _add_evidence_columns, _counts, _summary
from hipporeplayimm.evidence_reporting import EXACT_EVIDENCE_SUPPORT, TRUNCATED_EVIDENCE_SUPPORT
from hipporeplayimm.kd_reference import empirical_grid_prior

_EVENT_KEY = ["session", "event_index"]
_DEFAULT_MODELS = ("diffusion", "momentum")
_OBSERVATION_PARAMETER_COLUMNS = (
    "time_bin_s",
    "spike_rate_scale",
    "emission_likelihood_temperature",
    "emission_negative_binomial_overdispersion",
    "bin_size_cm",
    "smoothing_sigma_bins",
    "min_speed_cm_s",
    "clusterless_mark_smoothing_sigma_bins",
    "clusterless_mark_prior_count",
    "clusterless_mark_variance_floor",
    "clusterless_rate_floor_hz",
)
_OUTPUT_METADATA_COLUMNS = (
    "n_time",
    "n_spikes",
    "runtime_s",
    "bin_size_cm",
    "smoothing_sigma_bins",
    "min_speed_cm_s",
    "time_bin_s",
    "spike_rate_scale",
    "emission_likelihood_temperature",
    "emission_negative_binomial_overdispersion",
    "clusterless_mark_smoothing_sigma_bins",
    "clusterless_mark_prior_count",
    "clusterless_mark_variance_floor",
    "clusterless_rate_floor_hz",
)
_INTEGER_METADATA_COLUMNS = {"n_time", "n_spikes"}
_CATEGORICAL_PARAMETER_COLUMNS = {"state_space_momentum_candidate_source"}
_OPTIONAL_MOMENTUM_PARAMETER_DEFAULTS = {
    "state_space_momentum_predicted_candidate_top_k": 8,
    "state_space_momentum_candidate_source": "emission",
}
_EXACT_SPARSE_MOMENTUM_MODEL = "sorted-spike-state-space-momentum-exact-sparse"


@dataclass(frozen=True)
class _ModelSpec:
    short_name: str
    source_model: str
    output_model: str
    param_cols: tuple[str, ...]


_MODEL_SPECS = {
    "diffusion": _ModelSpec(
        short_name="diffusion",
        source_model="sorted-spike-state-space-diffusion",
        output_model="sorted-spike-state-space-diffusion-marginalized",
        param_cols=("state_space_diffusion_sigma_cm_sqrt_s",),
    ),
    "momentum": _ModelSpec(
        short_name="momentum",
        source_model="sorted-spike-state-space-momentum",
        output_model="sorted-spike-state-space-momentum-marginalized",
        param_cols=(
            "state_space_momentum_sigma_cm_sqrt_s",
            "state_space_momentum_initial_sigma_cm_sqrt_s",
            "state_space_momentum_velocity_decay",
            "state_space_momentum_candidate_top_k",
            "state_space_momentum_predicted_candidate_top_k",
            "state_space_momentum_candidate_source",
        ),
    ),
}


def marginalize_sweep(
    input_csv: str | Path,
    output: str | Path,
    *,
    models: tuple[str, ...] = _DEFAULT_MODELS,
    prior: str = "empirical",
    observation_parameters: str = "auto",
) -> dict[str, pd.DataFrame]:
    scores = pd.read_csv(input_csv)
    scores = _ensure_optional_momentum_parameter_columns(scores)
    if scores.empty:
        raise ValueError("state-space sweep scores are empty")
    rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []
    prior_rows: list[dict[str, object]] = []

    for model in models:
        spec = _MODEL_SPECS[_canonical_model(model)]
        grid, event_table, param_values, source = _grid_for_model(scores, spec, observation_parameters=observation_parameters)
        source_model = _source_model_from_rows(source, spec)
        weights, prior_kind = _prior_for_grid(grid, spec, param_values, prior)
        marginalized = logsumexp(grid + np.log(np.maximum(weights, np.finfo(float).tiny))[None, ...], axis=tuple(range(1, grid.ndim)))
        rows.extend(_marginalized_rows(spec, event_table, source, param_values, marginalized, prior_kind, source_model))
        best_rows.extend(_best_parameter_rows(spec, event_table, grid, param_values, source_model))
        prior_rows.extend(_prior_rows(spec, param_values, weights, prior_kind, source_model))

    outdir = Path(output)
    outdir.mkdir(parents=True, exist_ok=True)
    event_model_evidence = pd.DataFrame(rows)
    if "runtime_s" not in event_model_evidence:
        event_model_evidence["runtime_s"] = 0.0
    event_model_evidence = _add_evidence_columns(event_model_evidence)
    gridsearch_best_params = pd.DataFrame(best_rows)
    prior_weights = pd.DataFrame(prior_rows)
    event_model_evidence.to_csv(outdir / "state_space_marginalized_event_model_evidence.csv", index=False)
    _summary(event_model_evidence).to_csv(outdir / "state_space_marginalized_model_evidence_summary.csv", index=False)
    _counts(event_model_evidence).to_csv(outdir / "state_space_marginalized_best_model_counts.csv", index=False)
    gridsearch_best_params.to_csv(outdir / "state_space_marginalized_gridsearch_best_params.csv", index=False)
    prior_weights.to_csv(outdir / "state_space_marginalized_prior_weights.csv", index=False)
    for metric in ("log_evidence", "relative_log_evidence", "model_probability"):
        event_model_evidence.pivot_table(index=_EVENT_KEY, columns="model", values=metric, aggfunc="first").reset_index().to_csv(
            outdir / f"state_space_marginalized_event_model_pivot_{metric}.csv", index=False
        )
    return {
        "event_model_evidence": event_model_evidence,
        "model_evidence_summary": _summary(event_model_evidence),
        "best_model_counts": _counts(event_model_evidence),
        "gridsearch_best_params": gridsearch_best_params,
        "prior_weights": prior_weights,
    }


def _canonical_model(model: str) -> str:
    name = model.strip().lower()
    if name.startswith("sorted-spike-state-space-"):
        name = name.removeprefix("sorted-spike-state-space-")
    if name.endswith("-marginalized"):
        name = name.removesuffix("-marginalized")
    if name not in _MODEL_SPECS:
        raise ValueError(f"unsupported marginalized model {model!r}; choose from {sorted(_MODEL_SPECS)}")
    return name


def _grid_for_model(
    scores: pd.DataFrame,
    spec: _ModelSpec,
    *,
    observation_parameters: str,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    if observation_parameters not in {"auto", "all", "none"}:
        raise ValueError("observation_parameters must be one of: auto, all, none")
    missing_base = [col for col in (*_EVENT_KEY, "model", "log_evidence", *spec.param_cols) if col not in scores.columns]
    if missing_base:
        raise ValueError(f"sweep scores are missing columns needed for {spec.short_name}: {missing_base}")
    source = _source_rows_for_model(scores, spec)
    source_model = _source_model_from_rows(source, spec)

    param_cols = _parameter_columns_for_model(source, spec, observation_parameters)
    missing = [col for col in param_cols if col not in source.columns]
    if missing:
        raise ValueError(f"sweep scores are missing columns needed for {spec.short_name}: {missing}")

    grouped = (
        source.groupby([*_EVENT_KEY, *param_cols], dropna=False, as_index=False)
        .agg(**_aggregation_columns(source, param_cols))
        .sort_values([*_EVENT_KEY, *param_cols])
        .reset_index(drop=True)
    )
    event_table = grouped[_EVENT_KEY].drop_duplicates().sort_values(_EVENT_KEY).reset_index(drop=True)
    param_values = {col: _sorted_parameter_values(grouped[col], col) for col in param_cols}
    empty_params = [col for col, values in param_values.items() if values.size == 0]
    if empty_params:
        raise ValueError(f"{spec.source_model} has no finite values for parameter columns: {empty_params}")

    shape = (len(event_table), *(len(values) for values in param_values.values()))
    grid = np.full(shape, np.nan, dtype=float)
    event_index = {tuple(row): idx for idx, row in enumerate(event_table[_EVENT_KEY].itertuples(index=False, name=None))}
    param_index = {col: _parameter_index(values, col) for col, values in param_values.items()}
    for row in grouped.itertuples(index=False):
        event = tuple(getattr(row, col) for col in _EVENT_KEY)
        params = tuple(param_index[col][_parameter_key(getattr(row, col), col)] for col in param_cols)
        grid[(event_index[event], *params)] = float(row.log_evidence)
    missing_count = int(np.isnan(grid).sum())
    if missing_count:
        raise ValueError(f"{spec.source_model} grid has {missing_count} missing event/parameter scores")
    representatives = _event_representatives(grouped, event_table)
    representatives["model"] = source_model
    return grid, event_table, param_values, representatives


def _source_rows_for_model(scores: pd.DataFrame, spec: _ModelSpec) -> pd.DataFrame:
    for source_model in _source_model_candidates(spec):
        source = scores[scores["model"] == source_model].copy()
        if "status" in source.columns:
            source = source[source["status"] == "success"].copy()
        if not source.empty:
            return source
    raise ValueError(f"no successful rows for any of: {', '.join(_source_model_candidates(spec))}")


def _source_model_candidates(spec: _ModelSpec) -> tuple[str, ...]:
    if spec.short_name == "momentum":
        return (_EXACT_SPARSE_MOMENTUM_MODEL, spec.source_model)
    return (spec.source_model,)


def _source_model_from_rows(source: pd.DataFrame, spec: _ModelSpec) -> str:
    if "model" not in source or source.empty:
        return spec.source_model
    return str(source["model"].iloc[0])


def _parameter_columns_for_model(source: pd.DataFrame, spec: _ModelSpec, observation_parameters: str) -> tuple[str, ...]:
    cols: list[str] = list(spec.param_cols)
    if observation_parameters == "none":
        return tuple(cols)
    for col in _OBSERVATION_PARAMETER_COLUMNS:
        if col not in source.columns:
            continue
        values = _numeric_values(source[col])
        include = observation_parameters == "all" or values.size > 1
        if include and col not in cols:
            cols.append(col)
    return tuple(cols)


def _numeric_values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
    return np.unique(values) if values.size else np.asarray([], dtype=float)


def _sorted_parameter_values(series: pd.Series, column: str) -> np.ndarray:
    if column in _CATEGORICAL_PARAMETER_COLUMNS:
        values = sorted(
            {_normalize_categorical_parameter(value, column) for value in series.dropna()}
        )
        return np.asarray(values, dtype=object)
    return _sorted_numeric_values(series, column)


def _sorted_numeric_values(series: pd.Series, column: str) -> np.ndarray:
    values = _numeric_values(series)
    if values.size and not np.all(np.isfinite(values)):
        raise ValueError(f"parameter column {column!r} contains non-finite values")
    return np.asarray(sorted(float(value) for value in values), dtype=float)


def _parameter_index(values: np.ndarray, column: str) -> dict[object, int]:
    return {_parameter_key(value, column): idx for idx, value in enumerate(values)}


def _parameter_key(value: object, column: str) -> object:
    if column in _CATEGORICAL_PARAMETER_COLUMNS:
        return _normalize_categorical_parameter(value, column)
    return float(value)


def _normalize_categorical_parameter(value: object, column: str) -> str:
    if column != "state_space_momentum_candidate_source":
        return str(value)
    if pd.isna(value):
        return "emission"
    text = str(value).strip().lower().replace("_", "-")
    aliases = {
        "": "emission",
        "none": "emission",
        "null": "emission",
        "nan": "emission",
        "likelihood": "emission",
        "log-likelihood": "emission",
        "train-posterior": "posterior",
        "diffusion-posterior": "posterior",
        "first-order-posterior": "posterior",
    }
    normalized = aliases.get(text, text)
    if normalized not in {"emission", "posterior"}:
        raise ValueError("state_space_momentum_candidate_source must be 'emission' or 'posterior'")
    return normalized


def _ensure_optional_momentum_parameter_columns(scores: pd.DataFrame) -> pd.DataFrame:
    out = scores.copy()
    for column, default in _OPTIONAL_MOMENTUM_PARAMETER_DEFAULTS.items():
        if column not in out.columns:
            out[column] = default
        else:
            out[column] = out[column].where(out[column].notna(), default)
    return out


def _aggregation_columns(source: pd.DataFrame, param_cols: tuple[str, ...]) -> dict[str, tuple[str, object]]:
    blocked = {*_EVENT_KEY, "model", *param_cols}
    agg: dict[str, tuple[str, object]] = {"log_evidence": ("log_evidence", _unique_log_evidence)}
    if "runtime_s" in source.columns and "runtime_s" not in blocked:
        agg["runtime_s"] = ("runtime_s", "sum")
    for col in _OUTPUT_METADATA_COLUMNS:
        if col in source.columns and col not in blocked and col not in agg:
            agg[col] = (col, "first")
    return agg


def _event_representatives(grouped: pd.DataFrame, event_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for event in event_table.itertuples(index=False):
        mask = np.ones(len(grouped), dtype=bool)
        for col, value in zip(_EVENT_KEY, event, strict=True):
            mask &= grouped[col].to_numpy() == value
        group = grouped.loc[mask]
        row: dict[str, object] = {col: value for col, value in zip(_EVENT_KEY, event, strict=True)}
        for col in _OUTPUT_METADATA_COLUMNS:
            if col not in grouped.columns:
                continue
            numeric = pd.to_numeric(group[col], errors="coerce")
            if col == "runtime_s":
                row[col] = float(numeric.fillna(0.0).sum())
                continue
            values = np.unique(numeric.dropna().to_numpy(float))
            row[col] = float(values[0]) if values.size == 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _unique_log_evidence(values: pd.Series) -> float:
    finite = values.dropna().to_numpy(float)
    if finite.size == 0:
        return float("nan")
    if float(np.max(finite) - np.min(finite)) > 1e-7:
        raise ValueError("duplicate grid rows have conflicting log evidence")
    return float(finite[0])


def _prior_for_grid(grid: np.ndarray, spec: _ModelSpec, param_values: dict[str, np.ndarray], prior: str) -> tuple[np.ndarray, str]:
    shape = grid.shape[1:]
    if prior == "uniform":
        return np.full(shape, 1.0 / max(1, int(np.prod(shape))), dtype=float), "uniform"

    param_cols = tuple(param_values)
    dynamic_varying = [idx for idx, col in enumerate(param_cols) if col in spec.param_cols and param_values[col].shape[0] > 1]
    try:
        if len(dynamic_varying) == 1:
            collapsed = _collapse_grid_to_parameter_axes(grid, dynamic_varying)
            col = param_cols[dynamic_varying[0]]
            prior_values, _ = empirical_grid_prior({"sd_meters": param_values[col]}, collapsed)
            return _expand_prior(prior_values, param_values, dynamic_varying), _empirical_prior_kind(param_cols)
        if len(dynamic_varying) == 2:
            varying_cols = [param_cols[idx] for idx in dynamic_varying]
            sigma_col = next((col for col in varying_cols if "sigma" in col), None)
            decay_col = next((col for col in varying_cols if "decay" in col), None)
            if sigma_col is not None and decay_col is not None:
                collapsed = _collapse_grid_to_parameter_axes(grid, dynamic_varying)
                if varying_cols != [sigma_col, decay_col]:
                    source_axes = [varying_cols.index(sigma_col) + 1, varying_cols.index(decay_col) + 1]
                    collapsed = np.moveaxis(collapsed, source_axes, [1, 2])
                prior_values, _ = empirical_grid_prior({"sd_meters": param_values[sigma_col], "decay": param_values[decay_col]}, collapsed)
                target_axes = [param_cols.index(sigma_col), param_cols.index(decay_col)]
                return _expand_prior(prior_values, param_values, target_axes), _empirical_prior_kind(param_cols)
    except Exception:
        pass
    return np.full(shape, 1.0 / max(1, int(np.prod(shape))), dtype=float), "uniform_fallback"


def _collapse_grid_to_parameter_axes(grid: np.ndarray, keep_param_axes: list[int]) -> np.ndarray:
    keep_grid_axes = {axis + 1 for axis in keep_param_axes}
    collapsed = grid
    for axis in range(grid.ndim - 1, 0, -1):
        if axis not in keep_grid_axes:
            n_values = collapsed.shape[axis]
            collapsed = logsumexp(collapsed - np.log(float(max(n_values, 1))), axis=axis)
    return collapsed


def _empirical_prior_kind(param_cols: tuple[str, ...]) -> str:
    if any(col in _OBSERVATION_PARAMETER_COLUMNS for col in param_cols):
        return "empirical_dynamic_uniform_observation"
    return "empirical"


def _expand_prior(prior_values: np.ndarray, param_values: dict[str, np.ndarray], varying: list[int]) -> np.ndarray:
    param_cols = tuple(param_values)
    target_shape = tuple(param_values[col].shape[0] for col in param_cols)
    reshape = [1] * len(param_cols)
    for axis, size in zip(varying, prior_values.shape, strict=True):
        reshape[axis] = size
    prior = np.broadcast_to(prior_values.reshape(reshape), target_shape).astype(float)
    return prior / float(np.sum(prior))


def _marginalized_rows(
    spec: _ModelSpec,
    event_table: pd.DataFrame,
    source: pd.DataFrame,
    param_values: dict[str, np.ndarray],
    marginalized: np.ndarray,
    prior_kind: str,
    source_model: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    param_cols = tuple(param_values)
    observation_cols = tuple(col for col in param_cols if col in _OBSERVATION_PARAMETER_COLUMNS)
    dynamics_cols = tuple(col for col in param_cols if col not in observation_cols)
    grid_points = int(np.prod([len(values) for values in param_values.values()]))
    grid_description = ";".join(f"{col}={','.join(_format_parameter_value(value) for value in values)}" for col, values in param_values.items())
    for row_index, event in event_table.iterrows():
        representative = source.iloc[row_index]
        source_support = _source_evidence_support(spec, source_model)
        row: dict[str, object] = {
            "status": "success",
            "session": event["session"],
            "event_index": int(event["event_index"]),
            "model": spec.output_model,
            "requested_model": spec.output_model,
            "model_family": "trajectory",
            "log_evidence": float(marginalized[row_index]),
            "error": "",
            "diagnostic_state_space_marginalized_source_model": source_model,
            "diagnostic_state_space_marginalization_prior": prior_kind,
            "diagnostic_state_space_marginalization_grid_points": grid_points,
            "diagnostic_state_space_marginalized_dynamics_parameters": " ".join(dynamics_cols),
            "diagnostic_state_space_marginalized_observation_parameters": " ".join(observation_cols),
            "diagnostic_state_space_marginalization_grid": grid_description,
            "diagnostic_state_space_marginalized_source_evidence_support": source_support,
        }
        if spec.short_name == "momentum":
            if source_support == TRUNCATED_EVIDENCE_SUPPORT:
                # The legacy sorted-spike state-space momentum implementation uses
                # candidate-pruned second-order recursions, so its marginalized row
                # remains a truncated full-grid lower bound rather than exact evidence.
                row["diagnostic_state_space_momentum_evidence_support"] = TRUNCATED_EVIDENCE_SUPPORT
            else:
                row["diagnostic_state_space_sparse_momentum_evidence_support"] = EXACT_EVIDENCE_SUPPORT
        for col in _OUTPUT_METADATA_COLUMNS:
            if col not in source.columns:
                continue
            value = representative[col]
            if pd.isna(value):
                row[col] = np.nan
            elif col in _INTEGER_METADATA_COLUMNS:
                row[col] = int(value)
            else:
                row[col] = float(value)
        rows.append(row)
    return rows


def _source_evidence_support(spec: _ModelSpec, source_model: str) -> str:
    if spec.short_name == "momentum" and source_model != _EXACT_SPARSE_MOMENTUM_MODEL:
        return TRUNCATED_EVIDENCE_SUPPORT
    return EXACT_EVIDENCE_SUPPORT


def _best_parameter_rows(
    spec: _ModelSpec,
    event_table: pd.DataFrame,
    grid: np.ndarray,
    param_values: dict[str, np.ndarray],
    source_model: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    param_cols = list(param_values)
    for event_row, event in event_table.iterrows():
        flat_best = int(np.nanargmax(grid[event_row].reshape(-1)))
        indices = np.unravel_index(flat_best, grid.shape[1:])
        row: dict[str, object] = {
            "session": event["session"],
            "event_index": int(event["event_index"]),
            "source_model": source_model,
            "marginalized_model": spec.output_model,
            "best_log_evidence": float(grid[(event_row, *indices)]),
        }
        for col, idx in zip(param_cols, indices, strict=True):
            row[f"best_{col}"] = _parameter_output_value(param_values[col][idx])
        rows.append(row)
    return rows


def _prior_rows(
    spec: _ModelSpec,
    param_values: dict[str, np.ndarray],
    prior: np.ndarray,
    prior_kind: str,
    source_model: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    value_lists = [values.tolist() for values in param_values.values()]
    for indices in itertools.product(*(range(len(values)) for values in value_lists)):
        row: dict[str, object] = {
            "source_model": source_model,
            "marginalized_model": spec.output_model,
            "prior": prior_kind,
            "prior_weight": float(prior[indices]),
        }
        for col, idx in zip(param_values, indices, strict=True):
            row[col] = _parameter_output_value(param_values[col][idx])
        rows.append(row)
    return rows


def _parameter_output_value(value: object) -> object:
    if isinstance(value, str):
        return value
    try:
        scalar = value.item()
    except AttributeError:
        scalar = value
    if isinstance(scalar, (int, np.integer)):
        return int(scalar)
    if isinstance(scalar, (float, np.floating)):
        return float(scalar)
    return scalar


def _format_parameter_value(value: object) -> str:
    return str(value) if isinstance(value, str) else f"{float(value):g}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Marginalize selected state-space parameter-sweep evidence.")
    parser.add_argument("--input", required=True, help="state_space_evidence_sweep_event_scores.csv")
    parser.add_argument("--output", default="results/state-space-marginalized-evidence")
    parser.add_argument("--models", default=" ".join(_DEFAULT_MODELS), help="Models to marginalize: diffusion momentum")
    parser.add_argument("--prior", choices=("empirical", "uniform"), default="empirical")
    parser.add_argument(
        "--observation-parameters",
        choices=("auto", "all", "none"),
        default="auto",
        help=(
            "Observation hyperparameters to include in the marginalization grid: "
            "auto includes present columns with multiple values, all includes present columns even if singleton, "
            "and none reproduces dynamics-only marginalization."
        ),
    )
    args = parser.parse_args()
    models = tuple(_canonical_model(item) for item in args.models.replace(",", " ").split() if item.strip())
    tables = marginalize_sweep(args.input, args.output, models=models, prior=args.prior, observation_parameters=args.observation_parameters)
    print(tables["model_evidence_summary"].to_string(index=False))
    print("\nBest-model counts:")
    print(tables["best_model_counts"].to_string(index=False))
    print(f"\nRows: {len(tables['event_model_evidence'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
