from __future__ import annotations

from argparse import Namespace

import numpy as np
import pandas as pd

from scripts.augment_model_evidence_diagnostics import _safe_divide, augment, write_outputs


def test_safe_divide_coerces_csv_numeric_artifacts():
    result = _safe_divide(
        pd.Series(["4.0", "bad", "9.0"]),
        pd.Series(["2.0", "1.0", "0.0"]),
    )

    assert np.isclose(result.iloc[0], 2.0)
    assert np.isnan(result.iloc[1])
    assert np.isnan(result.iloc[2])


def test_augment_runtime_columns_tolerate_malformed_csv_scalars(tmp_path):
    scores_path = tmp_path / "scores.csv"
    pd.DataFrame(
        {
            "model": ["stationary", "diffusion", "stationary"],
            "runtime_s": ["2.0", "bad", "0.0"],
            "relative_log_evidence": ["4.0", "6.0", "bad"],
            "truncated_relative_log_evidence": ["1.0", "bad", "3.0"],
        }
    ).to_csv(scores_path, index=False)

    result = augment(
        Namespace(
            scores=scores_path,
            min_spikes=3,
            min_time_bins=2,
            min_candidate_log_mass=float(np.log(0.95)),
            max_terminal_entropy=float("inf"),
            hyperparameter_source="unit-test",
            selection_dataset="unit",
            selection_metric="relative_log_evidence",
        )
    )

    assert np.isclose(result.loc[0, "relative_log_evidence_per_runtime_s"], 2.0)
    assert np.isnan(result.loc[1, "runtime_s"])
    assert np.isnan(result.loc[1, "relative_log_evidence_per_runtime_s"])
    assert np.isnan(result.loc[2, "relative_log_evidence_per_runtime_s"])


def test_write_outputs_ignores_bad_runtime_values_in_summary(tmp_path):
    scores = pd.DataFrame(
        {
            "model": ["stationary", "stationary", "stationary"],
            "runtime_s": ["1.0", "bad", "3.0"],
            "relative_log_evidence_per_runtime_s": ["2.0", "bad", "6.0"],
            "event_reliable": [True, False, True],
            "event_low_spike_count": [False, True, False],
            "event_low_candidate_mass": [False, False, False],
            "event_too_few_time_bins": [False, False, False],
        }
    )

    write_outputs(scores, tmp_path)

    runtime = pd.read_csv(tmp_path / "model_runtime_summary.csv").iloc[0]
    assert int(runtime["rows"]) == 3
    assert np.isclose(runtime["mean_runtime_s"], 2.0)
    assert np.isclose(runtime["median_runtime_s"], 2.0)
    assert np.isclose(runtime["p95_runtime_s"], np.quantile([1.0, 3.0], 0.95))
    assert np.isclose(runtime["relative_log_evidence_per_runtime_s"], 4.0)
