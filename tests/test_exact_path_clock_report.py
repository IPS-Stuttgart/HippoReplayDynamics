import numpy as np
import pandas as pd
import pytest

from scripts.report_exact_path_clock_recovery import SUPPORTS, comparison_table, score_errors


def test_same_population_comparison_joins_and_errors():
    rows = []
    for repeat in range(3):
        for support in SUPPORTS:
            phi = 0.25 if support == "exhaustive" else 0.65
            rows.append(
                {
                    "dataset": "test",
                    "animal": "rat",
                    "session": "session",
                    "teacher": "bank",
                    "scenario": 0.25,
                    "repeat": repeat,
                    "support": support,
                    "phi_hat": phi,
                    "phi_low": phi - 0.05,
                    "phi_high": phi + 0.05,
                }
            )
    fits = pd.DataFrame(rows)
    result = comparison_table(fits)
    assert len(result) == 18
    np.testing.assert_allclose(result.estimate_error_vs_exact, 0.4)
    np.testing.assert_allclose(result.max_interval_endpoint_error_vs_exact, 0.4)
    assert not result.direction_agreement.any()
    for invalid in (fits.iloc[:-1], fits.iloc[:0], pd.concat([fits, fits.iloc[:1]])):
        with pytest.raises(ValueError):
            comparison_table(invalid)


def test_likelihood_residual_sign_and_clock_contrast():
    metadata = pd.DataFrame({"dataset": "test", "source_teacher": "bank", "scenario": 0.25, "generator": "physical", "row_index": range(3)})
    exact = np.full((3, 5), -10.0)
    mc = np.broadcast_to(exact, (2, 3, 3, 5)).copy()
    mc[..., 0] -= 0.6
    mc[..., 1] -= 0.2
    result = score_errors(metadata, exact, mc)
    np.testing.assert_allclose(result[result.axis.eq("physical")].mean_signed_score_error, -0.6)
    np.testing.assert_allclose(result[result.axis.eq("neural_minus_physical")].mean_signed_score_error, 0.4)
    np.testing.assert_allclose(result[result.axis.eq("stationary")].mean_signed_score_error, 0)
    assert result.n_events.eq(3).all()


def test_incomplete_or_nonfinite_likelihoods_rejected():
    metadata = pd.DataFrame({"dataset": "test", "source_teacher": "bank", "scenario": 0.5, "generator": "neural", "row_index": range(3)})
    exact = np.zeros((3, 5))
    mc = np.zeros((2, 3, 3, 5))
    with pytest.raises(ValueError):
        score_errors(metadata.iloc[:0], exact, mc)
    with pytest.raises(ValueError):
        score_errors(metadata, exact, mc[:, :2])
    mc[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        score_errors(metadata, exact, mc)
