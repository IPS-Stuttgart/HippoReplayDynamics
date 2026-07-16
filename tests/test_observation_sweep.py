import numpy as np
import pandas as pd

from hipporeplayimm.observation_sweep import (
    ObservationSweepConfig,
    observation_parameter_grid,
    summarize_observation_sweep,
)


def test_observation_parameter_grid_cartesian_product():
    config = ObservationSweepConfig(
        bin_sizes_cm=(4.0, 6.0),
        smoothing_sigmas_bins=(1.5,),
        min_speed_cm_s=(5.0,),
        min_occupancy_s=(0.01, 0.02),
        rate_floor_hz=(1e-5,),
        time_bin_ms=(2.0, 3.0),
        spike_rate_scales=(0.5, 1.0),
    )

    rows = observation_parameter_grid(config)

    assert len(rows) == 16
    assert rows[0]["sweep_id"] == 0
    assert rows[0]["bin_size_cm"] == 4.0
    assert rows[0]["min_occupancy_s"] == 0.01
    assert rows[0]["time_bin_s"] == 0.002
    assert rows[-1]["bin_size_cm"] == 6.0
    assert rows[-1]["min_occupancy_s"] == 0.02
    assert rows[-1]["time_bin_ms"] == 3.0
    assert rows[-1]["spike_rate_scale"] == 1.0


def test_observation_parameter_grid_allows_zero_smoothing_and_speed_threshold():
    config = ObservationSweepConfig(
        smoothing_sigmas_bins=(0.0,),
        min_speed_cm_s=(0.0,),
    )

    rows = observation_parameter_grid(config)

    assert len(rows) == 1
    assert rows[0]["smoothing_sigma_bins"] == 0.0
    assert rows[0]["min_speed_cm_s"] == 0.0


def test_observation_parameter_grid_rejects_nonpositive_values():
    config = ObservationSweepConfig(rate_floor_hz=(0.0,))

    try:
        observation_parameter_grid(config)
    except ValueError as exc:
        assert "rate_floor_hz" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_observation_parameter_grid_rejects_negative_nonnegative_values():
    cases = [
        (ObservationSweepConfig(smoothing_sigmas_bins=(-1.0,)), "smoothing_sigmas_bins"),
        (ObservationSweepConfig(min_speed_cm_s=(-1.0,)), "min_speed_cm_s"),
        (
            ObservationSweepConfig(negative_binomial_overdispersions=(-1.0,)),
            "negative_binomial_overdispersions",
        ),
    ]

    for config, field_name in cases:
        try:
            observation_parameter_grid(config)
        except ValueError as exc:
            message = str(exc)
            assert field_name in message
            assert "nonnegative" in message
        else:
            raise AssertionError(f"expected ValueError for {field_name}")


def test_observation_parameter_grid_rejects_nonfinite_values():
    cases = [
        (ObservationSweepConfig(bin_sizes_cm=(float("nan"),)), "bin_sizes_cm"),
        (ObservationSweepConfig(time_bin_ms=(float("inf"),)), "time_bin_ms"),
        (
            ObservationSweepConfig(
                negative_binomial_overdispersions=(float("nan"),)
            ),
            "negative_binomial_overdispersions",
        ),
        (ObservationSweepConfig(decode_bin_s=float("nan")), "decode_bin_s"),
    ]

    for config, field_name in cases:
        try:
            observation_parameter_grid(config)
        except ValueError as exc:
            message = str(exc)
            assert field_name in message
            assert "finite" in message
        else:
            raise AssertionError(f"expected ValueError for {field_name}")


def test_observation_parameter_grid_normalizes_numeric_overflow():
    cases = [
        (ObservationSweepConfig(bin_sizes_cm=(10**400,)), "bin_sizes_cm"),
        (ObservationSweepConfig(decode_bin_s=10**400), "decode_bin_s"),
    ]

    for config, field_name in cases:
        try:
            observation_parameter_grid(config)
        except ValueError as exc:
            message = str(exc)
            assert field_name in message
            assert "finite scalars" in message
        else:
            raise AssertionError(f"expected ValueError for {field_name}")


def test_observation_parameter_grid_rejects_boolean_numeric_values():
    cases = [
        (ObservationSweepConfig(bin_sizes_cm=(True,)), "bin_sizes_cm"),
        (ObservationSweepConfig(smoothing_sigmas_bins=(False,)), "smoothing_sigmas_bins"),
        (ObservationSweepConfig(decode_bin_s=True), "decode_bin_s"),
        (ObservationSweepConfig(n_folds=True), "n_folds"),
        (ObservationSweepConfig(simulation_events_per_model=True), "simulation_events_per_model"),
    ]

    for config, field_name in cases:
        try:
            observation_parameter_grid(config)
        except ValueError as exc:
            assert field_name in str(exc)
        else:
            raise AssertionError(f"expected ValueError for {field_name}")


def test_observation_parameter_grid_rejects_bare_string_sessions():
    config = ObservationSweepConfig(sessions="Rat1/Open1")

    try:
        observation_parameter_grid(config)
    except ValueError as exc:
        message = str(exc)
        assert "sessions" in message
        assert "sequence" in message
    else:
        raise AssertionError("expected ValueError for bare string sessions")


def test_observation_parameter_grid_rejects_invalid_session_sequences():
    cases = [
        ObservationSweepConfig(sessions=()),
        ObservationSweepConfig(sessions=("Rat1/Open1", " ")),
        ObservationSweepConfig(sessions=("Rat1/Open1", 1)),
    ]

    for config in cases:
        try:
            observation_parameter_grid(config)
        except ValueError as exc:
            assert "sessions" in str(exc)
        else:
            raise AssertionError("expected ValueError for invalid sessions")


def test_observation_parameter_grid_accepts_all_sessions_sentinel():
    config = ObservationSweepConfig(sessions=None)

    rows = observation_parameter_grid(config)

    assert len(rows) == 1


def test_summarize_observation_sweep_merges_overall_recovery():
    position_summary = pd.DataFrame(
        {
            "sweep_id": [0, 1],
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "median_posterior_mean_error_cm": [20.0, 10.0],
            "median_map_error_cm": [18.0, 12.0],
            "bin_size_cm": [4.0, 6.0],
            "smoothing_sigma_bins": [1.5, 2.0],
            "min_speed_cm_s": [5.0, 5.0],
            "min_occupancy_s": [0.02, 0.02],
            "rate_floor_hz": [1e-4, 1e-4],
            "time_bin_ms": [3.0, 3.0],
            "time_bin_s": [0.003, 0.003],
            "spike_rate_scale": [1.0, 1.0],
        }
    )
    simulation_summary = pd.DataFrame(
        {
            "sweep_id": [0, 0, 1],
            "session": ["Rat1/Open1", "Rat1/Open1", "Rat1/Open1"],
            "true_model": ["diffusion", "overall", "overall"],
            "recovery_accuracy": [0.25, 0.5, 0.75],
            "recovered_events": [1, 4, 6],
            "simulated_events": [4, 8, 8],
        }
    )

    summary = summarize_observation_sweep(position_summary, simulation_summary)

    assert list(summary["sweep_id"]) == [1, 0]
    assert np.isclose(summary.loc[summary["sweep_id"] == 1, "simulation_recovery_accuracy"].iloc[0], 0.75)
    assert np.isclose(summary.loc[summary["sweep_id"] == 0, "simulation_recovery_accuracy"].iloc[0], 0.5)
