import numpy as np
import pytest

from hipporeplayimm.simulation_recovery import (
    SimulationRecoveryConfig,
    _validate_recovery_runtime_limits,
)


@pytest.mark.parametrize(
    "event_count",
    [
        None,
        0,
        -1,
        1.5,
        "1.5",
        np.nan,
        np.inf,
        True,
        np.array(True),
        np.array(False),
        np.array([3]),
        np.array([True]),
    ],
)
def test_simulation_recovery_rejects_invalid_events_per_model(event_count):
    with pytest.raises(ValueError, match="events_per_model"):
        _validate_recovery_runtime_limits(
            SimulationRecoveryConfig(events_per_model=event_count)
        )


@pytest.mark.parametrize("event_count", [1, 3, 3.0, "3", "3.0", np.array(3)])
def test_simulation_recovery_accepts_integer_valued_events_per_model(event_count):
    _validate_recovery_runtime_limits(
        SimulationRecoveryConfig(events_per_model=event_count)
    )


@pytest.mark.parametrize(
    "event_count",
    [
        0,
        -1,
        1.5,
        "1.5",
        np.nan,
        np.inf,
        True,
        np.array(True),
        np.array(False),
        np.array([3]),
        np.array([True]),
    ],
)
def test_simulation_recovery_rejects_invalid_max_template_events(event_count):
    with pytest.raises(ValueError, match="max_template_events"):
        _validate_recovery_runtime_limits(
            SimulationRecoveryConfig(max_template_events=event_count)
        )


@pytest.mark.parametrize("event_count", [None, 1, 3, 3.0, "3", "3.0", np.array(3)])
def test_simulation_recovery_accepts_integer_valued_max_template_events(event_count):
    _validate_recovery_runtime_limits(
        SimulationRecoveryConfig(max_template_events=event_count)
    )


@pytest.mark.parametrize(
    "limit_value",
    [
        0,
        -1,
        1.5,
        "1.5",
        np.nan,
        np.inf,
        True,
        np.array(True),
        np.array(False),
        np.array([3]),
        np.array([True]),
    ],
)
def test_simulation_recovery_rejects_invalid_max_synthetic_events(limit_value):
    with pytest.raises(ValueError, match="max_synthetic_events"):
        _validate_recovery_runtime_limits(
            SimulationRecoveryConfig(max_synthetic_events=limit_value)
        )


@pytest.mark.parametrize("limit_value", [3, 3.0, "3", "3.0", np.array(3)])
def test_simulation_recovery_accepts_integer_valued_max_synthetic_events(limit_value):
    _validate_recovery_runtime_limits(
        SimulationRecoveryConfig(max_synthetic_events=limit_value)
    )


@pytest.mark.parametrize(
    "limit_value",
    [
        0.0,
        -0.1,
        np.nan,
        np.inf,
        True,
        np.array(True),
        np.array(False),
        np.array([1.0]),
        np.array([True]),
    ],
)
def test_simulation_recovery_rejects_invalid_max_runtime_s(limit_value):
    with pytest.raises(ValueError, match="max_runtime_s"):
        _validate_recovery_runtime_limits(SimulationRecoveryConfig(max_runtime_s=limit_value))


@pytest.mark.parametrize(
    "field_name",
    [
        "score_with_occupancy",
        "oracle_candidate_support",
        "continue_on_error",
        "progress_log",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        None,
        0,
        1,
        0.0,
        1.0,
        "False",
        "True",
        np.array([False]),
        np.array([True]),
    ],
)
def test_simulation_recovery_rejects_non_boolean_control_values(field_name, value):
    with pytest.raises(ValueError, match=field_name):
        _validate_recovery_runtime_limits(
            SimulationRecoveryConfig(**{field_name: value})
        )


@pytest.mark.parametrize(
    "value",
    [True, False, np.bool_(True), np.bool_(False), np.array(True), np.array(False)],
)
def test_simulation_recovery_accepts_scalar_boolean_control_values(value):
    _validate_recovery_runtime_limits(
        SimulationRecoveryConfig(
            score_with_occupancy=value,
            oracle_candidate_support=value,
            continue_on_error=value,
            progress_log=value,
        )
    )


def test_simulation_recovery_accepts_valid_runtime_limits():
    _validate_recovery_runtime_limits(
        SimulationRecoveryConfig(max_synthetic_events=3, max_runtime_s=1.25)
    )
