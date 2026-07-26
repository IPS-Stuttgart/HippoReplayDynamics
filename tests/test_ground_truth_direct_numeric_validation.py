import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.ground_truth import (
    first_post_ripple_well_visit,
    infer_well_locations_from_arrays,
)


def _position_and_wells() -> tuple[np.ndarray, pd.DataFrame]:
    times = np.linspace(0.0, 1.0, 11)
    position = np.column_stack(
        [times, np.zeros_like(times), np.zeros_like(times), np.zeros_like(times)]
    )
    wells = pd.DataFrame(
        {
            "well_id": [1],
            "well_x": [0.0],
            "well_y": [0.0],
            "n_estimates": [1],
        }
    )
    return position, wells


@pytest.mark.parametrize(
    ("argument", "value", "error"),
    [
        ("ripple_peak", np.nan, ValueError),
        ("ripple_peak", True, TypeError),
        ("visit_radius_cm", 0.0, ValueError),
        ("visit_radius_cm", "5", TypeError),
        ("min_dwell_s", -0.1, ValueError),
        ("min_dwell_s", False, TypeError),
        ("future_horizon_s", np.inf, ValueError),
        ("future_horizon_s", np.array([1.0]), TypeError),
    ],
)
def test_first_post_ripple_well_visit_rejects_invalid_direct_numeric_arguments(
    argument: str,
    value: object,
    error: type[Exception],
):
    position, wells = _position_and_wells()
    arguments = {
        "ripple_peak": 0.0,
        "visit_radius_cm": 1.0,
        "min_dwell_s": 0.1,
        "future_horizon_s": 1.0,
    }
    arguments[argument] = value

    with pytest.raises(error, match=argument):
        first_post_ripple_well_visit(position, wells, **arguments)


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (0.0, ValueError),
        (-1.0, ValueError),
        (np.nan, ValueError),
        (True, TypeError),
        ("1.0", TypeError),
        (np.array([1.0]), TypeError),
    ],
)
def test_infer_well_locations_rejects_invalid_direct_arrival_windows(
    value: object,
    error: type[Exception],
):
    position, _ = _position_and_wells()
    well_sequence = np.array([[0.0, 1.0], [1.0, 2.0]])

    with pytest.raises(error, match="well_arrival_window_s"):
        infer_well_locations_from_arrays(
            position,
            well_sequence,
            well_arrival_window_s=value,
        )


def test_direct_ground_truth_helpers_accept_numeric_scalar_wrappers():
    position, wells = _position_and_wells()

    visit = first_post_ripple_well_visit(
        position,
        wells,
        ripple_peak=np.array(0.0),
        visit_radius_cm=np.float32(1.0),
        min_dwell_s=np.float64(0.1),
        future_horizon_s=np.array(1.0),
    )
    inferred = infer_well_locations_from_arrays(
        position,
        np.array([[0.0, 1.0], [1.0, 2.0]]),
        well_arrival_window_s=np.float32(0.5),
    )

    assert visit is not None
    assert visit["well_id"] == 1
    assert len(inferred) == 1
