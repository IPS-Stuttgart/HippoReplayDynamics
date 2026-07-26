from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.ground_truth import (
    active_goal_at_time,
    first_post_ripple_well_visit,
    infer_well_locations_from_arrays,
)


def _position() -> np.ndarray:
    times = np.linspace(0.0, 1.0, 11)
    return np.column_stack(
        [times, np.zeros_like(times), np.zeros_like(times), np.zeros_like(times)]
    )


def _session(well_sequence: np.ndarray) -> ReplaySession:
    return ReplaySession(
        rat="Rat1",
        name="Open1",
        path=Path("unused"),
        position=_position(),
        spikes=np.empty((0, 2), dtype=float),
        tetrode_cell_ids=np.empty((0, 2), dtype=float),
        excitatory_neurons=np.array([], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.array([[0.0, 1.0]], dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=well_sequence,
        metadata={},
    )


@pytest.mark.parametrize("invalid_id", [1.5, np.nan, np.inf, True, "2.5"])
def test_infer_well_locations_rejects_invalid_well_ids(invalid_id: object) -> None:
    sequence = np.array([[0.0, invalid_id], [1.0, 2]], dtype=object)

    with pytest.raises(ValueError, match="well IDs"):
        infer_well_locations_from_arrays(
            _position(),
            sequence,
            well_arrival_window_s=0.5,
        )


@pytest.mark.parametrize("invalid_id", [1.5, np.nan, np.inf, True, "2.5"])
def test_active_goal_rejects_invalid_well_ids(invalid_id: object) -> None:
    sequence = np.array([[0.0, invalid_id], [1.0, 2]], dtype=object)

    with pytest.raises(ValueError, match="well IDs"):
        active_goal_at_time(_session(sequence), 0.5)


@pytest.mark.parametrize("invalid_id", [1.5, np.nan, np.inf, True, "2.5"])
def test_first_visit_rejects_invalid_well_ids(invalid_id: object) -> None:
    wells = pd.DataFrame(
        {
            "well_id": pd.Series([invalid_id], dtype=object),
            "well_x": [0.0],
            "well_y": [0.0],
            "n_estimates": [1],
        }
    )

    with pytest.raises(ValueError, match="well IDs"):
        first_post_ripple_well_visit(
            _position(),
            wells,
            ripple_peak=0.0,
            visit_radius_cm=1.0,
            min_dwell_s=0.1,
            future_horizon_s=1.0,
        )


def test_ground_truth_helpers_accept_integral_well_id_wrappers() -> None:
    sequence = np.array(
        [[0.0, np.float32(1.0)], [1.0, np.int64(2)]],
        dtype=object,
    )
    wells = pd.DataFrame(
        {
            "well_id": pd.Series([np.float32(1.0)], dtype=object),
            "well_x": [0.0],
            "well_y": [0.0],
            "n_estimates": [1],
        }
    )

    inferred = infer_well_locations_from_arrays(
        _position(),
        sequence,
        well_arrival_window_s=0.5,
    )
    goal = active_goal_at_time(_session(sequence), 0.5)
    visit = first_post_ripple_well_visit(
        _position(),
        wells,
        ripple_peak=0.0,
        visit_radius_cm=1.0,
        min_dwell_s=0.1,
        future_horizon_s=1.0,
    )

    assert inferred["well_id"].tolist() == [1]
    assert goal == 1
    assert visit is not None
    assert visit["well_id"] == 1
