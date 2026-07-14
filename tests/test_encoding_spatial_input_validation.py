from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm.data import ReplaySession
from hipporeplayimm.encoding import EncodingConfig, EncodingModel, fit_place_field_encoding


def _encoding_model() -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([0.0, 2.0]),
        y_edges=np.array([0.0, 2.0]),
        bin_centers=np.array([[1.0, 1.0]], dtype=float),
        rates_hz=np.empty((0, 1), dtype=float),
        occupancy_s=np.array([1.0], dtype=float),
        cell_ids=np.array([], dtype=int),
        config=EncodingConfig(),
    )


@pytest.mark.parametrize(
    "xy",
    [
        np.array([[True, False]], dtype=bool),
        np.array([["1.0", "1.0"]]),
        np.array([[1.0 + 2.0j, 1.0 + 0.0j]], dtype=complex),
    ],
)
def test_positions_to_flat_bins_rejects_lossy_coordinate_coercion(xy) -> None:
    with pytest.raises(TypeError, match="xy must be numeric"):
        _encoding_model().positions_to_flat_bins(xy)


def _session(tmp_path, position: np.ndarray) -> ReplaySession:
    return ReplaySession(
        rat="RatX",
        name="OpenX",
        path=tmp_path,
        position=position,
        spikes=np.empty((0, 2), dtype=float),
        tetrode_cell_ids=np.empty((0, 2), dtype=int),
        excitatory_neurons=np.array([], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.array([[0.0, 1.0]], dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
    )


@pytest.mark.parametrize(
    "position",
    [
        np.array([[False, False, False], [True, True, False]], dtype=bool),
        np.array([["0.0", "0.0", "0.0"], ["1.0", "1.0", "0.0"]]),
        np.array(
            [[0.0 + 0.0j, 0.0 + 1.0j, 0.0], [1.0 + 0.0j, 1.0, 0.0]],
            dtype=complex,
        ),
    ],
)
def test_place_field_fit_rejects_lossy_position_coercion(tmp_path, position) -> None:
    with pytest.raises(TypeError, match="position must be numeric"):
        fit_place_field_encoding(
            _session(tmp_path, position),
            EncodingConfig(min_speed_cm_s=0.0),
        )


def test_spatial_validation_patch_refreshes_stale_true_flag(monkeypatch) -> None:
    import hipporeplayimm.encoding as encoding
    import hipporeplayimm.encoding_grid_extra_columns as patch

    def stale_as_xy_array(xy, *, name="xy"):
        return np.asarray(xy, dtype=float)

    def stale_as_position_array(position):
        return np.asarray(position, dtype=float)

    monkeypatch.setattr(encoding, "_as_xy_array", stale_as_xy_array)
    monkeypatch.setattr(encoding, "_as_position_array", stale_as_position_array)
    monkeypatch.setattr(
        encoding,
        "_encoding_bool_validation_patch_applied",
        True,
        raising=False,
    )

    hipporeplayimm.apply_runtime_patches()

    assert getattr(encoding._as_xy_array, patch._AS_XY_ARRAY_WRAPPER_MARKER, False)
    assert getattr(
        encoding._as_position_array,
        patch._AS_POSITION_ARRAY_WRAPPER_MARKER,
        False,
    )
    with pytest.raises(TypeError, match="xy must be numeric"):
        encoding._as_xy_array(np.array([["1.0", "1.0"]]))
    with pytest.raises(TypeError, match="position must be numeric"):
        encoding._as_position_array(np.array([[True, False, False]]))
