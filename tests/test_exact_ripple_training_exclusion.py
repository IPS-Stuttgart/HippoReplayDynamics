from pathlib import Path
import sys
from types import ModuleType

import numpy as np

from hipporeplayimm.clusterless import (
    ClusterlessMarkConfig,
    fit_clusterless_mark_encoding,
)
from hipporeplayimm.data import ReplaySession, SpikeMarkData
from hipporeplayimm.encoding import EncodingConfig, fit_place_field_encoding
from hipporeplayimm.exact_ripple_training_exclusion import (
    _synchronize_aliases,
    retained_frame_durations,
)


_RATE_FLOOR = 1e-4


def _encoding_config() -> EncodingConfig:
    return EncodingConfig(
        bin_size_cm=10.0,
        smoothing_sigma_bins=0.0,
        min_speed_cm_s=0.0,
        min_occupancy_s=1e-6,
        rate_floor_hz=_RATE_FLOOR,
        arena_padding_cm=0.0,
        use_excitatory=True,
        exclude_ripple_intervals=True,
    )


def _session() -> ReplaySession:
    event_times = np.array([0.75, 1.5], dtype=float)
    return ReplaySession(
        rat="RatX",
        name="OpenX",
        path=Path("RatX/OpenX"),
        position=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 10.0, 0.0],
                [2.0, 20.0, 0.0],
            ],
            dtype=float,
        ),
        spikes=np.column_stack(
            [
                event_times,
                np.ones(event_times.shape, dtype=float),
            ]
        ),
        tetrode_cell_ids=np.array([[1.0, 1.0]], dtype=float),
        excitatory_neurons=np.array([1], dtype=int),
        inhibitory_neurons=np.empty(0, dtype=int),
        ripple_events=np.array(
            [[0.5, 1.5, 1.0, 0.0, 0.0, 0.0]],
            dtype=float,
        ),
        run_times=np.array([[0.0, 1.9]], dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
        spike_marks=SpikeMarkData(
            times=event_times,
            marks=np.array([[10.0], [20.0]], dtype=float),
            source_file="marks.mat",
            source_variable="marks",
            feature_names=("amplitude",),
            cell_ids=np.array([1, 1], dtype=int),
        ),
    )


def test_sorted_spike_encoding_subtracts_partial_ripple_frames() -> None:
    encoding = fit_place_field_encoding(_session(), _encoding_config())

    np.testing.assert_allclose(
        encoding.occupancy_s,
        np.array([0.5, 0.5]),
    )
    np.testing.assert_allclose(
        encoding.rates_hz[0],
        np.array([_RATE_FLOOR, 2.0]),
    )


def test_clusterless_encoding_uses_the_same_half_open_exclusion() -> None:
    encoding = fit_clusterless_mark_encoding(
        _session(),
        ClusterlessMarkConfig(
            encoding=_encoding_config(),
            mark_smoothing_sigma_bins=0.0,
            mark_likelihood="diagonal-gaussian",
        ),
    )

    np.testing.assert_allclose(
        encoding.occupancy_s,
        np.array([0.5, 0.5]),
    )
    np.testing.assert_allclose(
        encoding.effective_spike_count,
        np.array([0.0, 1.0]),
    )
    np.testing.assert_allclose(
        encoding.rate_hz,
        np.array([_RATE_FLOOR, 2.0]),
    )


def test_overlapping_exclusions_are_subtracted_as_a_union() -> None:
    retained = retained_frame_durations(
        np.array([0.0]),
        np.array([2.0]),
        np.array([[0.25, 1.25], [0.75, 1.5]]),
    )

    np.testing.assert_allclose(retained, np.array([0.75]))


def test_exact_ripple_alias_sync_is_limited_to_package_namespace(monkeypatch) -> None:
    external = ModuleType("hipporeplayimm_extension")
    package_probe = ModuleType("hipporeplayimm._exact_ripple_alias_probe")

    def previous():
        return None

    def replacement():
        return None

    external._frame_durations = previous
    package_probe._frame_durations = previous
    monkeypatch.setitem(sys.modules, external.__name__, external)
    monkeypatch.setitem(sys.modules, package_probe.__name__, package_probe)

    _synchronize_aliases("_frame_durations", previous, replacement)

    assert external._frame_durations is previous
    assert package_probe._frame_durations is replacement
