from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from scripts.analyze_tanni2022_wall_distance_replay import association_summary, wall_quartile_summary
from scripts.score_tanni2022_wall_distance_model_subset import evidence_decisions, select_balanced_model_subset

from hipporeplayimm.tanni2022 import (
    detect_ripple_candidates,
    local_poisson_code_gradient,
    nearest_wall_distance,
    posterior_from_log_likelihood,
    posterior_path_segments,
    read_tanni_position,
    read_tanni_session_metadata,
    read_tanni_sorted_spikes,
    ripple_envelope_robust_z,
)


def _write_minimal_nwb(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        settings = handle.create_group("general/data_collection/Settings/General")
        settings.create_dataset("animal", data=np.bytes_("RTEST"))
        settings.create_dataset("arena_size", data=np.array([100.0, 80.0]))
        recording = handle.create_group("acquisition/timeseries/recording1")
        tracking = recording.create_group("tracking")
        times = np.arange(10, dtype=float) / 10.0
        xy = np.column_stack((10.0 + times * 10.0, 20.0 + times * 5.0))
        tracking.create_dataset("ProcessedPos", data=np.column_stack((times, xy, np.full((10, 2), np.nan))))
        continuous = recording.create_group("continuous/processor102_100")
        continuous.create_dataset("downsampled_tetrode_data", data=np.zeros((1500, 2), dtype=np.int16))
        continuous.create_dataset("downsampled_timestamps", data=np.arange(1500, dtype=float) / 1500.0)
        info = continuous.create_group("downsampling_info")
        info.create_dataset("downsampled_sampling_rate", data=1500)
        electrode = recording.create_group("spikes/electrode1")
        electrode.create_dataset("timestamps", data=np.array([0.1, 0.2, 0.3, 0.4]))
        electrode.create_dataset("idx_keep", data=np.array([True, False, True, True]))
        clustering = electrode.create_group("clustering")
        clustering.create_dataset("manual_1", data=np.array([2, 1, 3], dtype=np.int16))


def test_tanni_nwb_readers_preserve_clock_and_manual_clusters(tmp_path: Path) -> None:
    path = tmp_path / "experiment_1.nwb"
    _write_minimal_nwb(path)

    metadata = read_tanni_session_metadata(path)
    position = read_tanni_position(path)
    spikes = read_tanni_sorted_spikes(path)

    assert metadata.animal == "RTEST"
    np.testing.assert_allclose(metadata.arena_size_cm, [100.0, 80.0])
    assert metadata.lfp_sample_rate_hz == 1500.0
    np.testing.assert_allclose(position.times_s, np.arange(10) / 10.0)
    assert position.valid.all()
    np.testing.assert_allclose(spikes.spikes[:, 0], [0.1, 0.4])
    np.testing.assert_array_equal(spikes.spikes[:, 1].astype(int), [1002, 1003])


def test_ripple_detector_merges_short_gaps_and_applies_peak_gate() -> None:
    times = np.arange(2000, dtype=float) / 1000.0
    z = np.zeros(2000, dtype=float)
    z[500:530] = 4.0
    z[530:540] = 0.0
    z[540:570] = 4.0
    z[550] = 12.0
    z[1200:1210] = 20.0

    events = detect_ripple_candidates(
        times,
        z,
        threshold_z=3.0,
        peak_threshold_z=10.0,
        min_duration_s=0.015,
        max_duration_s=0.25,
        merge_gap_s=0.02,
    )

    assert len(events) == 1
    assert events[0].start_time_s == 0.5
    assert np.isclose(events[0].end_time_s, 0.57)
    assert events[0].peak_time_s == 0.55


def test_channelwise_ripple_envelope_has_peak_at_injected_burst() -> None:
    sample_rate = 1500.0
    times = np.arange(3000) / sample_rate
    rng = np.random.default_rng(7)
    signal = rng.normal(scale=0.2, size=times.shape[0])
    burst = (times >= 0.9) & (times < 0.98)
    signal[burst] += 8.0 * np.sin(2.0 * np.pi * 200.0 * times[burst])

    envelope_z = ripple_envelope_robust_z(signal, sample_rate_hz=sample_rate)

    assert 0.9 <= times[int(np.argmax(envelope_z))] <= 0.98
    assert float(np.max(envelope_z)) > 10.0


def test_posterior_segments_report_physical_and_code_speed() -> None:
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    posterior = np.eye(3)
    rates = np.array([[1.0, 4.0, 9.0], [9.0, 4.0, 1.0]])
    occupancy = np.ones(3)

    segments = posterior_path_segments(
        posterior,
        centers,
        rates,
        occupancy,
        np.array([0.0, 0.02, 0.04]),
        np.array([100.0, 80.0]),
    )

    np.testing.assert_allclose(segments["physical_speed_cm_s"], [500.0, 500.0])
    np.testing.assert_allclose(segments["map_speed_cm_s"], [500.0, 500.0])
    np.testing.assert_allclose(segments["posterior_rms_independent_speed_cm_s"], [500.0, 500.0])
    assert np.all(segments["code_speed_sqrt_hz_per_s"] > 0.0)
    np.testing.assert_allclose(segments["wall_distance_cm"], [0.0, 0.0])
    np.testing.assert_allclose(segments["posterior_spread_cm"], [0.0, 0.0])


def test_wall_distance_and_local_code_gradient_are_geometric() -> None:
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    rates = np.array([[1.0, 4.0, 9.0], [9.0, 4.0, 1.0]])

    wall = nearest_wall_distance(np.array([[1.0, 4.0], [50.0, 40.0], [99.0, 79.0]]), np.array([100.0, 80.0]))
    gradient = local_poisson_code_gradient(centers, rates, neighbor_radius_cm=11.0)

    np.testing.assert_allclose(wall, [1.0, 40.0, 1.0])
    assert np.all(np.isfinite(gradient))
    assert np.all(gradient > 0.0)


def test_posterior_normalization_is_rowwise() -> None:
    posterior = posterior_from_log_likelihood(np.array([[0.0, 0.0], [-100.0, 0.0]]))

    np.testing.assert_allclose(posterior.sum(axis=1), 1.0)
    np.testing.assert_allclose(posterior[0], [0.5, 0.5])
    assert posterior[1, 1] > 0.999


def test_multianimal_association_is_animal_balanced() -> None:
    rows = []
    decoder_rows = []
    for animal_index in range(5):
        animal = f"R{animal_index}"
        wall = np.linspace(0.01, 0.99, 40)
        for index, distance in enumerate(wall):
            rows.append(
                {
                    "animal": animal,
                    "session": "D",
                    "event_index": index // 4,
                    "segment_index": index,
                    "wall_distance_normalized": distance,
                    "physical_speed_cm_s": 100.0 + 20.0 * distance,
                    "map_speed_cm_s": 100.0 + 20.0 * distance,
                    "posterior_rms_independent_speed_cm_s": 120.0 + 20.0 * distance,
                    "code_speed_sqrt_hz_per_s": 3.0,
                    "posterior_entropy": 2.0 + 0.01 * index,
                    "posterior_spread_cm": 10.0 + 0.01 * index,
                    "local_code_gradient_sqrt_hz_per_cm": 0.2 + 0.001 * index,
                    "local_occupancy_s": 1.0 + 0.01 * index,
                    "event_n_spikes": 10 + index % 4,
                    "event_n_active_cells": 4 + index % 3,
                    "peak_ripple_z": 12.0 + index % 5,
                }
            )
            decoder_rows.append(
                {
                    "animal": animal,
                    "wall_distance_normalized": distance,
                    "decoder_error_cm": 8.0 + distance,
                }
            )
    segments = pd.DataFrame(rows)
    associations = association_summary(segments, pd.DataFrame(), bootstrap_replicates=100, seed=3)
    quartiles = wall_quartile_summary(segments, pd.DataFrame(decoder_rows))

    physical = associations.loc[
        (associations["metric"] == "physical_speed_cm_s") & (associations["scope"] == "animal_balanced")
    ].iloc[0]
    assert physical["raw_spearman_r"] > 0.99
    assert physical["animals"] == 5
    assert quartiles.loc[quartiles["aggregation_level"] == "animal_balanced_median"].shape[0] == 4


def test_model_subset_selection_is_wall_balanced_and_deterministic() -> None:
    events = pd.DataFrame(
        {
            "animal": np.repeat(["R1", "R2"], 40),
            "session": "D",
            "event_index": np.tile(np.arange(40), 2),
            "median_wall_distance_cm": np.tile(np.linspace(1.0, 124.0, 40), 2),
        }
    )

    first = select_balanced_model_subset(events, events_per_animal=20, seed=11)
    second = select_balanced_model_subset(events, events_per_animal=20, seed=11)

    pd.testing.assert_frame_equal(first, second)
    assert first.groupby(["animal", "wall_quartile"]).size().eq(5).all()


def test_model_evidence_decisions_separate_ordered_and_fragmented() -> None:
    values = {
        1: {"stationary": 0.0, "diffusion": 2.0, "fragmented": 1.0, "first-order-imm": 10.0},
        2: {"stationary": 0.0, "diffusion": 1.0, "fragmented": 12.0, "first-order-imm": 3.0},
    }
    rows = [
        {"animal": "R1", "session": "D", "event_index": event, "model": model, "log_evidence": score}
        for event, scores in values.items()
        for model, score in scores.items()
    ]

    decisions = evidence_decisions(pd.DataFrame(rows), claim_margin=5.5).set_index("event_index")

    assert bool(decisions.loc[1, "ordered_trajectory_confident"])
    assert bool(decisions.loc[1, "imm_confident_over_fragmented"])
    assert bool(decisions.loc[2, "fragmented_confident"])
