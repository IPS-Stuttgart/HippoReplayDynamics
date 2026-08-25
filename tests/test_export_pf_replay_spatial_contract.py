from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.data import ReplaySession, RippleEvent
from hipporeplayimm.encoding import (
    EmissionConfig,
    EncodingConfig,
    build_emissions,
    emission_bin_schedule,
    fit_place_field_encoding,
)
from scripts.export_pf_replay_spatial_contract import (
    EventSelectionConfig,
    ExportedEvent,
    PointSpreadConfig,
    _pack_events,
    _require_clean_commit,
    estimate_prefix_decoder_point_spread_cm,
    select_lfp_only_events,
    verify_dataset_tree,
    verify_route_artifacts,
)


def _selection_session() -> ReplaySession:
    times = np.linspace(0.0, 10.0, 101)
    return ReplaySession(
        rat="RatX",
        name="OpenX",
        path=Path("."),
        position=np.column_stack(
            [times, times, np.zeros_like(times), np.zeros_like(times)]
        ),
        spikes=np.column_stack(
            [np.linspace(0.5, 9.5, 20), np.ones(20)]
        ),
        tetrode_cell_ids=np.array([[1, 1]]),
        excitatory_neurons=np.array([1]),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.array(
            [
                [4.0, 4.1, 4.05, 1.0, 0.0, 100.0],
                [5.0, 5.1, 5.05, 4.0, 0.0, -100.0],
                [6.0, 6.1, 6.05, 3.0, 0.0, -200.0],
                [7.0, 7.1, 7.05, 2.0, 0.0, 200.0],
            ]
        ),
        run_times=np.array([[0.0, 10.0]]),
        sleep_box_immobile_times=np.empty((0, 2)),
        sleep_times=np.empty((0, 2)),
        rem_times=np.empty((0, 2)),
        well_sequence=None,
        metadata={},
    )


def test_event_selection_uses_raw_lfp_power_not_decoder_or_z_score() -> None:
    routes = pd.DataFrame(
        {
            "movement_end_time_s": np.linspace(0.5, 3.0, 6),
        }
    )
    selected = select_lfp_only_events(
        _selection_session(),
        routes,
        EventSelectionConfig(
            events_per_session=2,
            minimum_training_duration_s=1.0,
            minimum_completed_routes=2,
        ),
    )

    assert [row["event_index"] for row in selected] == [1, 2]
    assert [row["selection_rank"] for row in selected] == [1, 2]
    assert [row["selection_metric"] for row in selected] == [4.0, 3.0]


def test_bounded_selected_emissions_match_dense_source_rows() -> None:
    session = _selection_session()
    encoding = fit_place_field_encoding(
        session,
        EncodingConfig(min_speed_cm_s=0.0),
        training_end_s=1.9,
    )
    holdout = RippleEvent(
        start=2.0,
        end=3.7,
        peak=2.85,
        raw_power=0.0,
        z_power_session=0.0,
        z_power_epoch=0.0,
    )
    config = EmissionConfig(time_bin_s=0.1)
    dense = build_emissions(session, encoding, holdout, config)
    _, source_times, _ = emission_bin_schedule(
        holdout.start,
        holdout.end,
        config.time_bin_s,
    )
    mask = np.arange(source_times.shape[0]) % 3 != 1
    bounded = build_emissions(
        session,
        encoding,
        holdout,
        config,
        time_bin_mask=mask,
        likelihood_time_chunk_size=2,
    )

    np.testing.assert_array_equal(bounded.times, dense.times[mask])
    np.testing.assert_array_equal(
        bounded.spike_counts,
        dense.spike_counts[mask],
    )
    np.testing.assert_allclose(
        bounded.log_likelihood,
        dense.log_likelihood[mask],
        rtol=1e-13,
        atol=1e-13,
    )
    assert bounded.metadata["emission_source_time_bins"] == dense.n_time
    assert bounded.metadata["emission_selected_time_bins"] == int(mask.sum())


def test_import_installed_poisson_wrapper_forwards_chunk_size() -> None:
    from hipporeplayimm import apply_runtime_patches
    from hipporeplayimm import encoding as encoding_module

    apply_runtime_patches()
    counts = np.array([[0, 1], [2, 0], [1, 1]], dtype=int)
    rates = np.array([[0.5, 1.5, 2.5], [1.0, 2.0, 3.0]])
    durations = np.array([0.1, 0.2, 0.3])
    dense = encoding_module._poisson_log_emissions(
        counts,
        rates,
        durations,
    )
    bounded = encoding_module._poisson_log_emissions(
        counts,
        rates,
        durations,
        time_chunk_size=1,
    )

    np.testing.assert_allclose(bounded, dense, rtol=1e-13, atol=1e-13)


def test_point_spread_defaults_to_fixed_120_second_holdout() -> None:
    config = PointSpreadConfig()

    assert config.holdout_window_s == 120.0
    assert config.likelihood_time_chunk_size == 32


def test_point_spread_failure_reports_event_and_support() -> None:
    context = "RatX/OpenX:event-7 selection_rank=3"
    with pytest.raises(
        ValueError,
        match=r"RatX/OpenX:event-7.*0 valid held-out RUN bins.*minimum is 2",
    ):
        estimate_prefix_decoder_point_spread_cm(
            _selection_session(),
            event_start_s=5.0,
            encoding_config=EncodingConfig(min_speed_cm_s=1e6),
            emission_config=EmissionConfig(),
            config=PointSpreadConfig(
                holdout_window_s=1.0,
                minimum_valid_bins=2,
            ),
            event_context=context,
        )


def _event(event_id: str, n_time: int, n_bins: int, well: str) -> ExportedEvent:
    return ExportedEvent(
        event_id=event_id,
        rat="RatX",
        session="RatX/OpenX",
        event_index=int(event_id[-1]),
        event_start_s=5.0,
        event_end_s=5.1,
        history_cutoff_s=4.9,
        decoder_training_cutoff_s=np.nextafter(5.0, -np.inf),
        field_available_s=np.full(7, 4.9),
        log_emissions=np.zeros((n_time, n_bins)),
        log_emission_offsets=np.arange(n_time, dtype=float),
        spatial_coordinates=np.column_stack(
            [np.arange(n_bins, dtype=float), np.zeros(n_bins)]
        ),
        nuisance_base=np.full(n_bins, 1.0 / n_bins),
        candidate_fields=np.ones((7, n_bins)),
        candidate_available=np.ones(7, dtype=bool),
        decoder_point_spread_cm=7.0,
        well_masses={well: 1.0},
        audit={},
    )


def test_packer_uses_nan_coordinates_and_log_zero_for_padding() -> None:
    arrays = _pack_events(
        [
            _event("event-0", 2, 3, "well-a"),
            _event("event-1", 1, 2, "well-b"),
        ]
    )

    assert arrays["log_emissions"].shape == (2, 2, 3)
    assert np.all(np.isneginf(arrays["log_emissions"][1, 1]))
    assert not arrays["time_mask"][1, 1]
    assert not arrays["active_spatial_mask"][1, 2]
    assert np.isnan(arrays["spatial_coordinates"][1, 2]).all()
    np.testing.assert_allclose(arrays["well_masses"].sum(axis=1), 1.0)


def test_dataset_tree_is_fully_verified_and_tampering_fails(tmp_path) -> None:
    root = tmp_path / "pfeiffer-foster"
    source = root / "Rat1" / "Open1" / "Position_Data.csv"
    source.parent.mkdir(parents=True)
    source.write_text("time,x,y\n0,1,2\n", encoding="utf-8")
    relative = source.relative_to(root).as_posix()
    record = {
        "path": relative,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "size_bytes": source.stat().st_size,
    }
    payload = {
        "dataset_root_name": root.name,
        "files": [record],
        "generated_at_utc": "2026-01-01T00:00:00+00:00",
        "session_count": 1,
        "sessions": [
            {
                "session": "Rat1/Open1",
                "missing_required_files": [],
                "required_files": [record],
                "optional_files": [],
            }
        ],
        "total_bytes": source.stat().st_size,
    }
    canonical = (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    path = root / "dataset_manifest.json"
    path.write_text(
        json.dumps(
            {
                **payload,
                "manifest_sha256_without_this_field": digest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_sha, report = verify_dataset_tree(root, path, digest)
    assert len(manifest_sha) == 64
    assert report["status"] == "pass"
    assert report["verified_file_count"] == 1
    assert report["verified_total_bytes"] == source.stat().st_size

    source.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match lock"):
        verify_dataset_tree(root, path, digest)


def test_dataset_tree_rejects_unlocked_extra_file(tmp_path) -> None:
    root = tmp_path / "pfeiffer-foster"
    root.mkdir()
    locked = root / "locked.bin"
    locked.write_bytes(b"locked")
    record = {
        "path": "locked.bin",
        "sha256": hashlib.sha256(locked.read_bytes()).hexdigest(),
        "size_bytes": locked.stat().st_size,
    }
    payload = {
        "dataset_root_name": root.name,
        "files": [record],
        "generated_at_utc": "2026-01-01T00:00:00+00:00",
        "session_count": 0,
        "sessions": [],
        "total_bytes": locked.stat().st_size,
    }
    canonical = (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    path = root / "dataset_manifest.json"
    path.write_text(
        json.dumps(
            {
                **payload,
                "manifest_sha256_without_this_field": digest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "unexpected.bin").write_bytes(b"x")

    with pytest.raises(ValueError, match="path mismatch"):
        verify_dataset_tree(root, path, digest)

def test_route_artifacts_require_clean_cutoff_safe_manifest(tmp_path) -> None:
    segments = tmp_path / "replay_behavior_route_segments.csv"
    points = tmp_path / "replay_behavior_route_segment_points.csv"
    segments.write_text("route_id\nr0\n", encoding="utf-8")
    points.write_text("route_id,x_cm,y_cm\nr0,0,0\n", encoding="utf-8")
    parameters = {"median_window_s": 0.167, "gaussian_sigma_s": 0.1}
    manifest = {
        "analysis": "replay_behavior_route_primitives",
        "producer_commit": "a" * 40,
        "producer_clean_worktree": True,
        "route_smoothing_scope": "within_completed_fill_interval",
        "parameters": parameters,
        "parameters_sha256": hashlib.sha256(
            json.dumps(
                parameters,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "output_sha256": {
            segments.name: hashlib.sha256(segments.read_bytes()).hexdigest(),
            points.name: hashlib.sha256(points.read_bytes()).hexdigest(),
        },
    }
    manifest_path = tmp_path / "route_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    verified = verify_route_artifacts(manifest_path, segments, points)
    assert verified["route_producer_commit"] == "a" * 40

    manifest["producer_clean_worktree"] = False
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="clean committed producer"):
        verify_route_artifacts(manifest_path, segments, points)


def test_claim_bearing_export_requires_clean_exact_commit() -> None:
    commit = "1" * 40
    assert _require_clean_commit(
        {"code_commit": commit, "git_dirty": False}
    ) == commit
    with pytest.raises(ValueError, match="clean committed worktree"):
        _require_clean_commit({"code_commit": commit, "git_dirty": True})
    with pytest.raises(ValueError, match="committed Git checkout"):
        _require_clean_commit({"code_commit": "unavailable", "git_dirty": False})
