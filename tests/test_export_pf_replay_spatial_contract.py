from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.data import ReplaySession
from scripts.export_pf_replay_spatial_contract import (
    EventSelectionConfig,
    ExportedEvent,
    _pack_events,
    _require_clean_commit,
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
