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
    verify_dataset_manifest,
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


def test_dataset_manifest_digest_is_verified_and_tampering_fails(tmp_path) -> None:
    payload = {"sessions": ["Rat1/Open1"], "total_bytes": 123}
    canonical = (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    path = tmp_path / "dataset_manifest.json"
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

    assert len(verify_dataset_manifest(path, digest)) == 64
    path.write_text(
        path.read_text(encoding="utf-8").replace("123", "124"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical digest"):
        verify_dataset_manifest(path, digest)


def test_claim_bearing_export_requires_clean_exact_commit() -> None:
    commit = "1" * 40
    assert _require_clean_commit(
        {"code_commit": commit, "git_dirty": False}
    ) == commit
    with pytest.raises(ValueError, match="clean committed worktree"):
        _require_clean_commit({"code_commit": commit, "git_dirty": True})
    with pytest.raises(ValueError, match="committed Git checkout"):
        _require_clean_commit({"code_commit": "unavailable", "git_dirty": False})
