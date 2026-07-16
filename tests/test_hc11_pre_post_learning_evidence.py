from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import score_hc11_pre_post_learning_evidence as learning  # noqa: E402
import score_hc11_webshare_native_ripple_evidence as hc11  # noqa: E402


def test_ripple_catalog_reads_published_all_state_group(tmp_path: Path) -> None:
    path = tmp_path / "session.ripplesALL.event.mat"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("ripples")
        group.create_dataset("times", data=np.array([[1.0, 2.0], [1.1, 2.2]]))
        group.create_dataset("peaks", data=np.array([1.05, 2.1]))
        group.create_dataset("peakNormedPower", data=np.array([5.0, 6.0]))

    catalog = learning.load_ripple_catalog(path)
    assert len(catalog) == 2
    np.testing.assert_allclose(catalog["start_time_s"], [1.0, 2.0])
    np.testing.assert_allclose(catalog["end_time_s"], [1.1, 2.2])
    np.testing.assert_allclose(catalog["peak_ripple_power_z"], [5.0, 6.0])


def _events(phase: str, offset: int, count: int) -> pd.DataFrame:
    rows = []
    for index in range(count):
        rows.append(
            {
                "phase": phase,
                "event_id": offset + index,
                "start_time_s": float(index),
                "end_time_s": float(index) + 0.08 + 0.001 * index,
                "peak_time_s": float(index) + 0.04,
                "duration_ms": 80.0 + index,
                "peak_ripple_power_z": 3.0 + 0.1 * index,
                "n_spikes": 8 + index,
                "n_active_units": 4 + index % 4,
                "model_score_that_must_not_be_used": 1000.0 - index,
            }
        )
    return pd.DataFrame(rows)


def test_strength_matching_is_balanced_deterministic_and_pre_evidence() -> None:
    pre = _events("PRE", 0, 30)
    post = _events("POST", 100, 35)
    first = learning.match_pre_post_events(pre, post, max_pairs=8, seed=42, pool_multiplier=3)
    second = learning.match_pre_post_events(pre, post, max_pairs=8, seed=42, pool_multiplier=3)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 16
    assert first.groupby("phase").size().to_dict() == {"POST": 8, "PRE": 8}
    assert first.groupby("match_pair_id")["phase"].nunique().eq(2).all()
    assert first["selection_rule"].eq("pre_evidence_random_pool_strength_matched").all()

    changed = pre.copy()
    changed["model_score_that_must_not_be_used"] *= -100.0
    rerun = learning.match_pre_post_events(changed, post, max_pairs=8, seed=42, pool_multiplier=3)
    assert rerun["event_id"].tolist() == first["event_id"].tolist()


def test_high_information_pool_is_pre_evidence_and_enriches_active_units() -> None:
    pre = _events("PRE", 0, 40)
    post = _events("POST", 100, 40)
    selected = learning.match_pre_post_events(
        pre,
        post,
        max_pairs=5,
        seed=3,
        pool_multiplier=2,
        pool_strategy="high_information",
    )
    assert len(selected) == 10
    assert selected["selection_rule"].eq("pre_evidence_high_information_pool_strength_matched").all()
    assert selected["n_active_units"].median() >= pre["n_active_units"].median()


def test_subset_encoding_preserves_space_and_selects_requested_cells() -> None:
    encoding = hc11.EncodingMap(
        name="pooled",
        unit_ids=(10, 20, 30),
        bin_edges_cm=np.array([0.0, 1.0, 2.0]),
        bin_centers_cm=np.array([0.5, 1.5]),
        occupancy_s=np.array([2.0, 3.0]),
        prior=np.array([0.4, 0.6]),
        rates_hz=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    )
    subset = learning.subset_encoding_map(encoding, (30, 10))
    assert subset.unit_ids == (30, 10)
    np.testing.assert_array_equal(subset.rates_hz, [[5.0, 6.0], [1.0, 2.0]])
    np.testing.assert_array_equal(subset.prior, encoding.prior)


def _evidence_rows(phase: str, population: str, values: dict[str, float]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model, log_evidence in values.items():
        rows.append(
            {
                "animal": "RatA",
                "session": "RatA_day1",
                "geometry": "linear",
                "phase": phase,
                "match_pair_id": 1,
                "event_id": 1 if phase == "PRE" else 2,
                "population": population,
                "model": model,
                "log_evidence": log_evidence,
                "status": "success",
                "mean_nonstationary_mode_probability": 0.8 if model == "first_order_imm" else np.nan,
                "fraction_time_map_nonstationary": 0.75 if model == "first_order_imm" else np.nan,
                "posterior_expected_path_length_cm": 40.0 if model == "first_order_imm" else np.nan,
                "posterior_net_displacement_cm": 30.0 if model == "first_order_imm" else np.nan,
                "n_spikes": 20,
                "n_active_units": 8,
                "n_encoding_units": 20,
                "duration_ms": 100.0,
            }
        )
    return rows


def test_fragmented_is_nonordered_and_post_pre_contrast_is_paired() -> None:
    pre_values = {"stationary": 0.0, "diffusion": 1.0, "fragmented": 9.0, "first_order_imm": 4.0}
    post_values = {"stationary": 0.0, "diffusion": 2.0, "fragmented": 3.0, "first_order_imm": 12.0}
    evidence = pd.DataFrame(
        _evidence_rows("PRE", "all", pre_values)
        + _evidence_rows("POST", "all", post_values)
    )
    decisions = learning.event_decisions(evidence, margin_threshold=5.5)
    pre = decisions.set_index("phase").loc["PRE"]
    post = decisions.set_index("phase").loc["POST"]
    assert pre["best_nonordered_model"] == "fragmented"
    assert pre["ordered_minus_nonordered"] == -5.0
    assert not bool(pre["ordered_confident"])
    assert post["best_model"] == "first_order_imm"
    assert bool(post["ordered_confident"])
    assert bool(post["imm_trajectory_active_candidate"])

    contrasts = learning.learning_contrasts(decisions)
    assert len(contrasts) == 1
    assert contrasts.iloc[0]["post_minus_pre_ordered_minus_nonordered"] == 14.0
    assert contrasts.iloc[0]["post_minus_pre_imm_trajectory_active_candidate"] == 1.0


def test_zero_event_gates_fail_non_vacuously() -> None:
    gates = learning.gate_summary([], pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 20)
    indexed = gates.set_index("gate")
    assert not bool(indexed.loc["ripple_event_sessions_present", "passed"])
    assert not bool(indexed.loc["required_models_complete", "passed"])
    assert not bool(indexed.loc["overall_technical", "passed"])
    assert not bool(indexed.loc["learning_dependent_trajectory_dynamics_supported", "passed"])


def test_offline_rate_groups_reject_unknown_scope(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(learning, "nrem_intervals", lambda _: np.array([[0.0, 10.0]]))
    monkeypatch.setattr(
        learning,
        "phase_intervals",
        lambda _: {"PRE": np.array([[0.0, 4.0]]), "POST": np.array([[6.0, 10.0]])},
    )
    spikes = hc11.SpikeData((1, 2), {1: np.array([1.0]), 2: np.array([2.0])})
    try:
        learning.offline_firing_rate_groups(tmp_path, spikes, spikes.unit_ids, scope="future")
    except ValueError as exc:
        assert "overall_session" in str(exc)
    else:
        raise AssertionError("unknown rate grouping scope should fail")


def test_generated_ripple_manifest_requires_explicit_qc_and_preserves_source(
    tmp_path: Path,
) -> None:
    root = tmp_path / "processed"
    session_dir = root / "RatA" / "RatA_day1"
    session_dir.mkdir(parents=True)
    event_path = tmp_path / "generated" / "RatA_day1.ripplesNREM.event.mat"
    event_path.parent.mkdir()
    event_path.touch()
    manifest = tmp_path / "events.csv"
    pd.DataFrame(
        [
            {
                "animal": "RatA",
                "session": "RatA_day1",
                "ripple_event_path": event_path,
                "event_source": "lfp_detected_validated",
                "detector_qc_passed": True,
            }
        ]
    ).to_csv(manifest, index=False)

    sessions = learning.resolve_ripple_event_sessions(root, manifest)
    assert len(sessions) == 1
    assert sessions[0].session_dir == session_dir.resolve()
    assert sessions[0].ripple_event_path == event_path.resolve()
    assert sessions[0].event_source == "lfp_detected_validated"

    table = pd.read_csv(manifest)
    table["detector_qc_passed"] = False
    table.to_csv(manifest, index=False)
    try:
        learning.resolve_ripple_event_sessions(root, manifest)
    except ValueError as exc:
        assert "detector_qc_passed is false" in str(exc)
    else:
        raise AssertionError("unvalidated generated ripple events should be rejected")


def test_paper_population_synchrony_events_require_mua_ripple_and_both_phases(
    monkeypatch,
    tmp_path: Path,
) -> None:
    unit_ids = (1, 2, 3, 4, 5)
    # PRE has one ripple-aligned burst and one equally strong burst without a ripple.
    # POST has one ripple-aligned burst. The non-ripple burst must not become an event.
    centers = (1.0, 2.5, 7.0)
    times_by_unit = {
        unit_id: np.sort(
            np.concatenate(
                [center + np.array([-0.010, 0.0, 0.010]) + unit_id * 0.0001 for center in centers]
            )
        )
        for unit_id in unit_ids
    }
    spikes = hc11.SpikeData(unit_ids, times_by_unit)
    monkeypatch.setattr(learning, "pyramidal_unit_ids", lambda *_: unit_ids)
    monkeypatch.setattr(
        learning,
        "phase_intervals",
        lambda _: {"PRE": np.array([[0.0, 4.0]]), "POST": np.array([[6.0, 10.0]])},
    )
    monkeypatch.setattr(
        learning,
        "nrem_intervals",
        lambda _: np.array([[0.0, 4.0], [6.0, 10.0]]),
    )
    monkeypatch.setattr(
        learning,
        "load_ripple_catalog",
        lambda _: pd.DataFrame(
            {
                "lfp_ripple_id": [0, 1],
                "start_time_s": [0.98, 6.98],
                "end_time_s": [1.02, 7.02],
                "peak_time_s": [1.0, 7.0],
                "peak_ripple_power_z": [6.0, 7.0],
            }
        ),
    )

    events = learning.detect_population_synchrony_events(
        tmp_path,
        spikes,
        unit_ids,
        tmp_path / "ripples.mat",
        min_event_spikes=5,
        min_event_active_units=5,
        mua_bin_s=0.001,
        mua_smoothing_s=0.015,
        mua_threshold_sd=3.0,
        min_duration_s=0.05,
        max_duration_s=0.50,
    )

    assert events.groupby("phase").size().to_dict() == {"POST": 1, "PRE": 1}
    assert events["lfp_ripple_count"].eq(1).all()
    assert events["population_n_active_units"].eq(5).all()
    assert events["n_active_units"].eq(5).all()
    assert events["peak_mua_z"].gt(3.0).all()
    assert events["event_definition"].eq(
        "paper_population_synchrony_with_lfp_ripple"
    ).all()
    assert not events["start_time_s"].between(2.0, 3.0).any()


def test_paper_population_synchrony_empty_result_preserves_schema(
    monkeypatch,
    tmp_path: Path,
) -> None:
    unit_ids = (1, 2, 3, 4, 5)
    spikes = hc11.SpikeData(unit_ids, {unit_id: np.array([1.0]) for unit_id in unit_ids})
    monkeypatch.setattr(learning, "pyramidal_unit_ids", lambda *_: unit_ids)
    monkeypatch.setattr(
        learning,
        "phase_intervals",
        lambda _: {"PRE": np.array([[0.0, 2.0]]), "POST": np.array([[3.0, 5.0]])},
    )
    monkeypatch.setattr(
        learning,
        "nrem_intervals",
        lambda _: np.array([[0.0, 2.0], [3.0, 5.0]]),
    )
    monkeypatch.setattr(
        learning,
        "load_ripple_catalog",
        lambda _: pd.DataFrame(
            columns=("lfp_ripple_id", "start_time_s", "end_time_s", "peak_time_s", "peak_ripple_power_z")
        ),
    )

    events = learning.detect_population_synchrony_events(
        tmp_path,
        spikes,
        unit_ids,
        tmp_path / "ripples.mat",
        min_event_spikes=5,
        min_event_active_units=5,
        mua_bin_s=0.001,
        mua_smoothing_s=0.015,
        mua_threshold_sd=3.0,
        min_duration_s=0.05,
        max_duration_s=0.50,
    )

    assert events.empty
    assert tuple(events.columns) == learning.EVENT_COLUMNS
