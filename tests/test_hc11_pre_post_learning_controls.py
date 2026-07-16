from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_hc11_pre_post_learning_controls as controls  # noqa: E402
import score_hc11_webshare_native_ripple_evidence as hc11  # noqa: E402


def _encoding(name: str, rates: np.ndarray) -> hc11.EncodingMap:
    n_bins = rates.shape[1]
    return hc11.EncodingMap(
        name=name,
        unit_ids=tuple(range(1, rates.shape[0] + 1)),
        bin_edges_cm=np.arange(n_bins + 1, dtype=float) * 4.0,
        bin_centers_cm=(np.arange(n_bins, dtype=float) + 0.5) * 4.0,
        occupancy_s=np.ones(n_bins),
        prior=np.full(n_bins, 1.0 / n_bins),
        rates_hz=np.asarray(rates, dtype=float),
    )


def test_map_permutation_is_shared_across_direction_maps() -> None:
    first = _encoding("negative", np.array([[1, 2], [3, 4], [5, 6]], dtype=float))
    second = _encoding("positive", np.array([[10, 20], [30, 40], [50, 60]], dtype=float))
    permuted = controls.permute_encoding_maps([first, second], np.array([2, 0, 1]))
    np.testing.assert_array_equal(permuted[0].rates_hz, [[5, 6], [1, 2], [3, 4]])
    np.testing.assert_array_equal(permuted[1].rates_hz, [[50, 60], [10, 20], [30, 40]])
    assert permuted[0].unit_ids == first.unit_ids


def test_whole_bin_shuffle_preserves_population_vectors_and_durations() -> None:
    counts = np.array([[1, 0], [0, 2], [3, 1]])
    edges = np.array([4.0, 4.01, 4.03, 4.06])
    shuffled, shuffled_edges = controls.shuffled_event(counts, edges, np.array([2, 0, 1]))
    np.testing.assert_array_equal(shuffled, counts[[2, 0, 1]])
    np.testing.assert_allclose(np.diff(shuffled_edges), np.diff(edges)[[2, 0, 1]])
    assert shuffled.sum() == counts.sum()


def test_train_only_posterior_predictive_diagnostics_do_not_depend_on_test_spikes() -> None:
    train_maps = [
        _encoding("negative", np.array([[8, 2, 1, 1], [1, 2, 8, 1]], dtype=float)),
        _encoding("positive", np.array([[1, 8, 2, 1], [1, 1, 2, 8]], dtype=float)),
    ]
    test_maps = [
        _encoding("negative", np.array([[6, 1, 1, 1], [1, 1, 6, 1]], dtype=float)),
        _encoding("positive", np.array([[1, 6, 1, 1], [1, 1, 1, 6]], dtype=float)),
    ]
    train_counts = np.array([[1, 0], [2, 0], [0, 1], [0, 2]])
    test_counts = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    edges = np.arange(5, dtype=float) * 0.01
    kwargs = {
        "topology": "linear",
        "track_length_cm": 16.0,
        "diffusion_sigma_cm_sqrt_s": 40.0,
        "stationary_sigma_cm": 2.0,
        "max_step_sigma": 4.0,
        "imm_mode_stickiness": 0.9,
    }
    first_scores, first_diagnostics = controls.posterior_marginal_predictive_scores(
        train_counts,
        test_counts,
        edges,
        train_maps,
        test_maps,
        model_kwargs=kwargs,
    )
    second_scores, second_diagnostics = controls.posterior_marginal_predictive_scores(
        train_counts,
        test_counts * 2,
        edges,
        train_maps,
        test_maps,
        model_kwargs=kwargs,
    )
    assert set(first_scores) == set(hc11.MODELS)
    assert all(np.isfinite(value) for value in first_scores.values())
    assert first_scores != second_scores
    assert first_diagnostics == second_diagnostics


def _control_row(control_type: str, replicate: int, scale: float) -> dict[str, object]:
    return {
        "animal": "RatA",
        "session": "RatA_day1",
        "geometry": "linear",
        "phase": "POST",
        "match_pair_id": 1,
        "event_id": 10,
        "population": "all",
        "control_type": control_type,
        "replicate": replicate,
        "status": "success",
        "best_model": "first_order_imm",
        "posterior_content_positive": True,
        "ordered_minus_nonordered": 12.0 * scale,
        "imm_minus_fragmented": 10.0 * scale,
        "mean_nonstationary_mode_probability": 0.8 * scale,
        "posterior_expected_path_length_cm": 50.0 * scale,
        "posterior_net_displacement_cm": 30.0 * scale,
    }


def test_event_summary_requires_map_order_and_heldout_support() -> None:
    rows = [_control_row("original", 0, 1.0)]
    rows.extend(_control_row("map_permutation", index, 0.2) for index in range(3))
    rows.extend(_control_row("time_shuffle", index, 0.1) for index in range(3))
    controls_frame = pd.DataFrame(rows)
    heldout = pd.DataFrame(
        [
            {
                "animal": "RatA",
                "session": "RatA_day1",
                "geometry": "linear",
                "phase": "POST",
                "match_pair_id": 1,
                "event_id": 10,
                "population": "all",
                "split_index": split,
                "heldout_stationary": -20.0,
                "heldout_diffusion": -15.0,
                "heldout_fragmented": -18.0,
                "heldout_first_order_imm": -10.0,
                "train_mean_nonstationary_mode_probability": 0.75,
            }
            for split in range(3)
        ]
    )
    summary = controls.build_event_summary(controls_frame, heldout, 5.5)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert bool(row["map_specific_content"])
    assert bool(row["time_order_sensitive_imm"])
    assert bool(row["heldout_imm_positive"])
    assert bool(row["validated_ordered_trajectory"])
    assert bool(row["validated_clean_imm"])


def test_split_is_deterministic_and_nonempty() -> None:
    unit_ids = (1, 2, 3, 4, 5)
    first = controls.split_unit_ids(unit_ids, 0.3, 9)
    second = controls.split_unit_ids(unit_ids, 0.3, 9)
    assert first == second
    assert first[0] and first[1]
    assert set(first[0]).isdisjoint(first[1])
    assert set(first[0]) | set(first[1]) == set(unit_ids)


def test_equal_animal_inference_requires_four_robustly_positive_animals() -> None:
    rows = []
    for animal, value in zip(("RatA", "RatB", "RatC", "RatD"), (0.1, 0.2, 0.3, 0.4), strict=True):
        rows.append(
            {
                "animal": animal,
                "population_contrast": "all",
                "metric": "post_minus_pre_validated_ordered_trajectory",
                "sessions": 2,
                "matched_pairs": 40,
                "estimate": value,
            }
        )
    inference, leave_one_out = controls.infer_equal_animal_learning_effects(
        pd.DataFrame(rows),
        n_bootstraps=2000,
        seed=4,
    )
    assert len(inference) == 1
    assert bool(inference.iloc[0]["positive_robust"])
    assert inference.iloc[0]["rat_bootstrap_ci_low"] > 0.0
    assert len(leave_one_out) == 4
    assert leave_one_out["estimate"].gt(0.0).all()

    three_rat = pd.DataFrame(rows[:-1])
    inference, _ = controls.infer_equal_animal_learning_effects(
        three_rat,
        n_bootstraps=2000,
        seed=4,
    )
    assert not bool(inference.iloc[0]["positive_robust"])


def test_control_rejects_scoring_parameter_mismatch() -> None:
    selection = pd.DataFrame(
        {
            "scoring_time_bin_s": [0.02, 0.02],
            "scoring_event_padding_s": [0.0, 0.0],
        }
    )
    controls.validate_selection_scoring_parameters(
        selection,
        time_bin_s=0.02,
        event_padding_s=0.0,
    )
    try:
        controls.validate_selection_scoring_parameters(
            selection,
            time_bin_s=0.01,
            event_padding_s=0.0,
        )
    except ValueError as exc:
        assert "scoring_time_bin_s mismatch" in str(exc)
    else:
        raise AssertionError("control must reject a scorer/control bin-width mismatch")
