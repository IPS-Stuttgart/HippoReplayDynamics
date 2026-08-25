from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.test_pf_replay_revision_discriminator import (
    FEATURES,
    _centroid_predictions,
    _fit_centroid_model,
    adjudication_interpretation,
    build_cross_validated_markov_models,
    calibrate_abstention,
    correlated_noisy_path,
    cross_validated_recovery,
    equal_animal_mean,
    finite_transition_surprise,
    PURE_LABELS,
    restricted_circular_template_null,
    retrospective_geometry_score,
)


def _json_path(points: list[list[float]]) -> str:
    return json.dumps(points, separators=(",", ":"))


def test_unavailable_historical_control_forces_nonadjudicative() -> None:
    assert (
        adjudication_interpretation(
            technical_passed=True,
            candidate_classifiable=True,
            historical_positive_control_exactly_reproducible=False,
            direction="retrospective_geometry",
        )
        == "technical_nonadjudicative"
    )


def test_retrospective_geometry_score_has_predeclared_sign() -> None:
    past = np.array([[0.0, 0.0], [-1.0, 0.0], [-2.0, 0.0]])
    future = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    _, _, past_score = retrospective_geometry_score(past, past, future)
    _, _, future_score = retrospective_geometry_score(future, past, future)

    assert past_score > 0.0
    assert future_score < 0.0


def test_equal_animal_mean_does_not_event_weight_large_animal() -> None:
    frame = pd.DataFrame(
        {
            "rat": ["large"] * 10 + ["small"],
            "value": [1.0] * 10 + [-1.0],
        }
    )
    assert equal_animal_mean(frame, "value") == 0.0


def test_restricted_circular_null_never_uses_zero_offset() -> None:
    rows = []
    for rat_index, rat in enumerate(("Rat1", "Rat2"), start=1):
        session = f"{rat}/Open1"
        for event_index in range(3):
            rows.append(
                {
                    "session": session,
                    "rat": rat,
                    "event_index": 10 * rat_index + event_index,
                    "event_peak_s": float(event_index),
                    "event_route_relation": "next_movement",
                    "circular_null_eligible": True,
                    "retrospective_geometry_score": 0.0,
                    "emission_path_xy_json": _json_path(
                        [[0.0, 0.0], [1.0 + event_index, 0.0], [2.0 + event_index, 0.0]]
                    ),
                    "past_template_xy_json": _json_path(
                        [[0.0, 0.0], [-1.0, float(event_index)], [-2.0, float(event_index)]]
                    ),
                    "future_template_xy_json": _json_path(
                        [[0.0, 0.0], [1.0, float(event_index)], [2.0, float(event_index)]]
                    ),
                }
            )
    null = restricted_circular_template_null(
        pd.DataFrame(rows),
        permutations=20,
        seed=8,
    )
    assert len(null) == 20
    assert null["equal_animal_mean"].notna().all()
    for offsets in null["offsets_json"].map(json.loads):
        assert offsets
        assert all(offset > 0 for offset in offsets)


def test_finite_transition_surprise_handles_out_of_support_target() -> None:
    model = {
        "origin": np.array([0.0, 0.0]),
        "bin_cm": 1.0,
        "alpha": 0.5,
        "support": {(0, 0), (1, 0)},
        "counts": {((0, 0), (1, 0)): 3},
        "outgoing": {(0, 0): 3},
        "target_categories": 3,
    }
    surprise, pairs = finite_transition_surprise(
        np.array([[0.1, 0.1], [100.0, 100.0]]),
        model,
    )
    assert pairs == 1
    assert np.isfinite(surprise)
    assert surprise > 0.0


def test_finite_transition_surprise_same_bin_is_finite_zero() -> None:
    model = {
        "origin": np.array([0.0, 0.0]), "bin_cm": 10.0, "alpha": 0.5,
        "support": {(0, 0)}, "counts": {}, "outgoing": {}, "target_categories": 2,
    }
    surprise, pairs = finite_transition_surprise(
        np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]), model
    )
    assert (surprise, pairs) == (0.0, 0)


def test_markov_training_excludes_held_route_fold() -> None:
    routes = pd.DataFrame(
        [
            {"session": "Rat1/Open1", "route_id": "a", "cv_fold": 0},
            {"session": "Rat1/Open1", "route_id": "b", "cv_fold": 1},
        ]
    )
    points = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "route_id": route_id,
                "point_index": point_index,
                "x_cm": x,
                "y_cm": y,
            }
            for route_id, coordinates in (
                ("a", [(0.0, 0.0), (1.0, 0.0)]),
                ("b", [(0.0, 0.0), (0.0, 1.0)]),
            )
            for point_index, (x, y) in enumerate(coordinates)
        ]
    )
    models = build_cross_validated_markov_models(
        routes,
        points,
        bin_cm=1.0,
        alpha=0.5,
    )
    held_zero = models[("Rat1/Open1", 0)]
    assert held_zero["training_route_count"] == 1
    assert ((0, 0), (1, 0)) not in held_zero["counts"]
    assert ((0, 0), (0, 1)) in held_zero["counts"]


def test_correlated_noise_is_seeded_and_anchored() -> None:
    base = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    first = correlated_noisy_path(
        base,
        n_points=9,
        radial_rms_cm=3.0,
        rho=0.75,
        rng=np.random.default_rng(17),
    )
    second = correlated_noisy_path(
        base,
        n_points=9,
        radial_rms_cm=3.0,
        rho=0.75,
        rng=np.random.default_rng(17),
    )
    assert np.allclose(first, second)
    assert np.allclose(first[0], base[0])


def _recovery_fixture() -> pd.DataFrame:
    centers = {
        "past_reversed": np.array([2.0, 0.2, 1.0, 0.1]),
        "future_plan": np.array([-2.0, 0.2, 1.0, 0.1]),
        "pe_disordered": np.array([0.0, 4.0, 5.0, 2.0]),
        "null_mismatched": np.array([0.0, 4.0, 1.0, 0.1]),
        "mixture_50_50": np.array([0.0, 0.8, 1.0, 0.1]),
    }
    rows = []
    event_index = 0
    for rat in ("Rat1", "Rat2", "Rat3"):
        for session_index in (1, 2):
            session = f"{rat}/Open{session_index}"
            for label, center in centers.items():
                for replicate in range(8):
                    jitter = (replicate - 3.5) * 0.002
                    rows.append(
                        {
                            "session": session,
                            "rat": rat,
                            "event_index": event_index,
                            "replicate": replicate,
                            "sample_kind": "mixture" if label == "mixture_50_50" else "pure",
                            "true_label": label,
                            **{
                                feature: float(value + jitter)
                                for feature, value in zip(FEATURES, center, strict=True)
                            },
                        }
                    )
                event_index += 1
    return pd.DataFrame(rows)


def test_recovery_calibration_never_contains_held_group() -> None:
    folds, confusion, _, _ = cross_validated_recovery(
        _recovery_fixture(),
        target_accuracy=0.50,
        minimum_coverage=0.10,
        minimum_mixture_abstention=0.0,
    )
    assert set(folds["scheme"]) == {
        "leave_one_animal_out",
        "leave_one_session_out",
    }
    assert len(confusion) > 0
    for row in folds.itertuples(index=False):
        assert str(row.held_out_group) not in json.loads(row.training_groups_json)


def test_infeasible_calibration_forces_abstention() -> None:
    fixture = _recovery_fixture()
    pure = fixture[fixture["sample_kind"].eq("pure")]
    mixture = fixture[fixture["sample_kind"].eq("mixture")]
    model = _fit_centroid_model(pure)
    minimum_margin, maximum_distance, calibration = calibrate_abstention(
        pure,
        mixture,
        model,
        target_accuracy=1.1,
        minimum_coverage=0.5,
        minimum_mixture_abstention=0.5,
    )
    assert calibration["calibration_feasible"] == 0.0
    assert minimum_margin == np.inf
    assert maximum_distance == -np.inf


def test_classifier_rejects_missing_nonfinite_and_all_degenerate_training() -> None:
    fixture = _recovery_fixture()
    pure = fixture[fixture["sample_kind"].eq("pure")]
    with pytest.raises(ValueError, match="missing pure classes"):
        _fit_centroid_model(pure[~pure["true_label"].eq(PURE_LABELS[-1])])

    nonfinite = pure.copy()
    nonfinite.loc[nonfinite.index[0], FEATURES[0]] = np.nan
    with pytest.raises(ValueError, match="nonempty and finite"):
        _fit_centroid_model(nonfinite)

    degenerate = pure.copy()
    degenerate.loc[:, FEATURES] = 1.0
    with pytest.raises(ValueError, match="all classifier training features are degenerate"):
        _fit_centroid_model(degenerate)


def test_classifier_rejects_nonfinite_prediction_features() -> None:
    fixture = _recovery_fixture()
    pure = fixture[fixture["sample_kind"].eq("pure")]
    model = _fit_centroid_model(pure)
    bad = pure.iloc[:1].copy()
    bad.loc[:, FEATURES[0]] = np.nan
    with pytest.raises(ValueError, match="prediction features must be finite"):
        _centroid_predictions(bad, model)
