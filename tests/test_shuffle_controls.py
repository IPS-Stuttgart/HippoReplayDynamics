from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.encoding import EncodingConfig, EncodingModel
from hipporeplayimm.shuffle_controls import (
    SHUFFLE_CONTROL_SCORE_COLUMNS,
    ShuffleControlConfig,
    _spatial_roll_rates,
    add_shuffle_p_values,
    score_shuffle_controls,
    shuffled_encoding,
)


SHUFFLE_SUMMARY_COLUMNS = {
    "shuffle_p_value",
    "shuffle_log_evidence_median",
    "shuffle_log_evidence_mean",
    "shuffle_log_evidence_std",
    "shuffle_count",
}


def _two_bin_encoding() -> EncodingModel:
    return EncodingModel(
        x_edges=np.array([0.0, 1.0, 2.0], dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        bin_centers=np.array([[0.5, 0.5], [1.5, 0.5]], dtype=float),
        rates_hz=np.ones((1, 2), dtype=float),
        occupancy_s=np.ones(2, dtype=float),
        cell_ids=np.array([10], dtype=int),
        config=EncodingConfig(),
    )


def test_independent_spatial_permutation_handles_empty_cell_set():
    encoding = EncodingModel(
        x_edges=np.array([0.0, 1.0, 2.0], dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        bin_centers=np.array([[0.5, 0.5], [1.5, 0.5]], dtype=float),
        rates_hz=np.empty((0, 2), dtype=float),
        occupancy_s=np.ones(2, dtype=float),
        cell_ids=np.array([], dtype=int),
        config=EncodingConfig(),
    )

    control = shuffled_encoding(
        encoding,
        mode="independent-spatial-permutation",
        random_seed=7,
    )

    assert control.rates_hz.shape == (0, 2)
    assert control.rates_hz.dtype == float
    np.testing.assert_array_equal(control.cell_ids, np.array([], dtype=int))
    np.testing.assert_allclose(control.occupancy_s, np.ones(2, dtype=float))


@pytest.mark.parametrize("random_seed", [-1, 1.5, True, float("nan")])
def test_shuffled_encoding_rejects_invalid_random_seed(random_seed) -> None:
    with pytest.raises(ValueError, match="random_seed"):
        shuffled_encoding(_two_bin_encoding(), random_seed=random_seed)  # type: ignore[arg-type]


def test_spatial_roll_avoids_identity_shift_on_multibin_grid() -> None:
    rates = np.arange(4.0, dtype=float).reshape(1, 4)

    rolled = _spatial_roll_rates(rates, (2, 2), np.random.default_rng(11))

    assert rolled.shape == rates.shape
    assert not np.array_equal(rolled, rates)


def test_spatial_roll_keeps_single_bin_grid_unchanged() -> None:
    rates = np.array([[3.5]], dtype=float)

    rolled = _spatial_roll_rates(rates, (1, 1), np.random.default_rng(11))

    np.testing.assert_array_equal(rolled, rates)


def test_spatial_roll_validates_rate_grid_shape_consistency() -> None:
    rates = np.arange(6.0, dtype=float).reshape(1, 6)

    with pytest.raises(ValueError, match="one column per spatial grid bin"):
        _spatial_roll_rates(rates, (2, 2), np.random.default_rng(1))


def test_score_shuffle_controls_preserves_schema_for_zero_shuffles() -> None:
    out = score_shuffle_controls(
        SimpleNamespace(session_id="Rat1/Open1"),
        _two_bin_encoding(),
        [2],
        {},
        control_config=ShuffleControlConfig(n_shuffles=0),
    )

    assert out.empty
    assert out.columns.tolist() == list(SHUFFLE_CONTROL_SCORE_COLUMNS)


@pytest.mark.parametrize(
    ("config_kwargs", "message"),
    [
        ({"mode": "not-a-control"}, "mode"),
        ({"n_shuffles": -1}, "n_shuffles"),
        ({"n_shuffles": 1.5}, "n_shuffles"),
        ({"n_shuffles": True}, "n_shuffles"),
        ({"n_shuffles": float("nan")}, "n_shuffles"),
        ({"random_seed": -1}, "random_seed"),
        ({"random_seed": 1.5}, "random_seed"),
        ({"random_seed": True}, "random_seed"),
        ({"random_seed": float("nan")}, "random_seed"),
    ],
)
def test_score_shuffle_controls_rejects_invalid_control_config(config_kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        score_shuffle_controls(
            SimpleNamespace(session_id="Rat1/Open1"),
            _two_bin_encoding(),
            [],
            {},
            control_config=ShuffleControlConfig(**config_kwargs),
        )


def test_score_shuffle_controls_materializes_generator_event_indices(monkeypatch) -> None:
    encoding = EncodingModel(
        x_edges=np.array([0.0, 1.0, 2.0], dtype=float),
        y_edges=np.array([0.0, 1.0], dtype=float),
        bin_centers=np.array([[0.5, 0.5], [1.5, 0.5]], dtype=float),
        rates_hz=np.ones((1, 2), dtype=float),
        occupancy_s=np.ones(2, dtype=float),
        cell_ids=np.array([10], dtype=int),
        config=EncodingConfig(),
    )
    call_events: list[int] = []

    def fake_build_emissions(session, control_encoding, event_index, emission_config):
        del session, control_encoding, emission_config
        call_events.append(int(event_index))
        return SimpleNamespace()

    class DummyModel:
        def score(self, emissions, bin_centers):
            del emissions, bin_centers
            return SimpleNamespace(
                model_name="dummy",
                log_likelihood=1.0,
                n_time=1,
                n_spikes=0,
            )

    monkeypatch.setattr("hipporeplayimm.shuffle_controls.build_emissions", fake_build_emissions)

    out = score_shuffle_controls(
        SimpleNamespace(session_id="Rat1/Open1"),
        encoding,
        (event_index for event_index in [2, 4]),
        {"dummy": DummyModel()},
        control_config=ShuffleControlConfig(n_shuffles=3, random_seed=11),
    )

    assert len(out) == 6
    assert call_events == [2, 4, 2, 4, 2, 4]
    assert out["event_index"].tolist() == [2, 4, 2, 4, 2, 4]
    assert out["control_index"].tolist() == [0, 0, 1, 1, 2, 2]


def test_add_shuffle_p_values_preserves_schema_when_control_scores_empty() -> None:
    real_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"],
            "event_index": [3],
            "model": ["sorted-spike-state-space-first-order-imm"],
            "log_evidence": [12.5],
        }
    )
    control_scores = pd.DataFrame(columns=["session", "event_index", "model", "log_evidence"])

    out = add_shuffle_p_values(real_scores, control_scores)

    assert SHUFFLE_SUMMARY_COLUMNS.issubset(out.columns)
    assert np.isnan(out.loc[0, "shuffle_p_value"])
    assert np.isnan(out.loc[0, "shuffle_log_evidence_median"])
    assert np.isnan(out.loc[0, "shuffle_log_evidence_mean"])
    assert np.isnan(out.loc[0, "shuffle_log_evidence_std"])
    assert np.isnan(out.loc[0, "shuffle_count"])


def test_add_shuffle_p_values_preserves_schema_when_real_scores_empty() -> None:
    real_scores = pd.DataFrame(columns=["session", "event_index", "model", "log_evidence"])
    control_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"],
            "event_index": [3],
            "model": ["sorted-spike-state-space-first-order-imm"],
            "log_evidence": [11.0],
        }
    )

    out = add_shuffle_p_values(real_scores, control_scores)

    assert out.empty
    assert SHUFFLE_SUMMARY_COLUMNS.issubset(out.columns)


def test_add_shuffle_p_values_returns_nan_for_nonfinite_real_log_evidence() -> None:
    real_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"],
            "event_index": [3],
            "model": ["sorted-spike-state-space-first-order-imm"],
            "log_evidence": [np.nan],
        }
    )
    control_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [3, 3],
            "model": ["sorted-spike-state-space-first-order-imm", "sorted-spike-state-space-first-order-imm"],
            "log_evidence": [11.0, 12.0],
        }
    )

    out = add_shuffle_p_values(real_scores, control_scores)

    assert np.isnan(out.loc[0, "shuffle_p_value"])
    assert out.loc[0, "shuffle_count"] == 2


def test_add_shuffle_p_values_ignores_nonfinite_control_log_evidence() -> None:
    real_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"],
            "event_index": [3],
            "model": ["sorted-spike-state-space-first-order-imm"],
            "log_evidence": [10.0],
        }
    )
    control_scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"] * 5,
            "event_index": [3] * 5,
            "model": ["sorted-spike-state-space-first-order-imm"] * 5,
            "log_evidence": [12.0, 8.0, np.nan, np.inf, -np.inf],
        }
    )

    out = add_shuffle_p_values(real_scores, control_scores)

    assert np.isclose(out.loc[0, "shuffle_p_value"], 2.0 / 3.0)
    assert out.loc[0, "shuffle_count"] == 2
    assert out.loc[0, "shuffle_log_evidence_median"] == 10.0
