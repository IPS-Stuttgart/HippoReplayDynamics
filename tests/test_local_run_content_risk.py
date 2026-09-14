import numpy as np
import pandas as pd
import pytest

from scripts.local_run_content_risk import BASE, FULL, correlation, evaluate_case, features, fit, predict, retain, summary


def test_fit_known_error_predictor_uses_only_declared_columns():
    rng = np.random.default_rng(42)
    x = pd.DataFrame(rng.uniform(size=(160, len(FULL))), columns=FULL)
    y = 5 + 20 * x.entropy
    state = fit(x, y, 100, "spikes_entropy")
    original = predict(x, state)
    changed = x.assign(heldout_score=1e9, endpoint_separation_cm=1e8, regional_tv=-999, a_truth_error_cm=0, b_truth_error_cm=1e9)
    np.testing.assert_array_equal(original, predict(changed, state))
    assert set(state["columns"]) == set(BASE)
    assert correlation(original, y) > 0.95
    with pytest.raises(ValueError, match="100"):
        fit(x.head(99), y[:99], 100, "full")


def test_population_own_features_ignore_other_populations_replay():
    grid = np.array([[0.0, 0.0], [8.0, 0.0], [0.0, 8.0], [8.0, 8.0]])
    rates = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    frame = pd.DataFrame({"a_x_cm": [2.0, 3.0], "a_y_cm": [4.0, 5.0], "a_spikes": [1, 3], "a_active": [1, 2], "a_entropy": [0.8, 0.6], "a_width_cm": [4.0, 3.0]})
    before = features(frame, "a", rates, grid)
    after = features(frame.assign(b_x_cm=999, b_y_cm=-999, b_entropy=0, regional_tv=1000), "a", rates, grid)
    pd.testing.assert_frame_equal(before, after)


def case():
    return pd.DataFrame(
        {
            "event_index": [3, 2, 1, 0],
            "risk_full": [4.0, 3.0, 2.0, 1.0],
            "endpoint_separation_cm": [40.0, 30.0, 20.0, 10.0],
            "regional_tv": [0.4, 0.3, 0.2, 0.1],
            "a_entropy": [0.8, 0.7, 0.6, 0.5],
            "b_entropy": [0.8, 0.7, 0.6, 0.5],
            "a_truth_error_cm": [8.0, 6.0, 4.0, 2.0],
            "b_truth_error_cm": [8.0, 6.0, 4.0, 2.0],
            "mean_truth_error_cm": [8.0, 6.0, 4.0, 2.0],
            "source": "run_test",
            "true_tile": [0, 1, 0, 1],
        }
    )


def test_half_selection_is_fixed_and_tie_broken_by_event_id():
    data = case()
    selected = retain(data.assign(risk_full=0.0), "risk_full")
    assert selected.event_index.tolist() == [0, 1]
    assert len(retain(data.head(3), "risk_full")) == 2
    with pytest.raises(ValueError):
        retain(data.iloc[:0], "risk_full")
    with pytest.raises(ValueError):
        retain(data.assign(event_index=0), "risk_full")
    with pytest.raises(ValueError):
        retain(data.assign(risk_full=np.nan), "risk_full")
    assert np.isnan(correlation(np.zeros(4), np.arange(4)))


def test_true_tile_reweighting_and_missing_tile_are_explicit():
    data = case()
    rows = pd.DataFrame(evaluate_case(data, "full")).set_index("metric")
    assert rows.loc["true_tile_reweighted_error_cm", "selected"] == 3
    assert rows.loc["mean_truth_error_cm", "baseline"] == 5
    data["true_tile"] = [0, 0, 1, 1]
    rows = pd.DataFrame(evaluate_case(data, "full")).set_index("metric")
    assert np.isnan(rows.loc["true_tile_reweighted_error_cm", "selected"])
    assert rows.loc["true_tile_reweighted_error_cm", "selected_tiles"] == 1
    assert rows.loc["true_tile_reweighted_error_cm", "original_tiles"] == 2


def gate_fixture():
    rows = []
    for model in ("spikes_entropy", "full"):
        for animal in range(4):
            for source in ("real", "run_test", "sim_matched", "sim_drift"):
                for metric in (
                    "endpoint_separation_cm",
                    "regional_tv",
                    "a_entropy",
                    "b_entropy",
                    "a_truth_error_cm",
                    "b_truth_error_cm",
                    "mean_truth_error_cm",
                    "true_tile_reweighted_error_cm",
                ):
                    if source == "real" and "error" in metric:
                        continue
                    rows.append(
                        {
                            "model": model,
                            "animal": str(animal),
                            "session": str(animal),
                            "source": source,
                            "split": 0,
                            "metric": metric,
                            "baseline": 10.0,
                            "selected": 8.0,
                            "reduction": 2.0,
                            "relative_reduction": 0.2,
                            "risk_correlation": 0.5,
                            "n_events": 100,
                            "n_selected": 50,
                        }
                    )
    return pd.DataFrame(rows)


def test_truth_worsening_and_broadening_block_advancement(tmp_path):
    good = gate_fixture()
    output = tmp_path / "good"
    output.mkdir()
    gates = pd.DataFrame(summary(good, output))
    assert gates.loc[gates.gate.eq("PF_development_advance"), "pass"].all()
    bad = good.copy()
    mask = bad.metric.isin(["b_entropy", "b_truth_error_cm"])
    bad.loc[mask, ["selected", "reduction"]] = [12.0, -2.0]
    output = tmp_path / "bad"
    output.mkdir()
    gates = pd.DataFrame(summary(bad, output))
    assert not gates.loc[gates.gate.eq("PF_development_advance"), "pass"].any()


def test_missing_true_tile_does_not_pass_via_nan_averaging(tmp_path):
    data = gate_fixture()
    data.loc[data.metric.eq("true_tile_reweighted_error_cm") & data.animal.eq("0"), ["selected", "reduction"]] = np.nan
    gates = pd.DataFrame(summary(data, tmp_path))
    assert not gates.loc[gates.gate.eq("PF_development_advance"), "pass"].any()
