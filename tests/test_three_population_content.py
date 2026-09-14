import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax

from scripts._provenance import file_sha256
from scripts.measure_three_population_content import (
    FULL_FEATURES,
    POOLED_FEATURES,
    measure_session,
    observed_features,
    partition_three,
    validation_outcomes,
)
from scripts.validate_three_population_content import (
    apply_models,
    equal_animal_weights,
    external_gates,
    predict_model,
    retained_half,
    summary_tables,
    train_model,
)


def synthetic(n=80):
    rng = np.random.default_rng(42)
    grid = np.array([[x, y] for x in range(0, 80, 8) for y in range(0, 80, 8)], float)
    rates = rng.uniform(.2, 15, (18, len(grid)))
    counts = rng.poisson(.2, (n, 18))
    a, b, c = partition_three(18, 4)
    features, decoded = observed_features(counts[:, a], counts[:, b], rates[a], rates[b], grid)
    outcomes = validation_outcomes(counts[:, c], rates[c], grid, decoded, np.full((n, 2), 30))
    data = pd.concat([features, outcomes], axis=1)
    data["dataset"], data["source"], data["split"], data["draw"] = "pfeiffer_foster", "real", 0, -1
    data["animal"] = [f"rat{i%4}" for i in range(n)]
    data["session"] = data.animal+"_session"
    data["event_index"] = np.arange(n)
    # Labels deliberately cover both outcomes for classifier fixture tests.
    data["c_support"] = np.arange(n)%3 == 0
    return data, (grid, rates, counts, a, b, c)


def test_partition_disjoint_equal_reproducible():
    for n in range(15, 31):
        groups = partition_three(n, 3)
        assert all(len(group) == n//3 for group in groups)
        assert len(np.unique(np.concatenate(groups))) == 3*(n//3)
        for a, b in zip(groups, partition_three(n, 3), strict=True):
            np.testing.assert_equal(a, b)
    with pytest.raises(ValueError):
        partition_three(14, 3)


def test_direct_poisson_and_fair_pooled_population():
    _, (grid, rates, counts, a, b, c) = synthetic()
    frame, decoded = observed_features(counts[:, a], counts[:, b], rates[a], rates[b], grid)
    likelihood = np.sum(counts[:, a, None]*np.log(rates[a])[None, :, :] - .02*rates[a][None, :, :], axis=1)
    expected = softmax(likelihood, axis=1)
    np.testing.assert_allclose(decoded["p"], expected)
    np.testing.assert_equal(frame.pooled_spikes, counts[:, np.r_[a, b]].sum(axis=1))
    np.testing.assert_equal(frame.pooled_active, (counts[:, np.r_[a, b]] > 0).sum(axis=1))
    assert set(POOLED_FEATURES).issubset(FULL_FEATURES)
    assert not any(name.startswith("c_") for name in FULL_FEATURES)
    assert frame.n_observed_cells.eq(len(a)+len(b)).all()


def test_c_perturbation_cannot_change_features_predictions_or_retention():
    data, (grid, rates, counts, a, b, c) = synthetic()
    models = {name: train_model(data, features) for name, features in
              dict(constant=[], pooled=POOLED_FEATURES, full=FULL_FEATURES).items()}
    before = apply_models(data, models)
    changed = counts.copy()
    changed[:, c] = 100
    features, decoded = observed_features(changed[:, a], changed[:, b], rates[a], rates[b], grid)
    np.testing.assert_allclose(features[FULL_FEATURES], data[FULL_FEATURES])
    outcomes = validation_outcomes(changed[:, c], rates[c], grid, decoded)
    after_data = data.copy()
    after_data[outcomes.columns] = outcomes
    after = apply_models(after_data, models)
    for column in [name for name in before if name.startswith(("prediction_", "retained_"))]:
        np.testing.assert_array_equal(before[column].to_numpy(), after[column].to_numpy())
    assert not np.allclose(before.c_spikes, after.c_spikes)


def test_prediction_serialization_and_test_data_not_used_for_scaling():
    data, _ = synthetic()
    model = train_model(data, FULL_FEATURES)
    restored = json.loads(json.dumps(model))
    np.testing.assert_equal(predict_model(data, model), predict_model(data, restored))
    predicted = predict_model(data.iloc[:1], model)
    appended = pd.concat([data.iloc[:1], data.iloc[1:2].assign(a_spikes=1e8)])
    np.testing.assert_allclose(predicted, predict_model(appended, model)[:1], rtol=1e-13, atol=1e-15)


def test_equal_rat_session_event_weights():
    frame = pd.DataFrame(dict(animal=["a", "a", "a", "b"], session=["x", "x", "y", "z"], event_index=[0, 1, 0, 0]))
    frame["weight"] = equal_animal_weights(frame)
    np.testing.assert_allclose(frame.groupby("animal").weight.sum(), [2, 2])
    np.testing.assert_allclose(frame.groupby("session").weight.sum(), [1, 1, 2])


def test_ties_fixed_coverage_and_no_target_use():
    frame, _ = synthetic(12)
    retained = retained_half(frame, np.zeros(len(frame)))
    selected = frame.assign(retained=retained).groupby("session").retained.sum()
    assert selected.eq(2).all()
    permuted = frame.sample(frac=1, random_state=3)
    alternate = retained_half(permuted, np.zeros(len(frame)))
    assert set(frame.loc[retained, "event_index"]) == set(permuted.loc[alternate, "event_index"])
    with pytest.raises(ValueError):
        retained_half(pd.concat([frame, frame]), np.zeros(24))


def test_one_class_prevalence_is_not_perfect_generalization():
    data, _ = synthetic()
    model = train_model(data.assign(c_support=False), FULL_FEATURES)
    assert model["kind"] == "constant"
    assert (predict_model(data, model) == 0).all()


def test_validation_never_vacuously_passes_or_self_audits():
    data, _ = synthetic()
    models = {name: train_model(data, features) for name, features in
              dict(constant=[], pooled=POOLED_FEATURES, full=FULL_FEATURES).items()}
    data = apply_models(data, models)
    sessions, animal, summary = summary_tables(data)
    catalog = pd.DataFrame(dict(candidates=[40, 40]))
    gates = external_gates(sessions, animal, summary, catalog, catalog, True).set_index("gate")
    assert not gates.loc["independent_reconstruction_audit", "passed"]
    assert not gates.loc["run_test_a_truth_error_not_worse", "passed"]
    assert not gates.loc["overall", "passed"]
    catalog.loc[0, "candidates"] = np.nan
    gates = external_gates(sessions, animal, summary, catalog, catalog, True).set_index("gate")
    assert not gates.loc["source_endpoint_coverage", "passed"]


def test_measure_smoke_preserves_whole_population_simulation_counts(tmp_path, monkeypatch):
    import scripts.measure_three_population_content as module
    _, (grid, rates, _, _, _, _) = synthetic()
    path = tmp_path/"cache.npz"
    base = np.ones((20, 18), int)
    np.savez_compressed(path, unit_qc_mask=np.ones(18, bool), cell_ids=np.arange(18),
        valid_spatial_bins=np.ones(len(grid), bool), occupancy_first_half_s=np.ones(len(grid)),
        bin_centers_cm=grid, rates_first_half_hz=rates, rates_second_half_hz=rates,
        supported_run_intervals=[[0, 200]], candidate_event_indices=[4, 9], candidate_offsets=[0, 10, 20],
        candidate_base_counts=base, candidate_base_starts_s=np.r_[np.arange(10)*.005+10, np.arange(10)*.005+20],
        candidate_base_durations_s=np.full(20, .005))
    windows = np.column_stack([np.arange(100), np.arange(100)+.02, np.full(100, 30), np.full(100, 30)])
    monkeypatch.setattr(module, "run_windows", lambda *args: (windows, np.ones((100, 18), int)))
    row = SimpleNamespace(artifact_path=str(path), artifact_sha256=file_sha256(path), dataset="pfeiffer_foster",
                          animal="rat", session="one", candidates=2)
    result = measure_session(row, tmp_path/"out", 4)
    assert result["candidates"] == 2
    table = pd.read_csv(tmp_path/"out/event_readouts.csv.gz")
    assert len(table) == 3*(2+100+4*2)
    archive = np.load(tmp_path/"out/audit_arrays.npz", allow_pickle=False)
    for key in archive.files:
        if key.startswith("split") and key.endswith("_counts"):
            np.testing.assert_equal(archive[key].sum(axis=1), [72, 72])
