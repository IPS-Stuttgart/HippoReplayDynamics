from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from hipporeplayimm.encoding import (
    EncodingConfig,
    EncodingModel,
    LogEmissionTensor,
)
from hipporeplayimm.models import EventScore
from scripts import test_pfeiffer_train_only_map_specific_mode_prediction as analysis


def _encoding() -> EncodingModel:
    return EncodingModel(
        x_edges=np.asarray([0.0, 1.0, 2.0]),
        y_edges=np.asarray([0.0, 1.0]),
        bin_centers=np.asarray([[0.5, 0.5], [1.5, 0.5]]),
        rates_hz=np.asarray(
            [
                [8.0, 1.0],
                [1.0, 8.0],
                [6.0, 2.0],
                [2.0, 6.0],
            ]
        ),
        occupancy_s=np.asarray([1.0, 2.0]),
        cell_ids=np.asarray([1, 2, 3, 4]),
        config=EncodingConfig(),
    )


def test_population_code_permutation_is_shared_and_deterministic() -> None:
    encoding = _encoding()
    first, first_hash = analysis.population_code_permuted_encoding(encoding, seed=3)
    second, second_hash = analysis.population_code_permuted_encoding(encoding, seed=3)

    np.testing.assert_array_equal(first.occupancy_s, encoding.occupancy_s)
    np.testing.assert_array_equal(first.bin_centers, encoding.bin_centers)
    np.testing.assert_array_equal(first.rates_hz, second.rates_hz)
    assert first_hash == second_hash
    # Every cell receives the same occupied-bin permutation.
    original_order = np.argsort(encoding.rates_hz, axis=1)
    permuted_order = np.argsort(first.rates_hz, axis=1)
    np.testing.assert_array_equal(original_order[:, ::-1], permuted_order)


class _DummyModel:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    def score(self, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> EventScore:
        self.calls += 1
        if self.name == analysis.IMM:
            posterior = np.log(np.asarray([[0.8, 0.2], [0.3, 0.7]]))
            diagnostics = {
                "state_space_imm_mode_posterior_over_time": "[[0.2,0.7,0.1],[0.3,0.6,0.1]]"
            }
            log_likelihood = 10.0
        else:
            posterior = np.log(np.asarray([[0.5, 0.5], [0.5, 0.5]]))
            diagnostics = {}
            log_likelihood = 1.0
        return EventScore(
            model_name=self.name,
            log_likelihood=log_likelihood,
            n_time=emissions.n_time,
            n_spikes=emissions.n_spikes,
            diagnostics=diagnostics,
            terminal_log_posterior=posterior[-1],
            trajectory_log_posterior=posterior,
        )


def test_event_scoring_never_invokes_model_with_heldout_emissions(monkeypatch) -> None:
    imm = _DummyModel(analysis.IMM)
    fragmented = _DummyModel(analysis.FRAGMENTED)
    monkeypatch.setattr(
        analysis,
        "_models",
        lambda state_config: {analysis.IMM: imm, analysis.FRAGMENTED: fragmented},
    )

    def fake_emissions(session, encoding, event_index, config):
        n_cells = len(encoding.cell_ids)
        offset = float(np.sum(encoding.cell_ids)) / 100.0
        return LogEmissionTensor(
            log_likelihood=np.asarray([[-1.0 - offset, -2.0], [-2.0, -1.0 - offset]]),
            spike_counts=np.ones((2, n_cells), dtype=int),
            times=np.asarray([0.002, 0.006]),
            dt=0.004,
            cell_ids=encoding.cell_ids,
            n_spikes=2 * n_cells,
        )

    monkeypatch.setattr(analysis, "build_emissions", fake_emissions)
    session = SimpleNamespace(
        session_id="Rat1/Open1",
        rat="Rat1",
        ripple=lambda index: SimpleNamespace(start=1.0, end=1.008),
    )
    real = _encoding()
    wrong, wrong_hash = analysis.population_code_permuted_encoding(real, seed=11)

    row = analysis._score_event(
        session=session,
        event_index=0,
        split_index=0,
        split_seed=1,
        train_cells=np.asarray([1, 2, 3]),
        test_cells=np.asarray([4]),
        real_encoding=real,
        wrong_encoding=wrong,
        wrong_map_sha256=wrong_hash,
        emission_config=object(),
        state_config=object(),
        margin_threshold=5.5,
    )

    assert row["status"] == "success"
    # One call per map, both before held-out emissions are scored directly.
    assert imm.calls == 2
    assert fragmented.calls == 2
    assert row["heldout_replay_spikes_used_for_latent_inference"] is False
    assert row["cell_sets_disjoint"] is True
    assert len(row["real_imm_training_posterior_sha256"]) == 64


def test_heldout_assembly_turnover_uses_training_defined_boundary() -> None:
    transition = np.zeros((11, 3, 3), dtype=float)
    transition[:, 0, 0] = 0.5
    transition[:, 1, 1] = 0.5
    transition[5] = 0.0
    transition[5, 0, 1] = 0.8
    transition[5, 0, 0] = 0.1
    transition[5, 1, 1] = 0.1
    score = SimpleNamespace(
        diagnostics={
            "state_space_imm_mode_transition_posterior_over_time": json.dumps(
                transition.tolist()
            )
        }
    )
    training_counts = np.ones((12, 3), dtype=int)
    heldout_counts = np.zeros((12, 2), dtype=int)
    heldout_counts[:6, 0] = 1
    heldout_counts[6:, 1] = 1

    result = analysis.heldout_assembly_turnover(
        score,
        training_counts,
        heldout_counts,
        window_bins=2,
        matched_controls=2,
    )

    assert result["assembly_turnover_evaluable"] is True
    assert result["assembly_boundary_transition_index"] == 5
    assert np.isclose(result["assembly_boundary_heldout_turnover_hellinger"], 1.0)
    assert np.isclose(result["assembly_control_heldout_turnover_median"], 0.0)
    assert np.isclose(result["heldout_assembly_turnover_excess"], 1.0)


def test_event_medians_and_clean_subset_use_training_split_values_only() -> None:
    rows = []
    for event_index in (0, 1):
        for split_index in (0, 1, 2):
            rows.append(
                {
                    "session": "Rat1/Open1",
                    "rat": "Rat1",
                    "event_index": event_index,
                    "cell_split_index": split_index,
                    "status": "success",
                    "train_map_specific_nonstationary_mass": event_index + split_index,
                    "real_frozen_heldout_delta_imm_minus_fragmented": 2 * event_index + split_index,
                    "wrong_frozen_heldout_delta_imm_minus_fragmented": split_index,
                    "map_specific_frozen_heldout_delta": 2 * event_index,
                    "real_train_mean_nonstationary_mode_probability": 0.7,
                    "wrong_train_mean_nonstationary_mode_probability": 0.4,
                    "real_imm_train_posterior_entropy": 2.0,
                    "train_cell_count": 7,
                    "test_cell_count": 3,
                    "train_spikes": 12,
                    "test_spikes": 5,
                    "n_time": 10,
                    "event_duration_s": 0.04,
                    "train_defined_clean_imm": split_index < 2,
                }
            )
    events = analysis.build_event_medians(pd.DataFrame(rows))

    assert events["completed_splits"].tolist() == [3, 3]
    assert events[analysis.PRIMARY_X].tolist() == [1.0, 2.0]
    assert events[analysis.PRIMARY_Y].tolist() == [1.0, 3.0]
    assert events["train_defined_clean_imm_fraction"].tolist() == [2 / 3, 2 / 3]
    assert events["train_defined_clean_imm_majority"].tolist() == [True, True]


def _passing_gate_inputs():
    hash_value = "a" * 64
    split_rows = []
    event_rows = []
    for index, rat in enumerate(("Rat1", "Rat2", "Rat3", "Rat4")):
        split_rows.append(
            {
                "status": "success",
                "actual_test_cell_fraction": 0.3,
                "cell_sets_disjoint": True,
                "heldout_replay_spikes_used_for_latent_inference": False,
                "selection_scope": "all_160_events_no_all_cell_clean_imm_selection",
                "real_imm_training_posterior_sha256": hash_value,
                "real_fragmented_training_posterior_sha256": hash_value,
                "wrong_imm_training_posterior_sha256": hash_value,
                "wrong_fragmented_training_posterior_sha256": hash_value,
            }
        )
        event_rows.append(
            {
                "session": f"{rat}/Open1",
                "rat": rat,
                "event_index": index,
                "completed_splits": 1,
                analysis.PRIMARY_X: 0.1 + index,
                analysis.PRIMARY_Y: 1.0 + index,
            }
        )
    associations = pd.DataFrame(
        [
            {
                "analysis_id": "primary_all_160_events",
                "core_adjusted_partial_spearman_rho": 0.5,
                "extended_adjusted_partial_spearman_rho": 0.4,
                "rat_cluster_bootstrap_ci_low": 0.1,
                "rat_cluster_bootstrap_ci_high": 0.8,
                "within_session_permutation_p_one_sided": 0.01,
            }
        ]
    )
    by_rat = pd.DataFrame(
        {"rat": ["Rat1", "Rat2", "Rat3", "Rat4"], "raw_spearman_rho": [0.1] * 4}
    )
    leave_one_out = pd.DataFrame(
        {
            "omitted_rat": ["Rat1", "Rat2", "Rat3", "Rat4"],
            "core_adjusted_partial_spearman_rho": [0.2] * 4,
        }
    )
    return (
        pd.DataFrame(split_rows),
        pd.DataFrame(event_rows),
        associations,
        by_rat,
        leave_one_out,
    )


def test_gate_requires_explicit_no_heldout_latent_inference() -> None:
    inputs = _passing_gate_inputs()
    gates = analysis.build_gate_summary(
        *inputs,
        expected_events=4,
        expected_splits=1,
        test_cell_fraction=0.3,
    ).set_index("gate")
    assert bool(gates.loc["overall_technical", "passed"])
    assert bool(
        gates.loc[
            "overall_population_generalizable_mode_allocation_hypothesis",
            "passed",
        ]
    )

    leaking = inputs[0].copy()
    leaking.loc[0, "heldout_replay_spikes_used_for_latent_inference"] = True
    gates = analysis.build_gate_summary(
        leaking,
        *inputs[1:],
        expected_events=4,
        expected_splits=1,
        test_cell_fraction=0.3,
    ).set_index("gate")
    assert not bool(gates.loc["heldout_spikes_never_used_for_latent_inference", "passed"])
    assert not bool(gates.loc["overall_technical", "passed"])
