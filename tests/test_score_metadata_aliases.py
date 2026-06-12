from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import hipporeplayimm.ground_truth as gt
from hipporeplayimm.benchmarks import BenchmarkConfig, _benchmark_config_metadata, _build_models
from hipporeplayimm.encoding import EmissionConfig, EncodingConfig
from hipporeplayimm.ground_truth import _emission_config_for_scores, _encoding_config_for_scores
from hipporeplayimm.pyrecest_score_metadata import pyrecest_config_kwargs_for_scores


def test_score_metadata_accepts_model_evidence_column_aliases():
    scores = pd.DataFrame(
        {
            "bin_size_cm": [6.0],
            "smoothing_sigma_bins": [2.25],
            "min_speed_cm_s": [7.5],
            "time_bin_s": [0.015],
            "spike_rate_scale": [2.5],
        }
    )

    encoding_config = _encoding_config_for_scores(
        scores,
        EncodingConfig(
            bin_size_cm=1.0,
            smoothing_sigma_bins=1.0,
            min_speed_cm_s=1.0,
        ),
    )
    emission_config = _emission_config_for_scores(
        scores,
        EmissionConfig(time_bin_s=0.02, spike_rate_scale=1.0),
    )

    assert encoding_config.bin_size_cm == pytest.approx(6.0)
    assert encoding_config.smoothing_sigma_bins == pytest.approx(2.25)
    assert encoding_config.min_speed_cm_s == pytest.approx(7.5)
    assert emission_config.time_bin_s == pytest.approx(0.015)
    assert emission_config.spike_rate_scale == pytest.approx(2.5)


def test_score_metadata_rejects_conflicting_canonical_and_legacy_values():
    scores = pd.DataFrame(
        {
            "encoding_bin_size_cm": [4.0],
            "bin_size_cm": [6.0],
        }
    )

    with pytest.raises(ValueError, match="encoding_bin_size_cm"):
        _encoding_config_for_scores(scores, EncodingConfig())

    emission_scores = pd.DataFrame(
        {
            "emission_time_bin_s": [0.015],
            "time_bin_s": [0.020],
            "emission_spike_rate_scale": [1.0],
            "spike_rate_scale": [2.0],
        }
    )

    with pytest.raises(ValueError, match="emission_time_bin_s"):
        _emission_config_for_scores(emission_scores, EmissionConfig())


def test_score_metadata_build_models_preserves_state_space_switch_tau():
    models = _build_models(
        BenchmarkConfig(
            emissions=EmissionConfig(time_bin_s=0.003),
            models=("state-space-first-order-imm",),
            state_space_imm_mode_stickiness=0.50,
            state_space_imm_switch_tau_s=0.060,
        )
    )

    model = models["state-space-first-order-imm"]
    assert model.config.imm_switch_tau_s == pytest.approx(0.060)
    assert model.config.imm_mode_stickiness == pytest.approx(np.exp(-0.003 / 0.060))


def test_benchmark_metadata_includes_pyrecest_hyperparameters():
    metadata = _benchmark_config_metadata(
        BenchmarkConfig(
            pyrecest_particles=73,
            pyrecest_alpha=0.42,
            pyrecest_beta=1.25,
            pyrecest_process_noise_sigma_cm_s=11.0,
            pyrecest_position_jump_sigma_cm=12.0,
            pyrecest_jump_probability=0.13,
            pyrecest_goal_reset_probability=0.14,
            pyrecest_position_proposal_probability=0.15,
            pyrecest_initial_velocity_sigma_cm_s=16.0,
            pyrecest_imm_mode_stickiness=0.61,
            pyrecest_imm_stationary_velocity_decay=0.21,
            pyrecest_imm_diffusion_velocity_decay=0.22,
            pyrecest_imm_momentum_velocity_decay=0.23,
            pyrecest_imm_jump_fraction=0.24,
            pyrecest_imm_jump_velocity_decay=0.25,
        )
    )

    assert metadata["pyrecest_particles"] == 73
    assert metadata["pyrecest_alpha"] == pytest.approx(0.42)
    assert metadata["pyrecest_beta"] == pytest.approx(1.25)
    assert metadata["pyrecest_process_noise_sigma_cm_s"] == pytest.approx(11.0)
    assert metadata["pyrecest_imm_jump_velocity_decay"] == pytest.approx(0.25)


def test_benchmark_metadata_includes_clusterless_mark_group_by():
    metadata = _benchmark_config_metadata(
        BenchmarkConfig(clusterless_mark_group_by="tetrode")
    )

    assert metadata["clusterless_mark_group_by"] == "tetrode"


def test_patched_build_models_preserves_state_space_imm_switch_tau() -> None:
    config = BenchmarkConfig(
        emissions=EmissionConfig(time_bin_s=0.02),
        state_space_imm_mode_stickiness=0.91,
        state_space_imm_switch_tau_s=0.5,
        models=("state-space-imm",),
    )

    model = _build_models(config)["state-space-imm"]

    assert model.config.imm_mode_stickiness == pytest.approx(np.exp(-0.02 / 0.5))
    assert model.config.imm_switch_tau_s == pytest.approx(0.5)


@pytest.mark.parametrize("bad_tau", [float("nan"), float("inf"), -0.001])
def test_patched_build_models_rejects_invalid_state_space_imm_switch_tau(
    bad_tau: float,
) -> None:
    config = BenchmarkConfig(
        state_space_imm_switch_tau_s=bad_tau,
        models=("state-space-imm",),
    )

    with pytest.raises(ValueError, match="state_space_imm_switch_tau_s"):
        _build_models(config)


def test_compare_ground_truth_restores_displacement_state_space_metadata_from_diagnostics(monkeypatch):
    scores = pd.DataFrame(
        {
            "session": ["s1"],
            "event_index": [0],
            "model": ["state-space-displacement-momentum"],
            "diagnostic_state_space_displacement_radius_bins": [4],
            "diagnostic_state_space_displacement_position_sigma_cm": [1.25],
            "diagnostic_state_space_displacement_transition_sigma_cm_sqrt_s": [33.0],
            "diagnostic_state_space_displacement_prior_sigma_cm": [9.5],
        }
    )
    ground_truth = pd.DataFrame(
        {
            "session": ["s1"],
            "event_index": [0],
            "true_well_id": [np.nan],
            "true_well_x": [np.nan],
            "true_well_y": [np.nan],
            "valid_label": [False],
        }
    )
    captured_configs = []

    class FakeEncoding:
        bin_centers = np.asarray([[0.0, 0.0], [1.0, 0.0]])
        cell_ids = np.asarray([1, 2])

    class FakeModel:
        def score(self, emissions, bin_centers):
            del emissions, bin_centers
            return SimpleNamespace(
                terminal_log_posterior=np.log(np.asarray([0.5, 0.5]))
            )

    def fake_build_models(config, session=None):
        del session
        captured_configs.append(config)
        return {"state-space-displacement-momentum": FakeModel()}

    monkeypatch.setattr(gt, "_load_or_generate_ground_truth", lambda *args, **kwargs: ground_truth)
    monkeypatch.setattr(gt, "load_open_field_sessions", lambda root: [SimpleNamespace(session_id="s1")])
    monkeypatch.setattr(gt, "fit_place_field_encoding", lambda *args, **kwargs: FakeEncoding())
    monkeypatch.setattr(gt, "build_emissions", lambda *args, **kwargs: SimpleNamespace(n_time=1))
    monkeypatch.setattr(
        gt,
        "infer_well_locations",
        lambda *args, **kwargs: pd.DataFrame(columns=["well_id", "well_x", "well_y"]),
    )
    monkeypatch.setattr(gt, "_build_models", fake_build_models)

    gt.compare_scores_to_ground_truth(
        "unused-root",
        scores,
        state_space_displacement_radius_bins=99,
        state_space_displacement_position_sigma_cm=99.0,
        state_space_displacement_transition_sigma_cm_sqrt_s=99.0,
        state_space_displacement_prior_sigma_cm=99.0,
    )

    assert len(captured_configs) == 1
    config = captured_configs[0]
    assert config.state_space_displacement_radius_bins == 4
    assert config.state_space_displacement_position_sigma_cm == pytest.approx(1.25)
    assert config.state_space_displacement_transition_sigma_cm_sqrt_s == pytest.approx(33.0)
    assert config.state_space_displacement_prior_sigma_cm == pytest.approx(9.5)


def test_score_metadata_recovers_pyrecest_hyperparameters_from_aliases():
    scores = pd.DataFrame(
        {
            "pyrecest_particles": [73],
            "diagnostic_pyrecest_alpha": [0.42],
            "pyrecest_beta": [1.25],
            "diagnostic_pyrecest_process_noise_sigma_cm_s": [11.0],
            "diagnostic_pyrecest_imm_mode_stickiness": [0.61],
            "pyrecest_imm_jump_velocity_decay": [0.25],
        }
    )

    kwargs = pyrecest_config_kwargs_for_scores(scores)

    assert kwargs["pyrecest_particles"] == 73
    assert kwargs["pyrecest_alpha"] == pytest.approx(0.42)
    assert kwargs["pyrecest_beta"] == pytest.approx(1.25)
    assert kwargs["pyrecest_process_noise_sigma_cm_s"] == pytest.approx(11.0)
    assert kwargs["pyrecest_imm_mode_stickiness"] == pytest.approx(0.61)
    assert kwargs["pyrecest_imm_jump_velocity_decay"] == pytest.approx(0.25)


def test_compare_ground_truth_restores_pyrecest_metadata_from_saved_scores(monkeypatch):
    scores = pd.DataFrame(
        {
            "session": ["s1"],
            "event_index": [0],
            "model": ["pyrecest-goal-particle-imm"],
            "pyrecest_particles": [73],
            "diagnostic_pyrecest_alpha": [0.42],
            "pyrecest_beta": [1.25],
            "diagnostic_pyrecest_process_noise_sigma_cm_s": [11.0],
            "pyrecest_position_jump_sigma_cm": [12.0],
            "diagnostic_pyrecest_jump_probability": [0.13],
            "pyrecest_goal_reset_probability": [0.14],
            "diagnostic_pyrecest_position_proposal_probability": [0.15],
            "pyrecest_initial_velocity_sigma_cm_s": [16.0],
            "diagnostic_pyrecest_imm_mode_stickiness": [0.61],
            "pyrecest_imm_stationary_velocity_decay": [0.21],
            "diagnostic_pyrecest_imm_diffusion_velocity_decay": [0.22],
            "pyrecest_imm_momentum_velocity_decay": [0.23],
            "diagnostic_pyrecest_imm_jump_fraction": [0.24],
            "pyrecest_imm_jump_velocity_decay": [0.25],
        }
    )
    ground_truth = pd.DataFrame(
        {
            "session": ["s1"],
            "event_index": [0],
            "true_well_id": [np.nan],
            "true_well_x": [np.nan],
            "true_well_y": [np.nan],
            "valid_label": [False],
        }
    )
    captured_configs = []

    class FakeEncoding:
        bin_centers = np.asarray([[0.0, 0.0], [1.0, 0.0]])
        cell_ids = np.asarray([1, 2])

    class FakeModel:
        def score(self, emissions, bin_centers):
            del emissions, bin_centers
            return SimpleNamespace(
                terminal_log_posterior=np.log(np.asarray([0.25, 0.75]))
            )

    def fake_build_models(config, session=None):
        del session
        captured_configs.append(config)
        return {"pyrecest-goal-particle-imm": FakeModel()}

    monkeypatch.setattr(gt, "_load_or_generate_ground_truth", lambda *args, **kwargs: ground_truth)
    monkeypatch.setattr(gt, "load_open_field_sessions", lambda root: [SimpleNamespace(session_id="s1")])
    monkeypatch.setattr(gt, "fit_place_field_encoding", lambda *args, **kwargs: FakeEncoding())
    monkeypatch.setattr(gt, "build_emissions", lambda *args, **kwargs: SimpleNamespace(n_time=1))
    monkeypatch.setattr(
        gt,
        "infer_well_locations",
        lambda *args, **kwargs: pd.DataFrame(columns=["well_id", "well_x", "well_y"]),
    )
    monkeypatch.setattr(gt, "_build_models", fake_build_models)

    gt.compare_scores_to_ground_truth(
        "unused-root",
        scores,
        pyrecest_particles=999,
        pyrecest_alpha=0.99,
        pyrecest_beta=0.98,
        pyrecest_process_noise_sigma_cm_s=97.0,
        pyrecest_position_jump_sigma_cm=96.0,
        pyrecest_jump_probability=0.95,
        pyrecest_goal_reset_probability=0.94,
        pyrecest_position_proposal_probability=0.93,
        pyrecest_initial_velocity_sigma_cm_s=92.0,
        pyrecest_imm_mode_stickiness=0.91,
        pyrecest_imm_stationary_velocity_decay=0.90,
        pyrecest_imm_diffusion_velocity_decay=0.89,
        pyrecest_imm_momentum_velocity_decay=0.88,
        pyrecest_imm_jump_fraction=0.87,
        pyrecest_imm_jump_velocity_decay=0.86,
    )

    assert len(captured_configs) == 1
    config = captured_configs[0]
    assert config.pyrecest_particles == 73
    assert config.pyrecest_alpha == pytest.approx(0.42)
    assert config.pyrecest_beta == pytest.approx(1.25)
    assert config.pyrecest_process_noise_sigma_cm_s == pytest.approx(11.0)
    assert config.pyrecest_position_jump_sigma_cm == pytest.approx(12.0)
    assert config.pyrecest_jump_probability == pytest.approx(0.13)
    assert config.pyrecest_goal_reset_probability == pytest.approx(0.14)
    assert config.pyrecest_position_proposal_probability == pytest.approx(0.15)
    assert config.pyrecest_initial_velocity_sigma_cm_s == pytest.approx(16.0)
    assert config.pyrecest_imm_mode_stickiness == pytest.approx(0.61)
    assert config.pyrecest_imm_stationary_velocity_decay == pytest.approx(0.21)
    assert config.pyrecest_imm_diffusion_velocity_decay == pytest.approx(0.22)
    assert config.pyrecest_imm_momentum_velocity_decay == pytest.approx(0.23)
    assert config.pyrecest_imm_jump_fraction == pytest.approx(0.24)
    assert config.pyrecest_imm_jump_velocity_decay == pytest.approx(0.25)


def test_score_metadata_rejects_conflicting_pyrecest_aliases():
    scores = pd.DataFrame(
        {
            "pyrecest_alpha": [0.42],
            "diagnostic_pyrecest_alpha": [0.43],
        }
    )

    with pytest.raises(ValueError, match="pyrecest_alpha"):
        pyrecest_config_kwargs_for_scores(scores)
