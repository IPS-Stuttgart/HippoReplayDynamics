import pandas as pd
import pytest

from hipporeplayimm.benchmarks import BenchmarkConfig, _benchmark_config_metadata
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
        EmissionConfig(time_bin_s=0.02),
    )

    assert encoding_config.bin_size_cm == pytest.approx(6.0)
    assert encoding_config.smoothing_sigma_bins == pytest.approx(2.25)
    assert encoding_config.min_speed_cm_s == pytest.approx(7.5)
    assert emission_config.time_bin_s == pytest.approx(0.015)


def test_score_metadata_rejects_conflicting_canonical_and_legacy_values():
    scores = pd.DataFrame(
        {
            "encoding_bin_size_cm": [4.0],
            "bin_size_cm": [6.0],
        }
    )

    with pytest.raises(ValueError, match="encoding_bin_size_cm"):
        _encoding_config_for_scores(scores, EncodingConfig())


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


def test_score_metadata_rejects_conflicting_pyrecest_aliases():
    scores = pd.DataFrame(
        {
            "pyrecest_alpha": [0.42],
            "diagnostic_pyrecest_alpha": [0.43],
        }
    )

    with pytest.raises(ValueError, match="pyrecest_alpha"):
        pyrecest_config_kwargs_for_scores(scores)
