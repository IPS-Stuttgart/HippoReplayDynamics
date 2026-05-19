import json

import pandas as pd

from hipporeplayimm.state_space import StateSpaceDecoderConfig
from hipporeplayimm.state_space_selection import (
    apply_state_space_parameter_values,
    load_state_space_decoder_config,
)


def test_load_state_space_decoder_config_from_selection_manifest(tmp_path):
    manifest = {
        "selected_parameters": {
            "state_space_diffusion_sigma_cm_sqrt_s": 70.0,
            "state_space_momentum_sigma_cm_sqrt_s": 75.0,
            "state_space_momentum_initial_sigma_cm_sqrt_s": 80.0,
            "state_space_momentum_velocity_decay": 0.9,
            "state_space_momentum_candidate_top_k": 96,
        }
    }
    (tmp_path / "state_space_parameter_selection_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    config = load_state_space_decoder_config(
        tmp_path,
        base=StateSpaceDecoderConfig(
            stationary_sigma_cm=3.0,
            max_step_sigma=5.0,
            imm_mode_stickiness=0.91,
        ),
    )

    assert config.stationary_sigma_cm == 3.0
    assert config.max_step_sigma == 5.0
    assert config.imm_mode_stickiness == 0.91
    assert config.diffusion_sigma_cm_sqrt_s == 70.0
    assert config.momentum_sigma_cm_sqrt_s == 75.0
    assert config.momentum_initial_sigma_cm_sqrt_s == 80.0
    assert config.momentum_velocity_decay == 0.9
    assert config.momentum_candidate_top_k == 96


def test_load_state_space_decoder_config_from_recommendation_csv(tmp_path):
    pd.DataFrame(
        {
            "recommendation_rank": [2, 1],
            "state_space_diffusion_sigma_cm_sqrt_s": [90.0, 60.0],
            "state_space_momentum_sigma_cm_sqrt_s": [92.0, 62.0],
            "state_space_momentum_initial_sigma_cm_sqrt_s": [94.0, 64.0],
            "state_space_momentum_velocity_decay": [0.95, 0.85],
            "state_space_momentum_candidate_top_k": [128, 64],
        }
    ).to_csv(tmp_path / "state_space_parameter_recommendation.csv", index=False)

    config = load_state_space_decoder_config(tmp_path)

    assert config.diffusion_sigma_cm_sqrt_s == 60.0
    assert config.momentum_sigma_cm_sqrt_s == 62.0
    assert config.momentum_initial_sigma_cm_sqrt_s == 64.0
    assert config.momentum_velocity_decay == 0.85
    assert config.momentum_candidate_top_k == 64


def test_apply_state_space_parameter_values_accepts_optional_config_fields():
    config = apply_state_space_parameter_values(
        StateSpaceDecoderConfig(),
        {
            "state_space_stationary_sigma_cm": 4.0,
            "state_space_max_step_sigma": 6.0,
            "state_space_imm_mode_stickiness": 0.97,
            "state_space_momentum_predicted_candidate_top_k": 11,
        },
    )

    assert config.stationary_sigma_cm == 4.0
    assert config.max_step_sigma == 6.0
    assert config.imm_mode_stickiness == 0.97
    assert config.momentum_predicted_candidate_top_k == 11
