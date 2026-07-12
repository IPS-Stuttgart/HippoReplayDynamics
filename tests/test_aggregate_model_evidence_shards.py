import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
_SCRIPT = _SCRIPTS_DIR / "aggregate_model_evidence_shards.py"
_SPEC = importlib.util.spec_from_file_location("aggregate_model_evidence_shards", _SCRIPT)
assert _SPEC is not None
aggregate_model_evidence_shards = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(aggregate_model_evidence_shards)

_validate_constant_settings = aggregate_model_evidence_shards._validate_constant_settings


def test_validate_constant_settings_accepts_consistent_shards():
    _validate_constant_settings(
        pd.DataFrame(
            [
                _row(event_index=0, spike_rate_scale=2.0),
                _row(event_index=1, spike_rate_scale=2.0),
            ]
        )
    )


def test_validate_constant_settings_rejects_mixed_spike_rate_scale():
    frame = pd.DataFrame(
        [
            _row(event_index=0, spike_rate_scale=1.0),
            _row(event_index=1, spike_rate_scale=2.0),
        ]
    )

    with pytest.raises(ValueError, match="spike_rate_scale"):
        _validate_constant_settings(frame)


def test_validate_constant_settings_rejects_mixed_clusterless_hyperparameters():
    frame = pd.DataFrame(
        [
            _row(event_index=0, clusterless_mark_prior_count=0.5),
            _row(event_index=1, clusterless_mark_prior_count=1.0),
        ]
    )

    with pytest.raises(ValueError, match="clusterless_mark_prior_count"):
        _validate_constant_settings(frame)


def test_validate_constant_settings_rejects_mixed_clusterless_rate_floor():
    frame = pd.DataFrame(
        [
            _row(event_index=0, clusterless_rate_floor_hz=1e-4),
            _row(event_index=1, clusterless_rate_floor_hz=1e-3),
        ]
    )

    with pytest.raises(ValueError, match="clusterless_rate_floor_hz"):
        _validate_constant_settings(frame)


@pytest.mark.parametrize(
    "column",
    [
        "diagnostic_candidate_diffusion_sigma_cm",
        "diagnostic_state_space_diffusion_sigma_cm_sqrt_s",
        "diagnostic_state_space_momentum_velocity_decay",
    ],
)
def test_validate_constant_settings_rejects_mixed_diagnostic_model_dynamics(column):
    first = _row(event_index=0)
    second = _row(event_index=1)
    first[column] = 1.0
    second[column] = 2.0

    with pytest.raises(ValueError):
        _validate_constant_settings(pd.DataFrame([first, second]))


def test_validate_constant_settings_compares_canonical_and_diagnostic_aliases():
    canonical = _row(event_index=0)
    canonical["state_space_diffusion_sigma_cm_sqrt_s"] = 85.0
    diagnostic = _row(event_index=1)
    diagnostic["diagnostic_state_space_diffusion_sigma_cm_sqrt_s"] = 120.0

    with pytest.raises(ValueError, match="state_space_diffusion_sigma_cm_sqrt_s"):
        _validate_constant_settings(pd.DataFrame([canonical, diagnostic]))


def test_validate_constant_settings_keeps_raw_and_effective_imm_stickiness_separate():
    first = _row(event_index=0)
    first["state_space_imm_mode_stickiness"] = 0.95
    first["state_space_effective_imm_mode_stickiness"] = 0.90
    second = _row(event_index=1)
    second["state_space_imm_mode_stickiness"] = 0.95
    second["diagnostic_state_space_imm_mode_stickiness"] = 0.90

    _validate_constant_settings(pd.DataFrame([first, second]))


def _row(
    *,
    event_index: int,
    spike_rate_scale: float = 1.0,
    clusterless_mark_prior_count: float = 1.0,
    clusterless_rate_floor_hz: float = 1e-4,
) -> dict[str, object]:
    return {
        "status": "success",
        "session": "RatX/OpenY",
        "event_index": event_index,
        "model": "clusterless-state-space-diffusion",
        "requested_model": "clusterless-state-space-diffusion",
        "model_family": "trajectory",
        "log_evidence": -1.0,
        "n_time": 3,
        "n_spikes": 5,
        "runtime_s": 0.0,
        "error": "",
        "bin_size_cm": 6.0,
        "smoothing_sigma_bins": 2.0,
        "min_speed_cm_s": 5.0,
        "time_bin_s": 0.003,
        "spike_rate_scale": spike_rate_scale,
        "clusterless_mark_smoothing_sigma_bins": 1.0,
        "clusterless_mark_prior_count": clusterless_mark_prior_count,
        "clusterless_mark_variance_floor": 1.0,
        "clusterless_rate_floor_hz": clusterless_rate_floor_hz,
    }
