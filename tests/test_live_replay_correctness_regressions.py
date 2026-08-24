from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from hipporeplayimm.continuous_time_imm_transition_patch import (
    _continuous_time_mode_transition_matrix,
    _wrap_trajectory_diagnostics,
    _wrap_trajectory_mode_transition_sequence,
)
from hipporeplayimm.models import EventScore
from hipporeplayimm.reverse_time_terminal_guard import (
    _score_reverse_with_supported_return_trajectory,
)


def test_ctmc_trajectory_wrapper_does_not_execute_legacy_duration_routing() -> None:
    config = SimpleNamespace(imm_switch_tau_s=0.1)
    base_transition = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=float)

    def legacy_helper(*args, **kwargs):
        del args, kwargs
        raise AssertionError("legacy duration routing must not run in CTMC mode")

    def base_helper(config, stickiness):
        del config, stickiness
        return base_transition

    wrapped = _wrap_trajectory_mode_transition_sequence(legacy_helper, base_helper)
    observed = wrapped(config, 0.9, np.asarray([0.025]))[0]
    expected = _continuous_time_mode_transition_matrix(base_transition, 0.025, 0.1)

    np.testing.assert_allclose(observed, expected, rtol=1.0e-12, atol=1.0e-12)


def test_legacy_trajectory_wrapper_delegates_before_duration_coercion() -> None:
    config = SimpleNamespace(imm_switch_tau_s=0.0)
    durations = object()
    sentinel = [np.eye(2, dtype=float)]

    def legacy_helper(got_config, stickiness, got_durations):
        assert got_config is config
        assert stickiness == 0.9
        assert got_durations is durations
        return sentinel

    def base_helper(*args, **kwargs):
        del args, kwargs
        raise AssertionError("CTMC base routing must not run in legacy mode")

    wrapped = _wrap_trajectory_mode_transition_sequence(legacy_helper, base_helper)

    assert wrapped(config, 0.9, durations) is sentinel


def _trajectory_diagnostic_helper(seen: list[np.ndarray]):
    def helper(
        emissions,
        bin_centers,
        config,
        transition_durations_s,
        *,
        valid_bin_mask=None,
        return_trajectory=True,
    ):
        del emissions, bin_centers, config, valid_bin_mask, return_trajectory
        seen.append(np.asarray(transition_durations_s, dtype=float))
        return (
            0.0,
            None,
            None,
            None,
            {"state_space_trajectory_imm_mode_stickiness_per_step": "legacy"},
        )

    return helper


def _diagnostic_module_stub() -> SimpleNamespace:
    return SimpleNamespace(
        _format_float_series=lambda values: ",".join(
            f"{float(value):.12g}" for value in np.asarray(values, dtype=float)
        )
    )


def test_ctmc_diagnostics_preserve_one_shot_duration_iterables() -> None:
    emissions = SimpleNamespace(n_time=3, dt=0.02)
    config = SimpleNamespace(imm_switch_tau_s=0.1)
    seen: list[np.ndarray] = []
    module = _diagnostic_module_stub()
    wrapped = _wrap_trajectory_diagnostics(_trajectory_diagnostic_helper(seen), module)

    result = wrapped(emissions, None, config, (value for value in (0.01, 0.03)))

    np.testing.assert_allclose(seen[0], np.asarray([0.01, 0.03]))
    expected_survival = np.exp(-np.asarray([0.01, 0.03]) / config.imm_switch_tau_s)
    assert result[-1]["state_space_trajectory_imm_mode_stickiness_per_step"] == module._format_float_series(
        expected_survival
    )


def test_ctmc_diagnostics_match_decoder_fallback_durations() -> None:
    emissions = SimpleNamespace(n_time=3, dt=0.02)
    config = SimpleNamespace(imm_switch_tau_s=0.1)
    seen: list[np.ndarray] = []
    module = _diagnostic_module_stub()
    wrapped = _wrap_trajectory_diagnostics(_trajectory_diagnostic_helper(seen), module)

    result = wrapped(emissions, None, config, iter(()))

    expected_durations = np.asarray([emissions.dt, emissions.dt])
    np.testing.assert_allclose(seen[0], expected_durations)
    expected_survival = np.exp(-expected_durations / config.imm_switch_tau_s)
    assert result[-1]["state_space_trajectory_imm_mode_stickiness_per_step"] == module._format_float_series(
        expected_survival
    )


def test_legacy_diagnostics_forward_duration_object_unchanged() -> None:
    emissions = SimpleNamespace(n_time=3, dt=0.02)
    config = SimpleNamespace(imm_switch_tau_s=0.0)
    durations = object()
    sentinel = (
        0.0,
        None,
        None,
        None,
        {"state_space_trajectory_imm_mode_stickiness_per_step": "legacy"},
    )

    def helper(
        got_emissions,
        bin_centers,
        got_config,
        transition_durations_s,
        *,
        valid_bin_mask=None,
        return_trajectory=True,
    ):
        del bin_centers, valid_bin_mask, return_trajectory
        assert got_emissions is emissions
        assert got_config is config
        assert transition_durations_s is durations
        return sentinel

    wrapped = _wrap_trajectory_diagnostics(helper, _diagnostic_module_stub())

    assert wrapped(emissions, None, config, durations) is sentinel


def _extensions_stub(result: EventScore) -> SimpleNamespace:
    return SimpleNamespace(
        copy_emissions_with_log_likelihood=lambda emissions, log_likelihood, reverse_time: emissions,
        score_replay_model_compat=lambda *args, **kwargs: result,
        _posterior_diagnostics=lambda terminal, bin_centers: {},
    )


def test_reverse_time_guard_uses_base_model_name_when_wrapper_name_is_missing() -> None:
    result = EventScore("diffusion", 0.0, 1, 0, diagnostics={})
    wrapper = SimpleNamespace(base_model=SimpleNamespace(name="diffusion"), name=None)
    emissions = SimpleNamespace(log_likelihood=np.array([[0.0]], dtype=float))

    scored = _score_reverse_with_supported_return_trajectory(
        _extensions_stub(result),
        wrapper,
        emissions,
        np.array([[0.0, 0.0]], dtype=float),
    )

    assert scored.model_name == "diffusion-reverse"


def test_reverse_time_guard_preserves_explicit_wrapper_name() -> None:
    result = EventScore("diffusion", 0.0, 1, 0, diagnostics={})
    wrapper = SimpleNamespace(base_model=SimpleNamespace(name="diffusion"), name="custom-reverse")
    emissions = SimpleNamespace(log_likelihood=np.array([[0.0]], dtype=float))

    scored = _score_reverse_with_supported_return_trajectory(
        _extensions_stub(result),
        wrapper,
        emissions,
        np.array([[0.0, 0.0]], dtype=float),
    )

    assert scored.model_name == "custom-reverse"
