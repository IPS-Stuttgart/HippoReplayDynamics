from __future__ import annotations

import numpy as np

from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.result_improvement_extensions import copy_emissions_with_log_likelihood
from hipporeplayimm.reverse_models import reverse_emissions


def _emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.array([[0.0, -1.0], [-2.0, -0.5], [-0.25, -3.0]], dtype=float),
        spike_counts=np.array([[0], [2], [1]], dtype=int),
        times=np.array([0.005, 0.020, 0.055], dtype=float),
        dt=0.02,
        cell_ids=np.array([7], dtype=int),
        n_spikes=3,
        bin_durations=np.array([0.010, 0.020, 0.030], dtype=float),
        transition_durations=np.array([0.015, 0.035], dtype=float),
        metadata={"source": "unit-test"},
    )


def test_reverse_emission_helpers_keep_time_coordinates_increasing() -> None:
    emissions = _emissions()
    copied = copy_emissions_with_log_likelihood(
        emissions,
        emissions.log_likelihood,
        reverse_time=True,
    )
    reversed_emissions = reverse_emissions(emissions)

    for output in (copied, reversed_emissions):
        np.testing.assert_allclose(output.log_likelihood, emissions.log_likelihood[::-1])
        np.testing.assert_allclose(output.bin_durations, emissions.bin_durations[::-1])
        np.testing.assert_allclose(output.transition_durations, emissions.transition_durations[::-1])
        np.testing.assert_allclose(output.times, emissions.times)
        assert np.all(np.diff(output.times) > 0.0)


def test_reverse_emission_time_patch_repairs_partial_reverse_model_state(monkeypatch) -> None:
    import hipporeplayimm.result_improvement_extensions as improved
    import hipporeplayimm.reverse_models as reverse_models
    from hipporeplayimm.time_order_patch import apply_reverse_emission_time_patch

    def stale_reverse_emissions(emissions: LogEmissionTensor) -> LogEmissionTensor:
        return LogEmissionTensor(
            log_likelihood=np.asarray(emissions.log_likelihood, dtype=float)[::-1].copy(),
            spike_counts=np.asarray(emissions.spike_counts)[::-1].copy(),
            times=np.asarray(emissions.times, dtype=float)[::-1].copy(),
            dt=emissions.dt,
            cell_ids=np.asarray(emissions.cell_ids).copy(),
            n_spikes=int(emissions.n_spikes),
            bin_durations=np.asarray(emissions.bin_durations, dtype=float)[::-1].copy(),
            transition_durations=np.asarray(emissions.transition_durations, dtype=float)[::-1].copy(),
            metadata=dict(getattr(emissions, "metadata", {}) or {}),
        )

    emissions = _emissions()
    monkeypatch.setattr(improved, "_time_order_patch_applied", True, raising=False)
    monkeypatch.setattr(reverse_models, "_time_order_patch_applied", False, raising=False)
    monkeypatch.setattr(reverse_models, "reverse_emissions", stale_reverse_emissions)

    apply_reverse_emission_time_patch()

    output = reverse_models.reverse_emissions(emissions)
    np.testing.assert_allclose(output.log_likelihood, emissions.log_likelihood[::-1])
    np.testing.assert_allclose(output.times, emissions.times)
    np.testing.assert_allclose(output.transition_durations, emissions.transition_durations[::-1])
    assert np.all(np.diff(output.times) > 0.0)
    assert getattr(improved, "_time_order_patch_applied") is True
    assert getattr(reverse_models, "_time_order_patch_applied") is True


def test_reverse_emission_time_patch_refreshes_stale_true_flags(monkeypatch) -> None:
    import hipporeplayimm.result_improvement_extensions as improved
    import hipporeplayimm.reverse_models as reverse_models
    from hipporeplayimm.time_order_patch import apply_reverse_emission_time_patch

    def stale_copy_emissions_with_log_likelihood(
        emissions: LogEmissionTensor,
        log_likelihood: np.ndarray,
        *,
        reverse_time: bool = False,
    ) -> LogEmissionTensor:
        likelihood = np.asarray(log_likelihood, dtype=float)
        counts = np.asarray(emissions.spike_counts)
        times = np.asarray(emissions.times, dtype=float)
        bin_durations = np.asarray(emissions.bin_durations, dtype=float)
        transition_durations = np.asarray(emissions.transition_durations, dtype=float)
        if reverse_time:
            likelihood = likelihood[::-1].copy()
            counts = counts[::-1].copy()
            times = times[::-1].copy()
            bin_durations = bin_durations[::-1].copy()
            transition_durations = transition_durations[::-1].copy()
        return LogEmissionTensor(
            log_likelihood=likelihood.copy(),
            spike_counts=counts.copy(),
            times=times.copy(),
            dt=emissions.dt,
            cell_ids=np.asarray(emissions.cell_ids).copy(),
            n_spikes=int(emissions.n_spikes),
            bin_durations=bin_durations.copy(),
            transition_durations=transition_durations.copy(),
            metadata=dict(getattr(emissions, "metadata", {}) or {}),
        )

    def stale_reverse_emissions(emissions: LogEmissionTensor) -> LogEmissionTensor:
        return stale_copy_emissions_with_log_likelihood(
            emissions,
            emissions.log_likelihood,
            reverse_time=True,
        )

    emissions = _emissions()
    monkeypatch.setattr(improved, "_time_order_patch_applied", True, raising=False)
    monkeypatch.setattr(reverse_models, "_time_order_patch_applied", True, raising=False)
    monkeypatch.setattr(improved, "copy_emissions_with_log_likelihood", stale_copy_emissions_with_log_likelihood)
    monkeypatch.setattr(reverse_models, "reverse_emissions", stale_reverse_emissions)

    apply_reverse_emission_time_patch()

    copied = improved.copy_emissions_with_log_likelihood(
        emissions,
        emissions.log_likelihood,
        reverse_time=True,
    )
    reversed_emissions = reverse_models.reverse_emissions(emissions)
    for output in (copied, reversed_emissions):
        np.testing.assert_allclose(output.log_likelihood, emissions.log_likelihood[::-1])
        np.testing.assert_allclose(output.times, emissions.times)
        np.testing.assert_allclose(output.transition_durations, emissions.transition_durations[::-1])
        assert np.all(np.diff(output.times) > 0.0)
    assert getattr(improved.copy_emissions_with_log_likelihood, "_time_order_patch_wrapped", False) is True
    assert getattr(reverse_models.reverse_emissions, "_time_order_patch_wrapped", False) is True
