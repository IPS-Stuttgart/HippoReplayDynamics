from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm import state_space_displacement_imm as displacement_imm
from hipporeplayimm import state_space_displacement_momentum as displacement_momentum
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space_model import StateSpaceDecoderConfig


def _nonuniform_emissions() -> LogEmissionTensor:
    return LogEmissionTensor(
        log_likelihood=np.zeros((3, 3), dtype=float),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.array([0.01, 0.03, 0.04], dtype=float),
        dt=0.02,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
        bin_durations=np.array([0.02, 0.02, 0.01], dtype=float),
        transition_durations=np.array([0.02, 0.01], dtype=float),
    )


@pytest.mark.parametrize(
    ("module", "score_name"),
    [
        (displacement_momentum, "_score_displacement_momentum_exact"),
        (displacement_imm, "_score_displacement_imm_exact"),
    ],
)
def test_finite_displacement_models_do_not_double_scale_variable_durations(
    monkeypatch: pytest.MonkeyPatch,
    module,
    score_name: str,
) -> None:
    emissions = _nonuniform_emissions()
    config = StateSpaceDecoderConfig(
        momentum_velocity_decay=1.0,
        displacement_radius_bins=1,
        displacement_position_sigma_cm=1.0,
        displacement_transition_sigma_cm_sqrt_s=1.0,
        displacement_prior_sigma_cm=1.0,
    )
    centers = np.array([[0.0], [1.0], [2.0]], dtype=float)

    reference_dts: list[float] = []
    transition_decays: list[float] = []
    original_duration_scale_at = module._duration_scale_at
    original_displacement_transition = module._displacement_transition_matrix

    def capture_duration_scale(durations, transition_index, reference_dt):
        reference_dts.append(float(reference_dt))
        return original_duration_scale_at(durations, transition_index, reference_dt)

    def capture_displacement_transition(vectors, *, sigma_cm, decay):
        transition_decays.append(float(decay))
        return original_displacement_transition(vectors, sigma_cm=sigma_cm, decay=decay)

    monkeypatch.setattr(module, "_duration_scale_at", capture_duration_scale)
    monkeypatch.setattr(module, "_displacement_transition_matrix", capture_displacement_transition)

    score = getattr(module, score_name)
    score(
        emissions,
        centers,
        config,
        emissions.transition_durations,
        return_trajectory=False,
    )

    # The displacement lattice represents motion over the nominal emission bin.
    # Actual transition durations therefore scale the spatial shift relative to
    # emissions.dt, while the displacement AR(1) update applies only physical
    # velocity decay. Multiplying by adjacent-duration ratios as well would apply
    # variable duration twice.
    assert reference_dts == pytest.approx([0.02, 0.02])
    assert transition_decays == pytest.approx([1.0, 1.0])
