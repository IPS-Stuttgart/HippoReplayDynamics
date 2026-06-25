from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from hipporeplayimm import shuffle_controls
from hipporeplayimm.encoding import EncodingConfig, EncodingModel, EmissionConfig, LogEmissionTensor
from hipporeplayimm.models import EventScore
from hipporeplayimm.shuffle_controls import ShuffleControlConfig, score_shuffle_controls
from hipporeplayimm.state_space_model import StateSpaceDecoderConfig, StateSpaceReplayModel


def test_score_shuffle_controls_passes_occupancy_to_state_space_models(monkeypatch) -> None:
    encoding = EncodingModel(
        x_edges=np.asarray([0.0, 1.0, 2.0]),
        y_edges=np.asarray([0.0, 1.0]),
        bin_centers=np.asarray([[0.5, 0.5], [1.5, 0.5]]),
        rates_hz=np.asarray([[0.1, 0.2]]),
        occupancy_s=np.asarray([0.0, 1.0]),
        cell_ids=np.asarray([7]),
        config=EncodingConfig(),
    )
    emissions = LogEmissionTensor(
        log_likelihood=np.asarray([[0.0, -1.0]]),
        spike_counts=np.zeros((1, 1), dtype=int),
        times=np.asarray([0.0]),
        dt=0.02,
        cell_ids=np.asarray([7]),
        n_spikes=0,
    )
    seen: dict[str, np.ndarray | None] = {"occupancy_s": None}

    def fake_build_emissions(session, control_encoding, event_index, emission_config):
        assert session.session_id == "Rat1/Open1"
        assert event_index == 3
        assert isinstance(emission_config, EmissionConfig)
        np.testing.assert_allclose(control_encoding.occupancy_s, encoding.occupancy_s)
        return emissions

    def fake_score(
        self,
        scored_emissions,
        bin_centers,
        candidate_indices=None,
        *,
        occupancy_s=None,
        return_trajectory=True,
    ):
        del candidate_indices, return_trajectory
        assert scored_emissions is emissions
        np.testing.assert_allclose(bin_centers, encoding.bin_centers)
        seen["occupancy_s"] = None if occupancy_s is None else np.asarray(occupancy_s, dtype=float)
        return EventScore(
            model_name=self.name,
            log_likelihood=-1.0,
            n_time=scored_emissions.n_time,
            n_spikes=scored_emissions.n_spikes,
        )

    monkeypatch.setattr(shuffle_controls, "build_emissions", fake_build_emissions)
    monkeypatch.setattr(StateSpaceReplayModel, "score", fake_score)

    model = StateSpaceReplayModel(
        mode="diffusion",
        config=StateSpaceDecoderConfig(
            mode="diffusion",
            valid_occupancy_threshold_s=0.5,
        ),
    )
    frame = score_shuffle_controls(
        SimpleNamespace(session_id="Rat1/Open1"),
        encoding,
        [3],
        {"diffusion": model},
        EmissionConfig(),
        ShuffleControlConfig(mode="cell-permutation", n_shuffles=1, random_seed=11),
    )

    assert len(frame) == 1
    np.testing.assert_allclose(seen["occupancy_s"], encoding.occupancy_s)
