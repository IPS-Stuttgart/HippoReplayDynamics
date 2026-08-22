from __future__ import annotations

import numpy as np

import hipporeplayimm.encoding as encoding
import hipporeplayimm.reverse_models as reverse_models
import hipporeplayimm.time_order_patch as time_order_patch


def _emissions() -> encoding.LogEmissionTensor:
    return encoding.LogEmissionTensor(
        log_likelihood=np.array([[0.0, -1.0], [-0.5, -2.0]], dtype=float),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 0.02], dtype=float),
        dt=0.02,
        cell_ids=np.array([1], dtype=int),
        n_spikes=0,
    )


def test_reverse_emissions_constructs_live_encoding_class(monkeypatch) -> None:
    live_class = encoding.LogEmissionTensor

    class StaleLogEmissionTensor(live_class):
        pass

    # Simulate importlib.reload(encoding): time_order_patch retains the class
    # object imported when that module was first loaded while encoding exposes a
    # newly defined LogEmissionTensor class.
    monkeypatch.setattr(time_order_patch, "LogEmissionTensor", StaleLogEmissionTensor)

    reversed_emissions = reverse_models.reverse_emissions(_emissions())

    assert type(reversed_emissions) is live_class
    np.testing.assert_allclose(
        reversed_emissions.log_likelihood,
        np.array([[-0.5, -2.0], [0.0, -1.0]], dtype=float),
    )
