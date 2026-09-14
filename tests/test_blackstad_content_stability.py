from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.prepare_blackstad_content_stability import convert


def test_native_names_preserve_identity_without_inventing_cell_types():
    raw = SimpleNamespace(animal="24101", session="session", path=Path("/fixture"),
        position_times_s=np.array([0., .02, .04]), position_cm=np.array([[1., 2], [2, 3], [3, 4]]),
        valid_position=np.array([True, False, True]),
        spikes_by_unit={"TT2": np.array([.02]), "TT1": np.array([.01, .03])},
        position_transform={"method": "native"})
    session, names = convert(raw)
    assert names == ["TT1", "TT2"]
    assert session.spikes[:, 1].tolist() == [1, 2, 1]
    assert np.isnan(session.position[1, 1:]).all()
    assert not len(session.excitatory_neurons)
    assert not len(session.inhibitory_neurons)
    assert session.metadata["source_dataset"] == "blackstad_moser"
    assert session.metadata["cell_type_source"] == "untyped_sorted_units"
    assert session.rat == "24101"
