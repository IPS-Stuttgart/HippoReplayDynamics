from __future__ import annotations

import sys
from types import ModuleType

import hipporeplayimm.spike_cell_id_emission_validation as validation


def test_spike_selection_alias_sync_stays_inside_package_namespace(monkeypatch) -> None:
    def previous(session, config):
        return session, config

    def replacement(session, config):
        return session, config

    setattr(replacement, validation._ORIGINAL_ATTR, previous)

    external = ModuleType("hipporeplayimm_extension")
    external._spikes_and_cell_ids_for_encoding = previous
    package_child = ModuleType("hipporeplayimm._spike_selection_alias_probe")
    package_child._spikes_and_cell_ids_for_encoding = previous

    monkeypatch.setitem(sys.modules, external.__name__, external)
    monkeypatch.setitem(sys.modules, package_child.__name__, package_child)

    validation._synchronize_spike_selection_aliases(replacement)

    assert external._spikes_and_cell_ids_for_encoding is previous
    assert package_child._spikes_and_cell_ids_for_encoding is replacement
