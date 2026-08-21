from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from hipporeplayimm import encoding_grid_extra_columns as patch


def test_encoding_alias_synchronizers_stay_inside_package_namespace(monkeypatch) -> None:
    foreign = ModuleType("hipporeplayimm_extension")
    package_probe = ModuleType("hipporeplayimm._encoding_alias_namespace_probe")

    aliases = (
        "_make_grid",
        "_speed_cm_s",
        "_interp_positions",
        "_as_xy_array",
        "_as_position_array",
        "_validate_encoding_config",
        "_time_bin_edges",
        "_poisson_log_emissions",
    )
    originals = {name: object() for name in aliases}
    replacements = {name: object() for name in aliases}
    for name in aliases:
        setattr(foreign, name, originals[name])
        setattr(package_probe, name, originals[name])

    monkeypatch.setitem(sys.modules, foreign.__name__, foreign)
    monkeypatch.setitem(sys.modules, package_probe.__name__, package_probe)

    patch._synchronize_make_grid_aliases(
        originals["_make_grid"],
        replacements["_make_grid"],
    )
    patch._synchronize_speed_aliases(
        originals["_speed_cm_s"],
        replacements["_speed_cm_s"],
    )
    patch._synchronize_interp_position_aliases(
        originals["_interp_positions"],
        replacements["_interp_positions"],
    )

    encoding = SimpleNamespace(
        **{
            name: replacements[name]
            for name in (
                "_as_xy_array",
                "_as_position_array",
                "_validate_encoding_config",
                "_time_bin_edges",
                "_poisson_log_emissions",
            )
        }
    )
    validation_originals = {
        "as_xy_array": originals["_as_xy_array"],
        "as_position_array": originals["_as_position_array"],
        "validate_encoding_config": originals["_validate_encoding_config"],
        "time_bin_edges": originals["_time_bin_edges"],
        "poisson_log_emissions": originals["_poisson_log_emissions"],
    }
    patch._synchronize_encoding_validation_aliases(encoding, validation_originals)

    for name in aliases:
        assert getattr(foreign, name) is originals[name]
        assert getattr(package_probe, name) is replacements[name]
