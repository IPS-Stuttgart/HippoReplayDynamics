"""Accept coordinate arrays with extra columns in encoding grid construction."""

from __future__ import annotations

import sys


def apply_encoding_grid_extra_columns_patch() -> None:
    """Install a grid builder that uses only the x/y coordinate columns."""

    from . import encoding

    if getattr(encoding, "_grid_extra_columns_patch_applied", False):
        return

    original_make_grid = encoding._make_grid

    def make_grid(xy, config):
        arr = encoding._as_xy_array(xy, name="xy")
        return original_make_grid(arr[:, :2], config)

    make_grid.__name__ = original_make_grid.__name__
    make_grid.__doc__ = original_make_grid.__doc__
    make_grid.__module__ = original_make_grid.__module__
    encoding._make_grid = make_grid
    _synchronize_make_grid_aliases(original_make_grid, make_grid)
    encoding._grid_extra_columns_patch_applied = True


def _synchronize_make_grid_aliases(original_make_grid, replacement_make_grid) -> None:
    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        if getattr(module, "_make_grid", None) is original_make_grid:
            module._make_grid = replacement_make_grid
