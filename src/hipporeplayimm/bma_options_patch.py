from __future__ import annotations


def apply_bma_options_patch() -> None:
    import hipporeplayimm.clusterless_ground_truth as module

    if getattr(module, "_bma_options_patch_applied", False):
        return

    names = module._CLUSTERLESS_KWARG_NAMES

    def keep_non_clusterless_options(options: dict[str, object]) -> dict[str, object]:
        return {key: value for key, value in options.items() if key not in names}

    module._drop_clusterless_kwargs = keep_non_clusterless_options
    module._bma_options_patch_applied = True
