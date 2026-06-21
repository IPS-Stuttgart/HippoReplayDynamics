"""Keep Bayesian-model-average options visible to ground-truth decoding."""

from __future__ import annotations


def apply_ground_truth_bma_options_patch() -> None:
    """Make the clusterless delegate filter only clusterless-specific options."""

    from . import clusterless_ground_truth as clusterless_gt

    if getattr(clusterless_gt, "_ground_truth_bma_options_patch_applied", False):
        return

    def filter_clusterless_options(options: dict[str, object]) -> dict[str, object]:
        clusterless_names = clusterless_gt._CLUSTERLESS_KWARG_NAMES
        return {key: value for key, value in options.items() if key not in clusterless_names}

    clusterless_gt._drop_clusterless_kwargs = filter_clusterless_options
    clusterless_gt._ground_truth_bma_options_patch_applied = True
