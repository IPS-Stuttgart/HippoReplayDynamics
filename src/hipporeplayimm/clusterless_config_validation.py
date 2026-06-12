"""Runtime validation for clusterless nested encoding configuration."""

from __future__ import annotations

from functools import wraps
import sys

from .encoding import EncodingConfig, _validate_encoding_config

_PATCH_MARKER = "_clusterless_encoding_config_validation_patch"


def apply_clusterless_encoding_config_validation_patch() -> None:
    """Validate ClusterlessMarkConfig.encoding before fitting clusterless marks."""

    import hipporeplayimm.clusterless as clusterless

    current = clusterless.fit_clusterless_mark_encoding
    if getattr(current, _PATCH_MARKER, False):
        previous = getattr(current, "__wrapped__", None)
        if previous is not None:
            _synchronize_aliases(previous, current)
        return

    previous = current

    @wraps(previous)
    def fit_clusterless_mark_encoding(session, config=None):
        _validate_nested_encoding_config(config)
        return previous(session, config)

    setattr(fit_clusterless_mark_encoding, _PATCH_MARKER, True)
    clusterless.fit_clusterless_mark_encoding = fit_clusterless_mark_encoding
    _synchronize_aliases(previous, fit_clusterless_mark_encoding)


def _validate_nested_encoding_config(config: object | None) -> None:
    encoding_config = EncodingConfig() if config is None else getattr(config, "encoding", None)
    if encoding_config is None:
        encoding_config = EncodingConfig()
    _validate_encoding_config(encoding_config)


def _synchronize_aliases(previous: object, patched: object) -> None:
    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        if getattr(module, "fit_clusterless_mark_encoding", None) is previous:
            module.fit_clusterless_mark_encoding = patched
