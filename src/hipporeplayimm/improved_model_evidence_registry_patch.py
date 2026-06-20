"""Runtime registry parity shim for the improved model-evidence script.

The improved script is also used as a standalone script.  When that script imports
``hipporeplayimm`` modules during startup, this shim installs a short-lived trace
hook and patches the script-level ``_models`` registry after it has been defined
but before ``main()`` is called.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import sys
from types import FrameType
from typing import Any

_PATCHED_FLAG = "_hipporeplayimm_improved_registry_patched"
_EXTRA_MODEL_NAMES = {
    "sorted-spike-state-space-trajectory-imm-anchored-exact-sparse",
    "sorted-spike-state-space-trajectory-imm-low-leak-exact-sparse",
    "sorted-spike-state-space-trajectory-imm-persistent-exact-sparse",
    "sorted-spike-state-space-displacement-momentum",
    "clusterless-state-space-displacement-momentum",
}
_EXTRA_TRAJECTORY_NAMES = set(_EXTRA_MODEL_NAMES)


def apply_improved_model_evidence_registry_patch() -> None:
    """Patch ``benchmark_model_evidence_improved.py`` when it is loading.

    The hook is installed only if that script is already on the call stack.  The
    already-running script frame also receives ``f_trace`` explicitly; otherwise
    ``sys.settrace`` would only affect future frames and could miss the remainder
    of the module body.
    """

    script_frames = _improved_script_frames_on_stack()
    if not script_frames:
        return
    previous_trace = sys.gettrace()

    def trace(frame: FrameType, event: str, arg: object):  # noqa: ANN001
        if event == "line" and _patch_frame_if_ready(frame):
            sys.settrace(previous_trace)
            if previous_trace is not None:
                return previous_trace(frame, event, arg)
            return None
        return trace

    for frame in script_frames:
        frame.f_trace = trace
    sys.settrace(trace)


def _improved_script_frames_on_stack() -> list[FrameType]:
    frames: list[FrameType] = []
    frame = sys._getframe()
    while frame is not None:
        if Path(str(frame.f_globals.get("__file__", ""))).name == "benchmark_model_evidence_improved.py":
            frames.append(frame)
        frame = frame.f_back
    return frames


def _patch_frame_if_ready(frame: FrameType) -> bool:
    globals_dict = frame.f_globals
    if Path(str(globals_dict.get("__file__", ""))).name != "benchmark_model_evidence_improved.py":
        return False
    if globals_dict.get(_PATCHED_FLAG):
        return True
    if "_models" not in globals_dict or "_state_space_config" not in globals_dict:
        return False
    _install_patch(globals_dict)
    return True


def _install_patch(globals_dict: dict[str, Any]) -> None:
    original_models = globals_dict["_models"]
    trajectory_set = globals_dict.get("_TRAJECTORY_MODELS")
    if isinstance(trajectory_set, set):
        trajectory_set.update(_EXTRA_TRAJECTORY_NAMES)

    def patched_models(args, session, encoding=None):  # noqa: ANN001
        requested = _requested_model_names(args, globals_dict.get("_ALIASES", {}))
        if not any(name in _EXTRA_MODEL_NAMES for name in requested):
            return original_models(args, session, encoding=encoding)

        base_names = [name for name in requested if name not in _EXTRA_MODEL_NAMES]
        base_models: dict[str, object] = {}
        if base_names:
            base_args = copy.copy(args)
            base_args.models = " ".join(base_names)
            base_models = original_models(base_args, session, encoding=encoding)

        out: dict[str, object] = {}
        for name in requested:
            if name in out:
                continue
            if name in base_models:
                out[name] = base_models[name]
            elif name in _EXTRA_MODEL_NAMES:
                out[name] = _build_extra_model(name, args, globals_dict)
            else:
                # Preserve the original error message for genuinely unknown names.
                return original_models(args, session, encoding=encoding)
        return out

    globals_dict["_models"] = patched_models
    globals_dict[_PATCHED_FLAG] = True


def _requested_model_names(args, aliases: object) -> list[str]:  # noqa: ANN001
    alias_map = aliases if isinstance(aliases, dict) else {}
    names: list[str] = []
    for raw in str(args.models).replace(",", " ").split():
        name = alias_map.get(raw.strip().lower(), raw.strip().lower())
        if name:
            names.append(name)
    if getattr(args, "include_clusterless_defaults", False):
        names.extend(
            [
                "clusterless-state-space-diffusion",
                "clusterless-state-space-momentum",
                "clusterless-state-space-imm",
            ]
        )
    return list(dict.fromkeys(names))


def _build_extra_model(name: str, args, globals_dict: dict[str, Any]) -> object:  # noqa: ANN001
    state_config = globals_dict["_state_space_config"]
    sorted_model_cls = globals_dict["SortedSpikeStateSpaceReplayModel"]
    clusterless_model_cls = globals_dict["ClusterlessStateSpaceReplayModel"]

    if name == "sorted-spike-state-space-trajectory-imm-anchored-exact-sparse":
        config = replace(
            state_config(args, "trajectory-imm-exact-sparse"),
            trajectory_imm_momentum_initial_probability=0.05,
            trajectory_imm_momentum_switch_probability=0.005,
        )
        return sorted_model_cls("trajectory-imm-exact-sparse", config=config, name=name)

    if name == "sorted-spike-state-space-trajectory-imm-low-leak-exact-sparse":
        config = replace(
            state_config(args, "trajectory-imm-exact-sparse"),
            trajectory_imm_momentum_initial_probability=0.01,
            trajectory_imm_momentum_switch_probability=0.001,
        )
        return sorted_model_cls("trajectory-imm-exact-sparse", config=config, name=name)

    if name == "sorted-spike-state-space-trajectory-imm-persistent-exact-sparse":
        config = replace(
            state_config(args, "trajectory-imm-exact-sparse"),
            trajectory_imm_mode_stickiness=0.985,
        )
        return sorted_model_cls("trajectory-imm-exact-sparse", config=config, name=name)

    if name == "sorted-spike-state-space-displacement-momentum":
        return sorted_model_cls(
            "displacement-momentum",
            config=state_config(args, "displacement-momentum"),
            name=name,
        )

    if name == "clusterless-state-space-displacement-momentum":
        return clusterless_model_cls(
            mode="displacement-momentum",
            config=state_config(args, "displacement-momentum"),
            mark_likelihood=args.clusterless_mark_likelihood,
        )

    raise ValueError(f"unknown improved model-evidence extra model: {name}")
