"""Derive pruned state-space candidate supports on active occupancy masks."""

from __future__ import annotations

import inspect

import numpy as np


def apply_occupancy_candidate_support_patch() -> None:
    """Keep benchmark, recovery, and post-hoc decode beams occupancy-aware."""

    from . import benchmarks as benchmark_module
    from . import ground_truth as ground_truth_module
    from . import ground_truth_candidate_support as ground_truth_candidate_module
    from . import simulation_recovery as recovery_module

    benchmark_module._call_candidate_indices = _call_candidate_indices
    benchmark_module._candidate_indices_for_model = _candidate_indices_for_model
    benchmark_module._score_train_joint_model = _score_train_joint_model
    benchmark_module._occupancy_candidate_support_patch_applied = True

    ground_truth_candidate_module._candidate_indices_for_model = _candidate_indices_for_model
    ground_truth_candidate_module._score_joint_for_ground_truth = _score_joint_for_ground_truth
    ground_truth_module._score_joint_for_ground_truth = _score_joint_for_ground_truth

    recovery_module._call_candidate_indices = _call_candidate_indices
    recovery_module._candidate_indices_for_model = _candidate_indices_for_model
    recovery_module._score_recovery_model = _score_recovery_model


def _candidate_indices_for_model(
    model: object,
    emissions,
    bin_centers: np.ndarray,
    *,
    occupancy_s: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Return candidate support, honoring active state-space occupancy masks."""

    valid_bin_mask = _valid_bin_mask_for_candidate_support(
        model,
        occupancy_s,
        _candidate_support_n_bins(emissions, bin_centers),
    )
    return _call_candidate_indices(
        model.candidate_indices,  # type: ignore[attr-defined]
        emissions,
        bin_centers,
        valid_bin_mask=valid_bin_mask,
    )


def _candidate_support_n_bins(emissions, bin_centers: np.ndarray) -> int:
    n_bins = getattr(emissions, "n_bins", None)
    if n_bins is not None:
        return int(n_bins)
    return int(np.asarray(bin_centers).shape[0])


def _is_state_space_model(model: object) -> bool:
    from .state_space import StateSpaceReplayModel

    return isinstance(model, StateSpaceReplayModel)


def _valid_bin_mask_for_candidate_support(
    model: object,
    occupancy_s: np.ndarray | None,
    n_bins: int,
) -> np.ndarray | None:
    if occupancy_s is None or not _is_state_space_model(model):
        return None
    config = getattr(model, "config", None)
    if config is None:
        return None
    from .state_space import _valid_bin_mask_from_occupancy

    return _valid_bin_mask_from_occupancy(
        occupancy_s,
        float(config.valid_occupancy_threshold_s),
        int(n_bins),
    )


def _call_candidate_indices(
    candidate_indices,
    emissions,
    bin_centers: np.ndarray,
    *,
    valid_bin_mask: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Call candidate_indices without swallowing implementation TypeErrors."""

    try:
        signature = inspect.signature(candidate_indices)
    except (TypeError, ValueError):
        kwargs = {"valid_bin_mask": valid_bin_mask} if valid_bin_mask is not None else {}
        return candidate_indices(emissions, bin_centers, **kwargs)

    parameters = tuple(signature.parameters.values())
    call_kwargs = _candidate_call_kwargs(signature, valid_bin_mask)
    if any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters):
        return candidate_indices(emissions, bin_centers, **call_kwargs)

    positional = tuple(
        parameter
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    if len(positional) >= 2:
        return candidate_indices(emissions, bin_centers, **call_kwargs)
    if "bin_centers" in signature.parameters:
        return candidate_indices(emissions, bin_centers=bin_centers, **call_kwargs)
    if "centers" in signature.parameters:
        return candidate_indices(emissions, centers=bin_centers, **call_kwargs)
    return candidate_indices(emissions, **call_kwargs)


def _candidate_call_kwargs(
    signature: inspect.Signature,
    valid_bin_mask: np.ndarray | None,
) -> dict[str, np.ndarray]:
    if valid_bin_mask is None:
        return {}
    parameter = signature.parameters.get("valid_bin_mask")
    accepts_keyword = parameter is not None and parameter.kind in (
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    accepts_var_keyword = any(
        current.kind == inspect.Parameter.VAR_KEYWORD
        for current in signature.parameters.values()
    )
    if accepts_keyword or accepts_var_keyword:
        return {"valid_bin_mask": np.asarray(valid_bin_mask, dtype=bool)}
    return {}


def _score_train_joint_model(
    model,
    train_emissions,
    joint_emissions,
    bin_centers,
    occupancy_s=None,
):
    from . import benchmarks as benchmark_module

    if _is_state_space_model(model):
        candidates = (
            _candidate_indices_for_model(
                model,
                train_emissions,
                bin_centers,
                occupancy_s=occupancy_s,
            )
            if benchmark_module._state_space_uses_candidate_support(model)
            else None
        )
        train_score = benchmark_module._score_state_space_model(
            model,
            train_emissions,
            bin_centers,
            candidates,
            occupancy_s,
        )
        joint_score = benchmark_module._score_state_space_model(
            model,
            joint_emissions,
            bin_centers,
            candidates,
            occupancy_s,
        )
        return train_score, joint_score
    if hasattr(model, "candidate_indices"):
        candidates = _candidate_indices_for_model(model, train_emissions, bin_centers)
        train_score = model.score(train_emissions, bin_centers, candidate_indices=candidates)
        joint_score = model.score(joint_emissions, bin_centers, candidate_indices=candidates)
        return train_score, joint_score
    return model.score(train_emissions, bin_centers), model.score(joint_emissions, bin_centers)


def _score_joint_for_ground_truth(
    model,
    train_emissions,
    joint_emissions,
    bin_centers: np.ndarray,
    *,
    occupancy_s: np.ndarray | None = None,
):
    from . import benchmarks as benchmark_module
    from . import ground_truth_candidate_support as ground_truth_candidate_module

    if _is_state_space_model(model):
        candidates = (
            _candidate_indices_for_model(
                model,
                train_emissions,
                bin_centers,
                occupancy_s=occupancy_s,
            )
            if benchmark_module._state_space_uses_candidate_support(model)
            else None
        )
        return ground_truth_candidate_module._score_state_space_joint_for_ground_truth(
            model,
            joint_emissions,
            bin_centers,
            candidates,
            occupancy_s,
        )
    if hasattr(model, "candidate_indices"):
        candidates = _candidate_indices_for_model(model, train_emissions, bin_centers)
        return model.score(joint_emissions, bin_centers, candidate_indices=candidates)
    return model.score(joint_emissions, bin_centers)


def _score_recovery_model(
    model: object,
    emissions,
    encoding,
    *,
    candidate_indices: list[np.ndarray] | None = None,
    score_with_occupancy: bool = True,
) -> object:
    """Score one synthetic event with occupancy-aware candidate support."""

    from . import simulation_recovery as recovery_module

    if isinstance(model, recovery_module.SortedSpikeStateSpaceReplayModel):
        kwargs: dict[str, object] = {}
        if model.mode == "momentum-exact-sparse":
            kwargs["return_trajectory"] = False
        if score_with_occupancy and model.mode in {"momentum", "imm"}:
            occupancy_candidates = _candidate_indices_for_model(
                model,
                emissions,
                encoding.bin_centers,
                occupancy_s=encoding.occupancy_s,
            )
            candidate_indices = (
                occupancy_candidates
                if candidate_indices is None
                else _union_candidate_indices(candidate_indices, occupancy_candidates)
            )
        if candidate_indices is not None:
            kwargs["candidate_indices"] = candidate_indices
        if score_with_occupancy:
            kwargs["occupancy_s"] = encoding.occupancy_s
        return model.score(emissions, encoding.bin_centers, **kwargs)
    if candidate_indices is not None:
        return model.score(  # type: ignore[attr-defined]
            emissions,
            encoding.bin_centers,
            candidate_indices=candidate_indices,
        )
    return model.score(emissions, encoding.bin_centers)  # type: ignore[attr-defined]


def _union_candidate_indices(
    left: list[np.ndarray],
    right: list[np.ndarray],
) -> list[np.ndarray]:
    if len(left) != len(right):
        raise ValueError("candidate support lists must have matching lengths")
    return [
        np.unique(
            np.concatenate(
                [
                    np.asarray(left_current, dtype=int),
                    np.asarray(right_current, dtype=int),
                ]
            )
        )
        for left_current, right_current in zip(left, right, strict=True)
    ]
