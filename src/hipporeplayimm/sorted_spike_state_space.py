"""Sorted-spike state-space replay decoder labels.

The current state-space replay baseline consumes Poisson emissions built from
sorted spike identities. This wrapper makes that observation model explicit in
model names and diagnostics so it is not confused with a true clusterless marked
point-process decoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .encoding import LogEmissionTensor
from .models import EventScore
from .state_space import StateSpaceDecoderConfig, StateSpaceReplayModel


@dataclass
class SortedSpikeStateSpaceReplayModel(StateSpaceReplayModel):
    """State-space replay model using sorted-spike Poisson emissions."""

    mode: str = "diffusion"
    config: StateSpaceDecoderConfig | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.name is None or self.name.startswith("state-space-"):
            self.name = f"sorted-spike-state-space-{self.mode}"

    def score(
        self,
        emissions: LogEmissionTensor,
        bin_centers: np.ndarray,
        candidate_indices: list[np.ndarray] | None = None,
        *,
        occupancy_s: np.ndarray | None = None,
    ) -> EventScore:
        score = super().score(
            emissions,
            bin_centers,
            candidate_indices=candidate_indices,
            occupancy_s=occupancy_s,
        )
        score.model_name = str(self.name)
        score.diagnostics["state_space_observation_model"] = "sorted-spike-poisson"
        score.diagnostics["clusterless_mark_likelihood"] = "not_implemented"
        return score
