import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from topological_replay_comparator import (  # noqa: E402
    COMPARISON_OUTPUT,
    EUCLIDEAN_DIFFUSION_MODEL,
    EVIDENCE_OUTPUT,
    GATE_OUTPUT,
    TOPO_GEODESIC_MODEL,
    TOPO_GRID_WALK_MODEL,
    TOPO_VALID_DIFFUSION_MODEL,
    geodesic_transition_matrix,
    write_topological_replay_outputs,
)


def test_geodesic_transition_respects_occupied_graph_barrier():
    grid_shape = (3, 3)
    centers = np.asarray(
        [[x, y] for x in range(grid_shape[0]) for y in range(grid_shape[1])],
        dtype=float,
    )
    valid_mask = np.ones(9, dtype=bool)
    valid_mask[4] = False

    transition, stats = geodesic_transition_matrix(
        grid_shape,
        valid_mask,
        centers,
        sigma_cm=10.0,
        max_distance_sigma=10.0,
        diagonal_neighbors=False,
    )

    assert transition.shape == (8, 8)
    assert np.allclose(np.asarray(transition.sum(axis=0)).ravel(), 1.0)
    assert int(stats["topological_graph_components"]) == 1
    assert int(stats["topological_graph_edges"]) == 8

    compact = {flat: idx for idx, flat in enumerate(np.flatnonzero(valid_mask))}
    left = compact[3]
    right = compact[5]
    straight_neighbor_probability = transition[right, left]
    around_barrier_probability = transition[compact[0], left]
    assert straight_neighbor_probability < around_barrier_probability


def test_topological_outputs_summarize_pairing_and_nonrequired_science_gates(tmp_path):
    evidence = pd.DataFrame(
        [
            _row("Rat1/Open1", 0, EUCLIDEAN_DIFFUSION_MODEL, 10.0),
            _row("Rat1/Open1", 0, TOPO_VALID_DIFFUSION_MODEL, 11.0, valid_fraction=0.7),
            _row("Rat1/Open1", 0, TOPO_GRID_WALK_MODEL, 9.0, valid_fraction=0.7),
            _row("Rat1/Open1", 0, TOPO_GEODESIC_MODEL, 18.0, valid_fraction=0.7),
            _row("Rat2/Open1", 1, EUCLIDEAN_DIFFUSION_MODEL, 6.0),
            _row("Rat2/Open1", 1, TOPO_VALID_DIFFUSION_MODEL, 5.0, valid_fraction=0.6),
            _row("Rat2/Open1", 1, TOPO_GRID_WALK_MODEL, 7.0, valid_fraction=0.6),
            _row("Rat2/Open1", 1, TOPO_GEODESIC_MODEL, 5.5, valid_fraction=0.6),
            _row("Rat3/Open1", 2, EUCLIDEAN_DIFFUSION_MODEL, 6.0),
            _row(
                "Rat3/Open1",
                2,
                TOPO_GEODESIC_MODEL,
                100.0,
                valid_fraction=0.6,
                evidence_comparable="False",
            ),
        ]
    )

    outputs = write_topological_replay_outputs(evidence, tmp_path, margin_threshold=5.5)

    assert set(outputs) == {EVIDENCE_OUTPUT, COMPARISON_OUTPUT, GATE_OUTPUT}
    summary = outputs[COMPARISON_OUTPUT]
    geodesic = summary[summary["topological_model"].eq(TOPO_GEODESIC_MODEL)].iloc[0]
    assert int(geodesic["paired_events"]) == 2
    assert int(geodesic["topological_wins"]) == 1
    assert int(geodesic["confident_topological_wins"]) == 1
    assert geodesic["median_delta_log_evidence"] == pytest.approx(3.75)

    best = summary[summary["topological_model"].eq("best_topological")].iloc[0]
    assert int(best["paired_events"]) == 2
    assert int(best["topological_wins"]) == 2

    gates = outputs[GATE_OUTPUT]
    overall = gates[gates["gate"].eq("overall")].iloc[0]
    assert bool(overall["passed"])
    diagnostic = gates[gates["gate"].eq("topological_geodesic_beats_euclidean_majority")].iloc[0]
    assert not bool(diagnostic["required_for_overall"])
    assert not bool(diagnostic["passed"])

    for filename in outputs:
        path = tmp_path / filename
        assert path.exists()
        assert path.stat().st_size > 0


def _row(
    session: str,
    event_index: int,
    model: str,
    log_evidence: float,
    *,
    valid_fraction: float | None = None,
    evidence_comparable: object = True,
) -> dict[str, object]:
    row: dict[str, object] = {
        "status": "success",
        "session": session,
        "event_index": event_index,
        "model": model,
        "requested_model": model,
        "model_family": "trajectory",
        "log_evidence": log_evidence,
        "n_time": 10,
        "n_spikes": 12,
        "runtime_s": 0.01,
        "evidence_comparable": evidence_comparable,
        "evidence_support": "exact_full_grid",
    }
    if valid_fraction is not None:
        row["diagnostic_valid_state_fraction"] = valid_fraction
    return row
