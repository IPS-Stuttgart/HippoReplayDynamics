#!/usr/bin/env python3
"""Plot compact evidence and posterior summaries for one replay event."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from hipporeplayimm.trajectory_metrics import trajectory_quality_metrics


def _select_event_scores(scores: pd.DataFrame, *, session: str, event_index: int) -> pd.DataFrame:
    """Select one event without lossy integer coercion of other score rows."""

    session_scores = scores.loc[scores["session"].astype(str) == session]
    if session_scores.empty:
        return session_scores
    try:
        numeric_event_indices = pd.to_numeric(session_scores["event_index"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"event_index values for session {session!r} must be numeric") from exc
    exact_match = numeric_event_indices.eq(event_index).fillna(False)
    return session_scores.loc[exact_match]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--event-index", type=int, required=True)
    parser.add_argument("--posterior", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    scores = pd.read_csv(args.scores)
    event_scores = _select_event_scores(scores, session=args.session, event_index=args.event_index)
    if event_scores.empty:
        raise SystemExit("No score rows matched the requested event.")

    n_rows = 1 + max(1, len(args.posterior))
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3.2 * n_rows), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    values = event_scores.set_index("model")["log_evidence"].sort_values()
    axes[0].barh(values.index.astype(str), values.to_numpy(float))
    axes[0].set_title(f"Model evidence: {args.session} event {args.event_index}")
    axes[0].set_xlabel("log evidence")

    for axis_index, posterior_path in enumerate(args.posterior, start=1):
        axis = axes[axis_index]
        artifact = np.load(posterior_path)
        log_post = artifact["trajectory_log_posteriors"] if "trajectory_log_posteriors" in artifact else artifact["log_posteriors"]
        normalized = log_post - logsumexp(log_post, axis=1)[:, None]
        posterior = np.exp(normalized)
        centers = artifact["bin_centers"]
        times = artifact["times"] if "times" in artifact else np.arange(log_post.shape[0])
        metrics = trajectory_quality_metrics(log_post, centers, times)
        image = axis.imshow(posterior.T, aspect="auto", origin="lower")
        fig.colorbar(image, ax=axis, label="posterior probability")
        axis.set_title(Path(posterior_path).name + f" | linearity={metrics['trajectory_posterior_mean_linearity']:.3f}")
        axis.set_xlabel("time bin")
        axis.set_ylabel("position bin")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
