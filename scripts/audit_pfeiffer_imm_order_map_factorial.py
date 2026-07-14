#!/usr/bin/env python3
"""Complete the Pfeiffer/Foster IMM order-by-map factorial on frozen events."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from hipporeplayimm.data import load_replay_session
from hipporeplayimm.encoding import EmissionConfig, EncodingConfig, fit_place_field_encoding
from hipporeplayimm.result_improvement_extensions import (
    ReplayEmissionCalibration,
    build_sorted_emissions_with_replay_calibration,
)

try:
    from scripts.audit_pfeiffer_imm_map_specific_posterior_content import (
        _stable_seed,
        permute_encoding,
    )
    from scripts.clean_imm_time_order_shuffle_control import (
        FRAGMENTED,
        FIRST_ORDER_IMM,
        EventKey,
        _score_emissions,
        _scoring_models,
        permute_emission_time_bins,
    )
except ModuleNotFoundError:
    from audit_pfeiffer_imm_map_specific_posterior_content import (
        _stable_seed,
        permute_encoding,
    )
    from clean_imm_time_order_shuffle_control import (
        FRAGMENTED,
        FIRST_ORDER_IMM,
        EventKey,
        _score_emissions,
        _scoring_models,
        permute_emission_time_bins,
    )


MAP_CONDITIONS = ("real_map", "population_code_permuted")
ORDER_CONDITIONS = ("original", "shuffled")
SCORE_COLUMNS = [
    "status",
    "failure_reason",
    "session",
    "rat",
    "event_index",
    "event_group",
    "map_condition",
    "order_condition",
    "shuffle_index",
    "model",
    "log_evidence",
    "duration_ms",
    "n_time",
    "n_spikes",
    "n_active_units",
    "runtime_s",
]


def _event_seed(seed: int, session: str, event_index: int) -> int:
    payload = f"{seed}:{session}:{event_index}:order-map".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def read_selection(path: str | Path) -> pd.DataFrame:
    """Read a frozen event selection, retaining only clean-IMM rows when labeled."""

    frame = pd.read_csv(path)
    required = {"session", "event_index"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"selection is missing required columns: {missing}")
    if "event_group" in frame:
        frame = frame[frame["event_group"].astype(str).eq("clean_imm")].copy()
    frame["session"] = frame["session"].astype(str)
    frame["event_index"] = pd.to_numeric(frame["event_index"], errors="raise").astype(int)
    if "rat" not in frame:
        frame["rat"] = frame["session"].str.split("/").str[0]
    frame["event_group"] = "clean_imm"
    return frame[["session", "rat", "event_index", "event_group"]].drop_duplicates().reset_index(drop=True)


def _score_session(task: dict[str, object]) -> list[dict[str, object]]:
    args = SimpleNamespace(**task["parameters"])
    session_id = str(task["session"])
    session = load_replay_session(Path(str(task["dataset_root"])) / session_id)
    real_encoding = fit_place_field_encoding(
        session,
        EncodingConfig(
            bin_size_cm=args.bin_size_cm,
            smoothing_sigma_bins=args.smoothing_sigma_bins,
            min_speed_cm_s=args.min_speed_cm_s,
            min_occupancy_s=args.min_occupancy_s,
            rate_floor_hz=args.rate_floor_hz,
        ),
    )
    encodings = {
        "real_map": real_encoding,
        "population_code_permuted": permute_encoding(
            real_encoding,
            condition="population_code_permuted",
            seed=_stable_seed(args.seed, session_id, "population_code_permuted"),
        ),
    }
    emission_config = EmissionConfig(
        time_bin_s=args.time_bin_s,
        spike_rate_scale=args.spike_rate_scale,
        likelihood_temperature=args.emission_likelihood_temperature,
        negative_binomial_overdispersion=args.emission_negative_binomial_overdispersion,
    )
    calibration = ReplayEmissionCalibration(
        gain_mode=args.replay_gain_mode,
        gain_prior_count=args.replay_gain_prior_count,
        max_gain=args.replay_gain_max_gain,
        emission_model=args.sorted_spike_emission_model,
        negative_binomial_dispersion=args.negative_binomial_dispersion,
    )
    models = _scoring_models(args)
    rows: list[dict[str, object]] = []
    for selected in task["events"]:
        event_index = int(selected["event_index"])
        ripple = session.ripple(event_index)
        window = SimpleNamespace(start=float(ripple.start), end=float(ripple.end))
        emissions = {
            condition: build_sorted_emissions_with_replay_calibration(
                session,
                encoding,
                window,
                emission_config,
                calibration=calibration,
            )
            for condition, encoding in encodings.items()
        }
        n_times = {condition: value.n_time for condition, value in emissions.items()}
        if len(set(n_times.values())) != 1:
            raise RuntimeError(f"map conditions changed the time grid for {session_id} event {event_index}")
        key = EventKey(
            session=session_id,
            rat=str(selected["rat"]),
            event_index=event_index,
            event_group="clean_imm",
        )
        for condition in MAP_CONDITIONS:
            scored = _score_emissions(
                models,
                emissions[condition],
                encodings[condition],
                key,
                score_kind="original",
                shuffle_index=-1,
            )
            rows.extend(_label_scores(scored, condition, "original"))
        rng = np.random.default_rng(_event_seed(args.seed, session_id, event_index))
        for shuffle_index in range(int(args.n_shuffles)):
            permutation = rng.permutation(next(iter(n_times.values())))
            for condition in MAP_CONDITIONS:
                shuffled = permute_emission_time_bins(emissions[condition], permutation)
                scored = _score_emissions(
                    models,
                    shuffled,
                    encodings[condition],
                    key,
                    score_kind="shuffle",
                    shuffle_index=shuffle_index,
                )
                rows.extend(_label_scores(scored, condition, "shuffled"))
    return rows


def _label_scores(
    rows: list[dict[str, object]],
    map_condition: str,
    order_condition: str,
) -> list[dict[str, object]]:
    return [
        {
            **row,
            "map_condition": map_condition,
            "order_condition": order_condition,
        }
        for row in rows
    ]


def build_event_decisions(scores: pd.DataFrame, *, expected_shuffles: int) -> pd.DataFrame:
    """Compute both order effects and their map-by-order interaction per event."""

    required = {
        "session",
        "event_index",
        "map_condition",
        "order_condition",
        "shuffle_index",
        "model",
        "log_evidence",
    }
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"factorial scores are missing required columns: {missing}")
    success = scores.copy()
    if "status" in success:
        success = success[success["status"].astype(str).eq("success")]
    rows: list[dict[str, object]] = []
    for (session, event_index), group in success.groupby(["session", "event_index"], sort=True):
        original: dict[str, float] = {}
        shuffled: dict[str, dict[int, float]] = {}
        for condition in MAP_CONDITIONS:
            map_rows = group[group["map_condition"].astype(str).eq(condition)]
            original[condition] = _paired_delta(
                map_rows[map_rows["order_condition"].astype(str).eq("original")]
            )
            shuffled[condition] = {}
            shuffle_rows = map_rows[map_rows["order_condition"].astype(str).eq("shuffled")]
            for shuffle_index, shuffle_group in shuffle_rows.groupby("shuffle_index", sort=True):
                delta = _paired_delta(shuffle_group)
                if np.isfinite(delta):
                    shuffled[condition][int(shuffle_index)] = delta
        common = sorted(set(shuffled["real_map"]).intersection(shuffled["population_code_permuted"]))
        real_values = np.asarray([shuffled["real_map"][index] for index in common], dtype=float)
        wrong_values = np.asarray(
            [shuffled["population_code_permuted"][index] for index in common],
            dtype=float,
        )
        real_median = float(np.median(real_values)) if real_values.size else np.nan
        wrong_median = float(np.median(wrong_values)) if wrong_values.size else np.nan
        real_advantage = original["real_map"] - real_median
        wrong_advantage = original["population_code_permuted"] - wrong_median
        interaction = real_advantage - wrong_advantage
        paired_interactions = (
            (original["real_map"] - real_values)
            - (original["population_code_permuted"] - wrong_values)
            if real_values.size
            else np.asarray([], dtype=float)
        )
        rows.append(
            {
                "session": str(session),
                "rat": _first_text(group, "rat") or str(session).split("/")[0],
                "event_index": int(event_index),
                "event_group": _first_text(group, "event_group") or "clean_imm",
                "real_original_delta_imm_minus_fragmented": original["real_map"],
                "real_shuffled_median_delta_imm_minus_fragmented": real_median,
                "wrong_original_delta_imm_minus_fragmented": original[
                    "population_code_permuted"
                ],
                "wrong_shuffled_median_delta_imm_minus_fragmented": wrong_median,
                "real_order_advantage": real_advantage,
                "wrong_order_advantage": wrong_advantage,
                "order_by_map_interaction": interaction,
                "paired_shuffle_interaction_median": (
                    float(np.median(paired_interactions))
                    if paired_interactions.size
                    else np.nan
                ),
                "paired_shuffle_interaction_positive_fraction": (
                    float(np.mean(paired_interactions > 0.0))
                    if paired_interactions.size
                    else np.nan
                ),
                "n_real_shuffles": int(len(shuffled["real_map"])),
                "n_wrong_shuffles": int(len(shuffled["population_code_permuted"])),
                "n_paired_shuffles": int(len(common)),
                "factorial_complete": bool(
                    len(common) == int(expected_shuffles)
                    and all(np.isfinite(value) for value in original.values())
                ),
            }
        )
    return pd.DataFrame(rows)


def _paired_delta(frame: pd.DataFrame) -> float:
    values = frame.groupby("model", sort=False)["log_evidence"].last()
    if FIRST_ORDER_IMM not in values or FRAGMENTED not in values:
        return np.nan
    return float(values[FIRST_ORDER_IMM] - values[FRAGMENTED])


def summarize(decisions: pd.DataFrame, group_column: str | None = None) -> pd.DataFrame:
    groups = [("all", decisions)] if group_column is None else decisions.groupby(group_column, sort=True)
    rows: list[dict[str, object]] = []
    for group_name, group in groups:
        interaction = pd.to_numeric(group["order_by_map_interaction"], errors="coerce").dropna()
        rows.append(
            {
                "scope": "all" if group_column is None else group_column,
                "group": group_name,
                "events": int(len(group)),
                "median_real_original_delta": _median(group, "real_original_delta_imm_minus_fragmented"),
                "median_real_shuffled_delta": _median(group, "real_shuffled_median_delta_imm_minus_fragmented"),
                "median_wrong_original_delta": _median(group, "wrong_original_delta_imm_minus_fragmented"),
                "median_wrong_shuffled_delta": _median(group, "wrong_shuffled_median_delta_imm_minus_fragmented"),
                "median_real_order_advantage": _median(group, "real_order_advantage"),
                "median_wrong_order_advantage": _median(group, "wrong_order_advantage"),
                "median_order_by_map_interaction": float(interaction.median()),
                "mean_order_by_map_interaction": float(interaction.mean()),
                "interaction_positive_fraction": float((interaction > 0.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def rat_bootstrap(decisions: pd.DataFrame, *, replicates: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rats = np.asarray(sorted(decisions["rat"].astype(str).unique()), dtype=object)
    by_rat = {
        rat: pd.to_numeric(
            decisions.loc[
                decisions["rat"].astype(str).eq(str(rat)),
                "order_by_map_interaction",
            ],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)
        for rat in rats
    }
    values = []
    for _ in range(int(replicates)):
        sampled = rng.choice(rats, size=len(rats), replace=True)
        values.append(float(np.median(np.concatenate([by_rat[str(rat)] for rat in sampled]))))
    series = pd.Series(values, dtype=float)
    return pd.DataFrame(
        [
            {
                "bootstrap_unit": "rat",
                "replicates": int(replicates),
                "seed": int(seed),
                "estimate": _median(decisions, "order_by_map_interaction"),
                "ci_low": float(series.quantile(0.025)),
                "ci_high": float(series.quantile(0.975)),
                "positive_fraction": float((series > 0.0).mean()),
            }
        ]
    )


def build_gates(
    scores: pd.DataFrame,
    decisions: pd.DataFrame,
    bootstrap: pd.DataFrame,
    *,
    expected_events: int,
    expected_shuffles: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(kind: str, gate: str, passed: bool, observed: object, criterion: str) -> None:
        rows.append(
            {
                "gate_type": kind,
                "gate": gate,
                "passed": bool(passed),
                "observed": observed,
                "criterion": criterion,
            }
        )

    failures = int((scores.get("status", "success").astype(str) != "success").sum()) if "status" in scores else 0
    add("technical", "no_scoring_failures", failures == 0, failures, "zero failed score rows")
    add("technical", "expected_events_present", len(decisions) == expected_events, len(decisions), f"{expected_events} frozen events")
    complete = decisions["factorial_complete"].map(bool) if not decisions.empty else pd.Series(dtype=bool)
    add("technical", "all_four_cells_complete", bool(not complete.empty and complete.all()), f"{int(complete.sum())}/{len(complete)}", "real/wrong map by original/shuffled order complete")
    shuffles = pd.to_numeric(decisions.get("n_paired_shuffles"), errors="coerce")
    add("technical", "paired_shuffle_count_complete", bool(not shuffles.empty and shuffles.eq(expected_shuffles).all()), "" if shuffles.empty else f"min={int(shuffles.min())}, max={int(shuffles.max())}", f"{expected_shuffles} identical whole-bin permutations per event and map")

    summary = summarize(decisions).iloc[0] if not decisions.empty else None
    interaction = float(summary["median_order_by_map_interaction"]) if summary is not None else np.nan
    ci_low = float(bootstrap["ci_low"].iloc[0]) if not bootstrap.empty else np.nan
    ci_high = float(bootstrap["ci_high"].iloc[0]) if not bootstrap.empty else np.nan
    rat_summary = summarize(decisions, "rat") if not decisions.empty else pd.DataFrame()
    rat_positive = int((pd.to_numeric(rat_summary.get("median_order_by_map_interaction"), errors="coerce") > 0.0).sum()) if not rat_summary.empty else 0
    add("interpretation", "median_interaction_positive", interaction > 0.0, interaction, "median (real order effect - wrong order effect) > 0")
    add("interpretation", "rat_bootstrap_interaction_ci_positive", ci_low > 0.0, f"[{ci_low}, {ci_high}]", "rat-bootstrap 95% CI excludes zero above")
    add("interpretation", "at_least_three_rats_positive", rat_positive >= 3, f"{rat_positive}/4", "at least 3 of 4 rat median interactions > 0")

    technical = [row for row in rows if row["gate_type"] == "technical"]
    if all(row["passed"] for row in technical):
        if ci_low > 0.0:
            verdict = "correct_map_strengthens_temporal_order_advantage"
        elif ci_high < 0.0:
            verdict = "wrong_map_strengthens_temporal_order_advantage"
        else:
            verdict = "temporal_order_advantage_largely_map_generic"
    else:
        verdict = "incomplete_factorial"
    rows.append(
        {
            "gate_type": "summary",
            "gate": "factorial_verdict",
            "passed": bool(all(row["passed"] for row in technical)),
            "observed": verdict,
            "criterion": "technical completion required; interaction sign and CI determine interpretation",
        }
    )
    return pd.DataFrame(rows)


def _median(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.median()) if not values.empty else np.nan


def _first_text(frame: pd.DataFrame, column: str) -> str:
    if column not in frame:
        return ""
    values = frame[column].dropna().astype(str)
    return str(values.iloc[0]) if not values.empty else ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _parameters(args: argparse.Namespace) -> dict[str, object]:
    excluded = {"selection", "dataset_root", "output_dir", "workers", "bootstrap_replicates"}
    return {key: value for key, value in vars(args).items() if key not in excluded}


def run(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    selection = read_selection(args.selection)
    tasks = [
        {
            "session": session,
            "events": group.to_dict("records"),
            "dataset_root": str(args.dataset_root),
            "parameters": _parameters(args),
        }
        for session, group in selection.groupby("session", sort=True)
    ]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
            results = list(executor.map(_score_session, tasks))
    else:
        results = [_score_session(task) for task in tasks]
    scores = pd.DataFrame([row for result in results for row in result])
    scores = scores.reindex(columns=SCORE_COLUMNS)
    decisions = build_event_decisions(scores, expected_shuffles=args.n_shuffles)
    summary = summarize(decisions)
    by_rat = summarize(decisions, "rat")
    by_session = summarize(decisions, "session")
    bootstrap = rat_bootstrap(
        decisions,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    gates = build_gates(
        scores,
        decisions,
        bootstrap,
        expected_events=len(selection),
        expected_shuffles=args.n_shuffles,
    )
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "pfeiffer_imm_order_map_factorial_scores.csv": scores,
        "pfeiffer_imm_order_map_factorial_event_decisions.csv": decisions,
        "pfeiffer_imm_order_map_factorial_summary.csv": summary,
        "pfeiffer_imm_order_map_factorial_by_rat.csv": by_rat,
        "pfeiffer_imm_order_map_factorial_by_session.csv": by_session,
        "pfeiffer_imm_order_map_factorial_rat_bootstrap.csv": bootstrap,
        "pfeiffer_imm_order_map_factorial_gate_summary.csv": gates,
    }
    for name, frame in outputs.items():
        frame.to_csv(outdir / name, index=False)
    verdict = gates.loc[gates["gate"].eq("factorial_verdict"), "observed"].iloc[0]
    report = [
        "# Pfeiffer/Foster IMM order by map factorial",
        "",
        f"**Verdict:** `{verdict}`",
        "",
        "The interaction is `(real original - real shuffled) - (wrong original - wrong shuffled)`.",
        "Whole population time bins use identical permutations under the real and wrong maps.",
        "A near-zero interaction means the IMM ordering advantage is generic temporal organization;",
        "a positive interaction means correct spatial content strengthens that advantage.",
        "",
    ]
    (outdir / "pfeiffer_imm_order_map_factorial_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )
    manifest = {
        "analysis": "pfeiffer_imm_order_by_map_factorial",
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "code_commit": _git_value(["rev-parse", "HEAD"]),
        "git_branch": _git_value(["branch", "--show-current"]),
        "selection": str(Path(args.selection).resolve()),
        "selection_sha256": _sha256(Path(args.selection)),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "selected_events": int(len(selection)),
        "maps": list(MAP_CONDITIONS),
        "orders": list(ORDER_CONDITIONS),
        "shuffle_unit": "whole_time_bin_population_vector",
        "identical_permutations_across_maps": True,
        "n_shuffles": int(args.n_shuffles),
        "seed": int(args.seed),
        "interaction": "(real_original-real_shuffled_median)-(wrong_original-wrong_shuffled_median)",
        "parameters": _parameters(args),
    }
    (outdir / "pfeiffer_imm_order_map_factorial_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-shuffles", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--state-space-stationary-sigma-cm", type=float, default=2.0)
    parser.add_argument("--state-space-diffusion-sigma-cm-sqrt-s", type=float, default=85.0)
    parser.add_argument("--state-space-max-step-sigma", type=float, default=4.0)
    parser.add_argument("--state-space-valid-occupancy-threshold-s", type=float, default=0.0)
    parser.add_argument("--state-space-imm-mode-stickiness", type=float, default=0.95)
    parser.add_argument("--state-space-imm-switch-tau-s", type=float, default=0.06)
    parser.add_argument("--state-space-momentum-sigma-cm-sqrt-s", type=float, default=85.0)
    parser.add_argument("--state-space-momentum-initial-sigma-cm-sqrt-s", type=float, default=85.0)
    parser.add_argument("--state-space-momentum-velocity-decay", type=float, default=0.95)
    parser.add_argument("--state-space-momentum-velocity-decay-tau-s", type=float, default=0.0)
    parser.add_argument("--state-space-momentum-candidate-top-k", type=int, default=32)
    parser.add_argument("--state-space-momentum-candidate-mass-threshold", type=float)
    parser.add_argument("--state-space-momentum-candidate-min-k", type=int, default=1)
    parser.add_argument("--state-space-momentum-candidate-max-k", type=int, default=0)
    parser.add_argument("--state-space-momentum-predicted-candidate-top-k", type=int, default=16)
    parser.add_argument("--state-space-momentum-candidate-source", default="emission")
    parser.add_argument("--time-bin-s", type=float, default=0.004)
    parser.add_argument("--spike-rate-scale", type=float, default=2.0)
    parser.add_argument("--emission-likelihood-temperature", type=float, default=0.3)
    parser.add_argument("--emission-negative-binomial-overdispersion", type=float, default=0.0)
    parser.add_argument("--sorted-spike-emission-model", default="poisson")
    parser.add_argument("--replay-gain-mode", default="none")
    parser.add_argument("--replay-gain-prior-count", type=float, default=10.0)
    parser.add_argument("--replay-gain-max-gain", type=float, default=20.0)
    parser.add_argument("--negative-binomial-dispersion", type=float, default=50.0)
    parser.add_argument("--bin-size-cm", type=float, default=6.0)
    parser.add_argument("--smoothing-sigma-bins", type=float, default=2.0)
    parser.add_argument("--min-speed-cm-s", type=float, default=5.0)
    parser.add_argument("--min-occupancy-s", type=float, default=0.02)
    parser.add_argument("--rate-floor-hz", type=float, default=1e-4)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
