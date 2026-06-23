#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.build_sota_comparator_pack import (
        DIFFUSION,
        FIRST_ORDER_IMM,
        FRAGMENTED,
        MOMENTUM_EXACT,
        STATIONARY,
        build_sota_comparator_event_table,
        build_sota_comparator_model_summary,
        build_sota_comparator_momentum_vs_diffusion_summary,
        read_event_model_evidence,
    )
except ModuleNotFoundError:
    from build_sota_comparator_pack import (
        DIFFUSION,
        FIRST_ORDER_IMM,
        FRAGMENTED,
        MOMENTUM_EXACT,
        STATIONARY,
        build_sota_comparator_event_table,
        build_sota_comparator_model_summary,
        build_sota_comparator_momentum_vs_diffusion_summary,
        read_event_model_evidence,
    )

SUMMARY_COLUMNS = [
    'dataset',
    'environment_type',
    'events',
    'trajectory_confident_claim_fraction',
    'nontrajectory_confident_claim_fraction',
    'mean_family_margin',
    'median_family_margin',
    'first_order_imm_raw_best_fraction',
    'momentum_raw_best_fraction',
    'momentum_vs_diffusion_median',
    'trajectory_family_raw_win_fraction',
    'trajectory_confident_claims',
    'nontrajectory_confident_claims',
    'ambiguous_family_events',
    'momentum_raw_wins',
    'momentum_confident_claims',
    'diffusion_confident_claims',
    'momentum_diffusion_ambiguous_events',
    'margin_threshold',
]

MODEL_SHORT_NAMES = {
    STATIONARY: 'stationary',
    DIFFUSION: 'diffusion',
    FRAGMENTED: 'fragmented',
    FIRST_ORDER_IMM: 'first_order_imm',
    MOMENTUM_EXACT: 'momentum_exact_sparse',
}

BOOTSTRAP_METRICS = [
    'trajectory_confident_claim_fraction',
    'nontrajectory_confident_claim_fraction',
    'mean_family_margin',
    'median_family_margin',
    'first_order_imm_raw_best_fraction',
    'momentum_raw_best_fraction',
    'momentum_vs_diffusion_median',
]


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        return bool(np.isfinite(numeric) and numeric != 0.0)
    return str(value).strip().lower() in {'1', '1.0', 'true', 't', 'yes', 'y', 'on'}


def _bool_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].map(_as_bool).astype(bool)


def _empty_summary_row(dataset: str, environment_type: str, margin_threshold: float) -> dict[str, object]:
    return {
        'dataset': dataset,
        'environment_type': environment_type,
        'events': 0,
        'trajectory_confident_claim_fraction': 0.0,
        'nontrajectory_confident_claim_fraction': 0.0,
        'mean_family_margin': np.nan,
        'median_family_margin': np.nan,
        'first_order_imm_raw_best_fraction': 0.0,
        'momentum_raw_best_fraction': 0.0,
        'momentum_vs_diffusion_median': np.nan,
        'trajectory_family_raw_win_fraction': 0.0,
        'trajectory_confident_claims': 0,
        'nontrajectory_confident_claims': 0,
        'ambiguous_family_events': 0,
        'momentum_raw_wins': 0,
        'momentum_confident_claims': 0,
        'diffusion_confident_claims': 0,
        'momentum_diffusion_ambiguous_events': 0,
        'margin_threshold': float(margin_threshold),
    }


def labeled_sota_event_table(
    event_model_evidence: pd.DataFrame,
    *,
    dataset: str,
    environment_type: str,
    margin_threshold: float = 5.5,
) -> pd.DataFrame:
    event_table = build_sota_comparator_event_table(
        event_model_evidence,
        margin_threshold=margin_threshold,
    ).copy()
    event_table.insert(0, 'environment_type', environment_type)
    event_table.insert(0, 'dataset', dataset)
    if 'animal' not in event_table:
        event_table['animal'] = event_table.get('rat', pd.Series('', index=event_table.index)).astype(str)
    return event_table


def _summarize_table(
    event_table: pd.DataFrame,
    *,
    dataset: str,
    environment_type: str,
    margin_threshold: float,
) -> dict[str, object]:
    complete = event_table[_bool_column(event_table, 'exact_core_complete')].copy()
    if complete.empty:
        return _empty_summary_row(dataset, environment_type, margin_threshold)

    events = int(len(complete))
    family_delta = pd.to_numeric(complete['delta_trajectory_minus_nontrajectory'], errors='coerce')
    momentum_delta = pd.to_numeric(complete['delta_momentum_exact_minus_diffusion'], errors='coerce').dropna()
    trajectory_claims = _bool_column(complete, 'trajectory_confident_vs_nontrajectory')
    nontrajectory_claims = _bool_column(complete, 'nontrajectory_confident_vs_trajectory')
    first_order_best = complete['best_exact_core_model'].astype(str).eq(FIRST_ORDER_IMM)
    momentum_best = complete['best_exact_core_model'].astype(str).eq(MOMENTUM_EXACT)
    momentum_confident = _bool_column(complete, 'momentum_exact_confident_vs_diffusion')
    diffusion_confident = _bool_column(complete, 'diffusion_confident_vs_momentum_exact')

    return {
        'dataset': dataset,
        'environment_type': environment_type,
        'events': events,
        'trajectory_confident_claim_fraction': float(trajectory_claims.mean()),
        'nontrajectory_confident_claim_fraction': float(nontrajectory_claims.mean()),
        'mean_family_margin': float(family_delta.mean()) if family_delta.notna().any() else np.nan,
        'median_family_margin': float(family_delta.median()) if family_delta.notna().any() else np.nan,
        'first_order_imm_raw_best_fraction': float(first_order_best.mean()),
        'momentum_raw_best_fraction': float(momentum_best.mean()),
        'momentum_vs_diffusion_median': float(momentum_delta.median()) if not momentum_delta.empty else np.nan,
        'trajectory_family_raw_win_fraction': float((family_delta > 0.0).mean()) if family_delta.notna().any() else 0.0,
        'trajectory_confident_claims': int(trajectory_claims.sum()),
        'nontrajectory_confident_claims': int(nontrajectory_claims.sum()),
        'ambiguous_family_events': int(events - trajectory_claims.sum() - nontrajectory_claims.sum()),
        'momentum_raw_wins': int((momentum_delta > 0.0).sum()) if not momentum_delta.empty else 0,
        'momentum_confident_claims': int(momentum_confident.sum()),
        'diffusion_confident_claims': int(diffusion_confident.sum()),
        'momentum_diffusion_ambiguous_events': int(events - momentum_confident.sum() - diffusion_confident.sum()),
        'margin_threshold': float(margin_threshold),
    }


def trajectory_family_comparison_summary(
    event_tables: pd.DataFrame,
    *,
    margin_threshold: float = 5.5,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, environment_type), group in event_tables.groupby(['dataset', 'environment_type'], sort=False):
        rows.append(
            _summarize_table(
                group,
                dataset=str(dataset),
                environment_type=str(environment_type),
                margin_threshold=margin_threshold,
            )
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def grouped_trajectory_family_summary(
    event_tables: pd.DataFrame,
    *,
    group_col: str,
    margin_threshold: float = 5.5,
) -> pd.DataFrame:
    if event_tables.empty:
        return pd.DataFrame(columns=['dataset', 'environment_type', group_col, *SUMMARY_COLUMNS[2:]])
    rows: list[dict[str, object]] = []
    for key, group in event_tables.groupby(['dataset', 'environment_type', group_col], sort=True):
        dataset, environment_type, group_value = key
        row = _summarize_table(
            group,
            dataset=str(dataset),
            environment_type=str(environment_type),
            margin_threshold=margin_threshold,
        )
        row[group_col] = str(group_value)
        rows.append(row)
    columns = ['dataset', 'environment_type', group_col, *SUMMARY_COLUMNS[2:]]
    return pd.DataFrame(rows, columns=columns)


def exact_core_model_hierarchy_summary(event_tables: pd.DataFrame, *, margin_threshold: float = 5.5) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for (dataset, environment_type), group in event_tables.groupby(['dataset', 'environment_type'], sort=False):
        summary = build_sota_comparator_model_summary(group, margin_threshold=margin_threshold).copy()
        summary.insert(0, 'environment_type', str(environment_type))
        summary.insert(0, 'dataset', str(dataset))
        summary['model_short_name'] = summary['model'].map(lambda model: MODEL_SHORT_NAMES.get(str(model), str(model)))
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def momentum_vs_diffusion_summary(event_tables: pd.DataFrame, *, margin_threshold: float = 5.5) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for (dataset, environment_type), group in event_tables.groupby(['dataset', 'environment_type'], sort=False):
        summary = build_sota_comparator_momentum_vs_diffusion_summary(group, margin_threshold=margin_threshold).copy()
        summary.insert(0, 'environment_type', str(environment_type))
        summary.insert(0, 'dataset', str(dataset))
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def rat_cluster_bootstrap_summary(
    event_tables: pd.DataFrame,
    *,
    n_bootstrap: int = 2000,
    random_seed: int = 1,
    margin_threshold: float = 5.5,
) -> pd.DataFrame:
    if n_bootstrap <= 0:
        raise ValueError('n_bootstrap must be positive')

    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(int(random_seed))
    for (dataset, environment_type), group in event_tables.groupby(['dataset', 'environment_type'], sort=False):
        complete = group[_bool_column(group, 'exact_core_complete')].copy()
        animals = sorted(complete['animal'].dropna().astype(str).unique()) if not complete.empty else []
        observed = _summarize_table(
            complete,
            dataset=str(dataset),
            environment_type=str(environment_type),
            margin_threshold=margin_threshold,
        )
        row: dict[str, object] = {
            'dataset': str(dataset),
            'environment_type': str(environment_type),
            'bootstrap_unit': 'animal',
            'bootstrap_replicates': int(n_bootstrap),
            'random_seed': int(random_seed),
            'observed_events': int(observed['events']),
            'observed_animals': int(len(animals)),
        }
        for metric in BOOTSTRAP_METRICS:
            row[f'observed_{metric}'] = observed[metric]

        if not animals:
            for metric in BOOTSTRAP_METRICS:
                row[f'{metric}_ci95_low'] = np.nan
                row[f'{metric}_ci95_high'] = np.nan
            rows.append(row)
            continue

        by_animal = {animal: complete[complete['animal'].astype(str).eq(animal)] for animal in animals}
        replicate_rows: list[dict[str, object]] = []
        for _ in range(int(n_bootstrap)):
            sampled = rng.choice(animals, size=len(animals), replace=True)
            sample = pd.concat([by_animal[animal] for animal in sampled], ignore_index=True)
            replicate_rows.append(
                _summarize_table(
                    sample,
                    dataset=str(dataset),
                    environment_type=str(environment_type),
                    margin_threshold=margin_threshold,
                )
            )
        replicates = pd.DataFrame(replicate_rows)
        for metric in BOOTSTRAP_METRICS:
            values = pd.to_numeric(replicates[metric], errors='coerce').to_numpy(dtype=float)
            row[f'{metric}_ci95_low'] = float(np.nanquantile(values, 0.025))
            row[f'{metric}_ci95_high'] = float(np.nanquantile(values, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def interpretation_summary(
    summary: pd.DataFrame,
    *,
    environment_1d: str = '1D_Z_track',
    environment_2d: str = '2D_open_field',
    min_interpretable_1d_events: int = 10,
    weak_fraction_gap: float = 0.25,
    similar_fraction_tolerance: float = 0.15,
) -> pd.DataFrame:
    columns = [
        'interpretation_rule',
        'paper_safe_interpretation',
        'avoid_overclaim',
        'environment_1d',
        'environment_2d',
        'events_1d',
        'events_2d',
        'trajectory_confident_claim_fraction_1d',
        'trajectory_confident_claim_fraction_2d',
        'trajectory_confident_claim_fraction_gap_1d_minus_2d',
        'first_order_imm_raw_best_fraction_1d',
        'first_order_imm_raw_best_fraction_2d',
        'first_order_imm_raw_best_fraction_gap_1d_minus_2d',
        'median_family_margin_1d',
        'median_family_margin_2d',
        'min_interpretable_1d_events',
        'weak_fraction_gap',
        'similar_fraction_tolerance',
    ]
    if summary.empty:
        row = {
            'interpretation_rule': 'unable_to_compare',
            'paper_safe_interpretation': 'No 1D-vs-2D interpretation is supported because the comparison summary is empty.',
            'avoid_overclaim': 'Do not claim IMM is only apparent in 2D without a robust 1D weak or negative result.',
            'environment_1d': environment_1d,
            'environment_2d': environment_2d,
            **_empty_interpretation_values(),
            'min_interpretable_1d_events': int(min_interpretable_1d_events),
            'weak_fraction_gap': float(weak_fraction_gap),
            'similar_fraction_tolerance': float(similar_fraction_tolerance),
        }
        return pd.DataFrame([row], columns=columns)

    one_d = summary[summary['environment_type'].astype(str).eq(environment_1d)]
    two_d = summary[summary['environment_type'].astype(str).eq(environment_2d)]
    if one_d.empty or two_d.empty:
        rule = 'unable_to_compare'
        wording = 'No 1D-vs-2D interpretation is supported because one comparison row is missing.'
        values = _empty_interpretation_values()
    else:
        one = one_d.iloc[0]
        two = two_d.iloc[0]
        events_1d = int(one['events'])
        events_2d = int(two['events'])
        family_1d = float(one['trajectory_confident_claim_fraction'])
        family_2d = float(two['trajectory_confident_claim_fraction'])
        imm_1d = float(one['first_order_imm_raw_best_fraction'])
        imm_2d = float(two['first_order_imm_raw_best_fraction'])
        family_gap = family_1d - family_2d
        imm_gap = imm_1d - imm_2d
        median_family_1d = float(one['median_family_margin'])
        median_family_2d = float(two['median_family_margin'])
        strong_family_1d = family_1d >= 0.5 and median_family_1d > 0.0
        similar_family = strong_family_1d and abs(family_gap) <= float(similar_fraction_tolerance)
        weaker_family = family_gap <= -float(weak_fraction_gap) or (family_1d < 0.5 and family_2d >= 0.5)
        less_imm = imm_gap <= -float(weak_fraction_gap)
        if events_1d < int(min_interpretable_1d_events):
            rule = 'feasibility_data_limitation'
            wording = 'Report the 1D result as a feasibility or data limitation, not negative evidence.'
        elif weaker_family:
            rule = 'weaker_1d_trajectory_family_signal'
            wording = 'This supports the hypothesis that 2D open-field replay exposes richer trajectory dynamics that are less apparent in constrained 1D settings.'
        elif similar_family and less_imm:
            rule = 'strong_trajectory_family_less_imm_dominance'
            wording = 'Trajectory replay may generalize, but the specific need for mode-flexible IMM may depend on environment geometry.'
        elif similar_family:
            rule = 'similarly_strong_trajectory_family_signal'
            wording = 'The trajectory-family signature generalizes beyond 2D open-field replay and may be a broader replay-dynamics feature.'
        elif strong_family_1d and less_imm:
            rule = 'strong_trajectory_family_less_imm_dominance'
            wording = 'Trajectory replay may generalize, but the specific need for mode-flexible IMM may depend on environment geometry.'
        else:
            rule = 'mixed_1d_result'
            wording = 'The 1D result is mixed; analyze by session, animal, track geometry, replay direction, and reward or proximity variables where available.'
        values = {
            'events_1d': events_1d,
            'events_2d': events_2d,
            'trajectory_confident_claim_fraction_1d': family_1d,
            'trajectory_confident_claim_fraction_2d': family_2d,
            'trajectory_confident_claim_fraction_gap_1d_minus_2d': family_gap,
            'first_order_imm_raw_best_fraction_1d': imm_1d,
            'first_order_imm_raw_best_fraction_2d': imm_2d,
            'first_order_imm_raw_best_fraction_gap_1d_minus_2d': imm_gap,
            'median_family_margin_1d': median_family_1d,
            'median_family_margin_2d': median_family_2d,
        }
    row = {
        'interpretation_rule': rule,
        'paper_safe_interpretation': wording,
        'avoid_overclaim': 'Do not claim IMM is only apparent in 2D until the 1D pipeline produces a robust negative or weak result.',
        'environment_1d': environment_1d,
        'environment_2d': environment_2d,
        **values,
        'min_interpretable_1d_events': int(min_interpretable_1d_events),
        'weak_fraction_gap': float(weak_fraction_gap),
        'similar_fraction_tolerance': float(similar_fraction_tolerance),
    }
    return pd.DataFrame([row], columns=columns)


def _empty_interpretation_values() -> dict[str, object]:
    return {
        'events_1d': 0,
        'events_2d': 0,
        'trajectory_confident_claim_fraction_1d': np.nan,
        'trajectory_confident_claim_fraction_2d': np.nan,
        'trajectory_confident_claim_fraction_gap_1d_minus_2d': np.nan,
        'first_order_imm_raw_best_fraction_1d': np.nan,
        'first_order_imm_raw_best_fraction_2d': np.nan,
        'first_order_imm_raw_best_fraction_gap_1d_minus_2d': np.nan,
        'median_family_margin_1d': np.nan,
        'median_family_margin_2d': np.nan,
    }


def gate_summary(summary: pd.DataFrame, event_tables: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append(
        {
            'gate': 'both_environment_rows_present',
            'passed': set(summary['environment_type'].astype(str)) >= {'1D_Z_track', '2D_open_field'},
            'observed': ' '.join(summary['environment_type'].astype(str)),
            'criterion': 'summary contains 1D_Z_track and 2D_open_field rows',
        }
    )
    complete = _bool_column(event_tables, 'exact_core_complete') if not event_tables.empty else pd.Series(dtype=bool)
    rows.append(
        {
            'gate': 'exact_core_rows_complete',
            'passed': bool(len(event_tables) > 0 and complete.all()),
            'observed': f'{int(complete.sum())}/{len(event_tables)}',
            'criterion': 'all compared events have the required exact-core model set',
        }
    )
    rows.append(
        {
            'gate': 'headline_metrics_present',
            'passed': bool(summary[SUMMARY_COLUMNS].notna().all(axis=None)) if not summary.empty else False,
            'observed': str(len(summary)),
            'criterion': 'all requested headline metric columns are populated',
        }
    )
    passed_count = sum(bool(row['passed']) for row in rows)
    rows.append(
        {
            'gate': 'overall',
            'passed': passed_count == len(rows),
            'observed': f'{passed_count}/{len(rows)} gates passed',
            'criterion': 'all 1D-vs-2D comparison gates pass',
        }
    )
    return pd.DataFrame(rows)


def build_comparison_outputs(
    evidence_1d: pd.DataFrame,
    evidence_2d: pd.DataFrame,
    *,
    dataset_1d: str = 'Olafsdottir2016',
    dataset_2d: str = 'PfeifferFoster',
    environment_1d: str = '1D_Z_track',
    environment_2d: str = '2D_open_field',
    margin_threshold: float = 5.5,
    rat_bootstrap_replicates: int = 2000,
    rat_bootstrap_random_seed: int = 1,
    min_interpretable_1d_events: int = 10,
    weak_fraction_gap: float = 0.25,
    similar_fraction_tolerance: float = 0.15,
) -> dict[str, pd.DataFrame]:
    one_d = labeled_sota_event_table(
        evidence_1d,
        dataset=dataset_1d,
        environment_type=environment_1d,
        margin_threshold=margin_threshold,
    )
    two_d = labeled_sota_event_table(
        evidence_2d,
        dataset=dataset_2d,
        environment_type=environment_2d,
        margin_threshold=margin_threshold,
    )
    event_tables = pd.concat([one_d, two_d], ignore_index=True)
    summary = trajectory_family_comparison_summary(event_tables, margin_threshold=margin_threshold)
    return {
        'compare_1d_2d_trajectory_family_event_table.csv': event_tables,
        'compare_1d_2d_trajectory_family_summary.csv': summary,
        'compare_1d_2d_trajectory_family_session_summary.csv': grouped_trajectory_family_summary(
            event_tables,
            group_col='session',
            margin_threshold=margin_threshold,
        ),
        'compare_1d_2d_trajectory_family_animal_summary.csv': grouped_trajectory_family_summary(
            event_tables,
            group_col='animal',
            margin_threshold=margin_threshold,
        ),
        'compare_1d_2d_exact_core_model_summary.csv': exact_core_model_hierarchy_summary(
            event_tables,
            margin_threshold=margin_threshold,
        ),
        'compare_1d_2d_momentum_vs_diffusion_summary.csv': momentum_vs_diffusion_summary(
            event_tables,
            margin_threshold=margin_threshold,
        ),
        'compare_1d_2d_trajectory_family_bootstrap_summary.csv': rat_cluster_bootstrap_summary(
            event_tables,
            n_bootstrap=rat_bootstrap_replicates,
            random_seed=rat_bootstrap_random_seed,
            margin_threshold=margin_threshold,
        ),
        'compare_1d_2d_interpretation_summary.csv': interpretation_summary(
            summary,
            environment_1d=environment_1d,
            environment_2d=environment_2d,
            min_interpretable_1d_events=min_interpretable_1d_events,
            weak_fraction_gap=weak_fraction_gap,
            similar_fraction_tolerance=similar_fraction_tolerance,
        ),
        'compare_1d_2d_trajectory_family_gate_summary.csv': gate_summary(summary, event_tables),
    }


def write_comparison_outputs(evidence_1d: pd.DataFrame, evidence_2d: pd.DataFrame, output: str | Path, **kwargs: object) -> dict[str, pd.DataFrame]:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    outputs = build_comparison_outputs(evidence_1d, evidence_2d, **kwargs)
    for filename, frame in outputs.items():
        frame.to_csv(out / filename, index=False)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare 1D Olafsdottir Z-track and 2D Pfeiffer/Foster trajectory-family metrics.')
    parser.add_argument('--evidence-1d', '--olafsdottir-event-model-evidence', dest='evidence_1d', required=True)
    parser.add_argument('--evidence-2d', '--pfeiffer-foster-event-model-evidence', dest='evidence_2d', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dataset-1d', default='Olafsdottir2016')
    parser.add_argument('--dataset-2d', default='PfeifferFoster')
    parser.add_argument('--environment-1d', default='1D_Z_track')
    parser.add_argument('--environment-2d', default='2D_open_field')
    parser.add_argument('--margin-threshold', type=float, default=5.5)
    parser.add_argument('--rat-bootstrap-replicates', type=int, default=2000)
    parser.add_argument('--rat-bootstrap-random-seed', type=int, default=1)
    parser.add_argument('--min-interpretable-1d-events', type=int, default=10)
    parser.add_argument('--weak-fraction-gap', type=float, default=0.25)
    parser.add_argument('--similar-fraction-tolerance', type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = write_comparison_outputs(
        read_event_model_evidence(args.evidence_1d),
        read_event_model_evidence(args.evidence_2d),
        args.output,
        dataset_1d=args.dataset_1d,
        dataset_2d=args.dataset_2d,
        environment_1d=args.environment_1d,
        environment_2d=args.environment_2d,
        margin_threshold=args.margin_threshold,
        rat_bootstrap_replicates=args.rat_bootstrap_replicates,
        rat_bootstrap_random_seed=args.rat_bootstrap_random_seed,
        min_interpretable_1d_events=args.min_interpretable_1d_events,
        weak_fraction_gap=args.weak_fraction_gap,
        similar_fraction_tolerance=args.similar_fraction_tolerance,
    )
    print(f'Wrote {len(outputs)} comparison tables to {Path(args.output)}')


if __name__ == '__main__':
    main()
