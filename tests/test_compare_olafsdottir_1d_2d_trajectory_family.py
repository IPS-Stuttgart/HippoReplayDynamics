from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_sota_comparator_pack import (
    DIFFUSION,
    FIRST_ORDER_IMM,
    FRAGMENTED,
    MOMENTUM_EXACT,
    STATIONARY,
)
from scripts.compare_olafsdottir_1d_2d_trajectory_family import (
    SUMMARY_COLUMNS,
    build_comparison_outputs,
    write_comparison_outputs,
)


def test_compare_1d_2d_trajectory_family_outputs(tmp_path: Path) -> None:
    evidence_1d = pd.DataFrame(
        [
            _score('R2142/2014-08-06/sleepPOST', 0, STATIONARY, 0.0),
            _score('R2142/2014-08-06/sleepPOST', 0, DIFFUSION, 10.0),
            _score('R2142/2014-08-06/sleepPOST', 0, FRAGMENTED, 3.0),
            _score('R2142/2014-08-06/sleepPOST', 0, FIRST_ORDER_IMM, 30.0),
            _score('R2142/2014-08-06/sleepPOST', 0, MOMENTUM_EXACT, 20.0),
            _score('R2142/2014-08-06/sleepPOST', 1, STATIONARY, 0.0),
            _score('R2142/2014-08-06/sleepPOST', 1, DIFFUSION, 12.0),
            _score('R2142/2014-08-06/sleepPOST', 1, FRAGMENTED, 25.0),
            _score('R2142/2014-08-06/sleepPOST', 1, FIRST_ORDER_IMM, 20.0),
            _score('R2142/2014-08-06/sleepPOST', 1, MOMENTUM_EXACT, 14.0),
        ]
    )
    evidence_2d = pd.DataFrame(
        [
            _score('Rat1/Open1', 0, STATIONARY, 0.0),
            _score('Rat1/Open1', 0, DIFFUSION, 10.0),
            _score('Rat1/Open1', 0, FRAGMENTED, 20.0),
            _score('Rat1/Open1', 0, FIRST_ORDER_IMM, 80.0),
            _score('Rat1/Open1', 0, MOMENTUM_EXACT, 40.0),
            _score('Rat1/Open1', 1, STATIONARY, 0.0),
            _score('Rat1/Open1', 1, DIFFUSION, 20.0),
            _score('Rat1/Open1', 1, FRAGMENTED, 5.0),
            _score('Rat1/Open1', 1, FIRST_ORDER_IMM, 60.0),
            _score('Rat1/Open1', 1, MOMENTUM_EXACT, 50.0),
            _score('Rat2/Open1', 2, STATIONARY, 0.0),
            _score('Rat2/Open1', 2, DIFFUSION, 20.0),
            _score('Rat2/Open1', 2, FRAGMENTED, 5.0),
            _score('Rat2/Open1', 2, FIRST_ORDER_IMM, 10.0),
            _score('Rat2/Open1', 2, MOMENTUM_EXACT, 70.0),
        ]
    )

    outputs = build_comparison_outputs(
        evidence_1d,
        evidence_2d,
        margin_threshold=5.5,
        rat_bootstrap_replicates=25,
        rat_bootstrap_random_seed=7,
    )

    summary = outputs['compare_1d_2d_trajectory_family_summary.csv']
    assert list(summary.columns[:10]) == SUMMARY_COLUMNS[:10]

    one_d = summary[summary['environment_type'].eq('1D_Z_track')].iloc[0]
    assert int(one_d['events']) == 2
    assert one_d['trajectory_confident_claim_fraction'] == 1.0
    assert one_d['nontrajectory_confident_claim_fraction'] == 0.0
    assert one_d['first_order_imm_raw_best_fraction'] == 0.5
    assert one_d['momentum_raw_best_fraction'] == 0.0
    assert one_d['momentum_vs_diffusion_median'] == 6.0

    two_d = summary[summary['environment_type'].eq('2D_open_field')].iloc[0]
    assert int(two_d['events']) == 3
    assert np.isclose(two_d['first_order_imm_raw_best_fraction'], 2 / 3)
    assert np.isclose(two_d['momentum_raw_best_fraction'], 1 / 3)
    assert two_d['momentum_vs_diffusion_median'] == 30.0

    exact_core = outputs['compare_1d_2d_exact_core_model_summary.csv']
    one_d_imm = exact_core[
        exact_core['environment_type'].eq('1D_Z_track')
        & exact_core['model'].eq(FIRST_ORDER_IMM)
    ].iloc[0]
    assert one_d_imm['model_short_name'] == 'first_order_imm'
    assert one_d_imm['raw_best_events'] == 1

    momentum = outputs['compare_1d_2d_momentum_vs_diffusion_summary.csv']
    assert int(momentum[momentum['environment_type'].eq('1D_Z_track')]['momentum_raw_wins'].iloc[0]) == 2

    bootstrap = outputs['compare_1d_2d_trajectory_family_bootstrap_summary.csv']
    assert set(bootstrap['environment_type']) == {'1D_Z_track', '2D_open_field'}
    assert (bootstrap['bootstrap_replicates'] == 25).all()

    gates = outputs['compare_1d_2d_trajectory_family_gate_summary.csv'].set_index('gate')
    assert bool(gates.loc['overall', 'passed'])

    interpretation = outputs['compare_1d_2d_interpretation_summary.csv'].iloc[0]
    assert interpretation['interpretation_rule'] == 'feasibility_data_limitation'
    assert 'not negative evidence' in interpretation['paper_safe_interpretation']
    assert 'Do not claim IMM is only apparent in 2D' in interpretation['avoid_overclaim']

    interpretable_outputs = build_comparison_outputs(
        evidence_1d,
        evidence_2d,
        margin_threshold=5.5,
        rat_bootstrap_replicates=5,
        min_interpretable_1d_events=1,
        weak_fraction_gap=0.1,
    )
    interpretable = interpretable_outputs['compare_1d_2d_interpretation_summary.csv'].iloc[0]
    assert interpretable['interpretation_rule'] == 'strong_trajectory_family_less_imm_dominance'

    written = write_comparison_outputs(
        evidence_1d,
        evidence_2d,
        tmp_path,
        margin_threshold=5.5,
        rat_bootstrap_replicates=10,
        rat_bootstrap_random_seed=3,
    )
    assert set(written) == set(outputs)
    for filename in written:
        assert (tmp_path / filename).is_file()


def _score(session: str, event_index: int, model: str, log_evidence: float) -> dict[str, object]:
    return {
        'status': 'success',
        'session': session,
        'event_index': event_index,
        'model': model,
        'log_evidence': log_evidence,
        'evidence_comparable': True,
        'evidence_support': 'exact_full_grid',
    }
