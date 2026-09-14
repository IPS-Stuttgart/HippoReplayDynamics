import numpy as np
import pandas as pd

from scripts.report_regional_prevalence_calibration import rat_mean, screen


def fixture():
    rows, fits = [], []
    for animal, session in [('a', '1'), ('a', '2'), ('b', '3'), ('c', '4')]:
        for side in ('high', 'low'):
            fits.append(dict(animal=animal, session=session, side=side, encoding='early_run', calibration='run_q3', status='available'))
        for source, panels in [('run_q4', range(5)), ('all_fixed_candidates', [-1]), ('full_accepted_segment', [-1])]:
            for panel in panels:
                rows.append(dict(animal=animal, session=session, encoding='early_run', calibration='run_q3', source=source,
                    panel='prevalence_' + str(panel) if panel >= 0 else 'unknown', high_status='available', low_status='available',
                    raw_mean_absolute_error=.1, corrected_mean_absolute_error=.03,
                    raw_discrepancy=.09, corrected_discrepancy=.03))
    return pd.DataFrame(rows), pd.DataFrame(fits)


def test_development_pass_cannot_become_validated_remedy():
    frame, fits = fixture()
    gates, _ = screen(frame, fits)
    result = gates.set_index('gate').passed
    assert result.pf_development_screen
    assert not result.independent_validation_complete
    assert not result.validated_remedy


def test_missing_or_incompatible_pairs_block_screen():
    frame, fits = fixture()
    frame.loc[0, 'corrected_mean_absolute_error'] = np.nan
    gates, metrics = screen(frame, fits)
    assert np.isnan(metrics['native_error_corrected'])
    assert not gates.set_index('gate').passed.pf_development_screen
    frame, fits = fixture()
    frame.loc[frame.source.eq('all_fixed_candidates'), 'high_status'] = 'incompatible_out_of_range'
    assert not screen(frame, fits)[0].set_index('gate').passed.pf_development_screen


def test_rat_weighting_and_no_vacuous_metrics():
    frame = pd.DataFrame(dict(animal=['a', 'a', 'b'], session=['1', '2', '3'], value=[0., 0., 1.]))
    assert rat_mean(frame, 'value') == .5
    assert np.isnan(rat_mean(frame.iloc[:0], 'value'))
