import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from scripts.recover_hc11_bilateral_sources import (
    download, extract_summaries, parse_checksums, parse_index, safe_relative, wake_immobility,
)
from scripts.report_hc11_source_recovery import (
    animal_ceiling, ca1_identity, independent_spike_check, optional_bool, source_field_comparison,
)


def test_index_is_authentic_and_unambiguous():
    assert parse_index("# CRCNS.org 'hc-11' dataset files\n data/a.tar.gz 12 (12 B)\n") == {'data/a.tar.gz': 12}
    for value in ('<html>Login</html>', '# dataset files\n a 2\n a 3', '# dataset files\n ../a 3'):
        with pytest.raises(ValueError):
            parse_index(value)


def test_checksums_reject_duplicates_and_traversal():
    digest = 'a' * 32
    assert parse_checksums(digest + '  ./docs/a.pdf\n') == {'docs/a.pdf': digest}
    with pytest.raises(ValueError):
        parse_checksums(digest + ' ../a')
    with pytest.raises(ValueError):
        parse_checksums((digest + ' a\n') * 2)


@pytest.mark.parametrize('name', ['/a', '../a', 'a/../../b', 'a\\b'])
def test_unsafe_members_never_extract(name):
    with pytest.raises(ValueError):
        safe_relative(name)


def test_bundle_contains_only_expected_unique_summaries(tmp_path):
    archive = tmp_path / 'bundle.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        entry = tarfile.TarInfo('folder/rat_1_sessInfo.mat')
        entry.size = 4
        tar.addfile(entry, io.BytesIO(b'data'))
    rows = extract_summaries(archive, tmp_path, ['rat_1'], 10)
    assert rows[0]['sha256'] == hashlib.sha256(b'data').hexdigest()
    assert extract_summaries(archive, tmp_path, ['rat_1'], 10) == rows
    with pytest.raises(ValueError):
        extract_summaries(archive, tmp_path, ['rat_2'], 10)
    with pytest.raises(ValueError):
        extract_summaries(archive, tmp_path, ['rat_1'], 3)
    (tmp_path / 'rat_1_sessInfo.mat').write_bytes(b'changed')
    with pytest.raises(ValueError):
        extract_summaries(archive, tmp_path, ['rat_1'], 10)


def test_completed_download_must_match_published_checksum(tmp_path):
    path = tmp_path / 'existing.pdf'
    path.write_bytes(b'data')
    assert download(None, 'unused', path, 4, hashlib.md5(b'data').hexdigest(), 0) == 'verified_existing'
    with pytest.raises(ValueError):
        download(None, 'unused', path, 4, '0' * 32, 0)


def test_partial_not_overwritten_when_server_ignores_range(tmp_path):
    path = tmp_path / 'a.pdf'
    Path(str(path) + '.partial').write_bytes(b'd')

    class Response:
        status = 200
        headers = {'Content-Length': '4'}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Client:
        def open(self, request):
            return Response()

    with pytest.raises(ValueError, match='range not honored'):
        download(Client(), 'https://example.org/a', path, 4, '0' * 32, 0)
    assert Path(str(path) + '.partial').read_bytes() == b'd'


def test_explicit_wake_required_and_no_tracking_gap_bridge():
    import numpy as np

    original = {'Position': {'TimeStamps': [0, .05, .1, .4, .45, .5], 'TwoDLocation': np.zeros((6, 2))},
                'Epochs': {'MazeEpoch': [[0, .5]], 'Wake': [[.05, .45]], 'Drowsy': [[0, .05], [.45, .5]]}}
    assert np.allclose(wake_immobility(original), [[.05, .1], [.4, .45]])
    original['Epochs']['Wake'] = []
    assert wake_immobility(original).shape == (0, 2)


def test_wake_availability_uses_documented_meters_and_never_repairs_clock():
    import numpy as np

    original = {'Position': {'TimeStamps': [0, .05, .1], 'TwoDLocation': [[0, 0], [.01, 0], [.01, 0]]},
                'Epochs': {'MazeEpoch': [[0, .1]], 'Wake': [[0, .1]]}}
    # One cm in 50 ms is moving, not immobile; the later stationary segment stays.
    assert np.allclose(wake_immobility(original), [[.05, .1]])
    original['Position']['TimeStamps'] = [0, .1, .05]
    with pytest.raises(ValueError, match='chronology'):
        wake_immobility(original)


def test_original_group_16_retained_extra_groups_not_assumed_CA1():
    mapping = {'left_ca1_groups': list(range(1,9)), 'right_ca1_groups': list(range(9,17))}
    assert ca1_identity(16,mapping) == ('right',8)
    mapping = {'left_ca1_groups': list(range(1,7)), 'right_ca1_groups': list(range(8,14))}
    assert ca1_identity(7,mapping) == (None,None)
    assert ca1_identity(14,mapping) == (None,None)
    assert ca1_identity(8,mapping) == ('right',1)
    with pytest.raises(ValueError):
        ca1_identity(1,{'left_ca1_groups':[1], 'right_ca1_groups':[1]})


def test_cohort_ceiling_cannot_promote_missing_EEG_or_few_units():
    rows = [
        {'animal':'A', 'eeg_present':True, 'pyramidal_left_upper_bound':50, 'pyramidal_right_upper_bound':50, 'eeg_original_clock_agrees':True},
        {'animal':'B', 'eeg_present':True, 'pyramidal_left_upper_bound':40, 'pyramidal_right_upper_bound':2, 'eeg_original_clock_agrees':True},
        {'animal':'C', 'eeg_present':False, 'pyramidal_left_upper_bound':30, 'pyramidal_right_upper_bound':30, 'eeg_original_clock_agrees':None},
        {'animal':'D', 'eeg_present':True, 'pyramidal_left_upper_bound':18, 'pyramidal_right_upper_bound':48, 'eeg_original_clock_agrees':False},
        {'animal':'D', 'eeg_present':True, 'pyramidal_left_upper_bound':41, 'pyramidal_right_upper_bound':0, 'eeg_original_clock_agrees':True},
    ]
    assert animal_ceiling(rows,10) == ['A','D']
    assert animal_ceiling(rows,10,require_clock=True) == ['A']


def test_independent_spike_verification_rejects_nonfinite_and_duplicate_identities(tmp_path):
    import h5py
    import numpy as np

    path = tmp_path / 'original.mat'
    with h5py.File(path,'w') as handle:
        group = handle.create_group('sessInfo/Spikes')
        group.create_dataset('SpikeTimes',data=[.1,.2,.3,.4])
        group.create_dataset('SpikeIDs',data=[101,1602,101,1602])
        group.create_dataset('PyrIDs',data=[101,1602])
    converted = {'UID':[1,2], 'times':[np.array([.1,.3]),np.array([.2,.4])]}
    audited = [{'unit_id':'1','original_cluster_id':'101'}, {'unit_id':'2','original_cluster_id':'1602'}]
    rows,pyr = independent_spike_check(path,converted,audited)
    assert pyr == {101,1602}
    assert all(row['max_spike_time_difference_s'] == 0 for row in rows)
    with pytest.raises(ValueError,match='duplicate original identity'):
        independent_spike_check(path,converted,[audited[0],audited[0]])
    converted['times'][1][0] = np.nan
    with pytest.raises(ValueError,match='converted spike chronology'):
        independent_spike_check(path,converted,audited)


def test_numeric_comparison_does_not_turn_missing_fields_into_agreement():
    import numpy as np

    original = {'Spikes': {'times': [1., np.nan]}, 'Position': {'MazeType': 'linear'}, 'Epochs': {}}
    previous = {'Spikes': {'times': [1., np.nan]}, 'Position': {'MazeType': 'linear'}, 'Epochs': {}}
    assert source_field_comparison(original, previous) == (1, True)
    previous['Spikes']['times'] = [2., np.nan]
    assert source_field_comparison(original, previous) == (1, False)
    previous['Spikes'] = {}
    assert source_field_comparison(original, previous) == (0, None)


def test_optional_source_boolean_preserves_false_and_missing_separately():
    assert optional_bool('False') is False
    assert optional_bool('True') is True
    assert optional_bool('') is None
    with pytest.raises(ValueError):
        optional_bool('unknown')
