import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from scripts.recover_hc11_bilateral_sources import (
    download, extract_summaries, parse_checksums, parse_index, safe_relative, wake_immobility,
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
