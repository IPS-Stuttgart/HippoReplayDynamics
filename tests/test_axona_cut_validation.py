import pytest

from hipporeplayimm.olafsdottir2016 import read_axona_cut


def test_read_axona_cut_rejects_truncated_label_payload(tmp_path):
    cut_path = tmp_path / "truncated.cut"
    cut_path.write_text(
        "Exact_cut_for: session.1 spikes: 3\n"
        "1 2\n",
        encoding="latin-1",
    )

    with pytest.raises(ValueError, match=r"declares 3 spikes but contains only 2 cluster labels"):
        read_axona_cut(cut_path)


def test_read_axona_cut_keeps_declared_number_when_payload_has_extra_labels(tmp_path):
    cut_path = tmp_path / "extra.cut"
    cut_path.write_text(
        "Exact_cut_for: session.1 spikes: 3\n"
        "1 2 3 4\n",
        encoding="latin-1",
    )

    result = read_axona_cut(cut_path)

    assert result.labels.tolist() == [1, 2, 3]
