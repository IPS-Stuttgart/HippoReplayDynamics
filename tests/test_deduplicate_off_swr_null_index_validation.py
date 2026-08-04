import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from deduplicate_off_swr_candidates import (  # noqa: E402
    build_one_per_source_group_decisions,
    write_off_swr_candidate_dedup_outputs,
)


@pytest.mark.parametrize("null_index", [True, False, 0.5, None, "not-an-index"])
def test_off_swr_candidate_dedup_rejects_invalid_null_index(null_index: object):
    validation = pd.DataFrame(
        [{"session": "Rat1/Open1", "event_index": 10, "null_index": null_index}]
    )

    with pytest.raises(ValueError, match="null_index must contain integer identifiers"):
        build_one_per_source_group_decisions(validation)


def test_off_swr_candidate_dedup_accepts_integer_valued_float_null_index():
    validation = pd.DataFrame(
        [{"session": "Rat1/Open1", "event_index": 10, "null_index": 1.0}]
    )

    decisions = build_one_per_source_group_decisions(validation)

    assert decisions["null_index"].eq(1).all()


def test_off_swr_candidate_merge_rejects_boolean_null_index_alias(tmp_path):
    validation = pd.DataFrame(
        [{"session": "Rat1/Open1", "event_index": 10, "null_index": 1}]
    )
    candidate_table = pd.DataFrame(
        [{"session": "Rat1/Open1", "event_index": 10, "null_index": True}]
    )

    with pytest.raises(ValueError, match="null_index must contain integer identifiers"):
        write_off_swr_candidate_dedup_outputs(
            validation_decisions=validation,
            candidate_table=candidate_table,
            output=tmp_path,
        )
