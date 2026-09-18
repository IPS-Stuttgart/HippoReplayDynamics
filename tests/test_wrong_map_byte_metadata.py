from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path("scripts").resolve()))
from compare_wrong_map_evidence_controls import wrong_map_model_evidence_attenuation


def test_wrong_map_attenuation_decodes_byte_backed_identifiers_and_status() -> None:
    model = "sorted-spike-state-space-diffusion"
    real = pd.DataFrame(
        [
            {
                "status": memoryview(b"success"),
                "session": b"Rat1/Open1",
                "event_index": 3,
                "model": bytearray(model.encode()),
                "log_evidence": 10.0,
            }
        ]
    )
    wrong = pd.DataFrame(
        [
            {
                "status": np.bytes_("success"),
                "session": memoryview(b"Rat1/Open1"),
                "event_index": 3,
                "model": np.bytes_(model),
                "requested_model": b"sorted-spike-state-space-diffusion",
                "map_session": bytearray(b"Rat1/Open2"),
                "log_evidence": 4.0,
            }
        ]
    )

    attenuation = wrong_map_model_evidence_attenuation(real, wrong)

    assert len(attenuation) == 1
    assert attenuation.loc[0, "rat"] == "Rat1"
    assert attenuation.loc[0, "session"] == "Rat1/Open1"
    assert attenuation.loc[0, "map_session"] == "Rat1/Open2"
    assert attenuation.loc[0, "model"] == model
    assert attenuation.loc[0, "requested_model"] == model
    assert attenuation.loc[0, "real_minus_wrong_log_evidence"] == 6.0



def test_wrong_map_attenuation_preserves_large_string_event_index() -> None:
    model = "sorted-spike-state-space-diffusion"
    event_index = str(2**53 + 1)
    real = pd.DataFrame(
        [{"session": "Rat1/Open1", "event_index": event_index, "model": model, "log_evidence": 10.0}]
    )
    wrong = pd.DataFrame(
        [{
            "session": "Rat1/Open1",
            "event_index": event_index,
            "model": model,
            "map_session": "Rat1/Open2",
            "log_evidence": 4.0,
        }]
    )

    attenuation = wrong_map_model_evidence_attenuation(real, wrong)

    assert attenuation.loc[0, "event_index"] == 2**53 + 1


def test_wrong_map_attenuation_rejects_unsafe_float_event_index() -> None:
    model = "sorted-spike-state-space-diffusion"
    unsafe = float(2**53)
    real = pd.DataFrame(
        [{"session": "Rat1/Open1", "event_index": unsafe, "model": model, "log_evidence": 10.0}]
    )
    wrong = pd.DataFrame(
        [{
            "session": "Rat1/Open1",
            "event_index": unsafe,
            "model": model,
            "map_session": "Rat1/Open2",
            "log_evidence": 4.0,
        }]
    )

    with pytest.raises(ValueError, match=r"floating-point event_index.*2\*\*53"):
        wrong_map_model_evidence_attenuation(real, wrong)
