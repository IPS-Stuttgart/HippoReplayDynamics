from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

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
