from __future__ import annotations

from pathlib import Path

import numpy as np

from hipporeplayimm.olafsdottir2016 import read_axona_egf


def test_read_axona_egf_ignores_standard_footer(tmp_path: Path) -> None:
    path = tmp_path / "footer.egf"
    signal = np.array([-2, -1, 0, 1], dtype=">i2")
    footer = b"data_" + b"end"
    path.write_bytes(
        b"sample_rate 4800 hz\r\n"
        b"data_start\r\n"
        + signal.tobytes()
        + b"\r\n"
        + footer
        + b"\r\n"
    )

    egf = read_axona_egf(path)

    np.testing.assert_array_equal(egf.signal, np.array([-2, -1, 0, 1], dtype=np.int16))
    np.testing.assert_allclose(egf.times_s, np.arange(4) / 4800.0)
