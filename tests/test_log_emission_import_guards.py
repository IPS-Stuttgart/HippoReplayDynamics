from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_log_emission_tensor_import_installs_summary_guards() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    script = textwrap.dedent(
        """
        import numpy as np
        from hipporeplayimm.encoding import LogEmissionTensor

        def make_tensor(**overrides):
            kwargs = {
                "log_likelihood": np.zeros((2, 2), dtype=float),
                "spike_counts": np.zeros((2, 1), dtype=int),
                "times": np.array([0.0, 0.02], dtype=float),
                "dt": 0.02,
                "cell_ids": np.array([1], dtype=int),
                "n_spikes": 0,
            }
            kwargs.update(overrides)
            return LogEmissionTensor(**kwargs)

        try:
            make_tensor(
                log_likelihood=np.array([[0.0, np.nan], [0.0, 0.0]], dtype=float),
            )
        except ValueError as exc:
            if "NaN" not in str(exc):
                raise SystemExit(f"wrong NaN error: {exc}")
        else:
            raise SystemExit("NaN log likelihood was accepted")

        try:
            make_tensor(spike_counts=np.array([[1], [0]], dtype=int), n_spikes=0)
        except ValueError as exc:
            if "n_spikes" not in str(exc):
                raise SystemExit(f"wrong n_spikes error: {exc}")
        else:
            raise SystemExit("mismatched n_spikes was accepted")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
