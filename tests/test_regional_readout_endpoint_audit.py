import hashlib
import json

import numpy as np
import pytest

from hipporeplayimm.regional_content_frontier import regional_log_bf
from scripts.audit_regional_readout_endpoint import independent_bf
from scripts.report_regional_readout_endpoint import verified_manifest


def test_independent_short_likelihood_reconstruction():
    rng = np.random.default_rng(100)
    rates = rng.uniform(.01, 25, (7, 40))
    counts = rng.poisson(.3, (520, 7))
    region = np.arange(40) < 12
    np.testing.assert_allclose(independent_bf(counts, rates, region),
                               regional_log_bf(counts, rates, region, exposure=.005), atol=1e-10)


def test_report_accepts_completed_hash_verified_outputs(tmp_path):
    (tmp_path/"result.csv").write_bytes(b"a,b\n1,2\n")
    value = {"status": "complete", "outputs": {"result.csv": hashlib.sha256(b"a,b\n1,2\n").hexdigest()}}
    (tmp_path/"manifest.json").write_text(json.dumps(value))
    assert verified_manifest(tmp_path) == value
    (tmp_path/"result.csv").write_bytes(b"a,b\n1,3\n")
    with pytest.raises(ValueError, match="changed output"):
        verified_manifest(tmp_path)


def test_report_rejects_incomplete_run(tmp_path):
    (tmp_path/"manifest.json").write_text(json.dumps({"status": "running"}))
    with pytest.raises(ValueError, match="incomplete"):
        verified_manifest(tmp_path)
