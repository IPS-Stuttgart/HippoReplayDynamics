import urllib.request

import pytest

from scripts.prepare_tirole_two_track_dataset import SafeRedirect, credentials, validate_file_metadata


@pytest.mark.parametrize("target,keep", [("https://datadryad.org/next", True), ("https://storage.example/next", False)])
def test_authorization_not_forwarded_cross_host(target, keep):
    req = urllib.request.Request("https://datadryad.org/api/download", headers={"Authorization": "Bearer synthetic-test-token"})
    result = SafeRedirect().redirect_request(req, None, 302, "redirect", {}, target)
    assert (result.get_header("Authorization") is not None) == keep


def test_no_plaintext_redirect():
    req = urllib.request.Request("https://datadryad.org/download")
    with pytest.raises(ValueError):
        SafeRedirect().redirect_request(req, None, 302, "redirect", {}, "http://example.test/next")


def test_credential_names_only_and_shell_not_executed(tmp_path):
    path = tmp_path / "test.env"
    path.write_text("export DRYAD_CLIENT_ID='fake id'\nDRYAD_CLIENT_SECRET='$(touch never-created)'\nUNRELATED=ignored\n")
    result = credentials(path)
    assert result == {"DRYAD_CLIENT_ID": "fake id", "DRYAD_CLIENT_SECRET": "$(touch never-created)"}
    assert not (tmp_path / "never-created").exists()


@pytest.mark.parametrize("name", ["../outside.mat", "/absolute.mat", "..", "a\\b.mat"])
def test_path_validation(name):
    with pytest.raises(ValueError):
        validate_file_metadata({"path": name, "digestType": "sha-256", "digest": "a" * 64, "size": 100})


def test_digest_validation():
    row = {"path": "RAT1_SESS1_extracted_clusters.mat", "digestType": "sha-256", "digest": "a" * 64, "size": 100}
    validate_file_metadata(row)
    row["digest"] = "z" * 64
    with pytest.raises(ValueError):
        validate_file_metadata(row)
