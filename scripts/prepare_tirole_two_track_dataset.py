#!/usr/bin/env python3
"""Fetch the pinned public two-track release; never archive API credentials."""

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

BASE = "https://datadryad.org"
DOI = "doi:10.5061/dryad.ksn02v76h"
VERSION = 238435


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlparse(newurl).scheme != "https":
            raise ValueError("refusing non-HTTPS download redirect")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if urllib.parse.urlparse(req.full_url).netloc != urllib.parse.urlparse(newurl).netloc:
            redirected.remove_header("Authorization")
        return redirected


def open_url(opener, url, data=None, headers=None):
    req = urllib.request.Request(url, data=data, headers={"User-Agent": "HippoReplayDynamics-two-track-audit/1.0", **(headers or {})})
    return opener.open(req, timeout=120)


def get_json(opener, url, data=None):
    body = urllib.parse.urlencode(data).encode() if data is not None else None
    with open_url(opener, url, body) as response:
        return json.load(response)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def credentials(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        tokens = shlex.split(line, comments=True)
        if tokens and tokens[0] == "export":
            tokens = tokens[1:]
        for token in tokens:
            if "=" in token:
                key, value = token.split("=", 1)
                if key in {"DRYAD_CLIENT_ID", "DRYAD_CLIENT_SECRET"}:
                    result[key] = value
    if set(result) != {"DRYAD_CLIENT_ID", "DRYAD_CLIENT_SECRET"}:
        raise ValueError("Dryad credential variable names missing")
    return result


def save_json(path, data):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def validate_file_metadata(row):
    name = row["path"]
    if Path(name).name != name or name in {"", ".", ".."} or "\\" in name:
        raise ValueError("unsafe release filename")
    if row["digestType"] != "sha-256" or len(row["digest"]) != 64 or any(c not in "0123456789abcdef" for c in row["digest"].lower()):
        raise ValueError("SHA256 release digest required")
    if int(row["size"]) <= 0:
        raise ValueError("empty release file")


def run(root, credential_file, include_lfp=False):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        http = urllib.request.build_opener(SafeRedirect())
        state = {"doi": DOI, "version": VERSION, "status": "running", "pid": os.getpid(), "created_at_utc": datetime.now(UTC).isoformat(), "files": []}
        state_path = root / "download_manifest.json"
        try:
            pages = []
            url = BASE + f"/api/v2/versions/{VERSION}/files"
            while url:
                page = get_json(http, url)
                pages.append(page)
                next_url = page["_links"].get("next", {}).get("href")
                url = BASE + next_url if next_url else None
            save_json(root / "release_file_metadata.json", pages)
            files = [r for page in pages for r in page["_embedded"]["stash:files"]]
            if len(files) != pages[0]["total"]:
                raise ValueError("release pagination incomplete")
            chosen = [r for r in files if include_lfp or "_CSC.mat" not in r["path"]]
            c = credentials(credential_file)
            response = get_json(
                http, BASE + "/oauth/token", data={"grant_type": "client_credentials", "client_id": c["DRYAD_CLIENT_ID"], "client_secret": c["DRYAD_CLIENT_SECRET"]}
            )
            token = response["access_token"]
            for row in chosen:
                validate_file_metadata(row)
                dest = root / row["path"]
                valid = dest.exists() and dest.stat().st_size == row["size"] and sha256(dest) == row["digest"]
                if not valid:
                    part = dest.with_suffix(dest.suffix + ".partial")
                    for attempt in range(5):
                        try:
                            with open_url(http, BASE + row["_links"]["stash:download"]["href"], headers={"Authorization": "Bearer " + token}) as download, part.open("wb") as out:
                                for block in iter(lambda: download.read(1024 * 1024), b""):
                                    out.write(block)
                            if part.stat().st_size != row["size"] or sha256(part) != row["digest"]:
                                raise ValueError("release size/digest mismatch: " + row["path"])
                            os.replace(part, dest)
                            break
                        except (urllib.error.URLError, TimeoutError, ValueError):
                            if attempt == 4:
                                raise
                            time.sleep(2**attempt)
                state["files"].append({"path": row["path"], "size": row["size"], "sha256": row["digest"], "verified": True})
                save_json(state_path, state)
                print("VERIFIED", row["path"], row["size"], flush=True)
            duplicate = defaultdict(list)
            for row in state["files"]:
                duplicate[row["sha256"]].append(row["path"])
            state["duplicate_hash_groups"] = [v for v in duplicate.values() if len(v) > 1]
            state["status"] = "complete"
            state["lfp_included"] = include_lfp
            state["total_release_files"] = len(files)
            state["completed_at_utc"] = datetime.now(UTC).isoformat()
            save_json(state_path, state)
            print("COMPLETE", len(chosen), "files; duplicate groups:", state["duplicate_hash_groups"], flush=True)
        except Exception as exc:
            state["status"] = "failed"
            state["error"] = type(exc).__name__ + ": " + str(exc)
            save_json(state_path, state)
            raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True, type=Path)
    p.add_argument("--credential-file", required=True, type=Path)
    p.add_argument("--include-lfp", action="store_true")
    a = p.parse_args()
    run(a.dataset_root, a.credential_file, a.include_lfp)
