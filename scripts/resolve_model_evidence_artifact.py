#!/usr/bin/env python3
"""Resolve a live model-evidence GitHub Actions artifact.

The behavior-alignment workflows used to pin one historical run forever. GitHub
artifacts expire, so a pinned run eventually becomes unusable even though newer
producer runs may exist. This helper selects the newest unexpired
``model-evidence-all-sessions-*`` artifact unless an explicit run/name is
requested.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEFAULT_PREFIX = "model-evidence-all-sessions-"


def select_artifact(
    artifacts: Iterable[Mapping[str, Any]],
    *,
    name_prefix: str = DEFAULT_PREFIX,
    artifact_name: str = "",
) -> Mapping[str, Any]:
    """Return the newest unexpired matching artifact."""

    requested = artifact_name.strip()
    candidates: list[Mapping[str, Any]] = []
    for artifact in artifacts:
        name = str(artifact.get("name", ""))
        if bool(artifact.get("expired", False)):
            continue
        if requested:
            if name != requested:
                continue
        elif not name.startswith(name_prefix):
            continue
        candidates.append(artifact)

    if not candidates:
        target = requested or f"{name_prefix}*"
        raise RuntimeError(
            f"no unexpired GitHub Actions artifact matching {target!r}; "
            "run the model-evidence-all-sessions producer first"
        )

    return max(
        candidates,
        key=lambda artifact: (
            str(artifact.get("created_at", "")),
            int(artifact.get("id", 0)),
        ),
    )


def _api_json(url: str, token: str) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "HippoReplayDynamics-artifact-resolver",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed GitHub API host
            payload = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"GitHub artifact lookup failed with HTTP {exc.code}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("GitHub artifact lookup returned an invalid payload")
    return payload


def _list_artifacts(repository: str, token: str, run_id: str) -> list[dict[str, Any]]:
    if "/" not in repository or repository.startswith("/") or repository.endswith("/"):
        raise ValueError("repository must use owner/name form")
    base = f"https://api.github.com/repos/{repository}/actions"
    if run_id:
        base += f"/runs/{int(run_id)}/artifacts"
    else:
        base += "/artifacts"

    artifacts: list[dict[str, Any]] = []
    page = 1
    while True:
        payload = _api_json(f"{base}?{urlencode({'per_page': 100, 'page': page})}", token)
        batch = payload.get("artifacts", [])
        if not isinstance(batch, list):
            raise RuntimeError("GitHub artifact lookup returned invalid artifact data")
        artifacts.extend(item for item in batch if isinstance(item, dict))
        if len(batch) < 100:
            break
        page += 1
    return artifacts


def _run_id_from_artifact(artifact: Mapping[str, Any], name_prefix: str) -> str:
    workflow_run = artifact.get("workflow_run")
    if isinstance(workflow_run, Mapping) and workflow_run.get("id") is not None:
        return str(int(workflow_run["id"]))

    name = str(artifact.get("name", ""))
    suffix = name.removeprefix(name_prefix)
    if name.startswith(name_prefix) and suffix.isdigit():
        return str(int(suffix))
    raise RuntimeError("selected artifact does not expose a workflow run id")


def resolve_artifact(
    repository: str,
    token: str,
    *,
    run_id: str = "",
    artifact_name: str = "",
    name_prefix: str = DEFAULT_PREFIX,
) -> tuple[str, str]:
    """Resolve and return ``(run_id, artifact_name)``."""

    requested_run = run_id.strip()
    artifacts = _list_artifacts(repository, token, requested_run)
    artifact = select_artifact(
        artifacts,
        name_prefix=name_prefix,
        artifact_name=artifact_name,
    )
    resolved_run = requested_run or _run_id_from_artifact(artifact, name_prefix)
    return resolved_run, str(artifact["name"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--artifact-name", default="")
    parser.add_argument("--name-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT", ""))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
    if not args.repository:
        raise SystemExit("--repository or GITHUB_REPOSITORY is required")
    if not token:
        raise SystemExit("GITHUB_TOKEN or GH_TOKEN is required")

    run_id, artifact_name = resolve_artifact(
        args.repository,
        token,
        run_id=args.run_id,
        artifact_name=args.artifact_name,
        name_prefix=args.name_prefix,
    )
    lines = [f"run_id={run_id}", f"artifact_name={artifact_name}"]
    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
