import re
from pathlib import Path


def test_workflow_dispatch_inputs_stay_within_github_limit():
    workflow_dir = Path(".github/workflows")
    offenders: dict[str, int] = {}
    for workflow in workflow_dir.glob("*.yml"):
        text = workflow.read_text(encoding="utf-8")
        if "workflow_dispatch:" not in text:
            continue
        input_count = _workflow_dispatch_input_count(text)
        if input_count > 25:
            offenders[str(workflow)] = input_count

    assert not offenders


def _workflow_dispatch_input_count(workflow: str) -> int:
    """Count first-level workflow_dispatch inputs in a GitHub Actions file."""

    match = re.search(
        r"workflow_dispatch:\n\s+inputs:\n(?P<inputs>.*?)(?:\n[a-zA-Z_][^\n]*:|\Z)",
        workflow,
        flags=re.S,
    )
    if match is None:
        return 0
    return len(
        re.findall(
            r"^      [A-Za-z0-9_]+:\s*$",
            match.group("inputs"),
            flags=re.M,
        )
    )
