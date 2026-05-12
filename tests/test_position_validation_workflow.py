from pathlib import Path


def test_position_validation_workflow_defaults_smoke_capped_full_uncapped():
    workflow = Path(".github/workflows/position-validation.yml").read_text(encoding="utf-8")

    assert "name: Validate position decoding" in workflow
    assert 'description: "Maximum windows per session; empty means smoke uses 500 and full is uncapped"' in workflow
    assert 'default: ""' in workflow
    assert 'if [ -z "${effective_max_windows}" ] && [ "${MODE}" = "smoke" ]; then' in workflow
    assert 'effective_max_windows="500"' in workflow
    assert 'args+=(--max-windows "${effective_max_windows}")' in workflow
    assert 'Position-validation max windows per session: ${effective_max_windows:-uncapped}' in workflow
