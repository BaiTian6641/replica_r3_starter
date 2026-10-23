"""Exact workspace-constant prompt evidence is visible without expected values."""

import json
from pathlib import Path

from tests.e3_runtime_helpers import make_runtime

def test_prompt_lists_current_constants_but_not_reference_value():
    import tempfile
    from tests.e3_runtime_helpers import make_case
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tools = make_case(root)
        # Replace the fixture with a set_constant-shaped workspace.
        (root / 'cases/runtime-case/workspace/config.json').write_text(
            'MAX_RETRIES = 1\nBACKOFF_SECONDS = 0.2\n')
        case_path = root / 'cases/runtime-case/case.json'
        case = json.loads(case_path.read_text())
        case['allowed_tools'] = ['read_file', 'set_constant', 'run_test']
        case['workspace_entry'] = 'workspace/config.json'
        case_path.write_text(json.dumps(case))
        tools = __import__('replica_lab.runtime.tools', fromlist=['FixtureWorkspace']).FixtureWorkspace(
            root / 'cases/runtime-case')
        runtime = make_runtime(tools, root / 'run')
        prompt = runtime._build_prompt(runtime.state)
        assert 'EXACT WORKSPACE CONSTANTS' in prompt
        assert 'MAX_RETRIES=1' in prompt and 'BACKOFF_SECONDS=0.2' in prompt
        assert '0.25' not in prompt
