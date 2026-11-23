import json

import pytest

from benchmark_io.unified_study import environment


def test_provider_ports_are_study_local_and_validated(tmp_path):
    plan = tmp_path/'study_plan.json'
    plan.write_text(json.dumps({'image': 'test-image', 'provider_port_base': 25110}))
    env = environment(tmp_path)
    assert env['ROBOMME_AGENTIC_PROVIDER_BASE'] == '25110'
    plan.write_text(json.dumps({'image': 'test-image'}))
    assert environment(tmp_path)['ROBOMME_AGENTIC_PROVIDER_BASE'] == '23110'
    for invalid in (0, 65500, True, '25110'):
        plan.write_text(json.dumps({'image': 'test-image', 'provider_port_base': invalid}))
        with pytest.raises(ValueError, match='port base'):
            environment(tmp_path)
