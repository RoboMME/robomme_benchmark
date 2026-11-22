import pytest

from benchmark_io.prepare_unified import select_configurations
from benchmark_io.protocol import setting_key


def test_hybrid_public_name_preserves_saved_study_identifier():
    assert setting_key('Hybrid') == setting_key('hybrid') == setting_key('demo&interaction')
    assert select_configurations(['hybrid'],[10]) == select_configurations(['demo&interaction'],[10])
    with pytest.raises(ValueError,match='distinct'):
        select_configurations(['hybrid','demo&interaction'],[10])


def test_fresh_r3_r5_creates_eight_independent_studies():
    rows=select_configurations(['interaction-only','demo&interaction'],[3,5])
    assert len(rows)==8 and len({r['key'] for r in rows})==8
    assert {r['budget'] for r in rows}=={3,5}
    assert {r['method'] for r in rows}=={'aspire','rax'}
    assert {r['setting'] for r in rows}=={'interaction-only','demo&interaction'}


def test_default_remains_r0_r10():
    rows=select_configurations()
    assert len(rows)==6
    assert all(r['budget']==(0 if r['setting']=='demo-only' else 10) for r in rows)


@pytest.mark.parametrize('budgets',[[],[3,3],[0],[20],[True]])
def test_invalid_budgets_fail_before_preparation(budgets):
    with pytest.raises(ValueError,match='budgets'):select_configurations(budgets=budgets)
