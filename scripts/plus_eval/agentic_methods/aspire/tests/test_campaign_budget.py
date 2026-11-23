import json
from types import SimpleNamespace

import pytest

from aspire.campaign import Campaign, prepare
from benchmark_io.protocol import Study


def campaign(tmp_path,setting='interaction-only',budget=2):
    study=Study(tmp_path/'study',method='aspire',setting=setting,budget=budget,
                records={'BinFill':[dict(seed=i,max_steps=100) for i in range(3)]})
    token=tmp_path/'token';token.write_text('test-placeholder')
    args=SimpleNamespace(output=tmp_path/'campaign',study=study.root,token_file=token,
        resume=False,gpus=['0'],provider_port_base=18110,demos=None)
    return Campaign(args)


def test_campaign_resume_preserves_spent_training_budget(tmp_path):
    session=campaign(tmp_path)
    attempt=session.study.reserve('BinFill',purpose='failed_setup')
    session.study.finish(attempt['id'],dict(success=False,control_steps=0))
    assert session.status()['tasks'][0]['starts_remaining']==1
    session.args.resume=True
    resumed=Campaign(session.args)
    assert resumed.status()['tasks'][0]['starts_remaining']==1


def test_fresh_native_campaign_can_be_prepared_without_api_or_simulator(tmp_path,monkeypatch):
    session=campaign(tmp_path)
    def forbidden(*args,**kwargs):raise AssertionError('No simulator starts during provisioning')
    monkeypatch.setattr(session,'run',forbidden)
    model,home,bridge=prepare(session)
    assert model=='us.anthropic.claude-opus-5-5'
    assert '512 timestamps' in (session.workspace/'docs/robomme-plus.md').read_text()
    assert (session.workspace/'.claude/skills/iterative-debugging/main-agent-prompt.md').is_file()
    assert json.loads((session.workspace/'campaign/campaign.json').read_text())['budget_per_task']==2
    assert session.study.attempts()==[]
    assert (bridge/'cap-harness').is_file()
    assert (home/'settings.json').is_file()


def test_test_label_cannot_bypass_the_development_study(tmp_path):
    from aspire.isolated import run_program
    session=campaign(tmp_path)
    with pytest.raises(ValueError,match='dataset must match'):
        run_program(study_root=session.study.root,output_root=tmp_path/'out',task='BinFill',
            episode=0,code='result = 1',image='unused',dataset='test')
    assert session.study.attempts()==[]
