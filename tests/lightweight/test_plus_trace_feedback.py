"""Only the demo adds a final-cue interval; execution keeps its original gate."""
from types import SimpleNamespace
import pytest
import torch
from robomme_plus.tasks.imitation.utils.tracing import RED_TARGET, final_feedback_ready


@pytest.mark.parametrize('control_timestep', [.05, .01])
def test_demo_waits_two_steps_and_resets_when_hidden(control_timestep):
    target=object()
    highlight=dict(is_highlighted=True,highlight_disk=object(),highlight_color_key=RED_TARGET)
    env=SimpleNamespace(selected_buttons=[target],_highlight_cache={id(target):highlight},
        _drawing=SimpleNamespace(phase='demonstration'),_trace_completion_started={},
        elapsed_steps=torch.tensor([100]),control_timestep=control_timestep)
    for _ in range(20):assert not final_feedback_ready(env)
    env.elapsed_steps+=1
    assert not final_feedback_ready(env)
    env.elapsed_steps+=1
    assert final_feedback_ready(env)
    env.elapsed_steps+=1
    highlight['is_highlighted']=False
    assert not final_feedback_ready(env)
    highlight['is_highlighted']=True
    assert not final_feedback_ready(env)
    env.elapsed_steps+=2
    assert final_feedback_ready(env)
    highlight['highlight_disk']=None
    assert not final_feedback_ready(env)


def test_execution_has_no_added_wait_but_still_requires_live_final_cue():
    target=object()
    highlight=dict(is_highlighted=True,highlight_disk=object(),highlight_color_key=RED_TARGET)
    env=SimpleNamespace(selected_buttons=[target],_highlight_cache={id(target):highlight},
        _drawing=SimpleNamespace(phase='execution'),_trace_completion_started={},
        elapsed_steps=torch.tensor([100]))
    for _ in range(20):assert final_feedback_ready(env)
    assert env._trace_completion_started=={}
    highlight['is_highlighted']=False
    assert not final_feedback_ready(env)
    highlight['is_highlighted']=True
    highlight['highlight_disk']=None
    assert not final_feedback_ready(env)
