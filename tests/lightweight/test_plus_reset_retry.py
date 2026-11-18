"""Reset recovery keeps the episode fixed and never replays policy actions."""

import gymnasium as gym
import pytest

from robomme_plus.evaluation.reset import ResetRetryWrapper


class Environment(gym.Env):
    def __init__(self, error=None):
        self.error = error
        self.closed = False
        self.requests = []

    def reset(self, **kwargs):
        self.requests.append(kwargs.copy())
        if self.error:
            kwargs['options']['mutable'].append('partial reset')
            raise self.error
        return {'frame': 'complete demo'}, {'execution_control_steps': 0}

    def step(self, action):
        raise RuntimeError('policy action failed')

    def close(self):
        self.closed = True


def test_reset_recovers_on_third_attempt_with_clean_environment_and_same_arguments():
    envs = [Environment(RuntimeError('demo grasp failed')) for _ in range(2)] + [Environment()]
    remaining = iter(envs[1:])
    env = ResetRetryWrapper(envs[0], rebuild=lambda: next(remaining))
    options = {'mutable': []}
    obs, info = env.reset(seed=123, options=options)
    assert obs == {'frame': 'complete demo'}
    assert info == {'execution_control_steps': 0, 'reset_attempts': 3}
    assert [e.requests[0]['seed'] for e in envs] == [123, 123, 123]
    assert options == {'mutable': []}
    assert envs[2].requests[0]['options'] == options
    assert all(e.closed for e in envs[:2]) and not envs[2].closed
    assert env.unwrapped is envs[2]


def test_reset_exhaustion_reraises_original_error_after_exactly_three_attempts():
    errors = [RuntimeError(str(i)) for i in range(3)]
    envs = [Environment(e) for e in errors]
    remaining = iter(envs[1:])
    env = ResetRetryWrapper(envs[0], rebuild=lambda: next(remaining))
    with pytest.raises(RuntimeError) as caught:
        env.reset(options={'mutable': []})
    assert caught.value is errors[-1]
    assert env.reset_attempts == 3
    assert all(e.closed for e in envs)


@pytest.mark.parametrize('error', [ValueError('wrong seed'), TypeError('bad options'),
                                 MemoryError(), RuntimeError('CUDA error: out of memory')])
def test_invalid_arguments_and_memory_errors_do_not_retry_in_same_process(error):
    def unexpected():
        pytest.fail('unexpected rebuild')
    env = ResetRetryWrapper(Environment(error), rebuild=unexpected)
    with pytest.raises(type(error)):
        env.reset(options={'mutable': []})
    assert env.reset_attempts == 1


def test_success_and_step_error_never_trigger_rebuild():
    def unexpected():
        pytest.fail('unexpected rebuild')
    env = ResetRetryWrapper(Environment(), rebuild=unexpected)
    assert env.reset()[1]['reset_attempts'] == 1
    with pytest.raises(RuntimeError, match='policy action'):
        env.step(None)
    assert env.reset_attempts == 1


def test_rebuild_errors_count_toward_total_attempt_limit():
    builds = []
    def rebuild():
        builds.append(1)
        raise RuntimeError('scene initialization failed')
    env = ResetRetryWrapper(Environment(RuntimeError('demo failed')), rebuild=rebuild)
    with pytest.raises(RuntimeError, match='scene initialization'):
        env.reset(options={'mutable': []})
    assert len(builds) == 2 and env.reset_attempts == 3


def test_training_cap_does_not_rebuild_after_one_failed_start():
    original=Environment(RuntimeError('setup failed'))
    rebuilt=[]
    env=ResetRetryWrapper(original,rebuild=lambda:rebuilt.append(True))
    env.limit_reset_attempts(1)
    with pytest.raises(RuntimeError,match='setup failed'):
        env.reset(options={'mutable':[]})
    assert env.reset_attempts==1 and not rebuilt


@pytest.mark.parametrize('limit',[0,4,True,1.5])
def test_invalid_reset_retry_caps_fail_before_simulation(limit):
    env=ResetRetryWrapper(Environment(),rebuild=lambda:Environment())
    with pytest.raises(ValueError):env.limit_reset_attempts(limit)
