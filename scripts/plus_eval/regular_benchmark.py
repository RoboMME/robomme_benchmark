"""Regular RoboMME splits through the baseline runners' builder protocol."""
import importlib
import json
from pathlib import Path
import sys


class CountedEnvironment:
    def __init__(self, env):
        self.env = env
        self.baseline = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def decorate(self, info):
        steps = max(0, int(self.env.get_wrapper_attr('steps_without_demonstration')) - self.baseline)
        return {**info, 'execution_control_steps': steps, 'execution_total_control_steps': steps}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.baseline = int(self.env.get_wrapper_attr('steps_without_demonstration'))
        return obs, self.decorate(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, self.decorate(info or {})

    def close(self):
        self.env.close()


def builder_class(source, dataset):
    if dataset not in ('test', 'ood_test'):
        raise ValueError('Regular baseline evaluation requires test or ood_test')
    source = Path(source).resolve()
    sys.path.insert(0, str(source.parent))
    package = importlib.import_module('robomme')
    if not Path(package.__file__).resolve().is_relative_to(source):
        raise ValueError('A different RoboMME package is already imported')
    importlib.import_module('robomme.robomme_env')
    native = importlib.import_module('robomme.env_record_wrapper').BenchmarkEnvBuilder
    selected_split = dataset

    class RegularBuilder:
        @classmethod
        def get_task_list(cls):
            return native.get_task_list(selected_split)

        def __init__(self, task, dataset='test', action_space='joint_angle', gui_render=False):
            if dataset != selected_split:
                raise ValueError('Evaluation split does not match the pinned builder')
            self.task, self.dataset = task, dataset
            self.max_steps = 1500 if dataset == 'ood_test' else 1300
            metadata = source / 'env_metadata' / dataset
            self.records = json.loads((metadata / f'record_dataset_{task}_metadata.json').read_text())['records']
            self.native = native(task, dataset=dataset, action_space=action_space, gui_render=gui_render,
                                 max_steps=self.max_steps, override_metadata_path=metadata)

        def get_episode_num(self):
            return self.native.get_episode_num()

        def episode_max_steps(self, episode):
            self.get_episode_metadata(episode)
            # Native termination remains authoritative; its constructor adds two
            # reset steps to the same split-specific budget used by CaM.
            return self.max_steps + 2

        def get_episode_metadata(self, episode):
            record = next(r for r in self.records if int(r['episode']) == episode)
            return {**record, 'task': self.task, 'benchmark': 'robomme', 'dataset': self.dataset,
                    'control_freq': 20, 'max_steps': self.max_steps + 2,
                    'native_max_steps_argument': self.max_steps}

        def make_env_for_episode(self, episode, **options):
            expected = self.get_episode_metadata(episode)
            if self.native.resolve_episode(episode) != (int(expected['seed']), expected['difficulty']):
                raise ValueError('Native identity disagrees with pinned metadata')
            return CountedEnvironment(self.native.make_env_for_episode(episode, **options))

    return RegularBuilder
