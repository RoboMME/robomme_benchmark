#!/usr/bin/env python3
"""One-shot Embodied-Astra: a successful train video followed by a new test episode."""
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(HERE.parent), str(ROOT / 'src')]

# An explicit module name avoids importing another method's eval_plus.py.
spec = importlib.util.spec_from_file_location('embodied_astra_base', HERE / 'eval_plus.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

from astra_image_sampling import MAX_ONE_SHOT_CONTEXT_IMAGES, image_count
from evaluation import common_parser, write_json
from one_shot_examples import (DEFAULT_TRAINING_ROOT, TRAINING_OBSERVATION_CONTEXT_VERSION,
                               TrainingExamples)

PROMPT_FILE = HERE / 'one_shot_prompt.txt'
IMAGE_ONLY_PROMPT_FILE = HERE / 'one_shot_images_prompt.txt'


def includes_training_state(args):
    return getattr(args, 'training_context', 'state') == 'state'


def test_seeds():
    seeds = set()
    for path in (ROOT / 'src/robomme_plus/env_metadata/test').glob('*.json'):
        document = json.loads(path.read_text())
        seeds.update(r['seed'] for r in document.get('records', []))
    return seeds


def training_examples(args):
    return TrainingExamples(args.training_root, max_images=args.training_max_images,
                            frame_stride=args.frame_stride, test_seeds=test_seeds(),
                            difficulty=getattr(args, 'training_difficulty', 'medium'))


def validate_training(args):
    examples = training_examples(args)
    tasks = args.tasks or list(dict.fromkeys(row['task'] for row in examples.records))
    report = {'training_samples': len(examples.records),
              'training_plan_sha256': examples.plan_sha256,
              'max_first_turn_images': MAX_ONE_SHOT_CONTEXT_IMAGES,
              'max_training_images': args.training_max_images, 'tasks': {}}
    for task in tasks:
        example = examples.get(task)
        report['tasks'][task] = (example.report if example else
                                 {'applied': False, 'reason': 'excluded_by_protocol'})
    report['passed'] = True
    write_json(args.output_dir / 'one_shot_training_validation.json', report)
    print(f"Verified {sum(r.get('applied', False) for r in report['tasks'].values())} "
          f"training examples from {len(examples.records)} samples; MoveCube has no example.")
    return report


class OneShotEmbodiedAstraEvaluation(base.EmbodiedAstraEvaluation):
    method = 'embodied-astra-fs'
    source_dependencies = (HERE / 'eval_plus.py', HERE / 'one_shot_examples.py',
                           PROMPT_FILE, IMAGE_ONLY_PROMPT_FILE,
                           HERE.parent / 'astra_image_sampling.py')

    def training_examples(self):
        return training_examples(self.args)

    def setup(self):
        self.adaptation_prompt = (PROMPT_FILE if includes_training_state(self.args)
                                  else IMAGE_ONLY_PROMPT_FILE).read_text()
        self.examples = None if self.args.tasks == ['MoveCube'] else self.training_examples()
        if self.examples is not None:
            tasks = self.args.tasks or list(dict.fromkeys(r['task'] for r in self.examples.records))
            for task in tasks:
                self.examples.selected_record(task)  # Fail setup if a medium sample is missing.
        self.training_content_key = None
        self.training_content = []
        super().setup()

    def configuration(self):
        configuration = super().configuration()
        selected = {}
        if self.examples is not None:
            tasks = self.args.tasks or list(dict.fromkeys(r['task'] for r in self.examples.records))
            for task in tasks:
                row, reason = self.examples.selected_record(task)
                selected[task] = ({k: row.get(k) for k in ('task', 'episode', 'seed', 'difficulty')}
                                  | {'selection': reason}) if row else {'selection': reason}
        configuration.update(
            original_prompts_preserved=False,
            demonstration_image_sampling={
                'max_initial_images': MAX_ONE_SHOT_CONTEXT_IMAGES,
                'training_max_images': self.args.training_max_images,
                'scope': 'first_turn_training_and_test',
                'sampling': 'uniform_with_phase_endpoints',
                'includes_current_wrist': True,
            },
            adaptation={
                'version': TRAINING_OBSERVATION_CONTEXT_VERSION if includes_training_state(self.args) else 1,
                'training_context': getattr(self.args, 'training_context', 'state'),
                'training_difficulty': getattr(self.args, 'training_difficulty', 'medium'),
                'examples_per_task': 1, 'training_selection': selected,
                'training_plan_sha256': self.examples.plan_sha256 if self.examples else None,
                'training_image_limit': self.args.training_max_images,
                'first_turn_image_limit': MAX_ONE_SHOT_CONTEXT_IMAGES,
                'test_video_image_limit': 'remaining first-turn budget after training images',
                'excluded_training_tasks': ['MoveCube'], 'insert_peg_episode': 1,
                'prompt': self.adaptation_prompt,
                'prompt_sha256': hashlib.sha256(self.adaptation_prompt.encode()).hexdigest(),
            })
        return configuration

    def system_prompt(self, info):
        return super().system_prompt(info) + '\n\n' + self.adaptation_prompt

    def demonstration_image_limit(self):
        training_count = self.example.report['input_images'] if self.example is not None else 0
        return MAX_ONE_SHOT_CONTEXT_IMAGES - training_count

    def start_episode(self, obs, info):
        self.example = self.examples.get(self.task) if self.examples is not None else None
        self.one_shot_report = (dict(self.example.report) if self.example else
                                {'applied': False, 'task': self.task, 'input_images': 0,
                                 'reason': 'excluded_by_protocol'})
        self.one_shot_report.update(max_first_turn_images=MAX_ONE_SHOT_CONTEXT_IMAGES,
                                    first_turn_images=None,
                                    training_context=getattr(self.args, 'training_context', 'state'),
                                    observation_context_version=(TRAINING_OBSERVATION_CONTEXT_VERSION
                                        if includes_training_state(self.args) else 1))
        self.one_shot_directory = self.recorder.directory
        super().start_episode(obs, info)

    def initial_content(self, obs, info):
        if self.example is None:
            training = [self.harness.text_item('TRAINING EXAMPLE: omitted for MoveCube.')]
        else:
            key = self.example.report['sample_sha256']
            if self.training_content_key != key:
                self.training_content = self.example.content(
                    self.harness, include_state=includes_training_state(self.args))
                self.training_content_key = key
            training = self.training_content
        content = training + [self.harness.text_item('CURRENT TEST EPISODE — new independent scene.')] + super().initial_content(obs, info)
        count = image_count(content)
        if count > MAX_ONE_SHOT_CONTEXT_IMAGES:
            raise ValueError('One-shot context exceeds its first-turn image budget')
        self.one_shot_report['first_turn_images'] = count
        self.one_shot_report['test_input_images'] = self.demo_sampling['input_images']
        return content

    def end_episode(self, result):
        super().end_episode(result)
        if getattr(self, 'one_shot_directory', None) == self.recorder.directory:
            result['one_shot_adaptation'] = self.one_shot_report
            write_json(self.recorder.directory / 'one_shot_adaptation.json', self.one_shot_report)


def main():
    parser = common_parser(__doc__)
    # Record this inference setting even before setup, when resume is checked.
    parser.set_defaults(training_observation_context_version=TRAINING_OBSERVATION_CONTEXT_VERSION)
    parser.add_argument('--model', default='gpt-6-astra')
    parser.add_argument('--frame-stride', type=int, default=3)
    parser.add_argument('--prompt-variant', choices=('base', 'lr_note'), default='lr_note')
    parser.add_argument('--training-root', type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument('--training-context', choices=('images', 'state'), default='images',
                        help='Training example images only (default), or images with robot state and calibration.')
    parser.add_argument('--training-difficulty', choices=('easy', 'medium', 'hard'), default='medium',
                        help='Select one example at this difficulty (default: medium).')
    parser.add_argument('--training-max-images', type=int, default=550,
                        help='Training video share of the 1,100-image first-turn context (default: 550).')
    parser.add_argument('--validate-training', action='store_true',
                        help='Check sample selection, phase labels and RGB inputs without simulator or API calls.')
    args = parser.parse_args()
    args.training_observation_context_version = (TRAINING_OBSERVATION_CONTEXT_VERSION
                                                if includes_training_state(args) else 1)
    if args.frame_stride < 1:
        parser.error('--frame-stride must be positive')
    if not 4 <= args.training_max_images <= MAX_ONE_SHOT_CONTEXT_IMAGES - 3:
        parser.error('--training-max-images must be between 4 and 1097')
    if args.validate_training:
        validate_training(args)
        return
    if not base.HARNESS.is_file():
        parser.error('Initialize the embodied-astra-robomme submodule.')
    asyncio.run(OneShotEmbodiedAstraEvaluation(args).run())


if __name__ == '__main__':
    main()
