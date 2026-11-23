"""Select training-only videos and preserve their demo/execution boundaries."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


DEFAULT_TRAINING_ROOT = Path('data/plus_train')
TRAINING_OBSERVATION_CONTEXT_VERSION = 2


def text_value(value):
    array = np.asarray(value).reshape(-1)
    if len(array) != 1:
        raise ValueError('Expected a single training metadata value')
    value = array[0]
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)


def numeric_value(group, name, shape, decimals=4):
    """Read an observed state/calibration field without silently inventing values."""
    if name not in group:
        raise ValueError(f'Missing training observation field: {group.name}/{name}')
    values = np.asarray(group[name][()], dtype=np.float64)
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError(f'Invalid training observation field: {group.name}/{name}; '
                         f'expected finite values with shape {shape}')
    return np.round(values, decimals).tolist()


def uniform_indices(start, stop, count):
    """Sample a nonempty phase including its first and last recorded frames."""
    length = stop - start
    if length <= 0 or count < min(2, length) or count > length:
        raise ValueError('Invalid phase sampling budget')
    if count == 1:
        return [start]
    return [start + i * (length - 1) // (count - 1) for i in range(count)]


def phase_sampling(frames, demonstration_frames, max_images, frame_stride):
    if frames < 1 or not 0 <= demonstration_frames < frames:
        raise ValueError('Training episode must contain an execution phase')
    if max_images < 4 or frame_stride < 1:
        raise ValueError('Training budget must be at least four images; stride must be positive')
    phases = [('video_demo', 0, demonstration_frames),
              ('execution', demonstration_frames, frames)]
    active = [(name, start, stop) for name, start, stop in phases if stop > start]
    desired = [min(stop - start, (stop - start - 1 + frame_stride - 1) // frame_stride + 1)
               for _, start, stop in active]
    allocation = [min(2, stop - start) for _, start, stop in active]
    total = min(max_images, sum(desired))
    # Allocate proportionally to each phase's sampled length, with two endpoints
    # guaranteed even for a brief execution following a long demonstration.
    while sum(allocation) < total:
        candidates = [i for i in range(len(active)) if allocation[i] < desired[i]]
        chosen = max(candidates, key=lambda i: desired[i] / (allocation[i] + 1))
        allocation[chosen] += 1
    return {name: uniform_indices(start, stop, count)
            for (name, start, stop), count in zip(active, allocation)}


@dataclass
class TrainingExample:
    report: dict
    phases: dict

    def content(self, harness, *, include_state=True):
        report = self.report
        content = [harness.text_item(
            f"BEGIN TRAINING EXAMPLE — task {report['task']}; training goal: {report['task_goal']}\n"
            "This is a separate successful training episode. Each image is front camera LEFT, "
            "wrist camera RIGHT. The following phases cover the complete recorded sample.")]
        if include_state:
            content.append(harness.text_item(
                'TRAINING CAMERA INTRINSICS: '
                + json.dumps(report['camera_calibration'], allow_nan=False)))
        if not self.phases.get('video_demo'):
            content.append(harness.text_item(
                'TRAINING VIDEO DEMO: absent. This task has no separate recorded demo phase.'))
        for phase, frames in self.phases.items():
            label = 'TRAINING VIDEO DEMO' if phase == 'video_demo' else 'TRAINING EXECUTION'
            detail = report['phases'][phase]
            content.append(harness.text_item(
                f"{label}: {len(frames)} chronological images spanning source frames "
                f"{detail['start_frame']}–{detail['stop_frame_exclusive'] - 1}. "
                f"Selected source indices: {detail['frame_indices']}. "
                f"Simulation times (seconds): {detail['times_seconds']}"))
            observations = detail['observations']
            if len(observations) != len(frames):
                raise ValueError('Training images and robot observations are not aligned')
            for frame, observation in zip(frames, observations):
                if include_state:
                    content.append(harness.text_item(
                        'TRAINING FRAME STATE AND CAMERA EXTRINSICS (for the next image): '
                        + json.dumps(observation, allow_nan=False)))
                content.append(harness.image_item(frame))
        content.append(harness.text_item(
            'END TRAINING EXAMPLE. Its execution was verified successful. '
            'The next section is a new independent test episode; use its own goal and observations.'))
        return content


class TrainingExamples:
    def __init__(self, root=DEFAULT_TRAINING_ROOT, max_images=550, frame_stride=3, test_seeds=(),
                 difficulty='medium'):
        self.root = Path(root)
        self.max_images, self.frame_stride = max_images, frame_stride
        if difficulty not in ('easy', 'medium', 'hard'):
            raise ValueError('Training difficulty must be easy, medium or hard')
        self.difficulty = difficulty
        if max_images < 4 or frame_stride < 1:
            raise ValueError('Invalid training video image budget or frame stride')
        plan = self.root / 'train_plan.json'
        raw = plan.read_bytes()
        self.plan_sha256 = hashlib.sha256(raw).hexdigest()
        self.records = json.loads(raw)
        if not isinstance(self.records, list) or not self.records:
            raise ValueError('Training plan must be a nonempty list')
        seeds = [r['seed'] for r in self.records]
        keys = [(r['task'], r['episode']) for r in self.records]
        if len(set(seeds)) != len(seeds) or len(set(keys)) != len(keys):
            raise ValueError('Training plan has duplicate seeds or episodes')
        if set(seeds) & set(test_seeds):
            raise ValueError('Training examples overlap the test seeds')
        self.cache = {}

    def selected_record(self, task):
        if task == 'MoveCube':
            return None, 'excluded_by_protocol'
        records = [r for r in self.records if r['task'] == task]
        if task == 'InsertPeg':
            selected = [r for r in records if r['episode'] == 1]
            reason = 'user_selected_episode_1_unlevelled'
        else:
            selected = [r for r in records if r.get('difficulty') == self.difficulty]
            reason = f'{self.difficulty}_difficulty'
        if len(selected) != 1:
            raise ValueError(f'{task}: expected exactly one training example ({reason})')
        return selected[0], reason

    def get(self, task):
        if task in self.cache:
            return self.cache[task]
        row, reason = self.selected_record(task)
        if row is None:
            return None
        file = self.root / 'h5' / f'record_dataset_{task}.h5'
        with h5py.File(file, 'r') as source:
            if source.attrs.get('split') != 'train' or source.attrs.get('task') != task:
                raise ValueError(f'{task}: expected a task-specific training H5 file')
            episode = source[f"episode_{row['episode']}"]
            if not bool(episode.attrs.get('success', False)):
                raise ValueError(f'{task}: training example is not verified successful')
            setup = episode['setup']
            if int(setup['seed'][()]) != row['seed'] or text_value(setup['task'][()]) != task:
                raise ValueError(f'{task}: training plan and H5 identity disagree')
            difficulty = text_value(setup['difficulty'][()])
            if difficulty != row.get('difficulty') and task != 'InsertPeg':
                raise ValueError(f'{task}: training plan and H5 difficulty disagree')
            if task != 'InsertPeg' and difficulty != self.difficulty:
                raise ValueError(f'{task}: H5 example is not {self.difficulty} difficulty')
            frames = int(episode.attrs['frames'])
            demonstration_frames = int(episode.attrs['demonstration_frames'])
            hz = int(setup['control_freq'][()])
            if hz <= 0:
                raise ValueError(f'{task}: invalid training control frequency')
            calibration = {
                f'{camera}_camera_K': numeric_value(
                    setup, f'{camera}_camera_intrinsic', (3, 3), decimals=2)
                for camera in ('front', 'wrist')
            }
            calibration.update(
                pixel_coordinates='local_to_each_camera_image_before_concatenation',
                extrinsics_convention='world_to_camera_opencv',
            )
            # Check the entire phase boundary; never infer labels from images,
            # action values, or privileged oracle/subgoal annotations.
            for i in range(frames):
                demo = bool(episode[f'timestep_{i}/info/is_video_demo'][()])
                if demo != (i < demonstration_frames):
                    raise ValueError(f'{task}: inconsistent training phase at frame {i}')
            selections = phase_sampling(frames, demonstration_frames, self.max_images, self.frame_stride)
            phases, reports = {}, {}
            digest = hashlib.sha256()
            for name, indices in selections.items():
                images, times, observations = [], [], []
                for i in indices:
                    timestep = episode[f'timestep_{i}']
                    front = np.asarray(timestep['obs/front_rgb'][()])
                    wrist = np.asarray(timestep['obs/wrist_rgb'][()])
                    if (front.dtype != np.uint8 or wrist.dtype != np.uint8 or
                            front.ndim != 3 or front.shape != wrist.shape or front.shape[-1] != 3):
                        raise ValueError(f'{task}: invalid RGB training frame {i}')
                    image = np.concatenate([front, wrist], axis=1)
                    image_size = [front.shape[1], front.shape[0]]
                    if calibration.setdefault('per_camera_image_size_wh', image_size) != image_size:
                        raise ValueError(f'{task}: inconsistent training image dimensions at frame {i}')
                    images.append(image)
                    digest.update(image.tobytes())
                    step = timestep['info'].get('native_control_step')
                    times.append(round((int(step[()]) if step is not None else i) / hz, 4))
                    observation = {
                        'source_frame': i, 'time_seconds': times[-1],
                        'tcp_pose_xyz_rpy': numeric_value(timestep['obs'], 'eef_state', (6,)),
                        'finger_positions_m': numeric_value(timestep['obs'], 'gripper_state', (2,)),
                    }
                    for camera in ('front', 'wrist'):
                        observation[f'{camera}_camera_extrinsic_world_to_camera'] = numeric_value(
                            timestep['obs'], f'{camera}_camera_extrinsic', (3, 4))
                    observations.append(observation)
                phases[name] = images
                reports[name] = {
                    'start_frame': 0 if name == 'video_demo' else demonstration_frames,
                    'stop_frame_exclusive': demonstration_frames if name == 'video_demo' else frames,
                    'frame_indices': indices, 'times_seconds': times,
                    'input_images': len(indices),
                    'observations': observations,
                }
            report = {
                'applied': True, 'task': task, 'episode': row['episode'], 'seed': row['seed'],
                'difficulty': row.get('difficulty'), 'selection': reason,
                'task_goal': text_value(setup['task_goal'][()]), 'split': 'train',
                'control_freq': hz, 'source_frames': frames,
                'source_demonstration_frames': demonstration_frames,
                'source_execution_frames': frames - demonstration_frames,
                'input_images': sum(len(v) for v in phases.values()),
                'max_training_images': self.max_images, 'frame_stride': self.frame_stride,
                'camera_layout': 'front_left_wrist_right', 'phases': reports,
                'observation_context_version': TRAINING_OBSERVATION_CONTEXT_VERSION,
                'camera_calibration': calibration,
                'training_plan_sha256': self.plan_sha256,
            }
            digest.update(json.dumps(report, sort_keys=True).encode())
            report['sample_sha256'] = digest.hexdigest()
        example = TrainingExample(report, phases)
        # Evaluation visits tasks sequentially. Retain only the current sample,
        # rather than keeping every task's decoded RGB video in memory.
        self.cache = {task: example}
        return example
