"""Reject invisible initial targets and audit actual front/wrist cue frames."""
MIN_PIXELS = 12  # Preserve the native RoboMME-Plus setup threshold.


def install(task, config):
    from robomme_plus.tasks.permanence.utils import tasks
    from robomme_plus.tasks.permanence.utils.environment import PermanenceTask
    from robomme_plus.common.sim import step_number

    config['target_visibility_filter'] = dict(minimum_pixels=MIN_PIXELS,
        cameras=['front', 'wrist'], scope='each target cube before masking',
        all_same_color_targets_checked_individually=True)
    original_setup, original_step = tasks.setup, PermanenceTask.step

    def state(env):
        return getattr(env, '_visible_shuffle', None) or getattr(env, '_static_shuffle', None)

    def passed(counts):
        return all(max(a,b)>=MIN_PIXELS for a,b in zip(counts['front'],counts['wrist']))

    def setup(env, settings):
        original_setup(env, settings)
        cubes=env._permanence_cubes[:settings['recipe']['total_targets']]
        counts=tasks.pixels(env,cubes)
        report=dict(minimum_pixels=MIN_PIXELS,target_names=[c.name for c in cubes],
            initial_pixels=counts,initial_filter_passed=passed(counts),
            peak_pixels={view:[0]*len(cubes) for view in ('front','wrist')},
            first_visible_control_step=[None]*len(cubes),samples=[],
            observed_before_mask_passed=False,masked_at_control_step=None,
            observation_phase='demonstration' if task.startswith('Video') else 'execution')
        env._unmask_visibility=report
        env._limit_diagnostics['target_visibility']=report
        if not report['initial_filter_passed']:
            raise RuntimeError(f'Visibility filter rejected {task} seed {env.seed}: {counts}')

    def step(env, action):
        result=original_step(env,action)
        report=getattr(env,'_unmask_visibility',None)
        shuffle=state(env)
        if report is None or shuffle is None or report['masked_at_control_step'] is not None:return result
        covered=shuffle['audit']['covered_step']
        if covered is not None:
            report['masked_at_control_step']=int(covered)
            # Video cues are generated before a policy receives the environment.
            # Reject an invalid cue, rather than score it as a policy failure.
            if task.startswith('Video') and not report['observed_before_mask_passed']:
                raise RuntimeError(f'Visibility filter rejected {task} video: a target was never visible')
            return result
        if task.startswith('Video') and not getattr(env,'demonstration_record_traj',False):return result
        cubes=env._permanence_cubes[:env._limit_config['recipe']['total_targets']]
        counts=tasks.pixels(env,cubes)
        now=int(step_number(env))
        report['samples'].append(dict(control_step=now,pixels=counts))
        for i in range(len(cubes)):
            for view in ('front','wrist'):
                report['peak_pixels'][view][i]=max(report['peak_pixels'][view][i],counts[view][i])
            if max(counts['front'][i],counts['wrist'][i])>=MIN_PIXELS and report['first_visible_control_step'][i] is None:
                report['first_visible_control_step'][i]=now
        report['observed_before_mask_passed']=passed(report['peak_pixels'])
        return result

    tasks.setup,PermanenceTask.step=setup,step
