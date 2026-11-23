"""Construct marked episodes with isolated mechanics and unchanged native goals."""
import copy
import re

from .namespace import TaskNamespace


def install(namespace, task, seed, profile):
    config = dict(copy.deepcopy(profile), task=task, seed=seed,
                  condition='mechanism_mix' if task == 'PatternLock' else 'mixed_motion')
    if profile.get('atomic') == 'container_speed_2x':
        namespace.load('behaviors.container_speed').install(config)
        return config
    if task in ('BinFill', 'PickXtimes', 'StopCube', 'SwingXtimes'):
        namespace.load('behaviors.counting').install(task, seed, config)
    elif task in ('VideoRepick', 'VideoUnmaskSwap', 'ButtonUnmaskSwap'):
        if task == 'ButtonUnmaskSwap':
            config.update(container_motion='original_robomme_plus',
                          style='original container motion after visible cube cluster swaps')
        else:
            namespace.load('behaviors.shared').install(task, seed, config)
        if task in ('ButtonUnmaskSwap', 'VideoUnmaskSwap'):
            namespace.load('behaviors.cube_shuffle').install(seed, config)
    elif task == 'PatternLock':
        namespace.load('behaviors.pattern_mechanism').install(seed, config)
    elif task == 'RouteStick':
        namespace.load('behaviors.tracing_variation').install(task, seed, config)
    elif task in ('ButtonUnmask', 'VideoUnmask'):
        namespace.load('behaviors.static_shuffle').install(task, seed, config)
    else:
        namespace.load('behaviors.place_motion').install(task, seed, config)
    if task in ('ButtonUnmask', 'ButtonUnmaskSwap', 'VideoUnmask', 'VideoUnmaskSwap'):
        namespace.load('behaviors.unmask_visibility').install(task, config)
    return config


def make_env(spec, *, seed, difficulty, profile, **options):
    import gymnasium as gym
    from mani_skill.utils.registration import TimeLimitWrapper

    class OwnedNamespace(gym.Wrapper):
        def close(self):
            try:
                return self.env.close()
            finally:
                namespace.close()

    namespace = TaskNamespace()
    env = None
    try:
        config = install(namespace, spec.name, seed, profile)
        module = re.sub(r'(?<!^)(?=[A-Z])', '_', spec.name).lower()
        cls = getattr(namespace.load(f'tasks.{spec.suite}.{module}'), spec.name)
        env = cls(seed=seed, difficulty=difficulty, **options)
        env.spec = copy.copy(gym.spec(spec.gym_id))
        env._mechanism_variation = config
        return OwnedNamespace(TimeLimitWrapper(env, max_episode_steps=None))
    except Exception:
        try:
            if env is not None:
                env.close()
        finally:
            namespace.close()
        raise
