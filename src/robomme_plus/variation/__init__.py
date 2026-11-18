"""Optional, seeded perceptual changes for selected test episodes."""

TASKS=frozenset(('BinFill','ButtonUnmask','ButtonUnmaskSwap','PatternLock','PickHighlight',
                 'PickXtimes','StopCube','SwingXtimes','VideoPlaceButton','VideoPlaceOrder',
                 'VideoRepick','VideoUnmask','VideoUnmaskSwap','RouteStick'))
PROFILES=('non_lighting','full')


def validate_profile(task,difficulty,profile):
    if profile is None:
        return
    if isinstance(profile,dict):
        from .atomic import validate
        validate(task,profile)
        if difficulty not in ('easy','medium','hard'):
            raise ValueError('Atomic perceptual variation requires a tiered episode')
        return
    expected={'medium':'non_lighting','hard':'full'}.get(difficulty)
    if task not in TASKS or not isinstance(profile,str) or profile not in PROFILES or profile!=expected:
        raise ValueError('perceptual_variation requires a supported task and non_lighting for medium / full for hard')


def configuration(task,seed,profile):
    """Sample independent appearance settings without consuming task RNG state."""
    if isinstance(profile,dict):
        from .atomic import configuration as atomic_configuration
        return atomic_configuration(task,seed,profile)
    if task not in TASKS or profile not in PROFILES:
        raise ValueError((task,profile))
    from .utils import rng_for
    from .lighting import LIGHTING
    rng=rng_for(seed,'perceptual-profile/'+task)
    config=dict(version=1,profile=profile,seed=int(seed),
                table=str(rng.choice(('wood','light_oak','steel','marble','concrete'))),
                lighting=str(rng.choice(LIGHTING)) if profile=='full' or task=='VideoUnmask' else 'normal')
    if task=='PickHighlight' and config['lighting'] not in ('normal','dim'):
        config['lighting']='normal'
    if task=='StopCube':
        from .clutter import sample_kinds
        config['clutter_kinds']=list(sample_kinds(seed))
        config['clutter_count']=len(config['clutter_kinds'])
    return config
