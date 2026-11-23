"""Seeded pickup position in front of one swing target, hard seeds only."""
import numpy as np
from .motion import rng_for


def relocate(controller,seed):
    from robomme_plus.common.sim import position,set_position
    from robomme_plus.tasks.counting.utils.motion import PICK_BUTTON_RADIUS
    env=controller.env;rng=rng_for(seed,'swing-front-pickup')
    original=position(env.target_cube,False).copy();origins=controller.slides.origins[:,:2]
    eye=np.array([.30,0.]);button=position(env.button,False)[:2]
    half=env.cube_half_size;cube_radius=np.sqrt(2)*half
    # The enlarged circle and translated dodge center remain inside this box.
    reach=controller.slides.grid_scale*[.06,.12]+controller.slides.radii[:,None]
    sides=rng.permutation(2);chosen=None
    for side in sides:
        direction=eye-origins[side];direction/=np.linalg.norm(direction)
        for forward in rng.uniform(.15,.215,80):
            point=origins[side]+direction*(forward/direction[0])
            if np.any(point<[-.40,-.34]) or np.any(point>[.14,.34]):continue
            distance=np.linalg.norm(point-[-.615,0.])
            if not .28<=distance<=.76:continue
            if np.linalg.norm(point-button)<PICK_BUTTON_RADIUS+cube_radius+.02:continue
            clearance=np.linalg.norm(np.maximum(np.abs(point-origins)-reach,0),axis=1)-.04-cube_radius
            if clearance.min()<.008:continue
            chosen=(int(side),point,clearance);break
        if chosen:break
    if chosen is None:raise RuntimeError('No reachable pickup site in front of either target')
    side,point,clearance=chosen
    xyz=np.r_[point,original[2]];set_position(env.target_cube,xyz)
    controller.layout['cube_xyz']=xyz.tolist()
    audit=dict(original_cube_xyz=original.tolist(),cube_xyz=xyz.tolist(),target_index=side,
        target_side='right' if side==0 else 'left',target_origin_xy=origins[side].tolist(),
        front_camera_eye_xy=eye.tolist(),minimum_ring_sweep_surface_clearance_m=float(clearance.min()),
        choice='seeded side, with reachable collision-clear fallback',cube_yaw_unchanged=True)
    controller._mixed_audit['pickup_layout']=audit
    controller.event('front_target_pickup_layout',**audit)
    return audit
