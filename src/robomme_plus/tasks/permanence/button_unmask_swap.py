"""ButtonUnmaskSwap: staged reveals and ordered container pickup.

New Easy episodes use finite motion. Medium and Hard independently enable
continuous motion with 50% probability, deterministically from the episode seed.
Explicit ``continuous_motion`` booleans override sampling for recording/replay.
With continuous motion, both buttons unlock pickup. Below 6 cm above the tabletop
containers, the gripper pauses container/cube pairs within 10 cm of its XY
projection. These neighbors stay paused until the gripper lifts clear; the rest
of the traffic moves at half speed meanwhile. Exposing a cube stops it while its
cover is held; placing the cover aside lets the uncovered cube rejoin traffic.
Neighbors can temporarily detour up to 8 cm outside the movement bounds during
pickup, retaining reachability and obstacle clearance, then return inside.
Covered pairs slowly rotate in yaw with seeded, smoothly varying 3-8 degree/s
target speeds. Rotation pauses during approach and pickup. Nearby traffic follows
a committed route past the stopped pair instead of pushing into it.
Replacing and releasing the cover resumes the covered pair. Both modes use the
compact workspace for new episodes. Saved mode/version/bounds restore each episode;
older metadata retains finite shuffles and the previous wide workspace.
"""

from mani_skill.utils.registration import register_env

from robomme_plus.registry import TASKS
from robomme_plus.tasks.permanence.utils.environment import PermanenceTask


@register_env(TASKS["ButtonUnmaskSwap"].gym_id)
class ButtonUnmaskSwap(PermanenceTask):
    PERCEPTUAL_VARIATIONS = True
    task_name = "ButtonUnmaskSwap"
