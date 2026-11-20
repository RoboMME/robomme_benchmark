# RoboMME-Plus: benchmark and data protocol

RoboMME-Plus evaluates robot policies on 16 tabletop tasks that require memory
as well as physical control. The robot may need to remember an earlier visual
event, keep track of objects over time, count events, or use information from a
video to follow the current language instruction. Appearance and scene behavior
can vary across episodes. A successful program must work on the current episode;
the objects, positions and event sequence in a training recording are not answers
for a new episode.

The released training set has 50 successful episodes: three per task, except
MoveCube, which has five. The test set has 50 episodes per task, 800 in total,
with training and test seeds disjoint within each task. Only the assigned training
tasks and resources are available during development.

There are three different sources of visual information:

- **Released training demonstration:** a fixed recording of a training episode.
  It can contain an initial task video followed by the expert robot's execution.
  It is available in AfD and AfD+I, and can be inspected repeatedly.
- **Task conditioning video:** earlier events supplied as part of the current
  episode's input, when that task has a video. It is available during both training
  interaction and testing, including AfI. It is not an additional training
  demonstration or the solution trajectory for the current execution.
- **Live execution observation:** the current camera views and robot state while
  your program controls the robot. The front camera observes the scene; the wrist
  camera moves with the robot. Read each method's input reference for the exact
  representation, calibration and control conventions.

The assigned adaptation setting determines what development may use:

| Setting | Released training recordings | Simulator starts per task |
| --- | --- | --- |
| Few-shot demonstrations (`AfD`) | Yes | 0 |
| Few-shot interactions (`AfI`) | No | At most 5 or 10 |
| Few-shot demonstrations+interactions (`AfD+I`) | Yes | At most 5 or 10 |

The assigned budget R is 5 or 10 for interaction settings and 0 for demonstrations
only. It counts attempts, not distinct seeds or demonstrations. The short setting
keys above are used by the tools and saved artifacts.
Every start counts, including a preview, debugging run, failed attempt, retry,
verification run or frozen-program rehearsal. Only the released training episode
indices are allowed: 0–2, or 0–4 for MoveCube. These indices identify the same
training episodes as the corresponding recordings. Reading an existing recording
does not start a simulator; starting an episode to obtain new observations does.
The operator's budget ledger is authoritative.

Read the current episode's language goal and control limits. The control clock
advances when actions execute, not while you inspect saved images or repeatedly
query an unchanged observation. A video frame index is not an execution step.
The overall step limit is a maximum; an episode may terminate earlier on success
or failure. Respect the returned terminal status. Public task rules, if supplied,
apply in addition to the language goal and overall horizon.

Phase labels identify the role of an observation, not its scene contents. The
first frame of a recording or runtime video is not guaranteed to be an empty
background. Conditioning footage and the live execution can differ in object
layout or tool geometry. A successful tool call or a normally exiting program
is distinct from the simulator's terminal task outcome; use the documented
result fields. Synthetic mock tests check assumed behavior, not real task success.

Use your method's existing development and policy-selection workflow within this
protocol. Observations expose sensors, calibration and robot state, not hidden
object identities, object poses or task solutions. Do not access simulator task
source, oracle code, other methods' solutions or held-out episodes. Keep code
versions, commands, observations, results and resource usage. Offline mock checks
are not simulator successes. Before testing, freeze the programs, prompts, tools,
library and harness. Within-episode memory is allowed; adaptive state must not
carry from one test episode to the next, and test results cannot revise the method.
