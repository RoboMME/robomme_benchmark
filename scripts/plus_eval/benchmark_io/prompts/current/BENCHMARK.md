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
  It is recorded data for offline analysis, available in `Demo-only` and
  `Hybrid`.
- **Task conditioning video:** earlier events supplied as part of the current
  episode's input, when that task has a video. It is available during both training
  interaction and testing, including `Interaction-only`. It is not an additional training
  demonstration or the solution trajectory for the current execution.
- **Live execution observation:** the current camera views and robot state while
  your program controls the robot. The front camera observes the scene; the wrist
  camera moves with the robot. Read each method's input reference for the exact
  representation, calibration and control conventions.

The assigned adaptation setting determines what development may use:

| Setting | Released Plus training recordings | Plus simulator starts per task |
| --- | --- | --- |
| `Zero-shot` | No | 0 |
| `Demo-only` | Yes | 0 |
| `Interaction-only` | No | At most 10 |
| `Hybrid` | Yes | At most 10 |

The main budget R is 10 for interaction settings and 0 for `Zero-shot`/`Demo-only`.
R1/R3/R5 are optional results; use the assigned budget, which is a total per task
across all its training seeds, workers, sessions and verification stages. It counts attempts, not distinct seeds or demonstrations. Use the same setting names in tools and saved artifacts.
Every Plus start counts, including a preview, debugging run, failed attempt, retry,
verification run or frozen-program rehearsal. Only the released training episode
indices are allowed: 0–2, or 0–4 for MoveCube. These indices identify the same
training episodes as the corresponding recordings. Reading an existing recording
does not start a simulator; starting an episode to obtain new observations does.
Replaying recorded actions through physics consumes this budget and is forbidden
in `Demo-only`. Agents may choose any released training seed; no even allocation
is required. The default interface rotates seeds when no choice is supplied.
Do not create another study or use another simulator entry point to obtain extra
experience. The operator's persistent ledger accounts for every accepted start.

The original RoboMME dataset and simulator may be used freely in every setting,
outside the Plus rollout budget. Keep their provenance separate from Plus data.
Use the resources exposed by your assigned harness; this permission does not
allow bypassing it or accessing Plus task implementations or held-out tests.

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
object identities, object poses or task solutions. Do not reverse-engineer Plus
task code, query Plus oracles, read other methods' Plus solutions or access Plus
held-out episodes. Do not handcraft solutions from Plus implementation details. Keep code
versions, commands, observations, results and resource usage. Offline mock checks
are not simulator successes. Before testing, freeze the programs, prompts, tools,
library and harness. Within-episode memory is allowed; adaptive state must not
carry from one test episode to the next, and test results cannot revise the method.

Save every code revision before running it. Retain prompts, visible responses,
commands/tool calls and outputs, child/resumed sessions, attempts/outcomes and API
usage. Preserve failed and interrupted work. Freeze the final code with hashes;
never edit a historical log or claim an unrecorded check was performed.

Public log exports may omit raw trajectory, image and video payloads. Preserve
text records, every code revision and attempt, and source/hash or manifest
references for omitted payloads. Include audit results in the submission Markdown.
