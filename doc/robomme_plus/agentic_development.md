# Agentic development interface

Use this interface during **development**. Held-out evaluation uses the existing
[fixed test protocol](../../src/robomme_plus/README.md#evaluate).
Keep your method's own planning, coding, verification and policy-selection workflow.

## Prepare once, outside the agent workspace

From the repository root in the simulator environment:

```bash
export PYTHONPATH="$PWD/src:$PWD/scripts/plus_eval"
python -m benchmark_io.prepare --root runs/my_agent/interaction-only \
  --method my_agent --setting interaction-only --budget 10
```

Settings are `zero-shot`, `demo-only`, `interaction-only`, `demo+interaction`.
The first two permit zero Plus simulator starts. The interaction allowance is **10
starts per task across all its training seeds combined**: not 10 per seed, worker,
verification stage or restart. R1/R3/R5 are optional results; the leaderboard uses
R10. Preparation pins training metadata and source hashes without a simulator
start. Record the training-data revision when supplying demos.

Original RoboMME's dataset and simulator may be used freely in every setting,
outside this Plus budget. Record their use separately; the wrapper here accounts
for Plus development only.

## Connect your harness

Run this code in your **trusted simulator service**, exposing its methods through
your harness's tools or RPC. The agent must not receive the Python session object,
ledger files, Plus simulator internals, Plus test data or unrestricted host access.

```python
from benchmark_io.training import TrainingSession

with TrainingSession("runs/my_agent/interaction-only", "BinFill") as sim:
    seeds = sim.status()["training_seeds"]
    observation = sim.reset(seed=seeds[1], policy_revision="<SAVED_CODE_HASH>")
    while not (observation["terminated"] or observation["truncated"]):
        action = agent.predict(observation)
        observation = sim.step(action)
```

| Call | Behavior |
| --- | --- |
| `status()` | Released seeds, budget, attempts used and remaining; no simulation. |
| `reset(policy_revision=..., seed=...)` | Reserve one attempt, then start the selected released episode. |
| `reset(policy_revision=...)` | Rotate through released episodes in order; the cursor survives resumes. Explicit choices leave the cursor unchanged. |
| `observe()` | Read the latest public observations; no physics step or new attempt. |
| `step(action)` | Execute one control action; native limits and termination apply. |
| `close()` | End the attempt, including an early stop; no refund. |

`episode=` may replace `seed=` to select a released training index. Do not supply
both. There is no required allocation among seeds. Actions default to `ee_pose`
(`[x,y,z,roll,pitch,yaw,gripper]`, metres/radians, -1 closed/+1 open); use
`action_space="joint_angle"` for seven joints plus gripper.

Observations contain phase-tagged front/wrist RGB-D, calibration, robot state,
goal, control limits and public success/error feedback. The initial task video
is an observation in every setting. Expert recordings are separate offline data
available only in demo settings. Simulator replay consumes the development
budget and is forbidden in demo-only.

The ledger commits before simulator construction. Failed setup, zero-action
previews, early stops, timeouts, retries and frozen-policy rehearsals all count.
Invalid requests rejected before reservation do not start an attempt. Exhausted
budgets block construction. Benchmark-defined behavior within an episode, such
as MoveCube's allowed retry, remains part of that episode.

## Accounting and isolation

All workers and resumed sessions must share the **same prepared study root**.
Its configuration and counters cannot be reset. Another directory does not grant
extra experience for that experiment. Independent runs must be reported separately
and must not pool evidence or select policies from test results.

This wrapper accounts for calls through it; it is not a Python sandbox. Your
harness must prevent uncounted Plus simulator entry points, ledger edits and
access to hidden Plus state or task implementation. Use an isolated agent process
and allowlisted RPC. Permitted original RoboMME access must remain separate.
Only the operator prepares studies, reads full metadata and freezes tests.
ASPIRE/RAX use the same `Study`/`Episode` backend through their native adapters.

## Preserve evidence

Use the [audit walkthrough and prompts](auditing.md) before development and again
when the full development record is ready.

The interface records attempts, seed choices, policy revisions, actions, sensor
metadata, rollout video and outcomes. Save code **before** its run. Your harness
must additionally retain prompts, visible model responses, commands/tool calls
and outputs, code history, child/resumed sessions, API usage and costs.
Provider-hidden reasoning is not required. Keep failed and interrupted records.

Freeze final programs, prompts, tools and harness with content hashes before
testing. Disclose human corrections with exact diffs and input provenance; do
not present corrected policies as unmodified agent output. Do not handcraft Plus
task solutions from implementation details or revise them using test feedback.

Submit a [harness audit](../skills/audit_robomme_plus_harness.md) and
[development-log audit](../skills/audit_robomme_plus_development.md) using the
[audit record](submission/model_example_plus.md#agentic-development-audit-record).
The wrapper alone does not certify compliance. The ASPIRE/RAX baselines'
150-minute deadline is a method configuration, not a universal benchmark limit.
