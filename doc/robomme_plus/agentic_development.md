# Agentic development guide

Use this guide if an agent develops your method on RoboMME-Plus: for example, a
coding agent that writes code-as-policies solutions from demonstrations or
rollouts, an agent that learns task-specific prompts for Astra, or a system that
improves its own harness.

One **run** means one method, one setting and one budget, for example
"ASPIRE · `Interaction-only` · 10 rollouts". Repeat the steps for each run.

## What is checked

After development, audit each run with the
[audit skill](../skills/audit_robomme_plus.md). It checks two things:

1. **No forbidden information.** Your agent never had access to the benchmark's
   task or simulator code, other methods' solutions, hidden simulator state, test
   episodes or results, demonstrations outside demo settings, or prompts that
   restate task-code details.
2. **Rollout budget.** At most 10 simulator starts per task (0 in `Zero-shot` and
   `Demo-only`), counting every start.

When you submit a pull request with your code, the RoboMME team re-runs the final
code on the 800 test episodes. Test-time optimization is currently not allowed; if
your method learns anything from test episodes, disclose it in the pull request.

How you build the harness and store records is up to you. This guide uses
`benchmark_io`, the tooling behind ASPIRE and RAX, as a working example; file
names such as `study.json` and `attempts.sqlite3` come from it and are not
required.

## Step 1: choose the setting and budget

| Setting | Agent may use | Simulator starts per task |
| --- | --- | --- |
| `Zero-shot` | Nothing from RoboMME-Plus | 0 |
| `Demo-only` | The 50 released demonstrations | 0 |
| `Interaction-only` | Rollouts on the training seeds | up to 10 |
| `Hybrid` | Demonstrations and rollouts | up to 10 |

The limit is per task, shared across all its training seeds, workers and
restarts. Every start counts, including crashes and quick tests of your own code.
The original RoboMME dataset and simulator may be used freely in every setting.

## Step 2: set up rollout counting

Every RoboMME-Plus simulator start must be recorded before it happens, with its
task, training seed, time and code revision, so the audit can count starts per task.

With `benchmark_io`, create one record (a "study root") per run, from the
repository root and outside anything your agent can see:

```bash
export PYTHONPATH="$PWD/src:$PWD/scripts/plus_eval"
python -m benchmark_io.prepare --root runs/aspire/interaction-only \
  --method aspire --setting interaction-only --budget 10
```

`--setting` takes `zero-shot`, `demo-only`, `interaction-only` or `hybrid`
(case-insensitive). This records the released training episodes and creates the
start counter. Use one `--root` per run and never delete it. In demo settings,
`--export-demos --demos <H5_DIR>` writes RGB frames, goals and phase labels under
`<ROOT>/demonstrations/`; you may instead supply any released modality.

## Step 3: build the harness

The harness sits between your agent and the simulator. Whatever you use, it must
guarantee four things:

1. **The agent cannot see benchmark code or test data.** Run the agent in a
   container or sandbox without this repository, without the installed
   `robomme_plus` package and without `env_metadata/test/`. Check from inside the
   sandbox: `python -c "import robomme_plus"` must fail and the repo path must not exist.
2. **The agent reads only the public documents:** the Markdown files in
   [`scripts/plus_eval/benchmark_io/prompts/current/`](../../scripts/plus_eval/benchmark_io/prompts/current/),
   the [data format](plus_training_data_format.md) and this guide. Give nothing else.
3. **Every simulator start goes through the counter.** Run the simulator in a
   trusted process outside the sandbox and give the agent only tools to check
   status, reset, observe, step and close. All workers share one record. With
   `benchmark_io`, this is `TrainingSession` ([reference below](#simulator-interface)).
4. **Everything is logged:** every prompt, response, tool call and output, and a
   saved copy of the code before each reset, identified by a revision ID (for
   example, its hash).

For a complete example, see ASPIRE's isolated worker,
[`aspire/isolated.py`](../../scripts/plus_eval/agentic_methods/aspire/isolated.py).
If a coding agent builds your harness, give it these instructions:

```text
Build the RoboMME-Plus development harness so that:
1. The developing agent runs in a sandbox without this repository, the installed
   robomme_plus package, test metadata or other methods' solutions. It receives
   only the allowed public documents.
2. Every RoboMME-Plus simulator start is recorded by a trusted process outside the
   sandbox (for example benchmark_io's TrainingSession); all workers share one record.
3. Every prompt, response, tool call and output is logged, and the code is saved
   before each reset with a revision ID.
4. Test results never reach the developing agent or policy selection.
```

## Step 4: give your agent its instructions

Start each task's development with a prompt like this, adapted to your method:

```text
You are developing a policy for the RoboMME-Plus task <TASK> in the <SETTING> setting.
The documents in <DOCS_DIR> are your only source of task information.
You may start at most <BUDGET> training episodes of this task with the simulator
tools. Every reset counts, including failed runs and tests of your own code.
Before each reset, save your code and pass its revision ID.
Do not look for the simulator's or the task's source code, test episodes or other
methods' solutions; they are unavailable by design.
Use only numbers you measured from observations, demonstrations or your own
rollouts; do not guess hidden task parameters.
When finished, put the final policy in <FINAL_DIR> and write a short report.
```

## Step 5: develop

Let the agent work. Do not look at, or give the agent, any test result. If you edit
the agent's code or the harness by hand, keep the diff.

We suggest setting a per-task time limit so development cannot run indefinitely;
[ASPIRE](../../scripts/plus_eval/agentic_methods/aspire/README.md) and
[RAX](../../scripts/plus_eval/agentic_methods/rax/README.md), for example, stop each task after 150 minutes.

## Step 6: evaluate your final code

Run your final policy on all 800 test episodes and record the results. Do not use
test results to change the policy or to choose among versions. The RoboMME team
re-runs this final code for the leaderboard.

## Step 7: keep the development record

Keep, and publish with the submission:

- **Agent logs:** prompts, visible model responses, commands or tool calls and
  their text outputs, including child and resumed sessions (provider-hidden
  reasoning is not required);
- **Code history:** every saved revision, including failed and interrupted work;
- **Rollout record:** every simulator start with its task, seed, time, code
  revision and outcome. With `benchmark_io` these are, for example, `study.json`,
  `attempts.sqlite3` and `benchmark_source.json` in the study root, plus each
  rollout's `attempt.json`, `outcome.json` and `events.jsonl`;
- **Final solutions** used for evaluation, with dependencies and evaluation commands;
- **An estimate** of development API cost.

Trajectories, images and videos (with `benchmark_io`: `actions.jsonl`,
`observations.jsonl`, `rollout.mp4`) are optional. Never remove development
records. Redact credentials in a separate copy, keeping the originals unchanged:

```bash
python scripts/plus_eval/privacy.py audit "<UNPACKED_EXPORT_DIR>" --apply
python scripts/plus_eval/privacy.py audit "<UNPACKED_EXPORT_DIR>"
```

This helper scans text files, not `.tar.gz` archives or SQLite databases. With
`benchmark_io`, per-task starts, unfinished starts and simulator steps come from
the study root:

```bash
python - runs/aspire/interaction-only <<'EOF'
import sqlite3, sys
db = sqlite3.connect(f"file:{sys.argv[1]}/attempts.sqlite3?mode=ro", uri=True)
for row in db.execute("SELECT task, count(*), sum(ended IS NULL), "
                      "sum(json_extract(outcome, '$.control_steps')) FROM attempts GROUP BY task"):
    print(row)
EOF
```

## Step 8: audit

Use a fresh coding-assistant session for each run. Point it to the run's folder
(agent logs, code history, rollout record and harness configuration) and send, from
the repository root:

```text
Use doc/skills/audit_robomme_plus.md to audit the <METHOD> run in the <SETTING>
setting with a per-task rollout budget of <BUDGET>, using the evidence in <RUN_DIR>.
Save the report to <REPORT_PATH>.
```

The report concludes **PASS** or **FAIL** on the two checks above. It does not
judge your method's design, development outcomes or cost accuracy. Disclose that it
is a self-review. You can also run the same prompt on your harness before step 5 as
a pre-flight check.

## Step 9: submit

Copy the [submission template](submission/model_example_plus.md), fill in the
results, link your final code and development record, and paste the audit report
at the bottom. Examples: [ASPIRE](<submission/aspire-plus (eval_by_RoboMME_team).md>)
and [RAX](<submission/rax-plus (eval_by_RoboMME_team).md>).

## Simulator interface

This section describes `benchmark_io`'s `TrainingSession`, the reference
implementation used by ASPIRE and RAX. It runs in your trusted simulator service.
Expose its methods to the agent through your harness's tools or RPC; never pass the
session object, record files, simulator internals, test data or host access to the
agent.

```python
from benchmark_io.training import TrainingSession

with TrainingSession("runs/aspire/interaction-only", "BinFill") as sim:
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
| `reset(policy_revision=...)` | Rotate through released episodes in order; the cursor survives resumes. |
| `observe()` | Read the latest public observations; no physics step or new attempt. |
| `step(action)` | Execute one control action; native limits and termination apply. |
| `close()` | End the attempt, including an early stop; no refund. |

- `episode=` may replace `seed=` to select a released training index; do not
  supply both. Any allocation among seeds is allowed.
- Actions default to `ee_pose` (`[x,y,z,roll,pitch,yaw,gripper]`, metres/radians,
  -1 closed/+1 open); use `action_space="joint_angle"` for seven joints plus gripper.
- Observations contain phase-tagged front/wrist RGB-D, calibration, robot state,
  goal, control limits and public success/error feedback. The task's conditioning
  video is an observation in every setting.
- The attempt is recorded before the simulator is built, so failed setup, previews,
  early stops, timeouts and retries all count. All workers and resumed sessions
  must share the same study root; a new directory grants no extra starts.
- For optional smaller budgets, see the
  [submission template](submission/model_example_plus.md#3-interaction-only).
