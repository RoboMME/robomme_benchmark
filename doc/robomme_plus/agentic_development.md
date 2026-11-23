# Agentic development guide

Use this guide if agents develop your method on RoboMME-Plus: writing task
solutions from demonstrations or rollouts, learning task-specific prompts, or
improving themselves (for example ASPIRE and RAX). Do the steps in order. The
[rules](../../src/robomme_plus/README.md#evaluation-protocol) are the reference.

One **run** means one method, one setting and one budget, for example
"ASPIRE · `Interaction-only` · 10 rollouts". Repeat the steps for each run.

## What is checked

One audit after development checks two things:

1. **No forbidden information.** Your agent never had access to the benchmark's
   task or simulator code, other methods' solutions, hidden simulator state, test
   episodes or results, demonstrations outside demo settings, or prompts that
   restate task-code details.
2. **Rollout budget.** At most 10 simulator starts per task (0 in `Zero-shot` and
   `Demo-only`), counting every start.

The RoboMME team re-runs your final code on the 800 tests, so you do not need to
freeze or fingerprint anything.

## Step 1: choose the setting and budget

| Setting | Agent may use | Simulator starts per task |
| --- | --- | --- |
| `Zero-shot` | Nothing from Plus | 0 |
| `Demo-only` | The 50 released demonstrations | 0 |
| `Interaction-only` | Training-seed rollouts | 10 (leaderboard) |
| `Hybrid` | Demonstrations and rollouts | 10 (leaderboard) |

The 10 starts are per task, shared across all its training seeds, workers and
restarts. Every start counts, including crashes and quick tests of your own code.
Original RoboMME's dataset and simulator may be used freely in every setting.

## Step 2: create the run's ledger

From the repository root, outside anything your agent can see:

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

The harness sits between your agent and the simulator. It must guarantee four things:

1. **The agent cannot see benchmark code or test data.** Run the agent in a
   container or sandbox without this repository, without the installed
   `robomme_plus` package and without `env_metadata/test/`. Check from inside the
   sandbox: `python -c "import robomme_plus"` must fail and the repo path must not exist.
2. **The agent reads only the public documents:** the Markdown files in
   [`scripts/plus_eval/benchmark_io/prompts/current/`](../../scripts/plus_eval/benchmark_io/prompts/current/),
   the [data format](plus_training_data_format.md) and this guide. Give nothing else.
3. **Every simulator start goes through the counter.** Run `TrainingSession`
   ([reference below](#simulator-interface)) in a trusted process outside the
   sandbox and expose only its `status`, `reset`, `observe`, `step` and `close` as
   the agent's tools. All workers use the same `--root`.
4. **Everything is logged:** every prompt, response, tool call and output, and a
   saved copy of the code before each `reset`, with its hash passed as `policy_revision`.

For a complete example, see ASPIRE's isolated worker,
[`aspire/isolated.py`](../../scripts/plus_eval/agentic_methods/aspire/isolated.py).
If a coding agent builds your harness, give it these instructions:

```text
Build the RoboMME-Plus development harness so that:
1. The developing agent runs in a sandbox without this repository, the installed
   robomme_plus package, test metadata or other methods' solutions. It receives
   only the allowed public documents.
2. Every Plus simulator start goes through benchmark_io's TrainingSession, from a
   trusted process outside the sandbox; all workers share one study root.
3. Every prompt, response, tool call and output is logged, and the code is saved
   before each reset with its hash passed as policy_revision.
4. Test results never reach the developing agent or policy selection.
```

## Step 4: give your agent its instructions

Start each task's development with a prompt like this, adapted to your method:

```text
You are developing a policy for the RoboMME-Plus task <TASK> in the <SETTING> setting.
The documents in <DOCS_DIR> are your only source of task information.
You may start at most <BUDGET> training episodes of this task with the simulator
tools. Every reset counts, including failed runs and tests of your own code.
Before each reset, save your code and pass its revision hash as policy_revision.
Do not look for the simulator's or the task's source code, test episodes or other
methods' solutions; they are unavailable by design.
Use only numbers you measured from observations, demonstrations or your own
rollouts; do not guess hidden task parameters.
When finished, put the final policy in <FINAL_DIR> and write a short report.
```

## Step 5: develop

Let the agent work. Do not look at, or give the agent, any test result. If you edit
the agent's code or the harness by hand, keep the diff.

## Step 6: evaluate your final code

Run your final policy on all 800 test episodes and record the results. Do not use
test results to change the policy or to choose among versions. The RoboMME team
re-runs this final code for the leaderboard.

## Step 7: preserve evidence

Keep, and publish with the submission:

- prompts, visible model responses, commands/tool calls and their text outputs,
  including child and resumed sessions (provider-hidden reasoning is not required);
- the full code history, including failed and interrupted work;
- the ledger: `study.json`, `attempts.sqlite3` and `benchmark_source.json` in the
  study root, plus each rollout's `attempt.json`, `outcome.json` and `events.jsonl`;
- the final solutions used for evaluation, with dependencies and evaluation commands;
- an estimate of development API cost.

Trajectories, images and videos (`actions.jsonl`, `observations.jsonl`,
`rollout.mp4`) are optional. Never remove development records. Redact credentials
in a separate copy, keeping the originals unchanged:

```bash
python scripts/plus_eval/privacy.py audit "<UNPACKED_EXPORT_DIR>" --apply
python scripts/plus_eval/privacy.py audit "<UNPACKED_EXPORT_DIR>"
```

This helper scans text files, not `.tar.gz` archives or SQLite databases.
Per-task starts, unfinished starts and simulator steps come from the ledger:

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
(agent logs, code history, ledger and harness configuration) and send, from the
repository root:

```text
Use doc/skills/audit_robomme_plus.md to audit the <METHOD> run in the <SETTING>
setting with a per-task rollout budget of <BUDGET>, using the evidence in <RUN_DIR>.
Save the report to <REPORT_PATH>.
```

The report concludes **PASS** or **FAIL** on the two checks above. It does not
judge freezing, your method's design, development outcomes or cost accuracy.
Disclose that it is a self-review. You can also run the same prompt on your
harness before step 5 as a pre-flight check.

## Step 9: submit

Copy the [submission template](submission/model_example_plus.md), fill in the
results, link your final code and development record, and paste the audit report
at the bottom. Examples: [ASPIRE](<submission/aspire-plus (eval_by_RoboMME_team).md>)
and [RAX](<submission/rax-plus (eval_by_RoboMME_team).md>).

## Simulator interface

`TrainingSession` runs in your trusted simulator service. Expose its methods to the
agent through your harness's tools or RPC; never pass the session object, ledger
files, simulator internals, test data or host access to the agent.

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
- The ASPIRE/RAX 150-minute per-task deadline is a method configuration, not a rule.
