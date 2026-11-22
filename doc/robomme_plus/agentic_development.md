# Agentic development interface

For [agentic development](../../src/robomme_plus/README.md#evaluation-protocol),
connect your harness to this interface for rollout accounting and public
observations. Your harness provides isolation and preserves development evidence.

## Prepare once, outside the agent workspace

From the repository root in the simulator environment:

```bash
export PYTHONPATH="$PWD/src:$PWD/scripts/plus_eval"
python -m benchmark_io.prepare --root runs/my_agent/interaction-only \
  --method my_agent --setting interaction-only --budget 10
```

Use `--setting zero-shot`, `--setting demo-only`, `--setting interaction-only`
or `--setting hybrid`. `Zero-shot` and `Demo-only` are fixed at 0 starts.
`--budget` is the total **per task across all its training seeds, workers and
sessions**: 10 for the leaderboard, or a smaller budget for optional results. Preparation
saves the released training settings and benchmark source hashes without starting a simulator.

In demo settings, `--export-demos --demos <H5_DIR>` writes each task's RGB frames,
goals and phase labels under `<ROOT>/demonstrations/`. You may instead supply any
released modality.

Original RoboMME's dataset and simulator may be used freely in every setting,
outside this budget. Record their use separately.

## Connect your harness

Run this code in your **trusted simulator service** and expose its methods to the
agent through your harness's tools or RPC. The agent must not receive the Python
session object, ledger files, Plus simulator internals, Plus test data or
unrestricted host access.

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

`episode=` may replace `seed=` to select a released training index; do not supply
both. Any allocation among seeds is allowed. Actions default to `ee_pose`
(`[x,y,z,roll,pitch,yaw,gripper]`, metres/radians, -1 closed/+1 open); use
`action_space="joint_angle"` for seven joints plus gripper.

Observations contain phase-tagged front/wrist RGB-D, calibration, robot state,
goal, control limits and public success/error feedback. The task's conditioning
video is an observation in every setting; expert recordings are offline data for
demo settings only.

The ledger commits before simulator construction, so failed setup, zero-action
previews, early stops, timeouts, retries and frozen-policy rehearsals all count.

For a complete example, see ASPIRE's isolated worker,
[`aspire/isolated.py`](../../scripts/plus_eval/agentic_methods/aspire/isolated.py):
only public sensor and robot data cross its JSON RPC, and the worker cannot
import or reset the simulator.

## Accounting and isolation

All workers and resumed sessions must share the **same prepared study root**. Its
configuration and counters cannot be reset, and a new directory grants no extra
experience. Report independent runs separately; never pool their evidence or
choose among them using test results.

The wrapper is not a sandbox. Isolate agents with allowlisted RPC; block uncounted
Plus simulator calls, ledger edits, hidden state, task source and test data,
including installed copies and mounts. Follow [protocol rule 2](../../src/robomme_plus/README.md#evaluation-protocol)
for readable documents, and keep original RoboMME access separate. Only the
operator prepares studies, reads full metadata and freezes tests.

## Preserve evidence

Run the [harness audit](auditing.md#1-review-the-harness) before development.
Save every code revision **before** running it, and keep:

- prompts, visible model responses, commands/tool calls and their text outputs,
  including child and resumed sessions (provider-hidden reasoning is not required);
- the full code history, including failed and interrupted work;
- the ledger: `study.json`, `attempts.sqlite3` and `benchmark_source.json` in the
  study root, plus each rollout's `attempt.json`, `outcome.json` and `events.jsonl`;
- frozen policy/harness manifests;
- development API usage and costs.

**Publish** the development text records and code history, and link runnable frozen
code in the submission. **Development trajectories, images and videos are optional
uploads.** Optional payloads include
`actions.jsonl`, `observations.jsonl`, `rollout.mp4` and embedded API media. Retain
source identifiers and hashes for omitted payloads; keep commands, code revisions,
attempt records and development outcomes. Redact a **separate, unpacked publication
copy**, keeping the original evidence and freeze manifests unchanged:

```bash
python scripts/plus_eval/privacy.py audit "<UNPACKED_EXPORT_DIR>" --apply
python scripts/plus_eval/privacy.py audit "<UNPACKED_EXPORT_DIR>"
```

This helper scans text files, not `.tar.gz` archives or SQLite databases. Check
other exported formats separately. Record redactions and original/exported file
hashes, then compress the reviewed copy. Never remove development records.

**Freeze** programs, prompts, tools and harness with content hashes before testing.
ASPIRE and RAX use `EvaluationStudy.freeze` in
[`benchmark_io/evaluation.py`](../../scripts/plus_eval/benchmark_io/evaluation.py).
It copies the policy, records its SHA-256 hashes and the development attempt
count, and refuses evaluation if either changes. Disclose human corrections with
diffs and input provenance. Do not derive solutions from Plus implementation
details or revise them using test feedback.

For optional results with a smaller budget R, either run independent experiments capped at R
attempts per task or extract checkpoints from a 10-attempt development run.
Report which approach you use. Save each task's code and dependencies before
**that task's** (R+1)th start;
offline revisions after the preceding attempt are allowed. Apply the
[per-task checkpoint rule](../../src/robomme_plus/README.md#evaluation-protocol)
and retain the full history, including subsequent development.

**Report usage** per task (starts, unfinished starts and simulator steps) from the ledger:

```bash
python - runs/my_agent/interaction-only <<'EOF'
import sqlite3, sys
db = sqlite3.connect(f"file:{sys.argv[1]}/attempts.sqlite3?mode=ro", uri=True)
for row in db.execute("SELECT task, count(*), sum(ended IS NULL), "
                      "sum(json_extract(outcome, '$.control_steps')) FROM attempts GROUP BY task"):
    print(row)
EOF
```

Run the [development-log audit](auditing.md#2-review-the-completed-development-record)
and submit [the required items](submission/model_example_plus.md#what-to-submit).
The RoboMME team evaluates the frozen code for verification and updates the
leaderboard. Test-dependent audit checks are completed using that verification run.
The wrapper alone does not certify compliance. The ASPIRE/RAX 150-minute per-task
deadline is a method configuration, not a benchmark rule.
