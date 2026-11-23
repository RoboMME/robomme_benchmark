# RoboMME-Plus submission templates

Follow the [evaluation protocol](../../../src/robomme_plus/README.md#evaluation-protocol).
Copy the [template](#template) into a new Markdown file, keep only the rows and
columns for the settings you report, and replace the bracketed placeholders.
Either track may use any setting. Each result covers all **800 test episodes
(16 tasks × 50)**. The leaderboard uses 10 rollouts per task for interaction
settings. Results with other rollout budgets are optional.

Report success rates as percentages with two decimal places. Only simulator success
counts; failures, timeouts, model stops and errors stay in the denominator. Report
estimated API costs in USD, development and evaluation separately; a reasonable
estimate (for example rounded to the nearest $10, excluding local compute) is
enough. Use N/A when no paid API is used.
Leave evaluation scores and costs PENDING until team verification is complete.

## What to submit

Open a pull request containing this Markdown. Public code, method-specific
checkpoints and at least one paper, blog post or project website are required for
PR acceptance. Agentic development also requires public release of the complete
development logs and the final solutions used for evaluation. Link released
artifacts at fixed revisions.

**Keep this Markdown self-contained.** Put the results, costs and the full audit
report in this file. Link only to external artifacts: code, checkpoints, final
solutions and development logs.

| Item | Required for | Contents |
| --- | --- | --- |
| Code and checkpoints | Every submission | Public runnable code and all method-specific checkpoints at fixed revisions, including prompts, tools, dependencies and installation/evaluation commands; methods without learned weights provide their final programs and prompts |
| Method description | Every submission | At least one public paper, blog post or project website explaining the method |
| Final evaluation solutions | Agentic development | Public final task programs, prompts, skills and tools, with their dependency versions and evaluation commands |
| Usage log | Simulator rollouts without agentic development (for example RL fine-tuning in `Interaction-only`) | Per-task simulator starts for each budget; agentic development records these in its development logs instead |
| Development logs | Agentic development | Public complete prompts, visible responses, commands/tool calls and text outputs, plus every code/prompt revision, including failed and interrupted work, and every simulator start; see [Keep the development record](../agentic_development.md#step-7-keep-the-development-record) |
| Audit record | Agentic development | The development audit, filled in at the [bottom of this file](#agentic-development-audit-record) |

Agentic development
includes agent-generated task solutions, task-specific prompts learned from
rollouts, and self-improving agents. Standard policy fine-tuning and direct control
with general prompts (such as Embodied Astra) mark the agentic-development-only
items N/A unless they also use agentic development.

- **Development trajectories and media are optional uploads.**
  Raw action/state trajectories, images and rollout videos may be omitted.
- **Keep and publish the required records above:** code history, text logs,
  development attempt metadata, outcomes and usage records, as applicable.
  Retain source identifiers and hashes for omitted payloads; preserve the code,
  commands, failed attempts and result records.

**Evaluation access.** If testing requires a closed API or gated model, email
[robomme2026@gmail.com](mailto:robomme2026@gmail.com) after opening the PR. Include its link,
confirm that the RoboMME team has the required access, and arrange private delivery
of a dedicated API credential with enough quota for one full **800-episode
evaluation per submitted setting**. Never put credentials in the PR. Proprietary
model weights may be accessed through their API; credentials are unnecessary for
models used only during development.

**Verification.** The RoboMME team re-runs the submitted final code on all 800
tests and uses that run for the leaderboard. For agentic development, the audit
must conclude **PASS** first; methods without agentic development mark it N/A.
Disclose self-reviews.

### Tracks and settings

| Track | Meaning |
| --- | --- |
| Non-Agentic | Fine-tuned robot policies (e.g., VLAs, WAMs) that directly predict actions |
| Agentic | Systems using pretrained foundation models for planning, action or coding |

| Setting | Adapts with | Rollouts per task |
| --- | --- | ---: |
| `Zero-shot` | Nothing from Plus: test directly, or develop only on original RoboMME | 0 |
| `Demo-only` | Released offline demonstrations, without Plus simulator interactions | 0 |
| `Interaction-only` | Training-seed interactions, without released demonstrations | 10 |
| `Hybrid` | Released demonstrations and training-seed interactions | 10 |

### Smaller rollout budgets

Results with fewer than 10 rollouts per task are optional and do not enter the
leaderboard; the ASPIRE and RAX references use 1, 3 and 5. For budget R, either run
a separate development capped at R attempts per task, or use each task's code as
saved before its (R+1)th start in a 10-attempt run, and state which. Each budget
evaluates all 800 tests and needs its own audit. Test outcomes cannot select checkpoints.

### Audit record

Agentic development pastes one full report per setting and budget under **Full
audit reports**, using the prompt in [How to audit](../agentic_development.md#step-8-audit).
Methods without agentic development replace the section's content with "Not
applicable: no agentic development."

## Template

# [METHOD] on RoboMME-Plus

[One paragraph: the model or agentic framework and its memory mechanism.]

- **Track:** [Agentic / Non-Agentic].
- **Code:** [Code](CODE_URL) · [Run instructions](RUN_INSTRUCTIONS_URL).
- **Final evaluation solutions:** [Final solutions](SOLUTION_URL) / N/A (no agentic development).
- **Development logs:** [Public archive](DEVELOPMENT_LOGS_URL) / N/A (no agentic development).
- **Paper / website / blog:** [Paper](PAPER_URL) · [Website](WEBSITE_URL) · [Blog](BLOG_URL)
- **Model / checkpoint:** [CHECKPOINT OR API MODEL AND VERSION, AT A FIXED REVISION]
- **Parameters:** [SIZE IN B, INCLUDING FROZEN PARAMETERS; SHARED WEIGHTS COUNTED ONCE] / N/A (API model)
- **Evaluated by:** [TEAM]

## Results

### Overall results

| Setting | Rollouts per task | Success | Development starts | Development API cost | Evaluation API cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Zero-shot` | 0 | ___/800 (___%) | 0 | [COST / N/A] | [COST / N/A] |
| `Demo-only` | 0 | ___/800 (___%) | 0 | [COST / N/A] | [COST / N/A] |
| `Interaction-only` | 10 | ___/800 (___%) | [STARTS] | [COST / N/A] | [COST / N/A] |
| `Hybrid` | 10 | ___/800 (___%) | [STARTS] | [COST / N/A] | [COST / N/A] |

Outcomes: [SUCCESSES, FAILURES AND TIMEOUTS, PER SETTING].

### Per-task success

| Suite | Task | `Zero-shot` | `Demo-only` | `Interaction-only` | `Hybrid` |
| --- | --- | ---: | ---: | ---: | ---: |
| Counting | BinFill | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Counting | PickXtimes | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Counting | SwingXtimes | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Counting | StopCube | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Permanence | VideoUnmask | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Permanence | VideoUnmaskSwap | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Permanence | ButtonUnmask | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Permanence | ButtonUnmaskSwap | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Reference | PickHighlight | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Reference | VideoRepick | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Reference | VideoPlaceButton | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Reference | VideoPlaceOrder | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Imitation | MoveCube | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Imitation | InsertPeg | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Imitation | PatternLock | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Imitation | RouteStick | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| **Overall** | **All tasks** | **___/800 (___%)** | **___/800 (___%)** | **___/800 (___%)** | **___/800 (___%)** |

### Reference results with fewer rollouts

| Setting | [N] rollouts | [N] rollouts | [N] rollouts |
| --- | ---: | ---: | ---: |
| `Interaction-only` | ___/800 (___%) | ___/800 (___%) | ___/800 (___%) |
| `Hybrid` | ___/800 (___%) | ___/800 (___%) | ___/800 (___%) |

**Budget construction:** [SEPARATE DEVELOPMENT RUNS / CODE SAVED BEFORE EACH TASK'S (R+1)TH START]

<details>
<summary>Per-task success with [N] rollouts</summary>

| Suite | Task | `Interaction-only` | `Hybrid` |
| --- | --- | ---: | ---: |
| Counting | BinFill | ___/50 (___%) | ___/50 (___%) |
| Counting | PickXtimes | ___/50 (___%) | ___/50 (___%) |
| Counting | SwingXtimes | ___/50 (___%) | ___/50 (___%) |
| Counting | StopCube | ___/50 (___%) | ___/50 (___%) |
| Permanence | VideoUnmask | ___/50 (___%) | ___/50 (___%) |
| Permanence | VideoUnmaskSwap | ___/50 (___%) | ___/50 (___%) |
| Permanence | ButtonUnmask | ___/50 (___%) | ___/50 (___%) |
| Permanence | ButtonUnmaskSwap | ___/50 (___%) | ___/50 (___%) |
| Reference | PickHighlight | ___/50 (___%) | ___/50 (___%) |
| Reference | VideoRepick | ___/50 (___%) | ___/50 (___%) |
| Reference | VideoPlaceButton | ___/50 (___%) | ___/50 (___%) |
| Reference | VideoPlaceOrder | ___/50 (___%) | ___/50 (___%) |
| Imitation | MoveCube | ___/50 (___%) | ___/50 (___%) |
| Imitation | InsertPeg | ___/50 (___%) | ___/50 (___%) |
| Imitation | PatternLock | ___/50 (___%) | ___/50 (___%) |
| Imitation | RouteStick | ___/50 (___%) | ___/50 (___%) |
| **Overall** | **All tasks** | **___/800 (___%)** | **___/800 (___%)** |

</details>

## Method details

- **Action space:** [`joint_angle` / `ee_pose` / `waypoint`].
- **Demo use:** [DEMONSTRATIONS AND MODALITIES USED, PER SETTING / NONE].
- **Training and adaptation:** [WHAT WAS TRAINED, PROMPTED OR DEVELOPED ON PLUS / NONE].
- **Inference:** [SAMPLING, CHUNKING, CONTEXT AND OTHER TEST-TIME SETTINGS].
- **Rollout usage:** [LINK TO PER-TASK SIMULATOR STARTS; ONLY FOR NON-AGENTIC METHODS THAT USE PLUS ROLLOUTS]

## Agentic development audit record

Auditor: [ASSISTANT MODEL AND VERSION], [DATE] ([SELF-REVIEW / INDEPENDENT]).

### Audit summary

| Setting | Budget per task | Development starts | 1. No forbidden information | 2. Rollout budget | Conclusion |
| --- | ---: | ---: | --- | --- | --- |
| `Demo-only` | 0 | 0 | [PASS / FAIL] | [PASS / FAIL] | **[PASS / FAIL]** |
| `Interaction-only` | 10 | [STARTS] | [PASS / FAIL] | [PASS / FAIL] | **[PASS / FAIL]** |
| `Hybrid` | 10 | [STARTS] | [PASS / FAIL] | [PASS / FAIL] | **[PASS / FAIL]** |

### Development starts per task

| Suite | Task | `Demo-only` starts | `Interaction-only` starts | `Hybrid` starts |
| --- | --- | ---: | ---: | ---: |
| Counting | BinFill | [STARTS] | [STARTS] | [STARTS] |
| Counting | PickXtimes | [STARTS] | [STARTS] | [STARTS] |
| Counting | SwingXtimes | [STARTS] | [STARTS] | [STARTS] |
| Counting | StopCube | [STARTS] | [STARTS] | [STARTS] |
| Permanence | VideoUnmask | [STARTS] | [STARTS] | [STARTS] |
| Permanence | VideoUnmaskSwap | [STARTS] | [STARTS] | [STARTS] |
| Permanence | ButtonUnmask | [STARTS] | [STARTS] | [STARTS] |
| Permanence | ButtonUnmaskSwap | [STARTS] | [STARTS] | [STARTS] |
| Reference | PickHighlight | [STARTS] | [STARTS] | [STARTS] |
| Reference | VideoRepick | [STARTS] | [STARTS] | [STARTS] |
| Reference | VideoPlaceButton | [STARTS] | [STARTS] | [STARTS] |
| Reference | VideoPlaceOrder | [STARTS] | [STARTS] | [STARTS] |
| Imitation | MoveCube | [STARTS] | [STARTS] | [STARTS] |
| Imitation | InsertPeg | [STARTS] | [STARTS] | [STARTS] |
| Imitation | PatternLock | [STARTS] | [STARTS] | [STARTS] |
| Imitation | RouteStick | [STARTS] | [STARTS] | [STARTS] |
| **Total** | **All tasks** | **[TOTAL]** | **[TOTAL]** | **[TOTAL]** |

### Full audit reports

<details>
<summary><code>[SETTING]</code> · [BUDGET] — [PASS / FAIL]</summary>

[PASTE THE FULL AUDIT REPORT FOR THIS SETTING AND BUDGET]

</details>

## Citations

- [CITATIONS]
