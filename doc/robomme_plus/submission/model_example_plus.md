# RoboMME-Plus submission templates

Follow the [evaluation protocol](../../../src/robomme_plus/README.md#evaluation-protocol).
Copy one section per setting you report and replace the bracketed placeholders.
Either track may use any setting. Each result covers all **800 test episodes
(16 tasks × 50)**. The leaderboard uses 10 rollouts per task for interaction
settings. Results with other rollout budgets are optional.

Report success rates as percentages with two decimal places. Report API costs in
USD, development and evaluation separately; use N/A when no paid API is used.
For estimates, rounding to the nearest $10 is sufficient. Disclose missing usage
from provider errors as a warning; see the [audit record](#agentic-development-audit-record).
Leave evaluation scores and costs PENDING until team verification is complete.

## What to submit

Open a pull request containing this Markdown. Link code and development logs at
fixed revisions so reviewers can inspect the submitted artifacts.

| Item | Required for | Contents |
| --- | --- | --- |
| Code | Every submission | Runnable method and frozen policy revision, including prompts, tools, dependencies and installation/evaluation commands |
| Usage log | Any Plus development rollouts | Per-task starts, failed/incomplete starts and simulator steps for each budget |
| Freeze manifest | Agentic development | Content hashes of the frozen programs, prompts, tools and harness, created before the first test episode |
| Development logs | Agentic development | Prompts, visible responses, commands/tool calls and text outputs, plus every code revision, including failed and interrupted work; see [Preserve evidence](../agentic_development.md#preserve-evidence) |
| Audit record | Agentic development | Both audits, filled in at the [bottom of this file](#agentic-development-audit-record) |

The last three items apply to [agentic development](../../../src/robomme_plus/README.md#evaluation-protocol).
This includes agent-generated task solutions, task-specific prompts learned from
rollouts, and self-improving agents. Standard policy fine-tuning and direct control
with general prompts (such as Embodied Astra) mark these items N/A unless they also
use agentic development. Audit requirements follow the development process.

- **Development trajectories and media are optional uploads.**
  Raw action/state trajectories, images and rollout videos may be omitted.
- **Keep and publish the required records above:** code history, text logs,
  development attempt metadata, outcomes and usage records, as applicable.
  Retain source identifiers and hashes for omitted payloads; preserve the code,
  commands, failed attempts and result records.

**Verification.** The RoboMME team runs the submitted frozen code to verify its
performance, then updates the leaderboard. For agentic development, both audits
must conclude **PASS** before the leaderboard update; test-dependent checks may
remain PENDING until team verification. Methods without agentic development mark them N/A.
Recover missing evidence or fix confirmed breaches and rerun the affected
development. Disclose self-reviews.


**Track (select one):**

- [ ] Non-Agentic Track: fine-tuned robot policies (e.g., VLAs, WAMs) that directly predict actions.
- [ ] Agentic Track: systems using pretrained foundation models for planning, action, or coding.


## 1. `Zero-shot`

Test the policy directly, or develop it only on original RoboMME, without Plus adaptation.

### [Model / framework name]

[Briefly describe the model or agentic framework and its memory mechanism.]

- **Track:** [Non-Agentic / Agentic]
- **Paper:** [Paper](PAPER_URL) / N/A
- **Website:** [Project website](WEBSITE_URL) / N/A
- **Blog:** [Blog post](BLOG_URL) / N/A
- **Code:** [Repository and frozen revision](CODE_URL)
- **Development logs:** [Public archive](DEVELOPMENT_LOGS_URL) / N/A
- **Model / version:** [CHECKPOINT OR API MODEL AND VERSION]
- **Parameters (Non-Agentic track):** [SIZE IN B / N/A]
- **Total evaluation API cost (USD):** [COST / N/A]

| Suite | Task | Successes | Success (%) |
| --- | --- | --- | --- |
| Counting | BinFill | ___/50 |  |
| Counting | PickXtimes | ___/50 |  |
| Counting | SwingXtimes | ___/50 |  |
| Counting | StopCube | ___/50 |  |
| Permanence | VideoUnmask | ___/50 |  |
| Permanence | VideoUnmaskSwap | ___/50 |  |
| Permanence | ButtonUnmask | ___/50 |  |
| Permanence | ButtonUnmaskSwap | ___/50 |  |
| Reference | PickHighlight | ___/50 |  |
| Reference | VideoRepick | ___/50 |  |
| Reference | VideoPlaceButton | ___/50 |  |
| Reference | VideoPlaceOrder | ___/50 |  |
| Imitation | MoveCube | ___/50 |  |
| Imitation | InsertPeg | ___/50 |  |
| Imitation | PatternLock | ___/50 |  |
| Imitation | RouteStick | ___/50 |  |
| **Overall** | **All tasks** | **___/800** |  |


## 2. `Demo-only`

Adapt using released offline demonstrations, without Plus simulator interactions.

### [Model / framework name]

[Briefly describe the model or agentic framework and its memory mechanism.]

- **Track:** [Non-Agentic / Agentic]
- **Paper:** [Paper](PAPER_URL) / N/A
- **Website:** [Project website](WEBSITE_URL) / N/A
- **Blog:** [Blog post](BLOG_URL) / N/A
- **Code:** [Repository and frozen revision](CODE_URL)
- **Development logs:** [Public archive](DEVELOPMENT_LOGS_URL) / N/A
- **Model / version:** [CHECKPOINT OR API MODEL AND VERSION]
- **Parameters (Non-Agentic track):** [SIZE IN B / N/A]
- **Adaptation:** [BRIEF DESCRIPTION]
- **Demo use:** [SELECTION AND MODALITIES]
- **Development API cost (USD):** [COST / N/A]
- **Total evaluation API cost (USD):** [COST / N/A]

| Suite | Task | Successes | Success (%) |
| --- | --- | --- | --- |
| Counting | BinFill | ___/50 |  |
| Counting | PickXtimes | ___/50 |  |
| Counting | SwingXtimes | ___/50 |  |
| Counting | StopCube | ___/50 |  |
| Permanence | VideoUnmask | ___/50 |  |
| Permanence | VideoUnmaskSwap | ___/50 |  |
| Permanence | ButtonUnmask | ___/50 |  |
| Permanence | ButtonUnmaskSwap | ___/50 |  |
| Reference | PickHighlight | ___/50 |  |
| Reference | VideoRepick | ___/50 |  |
| Reference | VideoPlaceButton | ___/50 |  |
| Reference | VideoPlaceOrder | ___/50 |  |
| Imitation | MoveCube | ___/50 |  |
| Imitation | InsertPeg | ___/50 |  |
| Imitation | PatternLock | ___/50 |  |
| Imitation | RouteStick | ___/50 |  |
| **Overall** | **All tasks** | **___/800** |  |


## 3. `Interaction-only`

Adapt through training-seed interactions, without released offline demonstrations.

### [Model / framework name]

[Briefly describe the model or agentic framework and its memory mechanism.]

- **Track:** [Non-Agentic / Agentic]
- **Paper:** [Paper](PAPER_URL) / N/A
- **Website:** [Project website](WEBSITE_URL) / N/A
- **Blog:** [Blog post](BLOG_URL) / N/A
- **Code:** [Repository and frozen revision](CODE_URL)
- **Development logs:** [Public archive](DEVELOPMENT_LOGS_URL) / N/A
- **Model / version:** [CHECKPOINT OR API MODEL AND VERSION]
- **Parameters (Non-Agentic track):** [SIZE IN B / N/A]
- **Adaptation:** [BRIEF DESCRIPTION]
- **Usage log:** [LINK TO ATTEMPTS AND SIMULATOR STEPS PER TASK, BY BUDGET]
- **Development API cost (USD):** 10 rollouts: [COST / N/A]
- **Total evaluation API cost (USD):** 10 rollouts: [COST / N/A]

| Suite | Task | Successes | Success with 10 rollouts (%) |
| --- | --- | --- | --- |
| Counting | BinFill |  |  |
| Counting | PickXtimes |  |  |
| Counting | SwingXtimes |  |  |
| Counting | StopCube |  |  |
| Permanence | VideoUnmask |  |  |
| Permanence | VideoUnmaskSwap |  |  |
| Permanence | ButtonUnmask |  |  |
| Permanence | ButtonUnmaskSwap |  |  |
| Reference | PickHighlight |  |  |
| Reference | VideoRepick |  |  |
| Reference | VideoPlaceButton |  |  |
| Reference | VideoPlaceOrder |  |  |
| Imitation | MoveCube |  |  |
| Imitation | InsertPeg |  |  |
| Imitation | PatternLock |  |  |
| Imitation | RouteStick |  |  |
| **Overall** | **All tasks** |  |  |


<details>
<summary>Optional results: other rollout budgets per task</summary>

Each budget evaluates all 800 tests. Replace [N] with each budget you report (fewer than 10 rollouts per task; the ASPIRE and RAX references use 1, 3 and 5), and
report successes and success rates per task. Remove unused columns.

| Suite | Task | [N] rollouts | [N] rollouts | [N] rollouts |
| --- | --- | --- | --- | --- |
| Counting | BinFill | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Counting | PickXtimes | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Counting | SwingXtimes | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Counting | StopCube | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Permanence | VideoUnmask | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Permanence | VideoUnmaskSwap | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Permanence | ButtonUnmask | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Permanence | ButtonUnmaskSwap | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Reference | PickHighlight | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Reference | VideoRepick | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Reference | VideoPlaceButton | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Reference | VideoPlaceOrder | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Imitation | MoveCube | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Imitation | InsertPeg | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Imitation | PatternLock | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Imitation | RouteStick | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| **Overall** | **All tasks** | **___/800 (___%)** | **___/800 (___%)** | **___/800 (___%)** |

| Budget per task | Development API cost (USD) | Evaluation API cost (USD) | Results and code |
| --- | --- | --- | --- |
| 1 rollout | ___ | ___ | [LINK] |
| 3 rollouts | ___ | ___ | [LINK] |
| 5 rollouts | ___ | ___ | [LINK] |

**Budget construction:** [SAVED CHECKPOINTS / SEPARATE DEVELOPMENT RUNS]

Both are valid: run an independent experiment capped at R attempts per task, or
extract checkpoints from a 10-attempt development run.
For each budget, link the usage log and frozen code revision and add a matching
audit below. Select each task's R checkpoint after at most R attempts and before
**that task's** (R+1)th start. It may include offline revisions using the preceding
attempts' feedback, or be saved immediately after an attempt. Tasks can freeze
independently; preserve the checkpoint's dependency versions and audit shared
components against the same per-task budget. Test outcomes cannot select checkpoints.

</details>


## 4. `Hybrid`

Adapt using released offline demonstrations and training-seed interactions.

### [Model / framework name]

[Briefly describe the model or agentic framework and its memory mechanism.]

- **Track:** [Non-Agentic / Agentic]
- **Paper:** [Paper](PAPER_URL) / N/A
- **Website:** [Project website](WEBSITE_URL) / N/A
- **Blog:** [Blog post](BLOG_URL) / N/A
- **Code:** [Repository and frozen revision](CODE_URL)
- **Development logs:** [Public archive](DEVELOPMENT_LOGS_URL) / N/A
- **Model / version:** [CHECKPOINT OR API MODEL AND VERSION]
- **Parameters (Non-Agentic track):** [SIZE IN B / N/A]
- **Adaptation:** [BRIEF DESCRIPTION]
- **Demo use:** [SELECTION AND MODALITIES]
- **Usage log:** [LINK TO ATTEMPTS AND SIMULATOR STEPS PER TASK, BY BUDGET]
- **Development API cost (USD):** 10 rollouts: [COST / N/A]
- **Total evaluation API cost (USD):** 10 rollouts: [COST / N/A]

| Suite | Task | Successes | Success with 10 rollouts (%) |
| --- | --- | --- | --- |
| Counting | BinFill |  |  |
| Counting | PickXtimes |  |  |
| Counting | SwingXtimes |  |  |
| Counting | StopCube |  |  |
| Permanence | VideoUnmask |  |  |
| Permanence | VideoUnmaskSwap |  |  |
| Permanence | ButtonUnmask |  |  |
| Permanence | ButtonUnmaskSwap |  |  |
| Reference | PickHighlight |  |  |
| Reference | VideoRepick |  |  |
| Reference | VideoPlaceButton |  |  |
| Reference | VideoPlaceOrder |  |  |
| Imitation | MoveCube |  |  |
| Imitation | InsertPeg |  |  |
| Imitation | PatternLock |  |  |
| Imitation | RouteStick |  |  |
| **Overall** | **All tasks** |  |  |


<details>
<summary>Optional results: other rollout budgets per task</summary>

Each budget evaluates all 800 tests. Replace [N] with each budget you report (fewer than 10 rollouts per task; the ASPIRE and RAX references use 1, 3 and 5), and
report successes and success rates per task. Remove unused columns.

| Suite | Task | [N] rollouts | [N] rollouts | [N] rollouts |
| --- | --- | --- | --- | --- |
| Counting | BinFill | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Counting | PickXtimes | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Counting | SwingXtimes | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Counting | StopCube | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Permanence | VideoUnmask | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Permanence | VideoUnmaskSwap | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Permanence | ButtonUnmask | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Permanence | ButtonUnmaskSwap | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Reference | PickHighlight | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Reference | VideoRepick | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Reference | VideoPlaceButton | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Reference | VideoPlaceOrder | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Imitation | MoveCube | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Imitation | InsertPeg | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Imitation | PatternLock | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| Imitation | RouteStick | ___/50 (___%) | ___/50 (___%) | ___/50 (___%) |
| **Overall** | **All tasks** | **___/800 (___%)** | **___/800 (___%)** | **___/800 (___%)** |

| Budget per task | Development API cost (USD) | Evaluation API cost (USD) | Results and code |
| --- | --- | --- | --- |
| 1 rollout | ___ | ___ | [LINK] |
| 3 rollouts | ___ | ___ | [LINK] |
| 5 rollouts | ___ | ___ | [LINK] |

**Budget construction:** [SAVED CHECKPOINTS / SEPARATE DEVELOPMENT RUNS]

Both are valid: run an independent experiment capped at R attempts per task, or
extract checkpoints from a 10-attempt development run.
For each budget, link the usage log and frozen code revision and add a matching
audit below. Select each task's R checkpoint after at most R attempts and before
**that task's** (R+1)th start. It may include offline revisions using the preceding
attempts' feedback, or be saved immediately after an attempt. Tasks can freeze
independently; preserve the checkpoint's dependency versions and audit shared
components against the same per-task budget. Test outcomes cannot select checkpoints.

</details>


## Agentic development audit record

Keep this section at the **bottom of your submission**, one per setting and
budget. [How to audit](../auditing.md) gives the procedure and prompts; full
report links are optional.

| Audit | Conclusion | Summary |
| --- | --- | --- |
| Development harness | [PASS / FAIL / PENDING] | [ONE LINE; REASON FOR ANY FAIL] |
| Development logs | [PASS / FAIL / PENDING] | [ONE LINE; REASON FOR ANY FAIL] |

- **Scope:** [METHOD, SETTING, BUDGET, TASKS AND RUN ID]
- **Versions:** [BENCHMARK/PROTOCOL AND HARNESS REVISIONS]
- **Frozen policy:** [CODE LINK AND FREEZE MANIFEST HASHES]
- **Development logs:** [PUBLIC ARCHIVE LINK]
- **Auditor / date:** [NAME OR ASSISTANT MODEL AND VERSION, DATE; SELF-REVIEW OR INDEPENDENT]
- **Coverage:** [RECORDS INSPECTED; OMITTED PAYLOADS AND ANY ADDITIONAL LOCAL EVIDENCE]
- **Agent-visible documentation:** [FILES AND REVISIONS FROM THE PROTOCOL'S ALLOWLIST]

Each audit concludes **PASS** only if H1–H7 pass and R1–R3 are PASS, WARN or N/A
with a reason. Any FAIL or GAP blocks PASS. Distinguish breaches from missing evidence.
Check `[x]` only when an item passes in both audits.
The [hard requirements](../auditing.md#hard-requirements) define H1–H7; the
skills give the checks.

### Hard requirements

- [ ] **H1 — No privileged information** · Harness: [PASS/FAIL] · Logs: [PASS/FAIL] · Evidence: [REFS]
- [ ] **H2 — No Plus task code or other methods' Plus solutions** · Harness: [PASS/FAIL] · Logs: [PASS/FAIL] · Evidence: [REFS]
- [ ] **H3 — No task-code details in prompts, skills or notes** · Harness: [PASS/FAIL] · Logs: [PASS/FAIL] · Evidence: [REFS]
- [ ] **H4 — Training seeds only; no test seeds, metadata or results** · Harness: [PASS/FAIL] · Logs: [PASS/FAIL] · Evidence: [REFS]
- [ ] **H5 — At most the rollout budget per task, every start counted** · Harness: [PASS/FAIL] · Logs: [PASS/FAIL] · Evidence: [REFS]
- [ ] **H6 — Demonstrations only in demo settings** · Harness: [PASS/FAIL] · Logs: [PASS/FAIL] · Evidence: [REFS]
- [ ] **H7 — Frozen before testing; no test-feedback tuning** · Harness: [PASS/FAIL] · Logs: [PASS/FAIL] · Evidence: [REFS]

### Required evidence

A GAP makes the audit FAIL. Development trajectories, images and videos are
optional uploads; retain source/hash references when omitting them.
API costs may be rough estimates: missing usage after a logged provider timeout
or service error is WARN when the estimate's basis and uncertainty are disclosed.
Keep unknown token usage unknown, not zero.

- [ ] **R1 — Complete logs and code history** · Harness: [PASS/GAP] · Logs: [PASS/GAP] · Evidence: [REFS]
- [ ] **R2 — Resource usage recorded and reconciled** · Harness: [PASS/WARN/GAP] · Logs: [PASS/WARN/GAP] · Evidence: [REFS]
- [ ] **R3 — Human corrections disclosed with diffs** · Harness: [PASS/GAP/N/A] · Logs: [PASS/GAP/N/A] · Evidence: [REFS]

### Per-task accounting

Include every task in scope for each reported budget. Use Unknown for unavailable
counts rather than zero. Failed/incomplete starts are included in Starts.

| Task | Budget | Starts | Failed/incomplete starts | Steps | Seed validity | Status |
| --- | --- | --- | --- | --- | --- | --- |
| [TASK] | [LIMIT] | [COUNT] | [COUNT] | [COUNT / UNKNOWN] | [STATUS] | [STATUS] |

### Warnings and required action

| Finding | Affected tasks/settings | Evidence | Required action |
| --- | --- | --- | --- |
| [FAIL, GAP, WARNING OR DISCLOSED CORRECTION; NONE IF NONE] | [SCOPE] | [REFERENCE] | [ACTION] |
