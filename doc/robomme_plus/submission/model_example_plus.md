# RoboMME-Plus submission templates

Follow the [evaluation protocol](../../../src/robomme_plus/README.md#evaluation-protocol).
Copy the template for each setting you report. Replace bracketed placeholders
and fill in the results table. Both Non-Agentic and Agentic tracks can use any
setting. For interaction settings, report results under a budget of at most
10 development rollouts per task.
Both `interaction-only` and `demo&interaction` may also report **1, 3 and 5
rollouts per task** as optional results. The leaderboard uses the 10-rollout
results. `zero-shot` and `demo-only` use zero simulator rollouts.

All 50 released demonstrations are available as recorded data for offline analysis
and learning in `demo-only` and `demo&interaction`.
Replaying their actions through the simulator consumes the development rollout
budget and is forbidden in the `demo-only` setting. State which demos
and modalities were used.
`zero-shot` and `interaction-only` do not use released demonstrations.

Each result uses **800 fixed test episodes: 16 tasks × 50 episodes**. Report
success rates as percentages with two decimal places. API costs are in USD;
report the evaluation total for all 800 episodes, and development cost separately.
Use N/A when no paid API is used. Label estimates or incomplete costs.

For submissions involving agentic development, provide:

- **Code link:** the method's source, harness and frozen policy revision.
- **Development-log link:** a public archive containing prompts, responses,
  commands/tool calls and text outputs, every code revision, all simulator-attempt
  records and API usage/costs, including failed and resumed work. **Raw trajectory
  data, images and videos do not need to be uploaded.** Keep source/hash or manifest
  references when omitting large payloads from logs.
- **Audit results:** include the harness and development-log audit verdicts,
  findings and warnings **at the bottom of the submission Markdown**, using the
  [audit record](#agentic-development-audit-record). Separate report links are optional.

Cover each reported setting and budget. If no agentic development was performed,
mark development logs and audits N/A and explain this briefly.


**Track (select one):**

- [ ] Non-Agentic Track: fine-tuned robot policies (e.g., VLAs, WAMs) that directly predict actions.
- [ ] Agentic Track: systems using pretrained foundation models for planning, action, or coding.


## 1. `zero-shot`

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
- **Results:** [ARTIFACTS LINK]

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


## 2. `demo-only`

Report this if you choose `demo-only`. Adapt using released offline demonstrations, without Plus simulator interactions.

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
- **Results:** [ARTIFACTS LINK]

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


## 3. `interaction-only`

Report this if you choose `interaction-only`. Adapt through training-seed interactions, without released offline demonstrations.

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
- **Results:** [ARTIFACTS LINK]

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
<summary>Optional results: 1, 3 and 5 rollouts per task</summary>

Each budget evaluates all 800 tests. Report successes and success rates per task;
use N/A for budgets you do not report.

| Suite | Task | 1 rollout | 3 rollouts | 5 rollouts |
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

Link actual attempts/steps and the frozen code revision for each budget. Include a matching audit verdict and evidence in the final audit section.
Choose and freeze policies before testing; test outcomes cannot select checkpoints.

</details>


## 4. `demo&interaction`

Report this if you choose `demo&interaction`. Adapt using released offline demonstrations and training-seed interactions.

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
- **Results:** [ARTIFACTS LINK]

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
<summary>Optional results: 1, 3 and 5 rollouts per task</summary>

Each budget evaluates all 800 tests. Report successes and success rates per task;
use N/A for budgets you do not report.

| Suite | Task | 1 rollout | 3 rollouts | 5 rollouts |
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

Link actual attempts/steps and the frozen code revision for each budget. Include a matching audit verdict and evidence in the final audit section.
Choose and freeze policies before testing; test outcomes cannot select checkpoints.

</details>


## Agentic development audit record

Keep this section at the **bottom of your submission**. Follow
[How to audit](../auditing.md) for the procedure and copy-and-paste prompts.
Include findings here for each setting and budget; external full reports are optional.

| Audit | Verdict | Summary of findings |
| --- | --- | --- |
| Development harness | [VERDICT] | [CONTROLS CHECKED AND GAPS] |
| Development logs | [VERDICT] | [RECORDS RECONCILED AND GAPS] |

Use **PASS WITHIN AUDITED SCOPE**, **VIOLATION FOUND** or **UNVERIFIED**. If an
audit is not completed, mark it **Pending report**. A harness audit checks the
interface; a log audit checks what happened during development.

- **Scope:** [METHOD, SETTING, BUDGET, TASKS AND RUN ID]
- **Versions:** [BENCHMARK/PROTOCOL, TRAINING DATA AND HARNESS REVISIONS]
- **Frozen policy:** [CODE LINK, COMMIT OR CONTENT HASHES]
- **Development logs:** [PUBLIC ARCHIVE LINK]
- **Auditor / date:** [NAME, DATE; SELF-REVIEW OR INDEPENDENT REVIEW]
- **Coverage:** [RECORDS INSPECTED; OMITTED PAYLOADS AND ANY ADDITIONAL LOCAL EVIDENCE]

For each item below, set **STATUS** to PASS, FAIL, UNVERIFIED or N/A and provide
a short evidence reference. Check `[x]` **only for PASS**; leave `[ ]` for the other
statuses and explain them. Any FAIL makes that audit's verdict VIOLATION FOUND;
otherwise any UNVERIFIED makes it UNVERIFIED. A pass needs all applicable checks
to pass. A checked box does not simply mean the check was performed.

### Harness checklist

- [ ] **Rule 1 — Complete text logs and code history — STATUS** · Evidence: [FILE:LINE]
- [ ] **Rule 1 — Shared budget counts every simulator start and retry — STATUS** · Evidence: [FILE:LINE]
- [ ] **Rule 2 — Inputs and agent access respect the setting — STATUS** · Evidence: [FILE:LINE]
- [ ] **Rule 2 — Only released training seeds/settings are exposed — STATUS** · Evidence: [FILE:LINE]
- [ ] **Rule 3 — Freeze and development/test separation are enforced — STATUS** · Evidence: [FILE:LINE]
- [ ] **Rule 4 — Attempts, steps and separate API costs are recorded — STATUS** · Evidence: [FILE:LINE]

### Development-log checklist

- [ ] **Rule 1 — All sessions, tool records and code revisions are accounted for — STATUS** · Evidence: [LOG/EVENT IDS]
- [ ] **Rule 1 — Starts reconcile across seeds/workers; budget respected — STATUS** · Evidence: [LEDGER/RECORD IDS]
- [ ] **Rule 2 — Recorded inputs, source access and demo use are permitted — STATUS** · Evidence: [LOG/EVENT IDS]
- [ ] **Rule 2 — Training seeds and full episode configurations match — STATUS** · Evidence: [METADATA/HASHES]
- [ ] **Rule 3 — Frozen policies trace to code history; no test-feedback tuning — STATUS** · Evidence: [HASHES/EVENT IDS]
- [ ] **Rule 4 — Per-task attempts/steps and API usage/costs reconcile — STATUS** · Evidence: [USAGE RECORDS]

### Per-task accounting

Include every task in scope for each reported budget. Use Unknown for unavailable
counts rather than zero. Failed/incomplete starts are included in Starts.

| Task | Budget | Starts | Failed/incomplete starts | Steps | Seed validity | Status |
| --- | --- | --- | --- | --- | --- | --- |
| [TASK] | [LIMIT] | [COUNT] | [COUNT] | [COUNT / UNKNOWN] | [STATUS] | [STATUS] |

Raw trajectories, images and videos are optional. Identify any specific finding
that cannot be verified from the available evidence.

### Warnings and required action

| Finding | Affected tasks/settings | Evidence | Required action |
| --- | --- | --- | --- |
| [GAP, VIOLATION OR DISCLOSED CORRECTION; NONE IF NONE] | [SCOPE] | [REFERENCE] | [ACTION] |
