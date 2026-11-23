# RoboMME-Plus submission templates

Follow the [evaluation protocol](../../../src/robomme_plus/README.md#evaluation-protocol).
Copy one section per setting you report and replace the bracketed placeholders.
Either track may use any setting. Each result covers all **800 test episodes
(16 tasks × 50)**. The leaderboard uses 10 rollouts per task for interaction
settings. Results with other rollout budgets are optional.

Report success rates as percentages with two decimal places. Report estimated API
costs in USD, development and evaluation separately; a reasonable estimate (for
example rounded to the nearest $10) is enough. Use N/A when no paid API is used.
Leave evaluation scores and costs PENDING until team verification is complete.

## What to submit

Open a pull request containing this Markdown. Public code, method-specific
checkpoints and at least one paper, blog post or project website are required for
PR acceptance. Agentic development also requires public release of the complete
development logs and the final solutions used for evaluation. Link released
artifacts at fixed revisions.

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
- **Code:** [Repository and final revision](CODE_URL)
- **Checkpoints:** [Released weights and fixed revision](CHECKPOINT_URL) / N/A (no learned weights)
- **Final evaluation solutions:** [Final solutions and revision](SOLUTION_URL) / N/A (no agentic development)
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
- **Code:** [Repository and final revision](CODE_URL)
- **Checkpoints:** [Released weights and fixed revision](CHECKPOINT_URL) / N/A (no learned weights)
- **Final evaluation solutions:** [Final solutions and revision](SOLUTION_URL) / N/A (no agentic development)
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
- **Code:** [Repository and final revision](CODE_URL)
- **Checkpoints:** [Released weights and fixed revision](CHECKPOINT_URL) / N/A (no learned weights)
- **Final evaluation solutions:** [Final solutions and revision](SOLUTION_URL) / N/A (no agentic development)
- **Development logs:** [Public archive](DEVELOPMENT_LOGS_URL) / N/A
- **Model / version:** [CHECKPOINT OR API MODEL AND VERSION]
- **Parameters (Non-Agentic track):** [SIZE IN B / N/A]
- **Adaptation:** [BRIEF DESCRIPTION]
- **Usage log:** [LINK TO SIMULATOR STARTS AND STEPS PER TASK, BY BUDGET] / In the development logs (agentic development)
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
| [N] rollouts | ___ | ___ | [LINK] |

**Budget construction:** [SAVED CHECKPOINTS / SEPARATE DEVELOPMENT RUNS]

For budget R, either run a separate development capped at R attempts per task, or
use each task's code as saved before its (R+1)th start in a 10-attempt run. State which.
For each budget, link its rollout usage (usage log or development logs) and final code revision and add a matching
audit below. Test outcomes cannot select checkpoints.

</details>


## 4. `Hybrid`

Adapt using released offline demonstrations and training-seed interactions.

### [Model / framework name]

[Briefly describe the model or agentic framework and its memory mechanism.]

- **Track:** [Non-Agentic / Agentic]
- **Paper:** [Paper](PAPER_URL) / N/A
- **Website:** [Project website](WEBSITE_URL) / N/A
- **Blog:** [Blog post](BLOG_URL) / N/A
- **Code:** [Repository and final revision](CODE_URL)
- **Checkpoints:** [Released weights and fixed revision](CHECKPOINT_URL) / N/A (no learned weights)
- **Final evaluation solutions:** [Final solutions and revision](SOLUTION_URL) / N/A (no agentic development)
- **Development logs:** [Public archive](DEVELOPMENT_LOGS_URL) / N/A
- **Model / version:** [CHECKPOINT OR API MODEL AND VERSION]
- **Parameters (Non-Agentic track):** [SIZE IN B / N/A]
- **Adaptation:** [BRIEF DESCRIPTION]
- **Demo use:** [SELECTION AND MODALITIES]
- **Usage log:** [LINK TO SIMULATOR STARTS AND STEPS PER TASK, BY BUDGET] / In the development logs (agentic development)
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
| [N] rollouts | ___ | ___ | [LINK] |

**Budget construction:** [SAVED CHECKPOINTS / SEPARATE DEVELOPMENT RUNS]

For budget R, either run a separate development capped at R attempts per task, or
use each task's code as saved before its (R+1)th start in a 10-attempt run. State which.
For each budget, link its rollout usage (usage log or development logs) and final code revision and add a matching
audit below. Test outcomes cannot select checkpoints.

</details>


## Agentic development audit record

Keep this section at the **bottom of your submission**, one per setting and
budget. [How to audit](../agentic_development.md#step-8-audit) gives the prompt; paste the report here.

**Conclusion:** [PASS / FAIL] — [ONE LINE; REASON FOR ANY FAIL]

- **Scope:** [METHOD, SETTING, BUDGET, TASKS AND RUN IDENTIFIER, IF ANY]
- **Development logs:** [PUBLIC ARCHIVE LINK]
- **Auditor / date:** [ASSISTANT MODEL AND VERSION, DATE; SELF-REVIEW OR INDEPENDENT]
- **Documents the agent received:** [FILES FROM THE PROTOCOL'S ALLOWED DOCUMENTATION]

| Check | Status | Evidence |
| --- | --- | --- |
| 1. No forbidden information: no task/simulator code, other methods' solutions, hidden state, test seeds/data/results, demos outside demo settings, or task-code details in prompts | [PASS / FAIL] | [FILE AND LINE, OR LOG ENTRY] |
| 2. Rollout budget: at most the per-task budget, every start counted, training seeds only | [PASS / FAIL] | [ROLLOUT RECORD REFERENCES] |

### Per-task rollouts

| Task | Budget | Starts | Seeds valid |
| --- | ---: | ---: | --- |
| [TASK] | [LIMIT] | [COUNT] | [YES / NO] |

**Notes:** [ONE LINE EACH, OR NONE]
