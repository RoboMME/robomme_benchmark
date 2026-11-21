# RoboMME-Plus submission templates

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

[Website](WEBSITE_URL) · [Paper](PAPER_URL)

[Briefly describe the model or agentic framework and its memory mechanism.]

- **Track:** [Non-Agentic / Agentic]
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

[Website](WEBSITE_URL) · [Paper](PAPER_URL)

[Briefly describe the model or agentic framework and its memory mechanism.]

- **Track:** [Non-Agentic / Agentic]
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

[Website](WEBSITE_URL) · [Paper](PAPER_URL)

[Briefly describe the model or agentic framework and its memory mechanism.]

- **Track:** [Non-Agentic / Agentic]
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

Each result evaluates all 800 tests under its stated maximum development budget.

| Budget per task | Successes / 800 | Success (%) | Development API cost (USD) | Evaluation API cost (USD) | Results and code |
| --- | --- | --- | --- | --- | --- |
| 1 rollout | ___/800 | ___ | ___ | ___ | [LINK] |
| 3 rollouts | ___/800 | ___ | ___ | ___ | [LINK] |
| 5 rollouts | ___/800 | ___ | ___ | ___ | [LINK] |

**Budget construction:** [SAVED CHECKPOINTS / SEPARATE DEVELOPMENT RUNS]

Link per-task results, actual attempts/steps and the frozen code revision for each
budget. Include a matching audit verdict and evidence in the final audit section.
Choose and freeze policies before testing; test outcomes cannot select checkpoints.

</details>


## 4. `demo&interaction`

Report this if you choose `demo&interaction`. Adapt using released offline demonstrations and training-seed interactions.

### [Model / framework name]

[Website](WEBSITE_URL) · [Paper](PAPER_URL)

[Briefly describe the model or agentic framework and its memory mechanism.]

- **Track:** [Non-Agentic / Agentic]
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

Each result evaluates all 800 tests under its stated maximum development budget.

| Budget per task | Successes / 800 | Success (%) | Development API cost (USD) | Evaluation API cost (USD) | Results and code |
| --- | --- | --- | --- | --- | --- |
| 1 rollout | ___/800 | ___ | ___ | ___ | [LINK] |
| 3 rollouts | ___/800 | ___ | ___ | ___ | [LINK] |
| 5 rollouts | ___/800 | ___ | ___ | ___ | [LINK] |

**Budget construction:** [SAVED CHECKPOINTS / SEPARATE DEVELOPMENT RUNS]

Link per-task results, actual attempts/steps and the frozen code revision for each
budget. Include a matching audit verdict and evidence in the final audit section.
Choose and freeze policies before testing; test outcomes cannot select checkpoints.

</details>


<details>
<summary>Evaluation rules — apply to all four templates</summary>

1. **Keep the full development history.** Save the complete process, including
   prompts, commands, code revisions, attempt records and outcomes. Every Plus simulator
   start counts: previews, failures, retries, demonstration replays, debugging,
   verification and model selection. Optional budgets of 1, 3 or 5 rollouts are
   totals, not additional allowances.
   The allowance covers all training seeds of a task combined, not each seed.
   Unused attempts cannot transfer between tasks. Simulator replays of recorded
   actions count toward this allowance.
   Follow the official step limits. Agentic methods must retain full development
   logs, follow the [shared development interface](../agentic_development.md),
   and submit the [harness audit](../../skills/audit_robomme_plus_harness.md) and
   [development-log audit](../../skills/audit_robomme_plus_development.md).
2. **Use only permitted resources.** Adapt to Plus using released demonstrations
   and/or training seeds, according to the setting. Do not access privileged
   information, reverse-engineer Plus task code, use test seeds during development,
   or handcraft task solutions from implementation details. Original RoboMME's
   [dataset](https://huggingface.co/datasets/Yinpei/robomme_data_h5) and
   [simulator](../../../src/robomme) may be used freely in every setting.
3. **Freeze everything before evaluation.** Finalize weights, programs, prompts,
   tools, libraries and the harness before testing. Reset memory between test
   episodes. Do not learn from test results or use them for policy selection.
   Test-time optimization is outside the standard protocol; report its results
   separately from these tables.
4. **Report resource usage.** Include the setting, demonstrations used, rollout budget, actual
   attempts and simulator steps per task, and separate development/evaluation API costs
   in USD. Compare methods at the same setting and rollout budget.

Use a shared model checkpoint or agentic framework across all 16 tasks. Keep
all 800 episodes in the denominator, including failures, timeouts, model stops,
and errors. Keep the default test variations and supplied execution limits
(`ceil(1.5 * oracle_policy_step) + 200` control steps). Report altered test
conditions separately. For repeated runs, report each inference seed and the
mean for the same setting and budget.

Task-provided test demonstration videos are normal observations, not adaptation
data. Original RoboMME data and simulator use is outside the Plus rollout budget.
Submit this Markdown with code/development-log links and the audit results below
in a pull request.

Reference: [evaluation guide](../../../src/robomme_plus/README.md) ·
[baseline results and running instructions](../robomme_plus_baseline_results.md) ·
[training data format and seed mapping](../plus_training_data_format.md).

</details>

## Agentic development audit record

Keep this section at the **bottom of your submission**. Follow
[How to audit](../auditing.md) for the procedure and copy-and-paste prompts.
Include findings here for each setting and budget; external full reports are optional.

- **Scope:** [METHOD, SETTING, BUDGET, TASKS AND RUN ID]
- **Versions:** [BENCHMARK/PROTOCOL, TRAINING DATA AND HARNESS REVISIONS]
- **Frozen policy:** [CODE LINK, COMMIT OR CONTENT HASHES]
- **Development logs:** [PUBLIC ARCHIVE LINK]
- **Auditor / date:** [NAME, DATE; SELF-REVIEW OR INDEPENDENT REVIEW]
- **Coverage:** [RECORDS INSPECTED; OMITTED PAYLOADS AND ANY ADDITIONAL LOCAL EVIDENCE]

| Audit | Verdict | Summary of findings |
| --- | --- | --- |
| Development harness | [VERDICT] | [CONTROLS CHECKED AND GAPS] |
| Development logs | [VERDICT] | [RECORDS RECONCILED AND GAPS] |

Use **PASS WITHIN AUDITED SCOPE**, **VIOLATION FOUND** or **UNVERIFIED**. If an
audit is not completed, mark it **Pending report**. A harness audit checks the
interface; a log audit checks what happened during development.

| Protocol rule | Harness finding and evidence | Development-log finding and evidence |
| --- | --- | --- |
| 1. Keep the full development history | [FINDING; FILE/LINE] | [FINDING; LOG/EVENT IDS] |
| 2. Use only permitted resources | [FINDING; FILE/LINE] | [FINDING; LOG/EVENT IDS] |
| 3. Freeze everything before evaluation | [FINDING; FILE/LINE] | [FINDING; HASHES] |
| 4. Report resource usage | [FINDING; FILE/LINE] | [FINDING; USAGE RECORDS] |

Include per-task attempt counts, failed/incomplete starts, simulator steps, seed
validity and the mapping to saved code revisions, inline or in the linked logs.
Raw trajectories, images and videos are optional. Identify any specific finding
that cannot be verified from the available evidence.

### Warnings and required action

[MISSING EVIDENCE, DISCLOSED CORRECTIONS AND REQUIRED FIXES; OR NONE]
