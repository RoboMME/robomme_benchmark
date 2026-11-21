# How to audit an agentic method

Run two reviews: **the harness before development**, then **the saved development
logs before submission**. Use a coding assistant that can read the method's code
and evidence files. The skills are Markdown instructions for that assistant;
there is no separate audit command or automatic certification.

Choose `zero-shot`, `demo-only`, `interaction-only` or `demo&interaction` first.
Use a Plus rollout budget of 0 for the first two, and 10 for the interaction
settings. R1/R3/R5 are optional. The budget is per task across all its training seeds.

## 1. Review the harness

Provide the harness source, launch configuration, exposed tools, container/mount
settings, input formats and benchmark/data revisions. Replace the placeholders
below and send this prompt from the benchmark repository root:

```text
Use doc/skills/audit_robomme_plus_harness.md to audit my development harness
against src/robomme_plus/README.md#evaluation-protocol.

Method: <METHOD>
Setting: <SETTING>
Plus rollout budget per task: <0 OR 10>
Tasks: <TASK LIST OR ALL 16>
Harness source and launch configuration: <PATHS>
Benchmark and training-data revisions: <REVISIONS>
Report destination: <OUTPUT_DIR>/harness_audit.md

Trace the inputs exposed to the agent and every Plus simulator entry point.
Check shared budget enforcement, seed selection, isolation, logging and the
development/test boundary. Original RoboMME data and simulator use is permitted.
Inspect files without running policies, simulators or model API calls. Write
the report only; do not modify the harness or evidence.
Start with the verdict, then scope, rule-by-rule evidence, warnings and required
fixes. Mark anything that cannot be established as UNVERIFIED.
```

Fix confirmed harness violations before development and repeat the review for
the changed code. Keep the report and audited revision. A harness pass describes
what the interface enforces; it does not establish what a later agent actually did.

## 2. Review the completed development record

Preserve prompts, visible model responses, commands/tool calls and outputs,
code revisions, simulator attempts/outcomes, API usage, and child/resumed sessions.
Include failed and interrupted attempts. Supply the final frozen policy hashes
and the harness report. Provider-hidden reasoning is not required.

```text
Use doc/skills/audit_robomme_plus_development.md to audit this saved experiment
against src/robomme_plus/README.md#evaluation-protocol.

Method: <METHOD>
Setting: <SETTING>
Plus rollout budget per task: <0 OR 10>
Tasks and run ID: <TASKS; RUN ID>
Complete development evidence: <LOG AND CODE-HISTORY PATHS>
Interaction ledger and native simulator records: <PATHS>
Frozen policies and manifests: <PATHS>
Harness report and audited revision: <PATH; REVISION>
Benchmark and training-data revisions: <REVISIONS>
Report destination: <OUTPUT_DIR>/development_audit.md

Reconcile all attempts across seeds, workers and resumed sessions. Inspect
actual inputs and tool access, distinguish offline analysis from Plus simulator
replay, and trace code history to the frozen policies. Check separation from
test feedback and reconcile resource usage. Do not import or run generated
policies, start simulators, call model APIs, or repair the evidence.
Write the report only. Start with the verdict, then scope, rule-by-rule evidence,
per-task attempts and unresolved gaps. Missing evidence is UNVERIFIED, not a pass.
State whether results can be reused, need more evidence, or require new runs.
```

For an already completed study, run both reviews against the **saved harness
revision and configuration**. Reviewing today's interface cannot prove how an
earlier run behaved. One report may cover several settings if each has its own
scope, evidence and verdict.

## 3. Submit the reports and evidence

Fill the [submission template](submission/model_example_plus.md#agentic-development-audit-record)
with links to the code, complete development record and both reports. Identify
the auditor, date and revisions; disclose whether the audit is a self-review.

Each report should use this order:

1. **Verdict:** PASS WITHIN AUDITED SCOPE, VIOLATION FOUND or UNVERIFIED.
2. **Scope:** method, setting, budget, tasks, run ID, revisions and coverage.
3. **Findings:** the protocol's four numbered rules, with file/line or event references.
4. **Warnings and action:** missing evidence, corrections and any required reruns.

For example:

```markdown
# <METHOD>: <harness or development-log> audit
**Verdict:** <PASS WITHIN AUDITED SCOPE / VIOLATION FOUND / UNVERIFIED>
**Scope:** <setting, budget, tasks, run ID, revisions>
**Auditor / date:** <name, date, self-review or independent review>
**Coverage:** <files and records inspected; exclusions>

| Protocol rule | Finding | Evidence | Status |
| --- | --- | --- | --- |
| 1. Keep the full development history | ... | <file:line / event ID> | ... |
| 2. Use only permitted resources | ... | ... | ... |
| 3. Freeze everything before evaluation | ... | ... | ... |
| 4. Report resource usage | ... | ... | ... |

## Warnings and required action
<Gaps, disclosed corrections, reuse decision and required fixes or reruns>
```

A report cannot certify unobserved behavior. Resolve violations and essential
evidence gaps before claiming compliance. Keep failed/interrupted records and
disclose human corrections rather than removing them from the submitted history.

Examples: [ASPIRE](<submission/aspire-plus (eval_by_RoboMME_team).md#development-audit>)
and [RAX](<submission/rax-plus (eval_by_RoboMME_team).md#development-audit>).
Their reports illustrate scoped conclusions and disclosed limitations.
