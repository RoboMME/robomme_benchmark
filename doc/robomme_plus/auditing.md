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
Use the submission template's structured harness checklist: verdict first, scope,
checkbox items with PASS/FAIL/UNVERIFIED/N/A and evidence, then a warnings/action
table. Check a box only for PASS. Mark anything not established as UNVERIFIED.
```

Fix confirmed harness violations before development and repeat the review for
the changed code. Keep the report and audited revision. A harness pass describes
what the interface enforces; it does not establish what a later agent actually did.

## 2. Review the completed development record

Preserve prompts, visible model responses, commands/tool calls and outputs,
code revisions, simulator attempts/outcomes, API usage, and child/resumed sessions.
Include failed and interrupted attempts. Supply the final frozen policy hashes
and the harness report. Provider-hidden reasoning is not required.

Public logs may omit raw trajectories, images and videos, including embedded
media/array payloads. Keep text records, call IDs, attempt metadata and source/hash
or manifest references for omitted payloads. State what the auditor inspected;
optional media being absent is not itself a violation.

```text
Use doc/skills/audit_robomme_plus_development.md to audit this saved experiment
against src/robomme_plus/README.md#evaluation-protocol.

Method: <METHOD>
Setting: <SETTING>
Plus rollout budget per task: <0 OR 10>
Tasks and run ID: <TASKS; RUN ID>
Complete development evidence: <LOG AND CODE-HISTORY PATHS>
Omitted trajectory/media payloads and retained references: <MANIFEST OR NONE>
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
Write the report only. Use the structured development-log checklist: verdict first,
scope, checkbox items with status/evidence, per-task accounting and warnings/action
tables. Check a box only for PASS. Public trajectory/media files are optional;
identify any specific finding that cannot be verified from the available records.
State whether results can be reused, need more evidence, or require new runs.
```

For an already completed study, run both reviews against the **saved harness
revision and configuration**. Reviewing today's interface cannot prove how an
earlier run behaved. One report may cover several settings if each has its own
scope, evidence and verdict.

## 3. Submit the reports and evidence

Link the code and public development logs in the
[submission template](submission/model_example_plus.md#agentic-development-audit-record).
Include both audits' verdicts, findings and warnings at the **bottom of the
submission Markdown**; separate full-report links are optional. Identify the
auditor, date, revisions and evidence inspected, and disclose self-review.

Each report should use this order:

1. **Verdict:** PASS WITHIN AUDITED SCOPE, VIOLATION FOUND or UNVERIFIED.
2. **Scope:** method, setting, budget, tasks, run ID, revisions and coverage.
3. **Checklist:** PASS/FAIL/UNVERIFIED/N/A and evidence for every required check.
4. **Per-task accounting:** required for the development-log audit.
5. **Warnings/action table:** missing evidence, corrections and any required reruns.

Use the template's [harness checklist](submission/model_example_plus.md#harness-checklist)
or [development-log checklist](submission/model_example_plus.md#development-log-checklist).
A checked box means PASS. Leave other boxes unchecked and state FAIL, UNVERIFIED
or N/A explicitly. Any FAIL makes the verdict VIOLATION FOUND; otherwise any
UNVERIFIED makes it UNVERIFIED. A pass requires all applicable checks to pass.

For example:

```markdown
# <METHOD>: <harness or development-log> audit
**Verdict:** <PASS WITHIN AUDITED SCOPE / VIOLATION FOUND / UNVERIFIED>
**Scope:** <setting, budget, tasks, run ID, revisions>
**Auditor / date:** <name, date, self-review or independent review>
**Coverage:** <files and records inspected; exclusions>

## Checklist

- [ ] **<Rule number and check> — <STATUS>** · Evidence: <file:line / event ID>
- [ ] **<Next check> — <STATUS>** · Evidence: <file:line / event ID>

## Per-task accounting (development-log audit)

| Task | Budget | Starts | Failed/incomplete starts | Steps | Seed validity | Status |
| --- | --- | --- | --- | --- | --- | --- |
| <task> | <limit> | <count> | <count> | <count or Unknown> | <status> | <status> |

## Warnings and required action

| Finding | Affected scope | Evidence | Required action |
| --- | --- | --- | --- |
| <gap or violation; None if none> | <tasks/settings> | <reference> | <action> |
```

A report cannot certify unobserved behavior. Resolve violations and essential
evidence gaps before claiming compliance. Keep failed/interrupted records and
disclose human corrections rather than removing them from the submitted history.

Examples: [ASPIRE](<submission/aspire-plus (eval_by_RoboMME_team).md#development-audit>)
and [RAX](<submission/rax-plus (eval_by_RoboMME_team).md#development-audit>).
Their reports illustrate scoped conclusions and disclosed limitations.
