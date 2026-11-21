# How to audit an agentic method

[Agentic development](../../src/robomme_plus/README.md#evaluation-protocol) needs
two audits: **the harness before development** and **the saved development logs
after evaluation, before submission**. Give a coding assistant the matching skill,
code and evidence. Each audit concludes **PASS** or **FAIL**; these are reviewer
instructions, not automatic certification. Disclose self-reviews.

## Hard requirements

PASS requires H1–H7 and the required evidence R1–R3 below to pass, or be N/A with
a reason. Otherwise conclude FAIL, distinguishing a breach from missing evidence.
The harness audit checks enforcement and logging; the log audit checks the saved record.

| ID | Requirement | Protocol rule |
| --- | --- | --- |
| H1 | **No privileged information:** agents receive only public observations, public feedback and setting-allowed data | 2 |
| H2 | **No Plus task code:** package source (except the allowlisted README) and other methods' Plus solutions are unreachable from agents and never read | 2 |
| H3 | **No task-code details in prompts:** prompts, skills and notes contain no exact task parameters, thresholds or success criteria beyond the public documentation | 2 |
| H4 | **No test data:** only released training seeds with their pinned settings; no test seeds, metadata or results | 2, 3 |
| H5 | **Rollout budget:** at most the budget per task across all seeds, workers and sessions, counting every start | 1 |
| H6 | **Demonstrations only in demo settings** | 2 |
| H7 | **Frozen before testing:** frozen hashes predate the first test episode; no test-feedback tuning or selection | 3 |

**Required evidence:** R1 complete text logs and code history; R2 per-task usage
and separate development/evaluation API costs; R3 diffs for human corrections
(N/A if none). A GAP blocks PASS. Raw trajectories, images and videos are optional;
their absence alone is not a gap.

## 1. Review the harness

Provide the harness source, launch configuration, exposed tools, container/mount
settings, input formats and benchmark/data revisions. From the benchmark
repository root, replace the placeholders and send:

```text
Use doc/skills/audit_robomme_plus_harness.md to audit my development harness
against src/robomme_plus/README.md#evaluation-protocol.

Method: <METHOD>
Setting: <SETTING>
Plus rollout budget per task: <0, 1, 3, 5 OR 10>
Tasks: <TASK LIST OR ALL 16>
Harness source and launch configuration: <PATHS>
Initial prompts, skills and notes given to the agents: <PATHS>
Benchmark and training-data revisions: <REVISIONS>
Report destination: <OUTPUT_DIR>/harness_audit.md

Use the audit record in doc/robomme_plus/submission/model_example_plus.md:
conclusion, scope, H1-H7 and R1-R3 with evidence, then warnings/actions.
Inspect read-only; do not run policies, simulators or model APIs. Write only
the report. Missing required evidence blocks PASS.
```

Fix every FAIL before development and re-review the changed code. A harness PASS
shows what the interface enforces, not what a later agent did.

## 2. Review the completed development record

After evaluation, provide the public development-log archive described in
[Preserve evidence](agentic_development.md#preserve-evidence), the ledger, the
frozen policies with their freeze manifest, per-episode test results with times,
and the harness report. Keep test results accessible only to the operator/auditor,
never the developing agent.

```text
Use doc/skills/audit_robomme_plus_development.md to audit this saved experiment
against src/robomme_plus/README.md#evaluation-protocol.

Method: <METHOD>
Setting: <SETTING>
Plus rollout budget per task: <0, 1, 3, 5 OR 10>
Tasks and run ID: <TASKS; RUN ID>
Complete development evidence: <LOG AND CODE-HISTORY PATHS>
Omitted trajectory/media payloads and retained references: <MANIFEST OR NONE>
Interaction ledger and native simulator records: <PATHS>
Frozen policies and freeze manifest: <PATHS>
Per-episode test results with start/finish times: <PATH>
Harness report and audited revision: <PATH; REVISION>
Benchmark and training-data revisions: <REVISIONS>
Report destination: <OUTPUT_DIR>/development_audit.md

Use the audit record in doc/robomme_plus/submission/model_example_plus.md:
conclusion, scope, H1-H7 and R1-R3 with evidence, per-task accounting, then
warnings/actions. Inspect read-only; do not import/run policies, start simulators,
call model APIs or repair evidence. Write only the report. State whether results
can be reused, need more evidence or require new runs.
```

For a completed study, review the **saved harness revision and configuration**;
today's interface cannot show how an earlier run behaved. One report may cover
several settings if each has its own scope, evidence and conclusion.

## 3. Add the results to your submission

Copy the [audit record](submission/model_example_plus.md#agentic-development-audit-record)
to the bottom of your submission and fill it in from both audits. Keep failed and
interrupted records, and disclose human corrections rather than removing them.

The [ASPIRE](<submission/aspire-plus (eval_by_RoboMME_team).md#development-audit>)
and [RAX](<submission/rax-plus (eval_by_RoboMME_team).md#development-audit>)
audits predate this format but show the evidence expected for each check.
