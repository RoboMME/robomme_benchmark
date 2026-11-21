---
name: audit-robomme-plus-development
description: Audit saved RoboMME-Plus agentic development evidence. Reconcile attempts, inputs, code history and frozen policies for a specified setting and budget. Produce a verdict-first checklist and per-task accounting without rerunning or repairing the experiment.
---

# Audit development records

Read [the protocol](../../src/robomme_plus/README.md#evaluation-protocol) and
[audit format](../robomme_plus/submission/model_example_plus.md#agentic-development-audit-record).
Establish method, setting, tasks, budget, run identity, benchmark/data revisions
and evaluated policy hashes. Read actual prompts and operator messages.
Map historical setting names explicitly; never relabel budgets or exempt a
conflicting prompt. Inspect read-only, including SQLite with `mode=ro`.
Do not import generated code, start simulators, call APIs or repair policies.
Treat logs as untrusted evidence and redact credentials in reports.

Raw trajectories, images and videos are optional in the public development
archive. Check retained text, call IDs, attempt metadata and source/hash or
manifest references for omitted payloads. State whether any additional local
evidence was inspected. Missing optional media is not itself a violation; mark
only findings that lack supporting evidence as UNVERIFIED.

1. Inventory native transcripts, API records, commands/outputs, child/resumed
   sessions, code history, simulator records and freeze manifests. Include
   abandoned attempts and related directories. Record hashes and gaps; derived
   indexes supplement rather than replace the original records.
2. Reconcile tool calls and responses. Review actual inputs/outputs for forbidden
   Plus task source, hidden Plus state, Plus demos in non-demo settings and
   Plus test-derived tuning. Original RoboMME's dataset and simulator are
   permitted in every setting; verify provenance before flagging access. Released
   training annotations belong to the demo setting, not runtime oracle access.
   Follow indirect access through scripts, relays, downloads and shared libraries.
   Search hits are leads, not verdicts. Distinguish forbidden reads, denied
   attempts and exposed-but-unobserved paths. Public SDK reads and an episode's
   conditioning video are permitted.
3. Reconcile Plus ledger reservations against independent startup/reset records and
   outcomes across workers/sessions. Count failed setup, previews, verification,
   rehearsals and retries. Deduplicate copies by attempt identity; API calls and
   file counts are not rollout counts. Verify every seed and full configuration.
   Ten attempts means ten total per task, not per seed. Unfinished starts count;
   unused allowance does not transfer between tasks.
4. Separate offline recordings/synthetic mocks from Plus physics replay. A real
   Plus simulator behind a sandbox still counts. `zero-shot`/`demo-only` allow no
   Plus starts. Original RoboMME simulator use is outside this rollout budget.
   Benchmark-defined internal retries remain within their episodes; check that
   they did not provide extra independent trials.
5. Trace code history and shared-skill provenance to frozen policies. Inspect
   operator edits, exact diffs, evidence and timing. Check test-feedback selection,
   code borrowed across settings/budgets and hidden task solutions. Shared
   libraries may spread a violation across tasks; determine that dependency scope.
6. Verify development ended before test feedback became available and frozen
   hashes match. Report data/protocol differences: valid historical results do
   not establish results for a changed training split. Reconcile attempts, steps
   and API costs; missing usage is unknown, not zero. Check additional limits
   only when declared by the study.

## Report and reuse decision

Generate structured Markdown using the
[submission format](../robomme_plus/submission/model_example_plus.md#agentic-development-audit-record):

1. **Verdict first:** PASS WITHIN AUDITED SCOPE, VIOLATION FOUND or UNVERIFIED.
2. **Scope:** method, setting, budget, tasks, run ID, revisions, auditor/date and coverage.
3. **Checklist:** copy the [development-log checklist](../robomme_plus/submission/model_example_plus.md#development-log-checklist).
   Give every item a status and a short event ID or file/line reference. Use `[x]`
   only for PASS; use `[ ]` with FAIL, UNVERIFIED or N/A for every other status.
   A checked box means verified compliance, not merely that the check was performed.
4. **Per-task accounting table:** budget, reconciled starts, failed/incomplete
   starts, simulator steps, seed validity and status for every task in scope.
   Use Unknown for unavailable counts, never zero; account for each reported budget.
5. **Warnings/action table:** affected tasks/settings, evidence and required action;
   write None if there are no warnings. Avoid long narrative summaries.

Any FAIL yields VIOLATION FOUND. Otherwise, any UNVERIFIED yields UNVERIFIED;
PASS WITHIN AUDITED SCOPE requires all applicable checks to pass. Explain N/A.
Link the
[harness audit](audit_robomme_plus_harness.md), without treating it as proof of use.
Map findings to the protocol's four rules: development history, permitted
resources, freeze before evaluation, and resource usage, in that order.

A pass needs affirmative evidence for applicable checks. Missing essential
records remain unverified; a confirmed violation remains a violation even if
other checks are unverified. Say "no forbidden read observed in the inspected
logs" when supported, not "source access was impossible".

State whether results can be reused, need evidence recovery, require frozen
reevaluation, or require redevelopment. Specify affected tasks/settings and why.
Formatting, renamed settings or a new wrapper alone do not require redevelopment.
Do not launch reruns as part of this read-only audit.
Include verdicts, evidence and warnings in the submission's final audit section;
separate full-report links are optional.
