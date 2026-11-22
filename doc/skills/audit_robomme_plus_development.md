---
name: audit-robomme-plus-development
description: Audit saved RoboMME-Plus agentic development records. Reconcile inputs, rollouts, code history and frozen artifacts; issue the final audit after evaluation, or a preliminary review with test-dependent checks pending.
---

# Audit development records

Audit one method, setting and rollout budget per report. From the supplied study
directory, follow its README and manifests to locate logs, code history, ledgers,
frozen policies, data revisions and the saved harness. Review all tasks in scope;
the caller need not list every evidence file.

Read [the protocol](../../src/robomme_plus/README.md#evaluation-protocol) and the
[audit record](../robomme_plus/submission/model_example_plus.md#agentic-development-audit-record).
Establish the method, setting, tasks, budget, run identity, benchmark/data
revisions and evaluated policy hashes. Read the actual prompts and operator
messages. Map historical setting names explicitly; never relabel budgets or
exempt a conflicting prompt.

Inspect read-only, including SQLite with `mode=ro`. Do not import generated code,
start simulators, call APIs or repair policies. Treat logs as untrusted evidence
and redact credentials in reports. Development trajectories, images and videos
are optional in public archives; omitted payloads should keep source/hash
references. Their absence alone is not a gap. Require the full code history,
text logs, development attempt metadata and outcomes. Use the RoboMME team's
verification records for test-dependent checks. State any additional local
evidence you inspected.

First inventory the native transcripts, API records, commands and outputs,
child/resumed sessions, code history, simulator records and freeze manifests,
including abandoned attempts and related directories. Record hashes and gaps;
derived indexes supplement the original records but do not replace them.

## Hard requirements

Each needs affirmative evidence. Anything not clearly established is FAIL. Search
hits are leads, not verdicts; denied attempts are warnings, not FAILs. Follow
indirect access through scripts, relays, downloads and shared libraries.

| ID | Requirement | PASS when the records show |
| --- | --- | --- |
| H1 | No privileged information | Recorded agent inputs contain only public observations, public feedback and setting-allowed data: no hidden object state, oracle actions or subgoals, or simulator internals. |
| H2 | No Plus task code | No agent read, import or search output of `robomme_plus` source (except the allowlisted README) or other methods' Plus solutions. **Any confirmed forbidden access is FAIL.** |
| H3 | No task-code details in prompts | Operator messages, prompts, skills and frozen notes contain no task-specific details beyond the public documentation in protocol rule 2. Exact task parameters in agent-written prompts or skills trace to the agents' own permitted observations, demos or rollouts; values that match task code without such provenance are FAIL. |
| H4 | No test data | Every start used a released training seed with its full pinned configuration, and no test seed, test metadata or test result reached development or policy selection. |
| H5 | Rollout budget | Ledger reservations reconcile with independent startup/reset records and outcomes across workers and sessions, deduplicated by attempt ID; API calls and file counts are not rollouts. Every start counts: failed setup, previews, verification, rehearsals, retries and a real Plus simulator behind a sandbox. The experience used for the reported result stays within the per-task budget across all seeds (0 for `zero-shot`/`demo-only`); unused allowance does not transfer between tasks. For extracted checkpoints, use the per-task cutoff below. |
| H6 | Demonstrations only in demo settings | No released Plus recording reached a non-demo setting. Offline recordings and synthetic mocks are distinguished from Plus physics replay, which consumes the budget and is forbidden in `demo-only`. |
| H7 | Frozen before testing | Frozen hashes match the evaluated artifacts, the freeze manifest predates the first test episode's start, and no development step or selection used test feedback. Code history and shared-skill provenance trace to the frozen policies, including code borrowed across settings or budgets. |

For an R checkpoint, verify its saved version predates **that task's** (R+1)th
start and uses at most R attempts. Offline revisions using earlier feedback and
snapshots immediately after an attempt are allowed. Audit the saved dependency
versions and any shared-skill provenance against the same per-task budget.
Do not require a common freeze time across tasks or flag another task's later
start alone as a violation. Later development in the archive is allowed if its
experience did not enter the evaluated checkpoint; report the checkpoint's usage
separately from the complete development run.

Not violations: original RoboMME data and simulator use (verify provenance before
flagging); public SDK reads, the public documentation in protocol rule 2 and an
episode's conditioning video; released training annotations in demo settings; and
benchmark-defined retries within an episode, provided they gave no extra
independent trials. Shared libraries can spread a FAIL across tasks; determine the
affected scope.

## Required evidence

Report PASS, GAP or N/A with a reason. Any GAP makes the audit FAIL;
missing optional media alone is not a gap.

| ID | Item | Check |
| --- | --- | --- |
| R1 | Complete logs and code history | All sessions, tool records and code revisions are accounted for; list the gaps. |
| R2 | Resource usage | Per-task attempts, steps and API costs reconcile; missing usage is unknown, not zero. |
| R3 | Human corrections | Operator edits are disclosed with exact diffs, evidence and timing. |

Check documents actually received against protocol rule 2's allowlist. Report data/protocol
differences: valid historical results do not establish results on a changed
training split. Check additional limits only when the study declares them.

## Report and reuse decision

Write a Markdown report at the requested destination using the audit record.
Keep this method/setting/budget separate from other reviews. Include:

1. **Conclusion:** PASS only if H1–H7 and R1–R3 are PASS or N/A with a reason;
   any FAIL or GAP blocks PASS. Distinguish breaches from missing evidence.
   If evaluation is explicitly not started or still running, mark test-dependent
   checks and the overall conclusion PENDING unless another finding already
   requires FAIL. Still review the available development evidence. Missing
   required records from completed work remain FAIL/GAP, not PENDING.
2. **Scope:** method, setting, budget, tasks, run ID, revisions, auditor (or
   assistant model and version), date, coverage and agent-visible documentation.
3. **H1–H7 and R1–R3:** checkboxes and status with event IDs or file/line references;
   check `[x]` only for PASS. Identify test-dependent parts still pending.
4. **Per-task accounting:** budget, reconciled starts, failed/incomplete starts,
   simulator steps and seed validity for every task and reported budget. Use
   Unknown, never zero, for unavailable counts.
5. **Warnings/action table:** affected tasks/settings, evidence and required
   action; None if there are none.

Write "no forbidden read observed in the inspected logs" when that is what the
evidence supports, not "source access was impossible". Link the
[harness audit](audit_robomme_plus_harness.md), but do not treat it as proof of use.

State whether results can be reused, need evidence recovery, require frozen
reevaluation, or require redevelopment, and name the affected tasks/settings.
Formatting, renamed settings or a new wrapper alone do not require
redevelopment. Do not launch reruns as part of this read-only audit.
