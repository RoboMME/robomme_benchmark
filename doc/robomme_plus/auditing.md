# How to audit an agentic method

[Agentic development](../../src/robomme_plus/README.md#evaluation-protocol) needs
two audits: **the harness before development** and **the saved development logs**.
This includes agent-generated solutions, task-specific prompts learned from
rollouts, and self-improving agents. Standard fine-tuning or direct control with
a general prompt, such as Embodied Astra, does not require these audits unless it also
uses agentic development.

Review the logs before submission; complete test-dependent checks using the
RoboMME team's verification run. Give a coding assistant the matching skill,
code and evidence. A completed audit concludes **PASS** or **FAIL**; test-dependent
checks remain **PENDING** until verification. These are reviewer instructions,
not automatic certification. Disclose self-reviews.

Use a fresh assistant session for each review, with **one method, setting and
budget per report**. Run the short prompts below from the repository root.
The skills define the checks and report format.

## Hard requirements

PASS requires H1–H7 and the required evidence R1–R3 below to pass, or be N/A with
a reason. Otherwise conclude FAIL, distinguishing a breach from missing evidence.
The harness audit checks enforcement and logging; the log audit checks the saved record.

| ID | Requirement | Protocol rule |
| --- | --- | --- |
| H1 | **No privileged information:** agents receive only public observations, public feedback and setting-allowed data | 2 |
| H2 | **No Plus task code:** package source (except the allowlisted README) and other methods' Plus solutions are unreachable from agents and never read | 2 |
| H3 | **No task-code details in prompts:** prompts, skills and notes contain no exact task parameters, thresholds or success criteria beyond the public documentation | 2 |
| H4 | **No test data:** only released training seeds with their full settings; no test seeds, metadata or results | 2, 3 |
| H5 | **Rollout budget:** at most the budget per task across all seeds, workers and sessions, counting every start | 1, 5 |
| H6 | **Demonstrations only in demo settings** | 2 |
| H7 | **Frozen before testing:** frozen hashes predate the first test episode; no test-feedback tuning or selection | 3 |

**Required evidence:** R1 complete text logs and code history; R2 per-task usage
and separate development/evaluation API costs; R3 diffs for human corrections
(N/A if none). A GAP blocks PASS. Development trajectories, images and videos are
optional uploads; their absence alone is not a gap. Code history, required text
logs, development attempt metadata and outcomes must remain.

## 1. Review the harness

Point to the harness directory containing its source, configuration and prompts.
Its README or manifests can reference inputs stored elsewhere.

```text
Use doc/skills/audit_robomme_plus_harness.md to audit the harness for <METHOD> in the <SETTING> setting with a per-task rollout budget of <BUDGET>, using the source and configuration in <HARNESS_DIR>. Save the review to <REPORT_PATH>.
```

Fix every FAIL before development and re-review the changed code. A harness PASS
shows what the interface enforces, not what a later agent did.

## 2. Review the completed development record

Point to one study's [complete evidence](agentic_development.md#preserve-evidence)
directory. Its README or manifests can reference shared inputs and results.
Review the saved harness revision used by that study.

Use Codex or Claude Code with the following prompt example to audit your development logs.

```text
Use doc/skills/audit_robomme_plus_development.md to audit the development logs for <METHOD> in the <SETTING> setting with a per-task rollout budget of <BUDGET>, using the evidence in <DEVELOP_DIR>. Save the review to <REPORT_PATH>.
```

The skill marks test-dependent checks PENDING until team verification is complete.
Test results must remain inaccessible to developing agents.

## 3. Add the results to your submission

Copy the [audit record](submission/model_example_plus.md#agentic-development-audit-record)
to the bottom of your submission and fill it in from both audits. Keep failed and
interrupted records, and disclose human corrections rather than removing them.

The [ASPIRE](<submission/aspire-plus (eval_by_RoboMME_team).md#agentic-development-audit-record>)
and [RAX](<submission/rax-plus (eval_by_RoboMME_team).md#agentic-development-audit-record>)
submissions include the returned audit reports and their current status.
