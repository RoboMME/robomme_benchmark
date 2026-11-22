# RAX on RoboMME-Plus (eval_by_RoboMME_team)

- **Track:** Agentic.
- **Code:** [Adapter and run instructions](../../../scripts/plus_eval/agentic_methods/rax/README.md); [frozen R10 solutions](../../../scripts/plus_eval/agentic_methods/rax/solutions/README.md).
- **Development logs:** Public upload pending (`<DEVELOPMENT_LOGS_URL>`). Complete code history, commands, responses, attempts and usage records are retained.
- **Paper:** `<PAPER_URL>`
- **Website:** `<WEBSITE_URL>`
- **Blog:** `<BLOG_URL>`
- **Development model:** Claude Opus 5.5 via AWS Bedrock, high reasoning effort.
- **Run:** `r10_20261121`; at most 10 development starts per task, shared across all training seeds; 150-minute task deadline.
- **Evaluation:** Frozen policies on the 800 released tests, with default variations. No paid model API calls during frozen execution.

## Results

| Setting | Budget per task | Success | Training starts | Development API estimate | Evaluation API cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Interaction-only` | 10 | 370/800 (46.25%) | 123 | ≈$90 | $0 |
| `Hybrid` | 10 | 326/800 (40.75%) | 121 | ≈$100 | $0 |

API estimates are rounded to the nearest $10 and exclude local compute. Exact usage is retained in the logs.

Failures, timeouts and infrastructure errors remain in the denominator.

| Task | Interaction-only | Hybrid |
| --- | ---: | ---: |
| BinFill | 31/50 (62.00%) | 25/50 (50.00%) |
| ButtonUnmask | 31/50 (62.00%) | 34/50 (68.00%) |
| ButtonUnmaskSwap | 15/50 (30.00%) | 10/50 (20.00%) |
| InsertPeg | 16/50 (32.00%) | 0/50 (0.00%) |
| MoveCube | 0/50 (0.00%) | 1/50 (2.00%) |
| PatternLock | 10/50 (20.00%) | 10/50 (20.00%) |
| PickHighlight | 18/50 (36.00%) | 19/50 (38.00%) |
| PickXtimes | 35/50 (70.00%) | 20/50 (40.00%) |
| RouteStick | 10/50 (20.00%) | 12/50 (24.00%) |
| StopCube | 26/50 (52.00%) | 33/50 (66.00%) |
| SwingXtimes | 32/50 (64.00%) | 33/50 (66.00%) |
| VideoPlaceButton | 31/50 (62.00%) | 19/50 (38.00%) |
| VideoPlaceOrder | 16/50 (32.00%) | 18/50 (36.00%) |
| VideoRepick | 33/50 (66.00%) | 31/50 (62.00%) |
| VideoUnmask | 28/50 (56.00%) | 29/50 (58.00%) |
| VideoUnmaskSwap | 38/50 (76.00%) | 32/50 (64.00%) |

## Agentic development audit record

The reports below review each setting separately. They are retrospective reviews of the saved harness and complete development record; numerical scores alone do not establish compliance.

### `Interaction-only` · R10

<details>
<summary>Harness audit — FAIL</summary>

**Retrospective harness audit — RAX · Interaction-only · R10**

**Conclusion: FAIL because required evidence remains incomplete.** The recorded development accounting reconciles, and the frozen policy/runtime hashes verify. However, the deployed image’s contents could not be independently established from the supplied archive, leaving source/test isolation and prevention of direct simulator bypasses unproven. Four evaluation reservations also lack finalized simulator outcome and step records. These are evidence gaps; this review did not establish a forbidden-source read, test-feedback tuning, demonstration exposure, or development-budget overrun.

**Scope and provenance**

- Method: RAX; setting: `Interaction-only`; budget: 10 starts per task across all released training seeds; all 16 tasks listed below. This is the archived independent R10 experiment, not a smaller-budget checkpoint.
- `<STUDY_ROOT>` denotes the supplied study archive. `<CASE_ROOT>` denotes `<STUDY_ROOT>/rax/interaction-only/R10`. All references below use these aliases.
- Auditor/date: Codex, GPT-6 family; exact backend build unavailable; 2026-11-22. This is a fresh retrospective assistant review of operator-provided evidence. External organizational independence is not established.
- The current evaluation protocol, submission audit-record format, development interface, and requested harness-audit procedure govern this review. Archived `demo&interaction` means public **Hybrid**; the naming change does not alter evidence or this case’s setting.
- The archived prior audit at `<STUDY_ROOT>/audit/harness_audit.md` was treated as a historical claim, not adopted as this review’s conclusion. No pre-development audit was performed now.
- Inspection was read-only, with every SQLite connection opened using `mode=ro`. No simulator, generated policy, development/evaluation API, or additional agent was started. No report or artifact was written.

The deployed implementation inspected was `<STUDY_ROOT>/runtime_source/`, not the current working implementation. Its manifest contains 722 entries, all rehashed successfully. The manifest SHA-256 is `29eb56530feaa42c022993970687a3d8b9a46cbf6da437ed63092e84a9699fae`. RAX provenance identifies upstream revision `d49c8ad605008bb5a87e909d0473cc3f5e3600a6` plus the recorded adapter/audit patches.

The benchmark fingerprint is `b34d910c7a8a8153145a88403826fb5427a076b766766d4f6edad0f6f7fd1a72`; its recorded files match the frozen benchmark snapshot. Preparation records identify commit `f8d5ec6b4c53ec57ae8f7204a9d2765b93c0b55a`, while dataset validation names `88c380fe70ca99e963bee13a24d0110ec5733bc4`. This review uses the verified per-file snapshot identity rather than assuming either label describes a clean checkout. The released-data provenance is `Yinpei/robomme_plus_data_h5@63cb9b5b62f5413993845c7d89b82b3bd58dc2fa`; recordings were not exported to this setting.

Every saved development launch pins image `sha256:d143d08aeca8a1791548cbdf2a3a0bd252222ea067868fc58c4cc09116c77f0b`. The archived Claude Code binary rehashes to `f390eaa0814e6a1bff74cbb47acb74cb9c89d30ced809fe0af5af00c7bfe8ebb`, matching `<STUDY_ROOT>/audit/cli_pin.json` and its recorded version 2.1.330.

**Coverage and limits**

All 16 task configurations, launch commands, demonstration contexts, development ledgers/results, history indexes and referenced history objects were checked. The review reconciled 123 development reservations with 123 native attempt/outcome pairs and their saved source revisions. It checked 3,115 history events, 595 distinct task-local revisions, and 1,635 referenced task-local objects; no missing or hash-invalid referenced object was found. All 123 reservations have a matching `SimulatorStart` snapshot saved before reservation.

All 1,307 development API request records were parsed and associated with task sessions. Their required request/response-event, completion, usage and cost files were checked, and all provider-body hashes matched `transport.json`. All 800 evaluation result records and reservations were inspected structurally. All 800 episode bundle copies—2,900 files—were rehashed against their frozen policies; 794 saved SDK copies also matched the frozen SDK hash.

This was not an exhaustive semantic development-log audit. Tool inputs and history records were scanned programmatically; initial MCP packets were sampled once per task. The full media-heavy MCP traces, images, videos and trajectories were not exhaustively reviewed or visually inspected. Provider-hidden reasoning was not required. No live container probe was run. Read-only Docker image inspection was attempted but denied by the environment.

**Agent-visible documentation**

The initial history snapshots preserve the exact documents and hashes. Their initial file sets agree across tasks except for the task-specific `TASK.md` text. They include:

- Native `AGENTS.md`, `CLAUDE.md`, `TASK.md`, `START_PROMPT.md`, `MCP_SESSION.md`, `MCP_CONTRACT.json`, and `api/runtime.py` under `<CASE_ROOT>/development/<Task>/agent-workspace/`.
- The installed `docs/robomme-protocol-v9/` copies of `BENCHMARK.md`, `DEMOS.md`, `RAX.md`, `PUBLIC_RULES.md`, `RUNTIME.md`, `README.md`, and synthetic fixtures with their manifests.
- The native skills `autonomous-control`, `avoid-overfitting`, `change-discipline`, `engineer-the-controller`, `evidence-and-regression`, `experience-memory`, `gemini`, `keep-trying`, `learn-from-incidents`, `manual-robot-control`, `match-precision-to-error`, `measure-success`, `object-localization`, `prioritize-by-funnel`, `rehearse-the-controller`, `robot-interface`, `spend-the-budget`, `verify-committal-actions`, and `wrist-camera-inspection`, including their local references.

The archived public Markdown matches the current public prompt bundle apart from the Hybrid naming changes in `BENCHMARK.md` and `README.md`. Public button deadlines therefore do not constitute private task-code disclosures. The inspected native instructions describe workflow and interfaces rather than private task parameters or solutions. Linked synthetic fixtures were inspected as evidence, not treated as authorization to access arbitrary linked files.

**Hard requirements**

- [x] **H1 — PASS: public observation boundary.** The trusted producer selects RGB-D, calibration and robot state; RAX translates those fields, the language goal, phase/time information, control limits and public feedback. Pixel geometry derives from public depth/calibration. Operator reward records remain host-side. References: `<STUDY_ROOT>/runtime_source/benchmark_io/episode.py:52`; `runtime_source/rax/services/robomme_plus/session.py:77`; `runtime_source/rax/services/robomme_plus/backend.py:127`; `runtime_source/rax/services/mcp_reply.py:39`.
- [ ] **H2 — FAIL, evidence gap: deployed-image source reachability is not fully established.** All 16 saved launches use a read-only root, network isolation, dropped capabilities and explicit mounts without the host benchmark checkout or other solutions. The Dockerfile and builder describe an offline image without simulator source. However, no digest-bound image filesystem/package inventory or corresponding image build record was located in the supplied study evidence. `<STUDY_ROOT>/audit/preflight.json:15` supplies only summarized negative package probes. With direct image inspection unavailable, that summary cannot establish absence of installed or otherwise reachable forbidden copies. No actual forbidden read was identified. References: `<STUDY_ROOT>/runtime_source/rax/harness/claude_cli/auto_research.py:200`; `runtime_source/rax/scripts/build_agent_image.py:81`; `<CASE_ROOT>/development/<Task>/launch_command.json:1`.
- [x] **H3 — PASS: no private task details found in supplied prompts/skills.** The operator prompt, task-context producer, initial documents and local skill references contain public rules, generic development guidance and interface conventions. References: `<STUDY_ROOT>/runtime_source/rax/services/robomme_plus/benchmark.py:61`; `<CASE_ROOT>/development/<Task>/development_prompt.txt:1`; `development/<Task>/audit/history/events.jsonl:1`; `<STUDY_ROOT>/protocol/PUBLIC_RULES.md:1`.
- [ ] **H4 — FAIL, evidence gap: training selection passes, but complete test-data isolation remains unproven.** All 123 recorded starts select pinned released training records, including their full settings. The service uses the training dataset and requires an operator-only frozen capability for tests. Agent mounts omit the study ledger, full metadata and evaluation directories. Nevertheless, H2’s unresolved image contents prevent establishing that packaged test metadata is unreachable everywhere. References: `<STUDY_ROOT>/runtime_source/benchmark_io/episode.py:95`; `runtime_source/rax/services/robomme_plus/session.py:44`; `<CASE_ROOT>/study/study.json:1`; `study/attempts.sqlite3`, tables `attempts` and `config`.
- [ ] **H5 — FAIL, evidence gap: recorded accounting passes; complete bypass prevention is not established.** The shared SQLite transaction reserves before environment construction, retains failed/abandoned reservations, validates full metadata and limits training reset recovery to one attempt. Rehearsals use the same path; episode closure has no implicit reset. Recorded usage is 67 interactive starts plus 56 rehearsals, all within R10. Two invalid-bundle rehearsal requests were rejected before reservation. The intended container/RPC boundary blocks direct host simulator access, but absence of an alternate simulator installation inside the deployed image remains unverified under H2. References: `<STUDY_ROOT>/runtime_source/benchmark_io/protocol.py:103`; `runtime_source/benchmark_io/episode.py:95`; `runtime_source/benchmark_io/episode.py:120`; `runtime_source/rax/services/robomme_plus/session.py:44`; `runtime_source/rax/services/controller/sandbox.py:108`.
- [x] **H6 — PASS: released demonstration exports are absent.** Every task has `demonstration_context=none`, zero expert demonstrations, and an empty released-demonstration directory. No recording dataset mount appears in the saved launches. The setting gate rejects released-demo access; conditioning videos remain permitted episode observations. References: `<STUDY_ROOT>/runtime_source/rax/services/robomme_plus/benchmark.py:84`; `<CASE_ROOT>/development/<Task>/demonstration-context.json:1`; `development/<Task>/configuration.json:1`.
- [x] **H7 — PASS: frozen policies and execution files precede testing.** All 16 frozen policy trees and the 757 shared execution/protocol/plan hashes match. Each manifest predates its first test reservation, and every manifest’s case-wide development count remains 123. All 800 episode policy copies match. The operator scheduler snapshot was hashed before testing, its worker implementation matches the frozen worker, and its manifest verifies. References: `<STUDY_ROOT>/runtime_source/benchmark_io/evaluation.py:39`; `runtime_source/benchmark_io/unified_study.py:193`; `<CASE_ROOT>/evaluation/<Task>/frozen_evaluation.json:1`; `<STUDY_ROOT>/runtime/operator_scheduler_20/manifest.json:1`.

**Required evidence**

- [x] **R1 — PASS for inspected development capture and structural retention.** Native transcripts, API request/visible-response records, tool events and content-addressed code history are retained. Synchronous hooks capture tool activity, and the trusted simulator records a source snapshot before each start. The archive has one recorded top-level session per task, no observed child-agent calls, and no recorded hook failures. This does not claim exhaustive semantic review of every command/output. References: `<STUDY_ROOT>/runtime_source/benchmark_io/development_evidence.py:29`; `runtime_source/benchmark_io/development_evidence.py:80`; `runtime_source/benchmark_io/evidence_hook.py:7`; `<CASE_ROOT>/development/<Task>/audit/history/events.jsonl:1`.
- [ ] **R2 — GAP: development usage reconciles, but four evaluation starts lack finalized simulator usage records.** Development starts, outcomes, steps, task attribution, API usage and estimated costs reconcile. Evaluation has 800 unique reservations and 800 result records, but four reservations have NULL `ended` and `outcome`, with no saved simulator `outcome.json`. Their result records show startup RPC failures and no controller execution; missing simulator step outcomes are retained as Unknown rather than replaced with assumed zero. References: `<CASE_ROOT>/evaluation/<Task>/evaluation_attempts.sqlite3`; `evaluation/episodes/BinFill/episode_012/result.json:15`; corresponding ButtonUnmask episode 012, RouteStick episode 011 and VideoPlaceButton episode 010 records; `<STUDY_ROOT>/runtime_source/benchmark_io/episode.py:221`.
- [x] **R3 — PASS: relevant operator corrections are disclosed and reconstructable.** Prelaunch interface edits have a patch and added-file copies. The provider-port correction has before/after source and manifests; its after-source matches the deployed runtime. The evaluation scheduler adjustment has a patch, launch record and hashed snapshot. No RAX generated-policy correction was identified. References: `<STUDY_ROOT>/audit/operator_interface_changes.patch:1`; `audit/corrections/provider_ports/correction.json:1`; `audit/rax_evaluation_start/operator_scheduler_changes.patch:1`. No top-level `operator/` directory exists. The separate ASPIRE/Hybrid MoveCube finalization is outside this case.

**Per-task development accounting**

“Failed/incomplete” counts distinct starts without confirmed success, including early closes; it is not an additional allowance. All 123 reservations have finalized development outcomes. Ten starts closed before native termination, including two with zero execution steps; all remain spent. Seed validity includes agreement with the pinned episode settings. The accounting status below concerns recorded usage, not the unresolved global isolation checks.

| Task | Budget | Starts | Failed/incomplete | Steps | Seed validity | Dev API requests | Dev USD, estimated | Accounting |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| BinFill | 10 | 8 | 2 | 4,962 | PASS | 63 | 3.72 | PASS |
| ButtonUnmask | 10 | 6 | 2 | 1,929 | PASS | 78 | 5.07 | PASS |
| ButtonUnmaskSwap | 10 | 7 | 3 | 3,194 | PASS | 64 | 4.63 | PASS |
| InsertPeg | 10 | 5 | 3 | 3,132 | PASS | 80 | 5.36 | PASS |
| MoveCube | 10 | 5 | 5 | 1,345 | PASS | 67 | 4.30 | PASS |
| PatternLock | 10 | 8 | 6 | 2,733 | PASS | 79 | 5.14 | PASS |
| PickHighlight | 10 | 10 | 2 | 11,487 | PASS | 98 | 6.97 | PASS |
| PickXtimes | 10 | 9 | 4 | 6,490 | PASS | 78 | 6.00 | PASS |
| RouteStick | 10 | 8 | 6 | 3,655 | PASS | 84 | 5.30 | PASS |
| StopCube | 10 | 9 | 2 | 1,564 | PASS | 91 | 6.28 | PASS |
| SwingXtimes | 10 | 9 | 3 | 4,180 | PASS | 67 | 5.24 | PASS |
| VideoPlaceButton | 10 | 10 | 8 | 1,982 | PASS | 123 | 8.44 | PASS |
| VideoPlaceOrder | 10 | 9 | 6 | 992 | PASS | 123 | 8.81 | PASS |
| VideoRepick | 10 | 6 | 1 | 1,412 | PASS | 63 | 3.75 | PASS |
| VideoUnmask | 10 | 7 | 0 | 3,865 | PASS | 59 | 3.45 | PASS |
| VideoUnmaskSwap | 10 | 7 | 1 | 4,285 | PASS | 90 | 5.47 | PASS |
| **Total** | **10/task** | **123** | **54** | **57,207** | **All recorded starts valid** | **1,307** | **87.93** | **Recorded accounting passes** |

Basis: `<CASE_ROOT>/study/attempts.sqlite3`, its `attempt_provenance` table, and `development/<Task>/native/results/*/sim/recording/{attempt,outcome}.json`. No development outcome exceeded its pinned execution-step limit. The source-access scan identified no forbidden-source access in recorded tool inputs; the installed-image gap applies to every task.

**Frozen-test reconciliation**

The SHA-256 prefixes below identify `<CASE_ROOT>/evaluation/<Task>/frozen_evaluation.json`; full hashes and per-file policy hashes remain in those manifests. All listed hashes, frozen policy files and episode bundle copies verified. Each manifest’s `development_attempt_count=123` is the **case-wide total**, not 123 starts for that task.

| Task | Freeze manifest SHA-256 prefix | Test reservations / finalized simulator outcomes | Recorded execution steps |
| --- | --- | ---: | ---: |
| BinFill | `000dff9b44a637ff` | 50 / 49 | 33,140 + Unknown |
| ButtonUnmask | `0b29f97c181e173d` | 50 / 49 | 16,880 + Unknown |
| ButtonUnmaskSwap | `f84c8a44525f498a` | 50 / 50 | 37,657 |
| InsertPeg | `7ed8345ee7421a26` | 50 / 50 | 32,628 |
| MoveCube | `00dbd7d88afee5fa` | 50 / 50 | 18,346 |
| PatternLock | `e9daacad6977b5fc` | 50 / 50 | 15,872 |
| PickHighlight | `1b08e6a8561796a5` | 50 / 50 | 47,985 |
| PickXtimes | `563728de7e5d6cdc` | 50 / 50 | 51,011 |
| RouteStick | `7968d15921194f27` | 50 / 49 | 21,758 + Unknown |
| StopCube | `a18da237b4dbcf00` | 50 / 50 | 14,721 |
| SwingXtimes | `54444b650c167a36` | 50 / 50 | 21,839 |
| VideoPlaceButton | `83bcfca0f6b46fb0` | 50 / 49 | 8,951 + Unknown |
| VideoPlaceOrder | `6a4d1aebb513a42c` | 50 / 50 | 4,703 |
| VideoRepick | `c8e3c18fcd663816` | 50 / 50 | 16,500 |
| VideoUnmask | `0b3dede2cc04f1d7` | 50 / 50 | 21,481 |
| VideoUnmaskSwap | `36bf2d4c2aeeb66a` | 50 / 50 | 36,619 |
| **Total** | **All verified** | **800 / 796** | **400,091 + four Unknown records** |

The four unfinished simulator reservations remain represented as failed test results, with no repeated episode reservation. They remain in the 800-episode denominator. The archive’s top-level `infrastructure_errors=0` does not describe these nested `formal_outcome=error` startup failures: the aggregator counts a different top-level field. Saved scores are not certified by this harness review.

**Usage and cost basis**

Development used AWS Bedrock model `us.anthropic.claude-opus-5-5`, high effort. All 1,307 generation requests have complete recorded usage/cost records and reconcile with task-local relay usage and `<CASE_ROOT>/api_cost_summary.json`:

| Token category | Recorded tokens | USD per million |
| --- | ---: | ---: |
| Uncached input | 21,378 | 4.40 |
| Output | 1,335,933 | 22.00 |
| Cache-read input | 144,653,411 | 0.22 |
| Five-minute cache creation | 0 | 5.50 |
| One-hour cache creation | 3,025,544 | 8.80 |

The resulting development estimate is **$87.93312682**, using the archived standard US cross-region rate card effective 2026-11-09. This is provider-reported usage multiplied by recorded list prices, not an invoice. No missing generation usage or unresolved cache-creation tokens were found. Simulator/GPU expenditure and this retrospective review’s cost are not included.

**Evaluation paid API usage: 0 requests; $0.00.** This is supported by the frozen offline controller path, network-isolated controller configuration, disabled external perception, and all 800 result records reporting zero API calls, charged/reported tokens and unresolved token reservations. It is independently consistent with `<CASE_ROOT>/evaluation/api_usage.json`; the summary alone was not used to establish zero cost. The four missing simulator outcomes do not create missing model usage because those starts never reached controller execution and the frozen startup path makes no paid API call.

**Warnings and required action**

| Finding | Affected scope | Evidence | Smallest required action |
| --- | --- | --- | --- |
| **Required evidence gap: image contents not independently established; H2/H4/H5 FAIL** | All 16 tasks; shared deployed image | `<STUDY_ROOT>/audit/preflight.json:15`; `runtime_source/rax/scripts/build_agent_image.py:81`; saved launch commands | Supply a read-only image filesystem/package inventory or export and build provenance tied to the exact deployed digest. Check forbidden source, packaged test metadata, archives and direct simulator entry points, then re-audit. No forbidden contents are asserted here. |
| **Required evidence gap: four unfinished evaluation usage records; R2 GAP** | BinFill episode 012; ButtonUnmask episode 012; RouteStick episode 011; VideoPlaceButton episode 010 | Their evaluation ledgers, startup-error results and missing simulator outcome files | Preserve originals and add a provenance-linked reconciliation record establishing termination and execution-step usage from surviving evidence. Keep unavailable values Unknown. Do not rerun reserved tests merely to replace missing historical records. |
| **Summary classification warning** | Evaluation failure summary | `<STUDY_ROOT>/runtime_source/benchmark_io/unified_study.py:251`; four nested startup-error results | Disclose the nested failures alongside the summary and retain all 800 results in the denominator. |
| **Disclosed operator changes** | Shared startup/scheduling infrastructure | `<STUDY_ROOT>/audit/corrections/provider_ports/`; `audit/rax_evaluation_start/` | Retain the existing before/after records, patches and hashed scheduler snapshot. No RAX policy correction was found. |

PASS remains blocked until the required evidence gaps are resolved. This report makes no claim that retrospective review supplied pre-development approval.

</details>

**Development audit: PENDING.** The independent review is running or queued; its returned report will be inserted here.

### `Hybrid` · R10

**Harness audit: PENDING.** The independent review is running or queued; its returned report will be inserted here.

**Development audit: PENDING.** The independent review is running or queued; its returned report will be inserted here.
