# ASPIRE on RoboMME-Plus (eval_by_RoboMME_team)

- **Track:** Agentic.
- **Code:** [Adapter and run instructions](../../../scripts/plus_eval/agentic_methods/aspire/README.md); [frozen R10 solutions](../../../scripts/plus_eval/agentic_methods/aspire/solutions/README.md).
- **Development logs:** Public upload pending (`<DEVELOPMENT_LOGS_URL>`). Complete code history, commands, responses, attempts and usage records are retained.
- **Paper:** `<PAPER_URL>`
- **Website:** `<WEBSITE_URL>`
- **Blog:** `<BLOG_URL>`
- **Development model:** Claude Opus 5.5 via AWS Bedrock, high reasoning effort.
- **Training data:** `Yinpei/robomme_plus_data_h5@63cb9b5b62f5413993845c7d89b82b3bd58dc2fa`.
- **Run:** `r10_20261121`; at most 10 development starts per task, shared across all training seeds; 150-minute task deadline.
- **Evaluation:** Frozen policies on the 800 released tests, with default variations. No paid model API calls during frozen execution.

## Results

| Setting | Budget per task | Success | Training starts | Development API estimate | Evaluation API cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `interaction-only` | 10 | 302/800 (37.75%) | 157 | ≈$160 | $0 |
| `Hybrid` | 10 | 342/800 (42.75%) | 134 | ≈$160 | $0 |

API estimates are rounded to the nearest $10 and exclude local compute. Exact usage is retained in the logs.
Requests with incomplete usage records: interaction-only: 1, Hybrid: 2. Estimates use the recorded portion; missing usage is unknown.

Failures, timeouts and infrastructure errors remain in the denominator.

| Task | Interaction-only | Hybrid |
| --- | ---: | ---: |
| BinFill | 25/50 (50.00%) | 41/50 (82.00%) |
| ButtonUnmask | 35/50 (70.00%) | 28/50 (56.00%) |
| ButtonUnmaskSwap | 4/50 (8.00%) | 14/50 (28.00%) |
| InsertPeg | 12/50 (24.00%) | 4/50 (8.00%) |
| MoveCube | 3/50 (6.00%) | 1/50 (2.00%) |
| PatternLock | 3/50 (6.00%) | 10/50 (20.00%) |
| PickHighlight | 17/50 (34.00%) | 11/50 (22.00%) |
| PickXtimes | 24/50 (48.00%) | 11/50 (22.00%) |
| RouteStick | 10/50 (20.00%) | 23/50 (46.00%) |
| StopCube | 40/50 (80.00%) | 29/50 (58.00%) |
| SwingXtimes | 27/50 (54.00%) | 33/50 (66.00%) |
| VideoPlaceButton | 9/50 (18.00%) | 9/50 (18.00%) |
| VideoPlaceOrder | 15/50 (30.00%) | 20/50 (40.00%) |
| VideoRepick | 24/50 (48.00%) | 36/50 (72.00%) |
| VideoUnmask | 27/50 (54.00%) | 34/50 (68.00%) |
| VideoUnmaskSwap | 27/50 (54.00%) | 38/50 (76.00%) |

## Agentic development audit record

The reports below review each setting separately. They are retrospective reviews of the saved harness and complete development record; numerical scores alone do not establish compliance.

### `interaction-only` · R10

<details>
<summary>Harness audit — FAIL</summary>

# ASPIRE `interaction-only` R10 — retrospective harness audit

**Conclusion: FAIL.** R2 has missing development API usage, and H7 lacks sufficient evidence that external provider weights and environments were frozen before testing. These are evidence gaps; this review found no confirmed H1–H6 breach in the inspected evidence. Matching source and policy hashes do not resolve either gap.

| Audit | Conclusion | Summary |
| --- | --- | --- |
| Development harness | FAIL | Incomplete API accounting and external provider freeze provenance. |
| Development logs | Not separately certified | Accounting and integrity reconciliation performed below; semantic log review was sampled. |

## Scope and provenance

- **Method / setting / budget:** ASPIRE, `interaction-only`, independent experiment capped at 10 starts per task; all 16 tasks.
- **Run:** `<CASE_ROOT>` = `<STUDY_ROOT>/aspire/interaction-only/R10`; `<STUDY_ROOT>` denotes the supplied `r10_20261121` archive.
- **Auditor / date:** Codex, GPT-6; 2026-11-22. Exact serving build and organizational independence are not established.
- **Review basis:** supplied current harness-audit procedure, benchmark rules and submission template. This is a fresh retrospective review. The historical PASS statement in `<STUDY_ROOT>/audit/harness_audit.md:3` was treated as an evidence claim, not adopted as this review’s conclusion.
- **Naming:** public `Hybrid` corresponds to archived `demo&interaction`. The inspected differences in the shared BENCHMARK and README prompt documents are naming changes; they do not alter this case’s evidence.
- **Development model:** `us.anthropic.claude-opus-5-5`, high reasoning; Claude Code 2.1.330. The retained CLI matches its recorded SHA-256.
- **Benchmark provenance:** preflight records repository revision `f8d5ec6b4c53ec57ae8f7204a9d2765b93c0b55a` plus archived changes. Dataset validation separately records benchmark commit `88c380fe70ca99e963bee13a24d0110ec5733bc4`. These labels were retained without assuming they identify identical trees; the deployed package’s verified content fingerprint is authoritative here.
- **Training dataset:** `Yinpei/robomme_plus_data_h5@63cb9b5b62f5413993845c7d89b82b3bd58dc2fa`. Recordings were present on the operator side but were not exported or mounted into this interaction-only agent.
- **ASPIRE upstream:** `fa00f0ef2bc1d91d594b9aa502c22133c9f538f7`, with archived adapters and patches.
- **Runtime manifest SHA-256:** `29eb56530feaa42c022993970687a3d8b9a46cbf6da437ed63092e84a9699fae`; all 722 listed files match.
- **Benchmark content fingerprint:** `b34d910c7a8a8153145a88403826fb5427a076b766766d4f6edad0f6f7fd1a72`; all 215 listed files match both development and evaluation pins.
- **Container image:** `sha256:d143d08aeca8a1791548cbdf2a3a0bd252222ea067868fc58c4cc09116c77f0b`.

Evidence: `<STUDY_ROOT>/audit/preflight.json:1`, `audit/dataset_validation.json:1`, `audit/cli_pin.json:1`, `runtime_source_manifest.json:1`, `study_plan.json:1`; `<CASE_ROOT>/study/benchmark_source.json:1`.

**Coverage:** reconciled every development ledger entry and all 800 saved evaluation executions. Parsed all 2,047 development API accounting envelopes, 4,191 audit events and 2,207 campaign snapshots. Verified every stored object hash in both history stores, with no missing referenced objects. Scanned all 2,046 recorded tool inputs for source/test-access indicators and reviewed matching candidates. Semantic reading of prompts, commands, responses and individual trace outputs was sampled; media were not watched, and all API bodies were not exhaustively read. This is complete task coverage for the stated accounting and hash checks, not an exhaustive development-log certification.

The review used the archived runtime, read-only file inspection and SQLite `mode=ro`. It did not execute policies, start simulators, call development/evaluation APIs, change artifacts or spawn agents. The operator will save this response as the requested report; no report file was written.

## Agent-visible documentation and boundaries

The initial archived workspace contains:

- `CLAUDE.md`, campaign task/index configuration and generated worker assignments.
- Original ASPIRE coordinator, worker and verifier prompts, bookkeeping scripts and the six initial generic skills: README, localization, grasping, task exploration, transport and manipulation.
- ASPIRE `docs/api-reference.md`, `docs/run-artifacts.md` and `docs/robomme-plus.md`.
- Public protocol v9 README, BENCHMARK, DEMOS, ASPIRE, PUBLIC_RULES, RUNTIME and fixture README, plus synthetic interface fixtures and bundle manifests.

The installed protocol payloads match their archived originals. The synthetic observation fixture contains blank images and invented state/calibration, rather than benchmark scenes. The supplied current public bundle explicitly publishes the button timing rules; their appearance in the archived prompts is therefore not an H3 breach. Permission to read public documentation was not treated as permission to follow links into simulator source, results or other solutions.

All three recorded coding-agent launches use the pinned image, `--network=none`, a read-only container root and explicit case-local workspace/home, bridge, publication and audit-socket mounts. Later launches additionally mount expired task directories read-only. No demonstrations, benchmark checkout, test directory, other case or Docker socket appears in those mount lists. The host bridge exposes status, budgeted run submission and the logged model relay. Policy workers receive a separate public-tool RPC and no reset endpoint.

Evidence: `<CASE_ROOT>/development/launch_commands.jsonl:1`; `<STUDY_ROOT>/runtime_source/aspire/campaign.py:167`, `campaign.py:271`, `development_runtime.py:27`, `isolated.py:71`, `program_worker.py:24`.

## Hard requirements

- [x] **H1 — PASS: No privileged information.** The producer selects RGB-D, camera calibration, joint/TCP/gripper state and conditioning frames. Robot-only transforms supply calibration. The adapter constructs public contract objects and limited task/controller metadata; the worker receives only registered public tools through data-only serialization. Planning context contains robot calibration, rather than scene objects. Public success and termination feedback remain permitted. No privileged-state crossing was found in the reviewed interface and sampled payloads. Evidence: `<STUDY_ROOT>/runtime_source/benchmark_io/episode.py:52`, `episode.py:161`; `runtime_source/aspire/ASPIRE/src/cap_harness/robomme_plus/adapter.py:66`, `adapter.py:75`, `adapter.py:154`; `runtime_source/aspire/isolated.py:83`, `isolated.py:175`; `runtime_source/aspire/wire.py:71`.
- [x] **H2 — PASS: No reachable Plus task code or other methods’ Plus solutions established by the deployed boundary.** Recorded mounts, offline networking, restricted bridge routes and policy-worker runtime copying keep the operator’s simulator checkout, installed host packages, archives and other cases outside the agent boundary. The archived negative installed-package probe corroborates this; the conclusion also rests on inspection of the actual launch and RPC code. Tool-input scanning found generic ASPIRE package inspection attempts, not demonstrated Plus-source access. No fresh container probe was performed. Evidence: `<CASE_ROOT>/development/launch_commands.jsonl:1`, `development/audit/history/events.jsonl`; `<STUDY_ROOT>/runtime_source/aspire/development_runtime.py:27`, `isolated.py:71`, `isolated.py:193`; `audit/preflight.json:1`.
- [x] **H3 — PASS: No prohibited task-code details in supplied prompts established.** Initial method documents and skills contain generic workflow/API material; Plus-specific instructions derive from the public bundle. Subsequent notes inspected during this review concern development observations and workflow. Published timing rules are within the current allowlist. No operator-supplied implementation-derived task solution was found. Evidence: `<CASE_ROOT>/development/audit/history/events.jsonl:1`; `<STUDY_ROOT>/protocol/ASPIRE.md:1`, `protocol/PUBLIC_RULES.md:1`, `runtime_source/aspire/development_runtime.py:73`, `runtime_source/aspire/campaign.py:182`.
- [x] **H4 — PASS: Released training episodes only; test inputs isolated.** All 157 ledger task/episode/seed combinations match the pinned training configuration. SQLite and JSON configurations agree. Runtime construction compares the complete normalized metadata record, including variations, before reset. Train/test seed sets are disjoint for every task. Test records and results remain operator-side. Evidence: `<CASE_ROOT>/study/study.json:1`, `study/attempts.sqlite3` tables `config` and `attempts`; `<STUDY_ROOT>/runtime_source/benchmark_io/episode.py:109`, `runtime_source/benchmark_io/protocol.py:88`.
- [x] **H5 — PASS: Persistent per-task rollout ceiling.** `BEGIN IMMEDIATE` atomically checks and reserves against the common task ledger before simulator construction. Failed, preview, verification and early-stop starts remain spent. Training reset retries are limited to one; stepping cannot implicitly reset. Agent and policy-worker isolation blocks direct simulator construction. All 157 submitted-program hashes were archived before reservation; requests, native attempt records, outcomes and event step totals reconcile without discrepancies. Evidence: `<STUDY_ROOT>/runtime_source/benchmark_io/protocol.py:103`, `runtime_source/benchmark_io/episode.py:95`, `episode.py:120`, `episode.py:196`; `runtime_source/aspire/campaign.py:103`; `<CASE_ROOT>/study/attempts.sqlite3`, `development/runs/`.
- [x] **H6 — PASS: No released demonstrations exposed.** Both export and mount branches exclude `interaction-only`; none of the three actual agent launch records includes a demonstration mount. Conditioning video comes from the current budgeted episode and is allowed. Evidence: `<STUDY_ROOT>/runtime_source/aspire/campaign.py:172`, `campaign.py:283`, `runtime_source/benchmark_io/protocol.py:152`; `<CASE_ROOT>/development/launch_commands.jsonl:1`.
- [ ] **H7 — FAIL, evidence gap: Complete runtime freezing is not established.** Policy and source freezing checks pass, as detailed below. However, provider launch records identify external Python environments, vendor roots and SAM3/Contact-GraspNet checkpoint files outside the hashed execution tree. The inspected freeze and provider records do not preserve contemporaneous content identities for those loaded weights and installations. Source hashes and dependency lockfiles alone do not establish their actual frozen contents. No post-freeze modification is alleged. Evidence: `<CASE_ROOT>/evaluation/frozen_evaluation.json:1`; `<STUDY_ROOT>/runtime_source/benchmark_io/evaluation.py:15`, `runtime_source/benchmark_io/full_study.py:184`, `runtime/providers/gpu0_sam3.json:1`, `runtime/providers/gpu0_graspnet.json:1`; `runtime_source/aspire/ASPIRE/src/cap_harness/providers/sam3/service.py:228`, `runtime_source/aspire/ASPIRE/src/cap_harness/providers/graspnet/service.py:193`.

## Required evidence

- [x] **R1 — PASS for inspected logging capability and retained accounting/history.** Native coordinator and 16 child transcripts, API transport/event records, commands, tool events, failed work and content-addressed revisions are retained. All 1,143 campaign-history objects and 1,296 audit-history objects match their names; all referenced objects exist. Every development start has saved code, native attempt/outcome records and step events. Two interrupted tool calls lack matching post-tool events, and two VideoUnmaskSwap runs lack high-level result/publication records; their native outcomes, commands and errors remain available. These limitations are disclosed rather than treated as absent starts. This does not certify exhaustive semantic log review. Evidence: `<CASE_ROOT>/development/history/`, `development/audit/history/`, `development/claude-home/projects/`, `development_api/`; `<STUDY_ROOT>/runtime_source/benchmark_io/development_evidence.py:23`, `runtime_source/aspire/campaign.py:79`.
- [ ] **R2 — GAP: Development usage is incomplete.** Starts and steps reconcile, but request `1810_bffa0bb5`, attributable to PatternLock, has empty `usage.json`, `usage_complete=false`, a null estimated cost, a transport timeout and no completion record. Its cost remains unknown. `pending_usage_requests=0` only indicates that a cost file exists; it does not establish complete usage. Evidence: `<CASE_ROOT>/development_api/1810_bffa0bb5/{request,transport,error,usage,cost}.json:1`, `api_cost_summary.json:1`; `<STUDY_ROOT>/runtime_source/benchmark_io/costs.py:45`, `costs.py:65`.
- [x] **R3 — PASS: Operator corrections disclosed with recoverable changes.** `<STUDY_ROOT>/operator/` is absent. Relevant records instead reside under `audit/`: initial interface patch/new files, provider-port correction with before/after source, and scheduler changes. The port correction preceded this case’s model calls and simulator starts and records no generated-policy changes. The separate MoveCube submission-only correction explicitly belongs to archived `demo&interaction`/Hybrid and was not attributed to this interaction-only case. Evidence: `<STUDY_ROOT>/audit/operator_interface_changes.patch:1`, `audit/operator_added_files/`, `audit/corrections/provider_ports/correction.json:1`, `audit/corrections/provider_ports/before/`, `audit/corrections/provider_ports/after/`, `audit/rax_evaluation_start/operator_scheduler_changes.patch:1`, `audit/movecube_submission_20261121/freeze_handoff.json:1`.

## Per-task accounting and frozen-policy reconciliation

“Failed” means the native development outcome has `success=false`, including early exits. Every start has a closed ledger record; unclosed starts are zero throughout. Forty-seven episodes closed without native termination/truncation and remain charged. Fifteen starts used zero control steps and remain charged.

All rows have valid pinned training episode/seed pairs. Every saved test program matches that task’s frozen policy for all 50 executions. Hash prefixes below identify entries in the full freeze manifest; they are not substitute full digests.

| Task | Budget | Starts | Failed / unclosed | Steps | Seeds | Frozen policy SHA-256 prefix | Test hash matches |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| BinFill | 10 | 10 | 6 / 0 | 4,313 | Valid | `958e2d6c5c25` | 50/50 |
| ButtonUnmask | 10 | 10 | 5 / 0 | 5,770 | Valid | `42985e8d6907` | 50/50 |
| ButtonUnmaskSwap | 10 | 10 | 8 / 0 | 4,953 | Valid | `500b36b6e5da` | 50/50 |
| InsertPeg | 10 | 10 | 8 / 0 | 5,243 | Valid | `d2a6b765d806` | 50/50 |
| MoveCube | 10 | 10 | 9 / 0 | 1,475 | Valid | `28fa85dc8b24` | 50/50 |
| PatternLock | 10 | 10 | 9 / 0 | 3,015 | Valid | `84181c0192ed` | 50/50 |
| PickHighlight | 10 | 10 | 5 / 0 | 7,447 | Valid | `50cdfe3a5dd0` | 50/50 |
| PickXtimes | 10 | 10 | 5 / 0 | 7,225 | Valid | `582bcfe59bb5` | 50/50 |
| RouteStick | 10 | 10 | 9 / 0 | 2,832 | Valid | `d2fa82548b23` | 50/50 |
| StopCube | 10 | 10 | 2 / 0 | 1,865 | Valid | `f36a7850e9af` | 50/50 |
| SwingXtimes | 10 | 10 | 6 / 0 | 4,188 | Valid | `3a37fac8b590` | 50/50 |
| VideoPlaceButton | 10 | 10 | 4 / 0 | 776 | Valid | `af4c87e4f417` | 50/50 |
| VideoPlaceOrder | 10 | 10 | 9 / 0 | 772 | Valid | `7bb4bc82cbcc` | 50/50 |
| VideoRepick | 10 | 7 | 3 / 0 | 791 | Valid | `8e44e108122f` | 50/50 |
| VideoUnmask | 10 | 10 | 3 / 0 | 2,733 | Valid | `07eb6eaff593` | 50/50 |
| VideoUnmaskSwap | 10 | 10 | 7 / 0 | 1,991 | Valid | `89d59adfec5f` | 50/50 |
| **Total** | **160 maximum** | **157** | **98 / 0** | **55,389** | **All valid** | **16 policies** | **800/800** |

Sources: `<CASE_ROOT>/study/attempts.sqlite3`, `development/runs/<Task>/attempt_*/request.json`, corresponding native `attempt.json`, `outcome.json` and `events.jsonl`; `evaluation/policy/`, `evaluation/episodes/` and `evaluation/frozen_evaluation.json:1`.

Source-access enforcement is common to all 16 workers: the same offline campaign boundary and public-tool policy-worker boundary apply. The all-task tool-input scan found no demonstrated forbidden source/test access; this is not a claim that every log sentence was reviewed.

**Freeze chronology and integrity:**

- Development starts: 2026-11-21 22:58:15 through final recorded closure at 2026-11-22 02:11:13 UTC.
- Last recorded development API completion: 02:15:22 UTC.
- Freeze manifest creation: 02:15:31 UTC, recording 157 development starts.
- First saved test start: 04:05:13 UTC; last closure: 06:27:39 UTC.
- All 17 frozen policy/selection files and all 757 execution entries match. Selected policies also match their original selected workspace/deadline copies. All 800 evaluation ledger entries are closed, with 50 unique episodes per task. Six recorded infrastructure errors remain in the 800-result denominator.
- Freeze manifest SHA-256: `af1fd98dde91910aa4e850862f48805d31cc7d240f2f37a1edf4f554c94c51d1`.

These checks establish the saved policy/source chain and pre-test chronology. They do not resolve the external provider-asset gap or constitute independent team performance verification.

## API usage and cost basis

Development has **2,047 recorded requests: 2,046 with complete usage and one unknown**. The known estimated subtotal is **USD 158.55272642**, plus the unknown cost of PatternLock request `1810_bffa0bb5`. The complete development total is therefore **unknown**.

Task attribution joins provider response IDs to native worker transcripts. For 23 requests lacking that join, the initial request’s worker assignment or coordinator prompt identifies the scope. Shared coordinator cost is reported separately rather than allocated arbitrarily.

| Development scope | Requests | Known estimated USD | Missing usage |
| --- | ---: | ---: | --- |
| BinFill | 113 | 8.45115854 | None found |
| ButtonUnmask | 77 | 5.58744714 | None found |
| ButtonUnmaskSwap | 122 | 10.26932808 | None found |
| InsertPeg | 131 | 10.15263612 | None found |
| MoveCube | 142 | 12.32459624 | None found |
| PatternLock | 147 | 10.40664196 | **One request; additional cost unknown** |
| PickHighlight | 135 | 10.40259352 | None found |
| PickXtimes | 75 | 6.81968122 | None found |
| RouteStick | 177 | 14.20077054 | None found |
| StopCube | 53 | 4.18308352 | None found |
| SwingXtimes | 110 | 8.00077432 | None found |
| VideoPlaceButton | 101 | 7.82447116 | None found |
| VideoPlaceOrder | 198 | 15.48423096 | None found |
| VideoRepick | 102 | 7.54348342 | None found |
| VideoUnmask | 89 | 8.09729778 | None found |
| VideoUnmaskSwap | 130 | 9.75427046 | None found |
| Shared coordinator | 145 | 9.05026144 | None found |
| **Total** | **2,047** | **158.55272642 + unknown** | **One request** |

The subtotal uses provider-reported usage and the archived AWS Bedrock standard US cross-region list-price card effective 2026-11-09. It is an estimate, not an invoice. Recorded token totals are 56,945 ordinary input, 2,613,000 output, 237,093,151 cache-read and 5,529,054 one-hour cache-creation tokens; five-minute creation is zero. Rates per million tokens are USD 4.40, 22.00, 0.22, 8.80 and 5.50 respectively. Missing usage is excluded from these known totals and is not interpreted as zero.

**Evaluation paid API cost: USD 0.00; requests: 0.** This is supported by the frozen execution path, not solely the aggregate’s zero field: evaluation runs fixed programs in offline containers; the registered tools route to local sensor/control, SAM3, Contact-GraspNet, PyRoki and cuRobo services, without the development model relay. All 800 retained worker commands use offline networking. Recorded development API calls end before freezing; optional Gemini is disabled. Local computation costs are outside this API-cost figure. The H7 provenance gap remains applicable despite the absence of paid evaluation calls.

Evidence: `<CASE_ROOT>/development_api/*/{transport,events,usage,cost}.json*`, `development/claude-home/projects/`, `api_cost_summary.json:1`, `evaluation/api_usage.json:1`; `<STUDY_ROOT>/runtime_source/benchmark_io/unified_study.py:173`, `runtime_source/aspire/isolated.py:161`, `isolated.py:193`, `runtime_source/aspire/program_worker.py:24`.

## Warnings and required action

| Finding | Affected scope | Evidence | Smallest required action |
| --- | --- | --- | --- |
| **R2 GAP — unknown development usage** | PatternLock; complete case cost | `<CASE_ROOT>/development_api/1810_bffa0bb5/` | Recover contemporaneous provider usage/billing evidence for this request and reconcile it. Otherwise retain unknown usage and FAIL; an empty usage object cannot substantiate zero cost. |
| **H7 FAIL — incomplete provider freeze evidence** | All 16 tasks using external providers | `<CASE_ROOT>/evaluation/frozen_evaluation.json:1`; `<STUDY_ROOT>/runtime/providers/`; `runtime_source/benchmark_io/full_study.py:184` | Recover contemporaneous identities/hashes for loaded checkpoints and actual provider/vendor environments. If unavailable, freeze the complete stack and perform a separately identified evaluation of the unchanged selected programs, without test-driven selection; preserve the historical gap. |
| **Nonblocking finalization defect** | VideoUnmaskSwap attempts 002 and 003 | `<CASE_ROOT>/development/runs/VideoUnmaskSwap/attempt_002_924ec2dd/operator.log`, `attempt_003_9d66c3a9/operator.log` | Preserve their native outcomes and charges. For future runs, handle `episode_terminated` in recorder finalization so publication/result creation survives; do not alter historical evidence. |
| **Disclosed operator corrections** | Shared infrastructure | `<STUDY_ROOT>/audit/corrections/provider_ports/`, `audit/rax_evaluation_start/` | Retain existing before/after source and scheduler records with the report. No interaction-only policy correction was established. |

PASS remains blocked by H7 and R2. No artifacts were changed during this review.

</details>

**Development audit: PENDING.** The independent review is running or queued; its returned report will be inserted here.

### `Hybrid` · R10

**Harness audit: PENDING.** The independent review is running or queued; its returned report will be inserted here.

**Development audit: PENDING.** The independent review is running or queued; its returned report will be inserted here.
