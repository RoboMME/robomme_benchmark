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

API estimates are rounded to the nearest $10 and exclude local compute. Recorded usage is retained in the logs; missing provider-error usage remains unknown.

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

Both settings passed the follow-up harness and development reviews. The reports below disclose the self-review, evidence recovery and warnings. Original audit reports and native records are retained.

### `Interaction-only` · R10

<details>
<summary>Harness audit — PASS with warnings</summary>

# RAX · Interaction-only · R10 — harness audit

**Conclusion: PASS, with the warnings below.** No benchmark-rule breach was established. Required accounting and freezing checks are supported by the original records and supplemental evidence.

- **Scope:** `r10_20261121`, all 16 tasks; at most ten starts per task across its training seeds. All 800 tests are included.
- **Auditor/date:** Codex (GPT-6 family), 2026-11-22; follow-up self-review, not an independent certification. Earlier reports are retained in the study archive.
- **Versions:** frozen runtime manifest SHA-256 `29eb56530feaa42c022993970687a3d8b9a46cbf6da437ed63092e84a9699fae`; benchmark source manifest SHA-256 `01b96e2565ba99a1ce8e9613f2db1139d66ebb115e51e93bf0a45da7c4a385b7`. The evidence index records the audit-skill hashes and all policy freeze manifests remain in the study.
- **Earlier harness review:** the prepared harness passed on 2026-11-21 (`<STUDY_ROOT>/audit/harness_audit.md`). This follow-up checks the deployed records; it does not substitute that preflight for evidence of actual behavior.
- **Evidence roots:** `<STUDY_ROOT>` = `r10_20261121`; `<CASE_ROOT>` = this method/setting/R10 directory; `<AUDIT_ROOT>` = `reaudit_20261122`. Native archived `demo&interaction` denotes public Hybrid.
- **Coverage:** every development ledger/native outcome pair, 3,115 history events, 1,635 referenced history objects, every frozen manifest and all 800 saved test policies. Structural checks cover all tasks; semantic review targets prompts, boundaries, corrections and flagged events. No claim is made to have manually read every repeated log byte or inspected optional media.
- **Method:** read-only study inspection. No simulator was started, policy executed or development/evaluation API called. Native logs, weights, policies, ledgers and results were not changed.
- **Agent-visible documents:** the saved public protocol bundle, native method instructions, SDK/interface material and setting-allowed observations/demos. No other Plus task implementation or test results are allowed.

## Checklist

- [x] **H1 — PASS.** The frozen trusted observation translator exposes RGB-D, calibration, robot state, goals and public terminal feedback; no object poses, oracle actions or subgoals. See <STUDY_ROOT>/runtime_source/benchmark_io/episode.py:52 and the method adapter.
- [x] **H2 — PASS.** All 16 saved launches use the verified immutable image, network=none, a read-only root and explicit mounts. The complete seven-layer/package inventory and launch review resolve the former image-inspection gap. No forbidden read was observed in the inspected logs. See <AUDIT_ROOT>/evidence/image_inventory.json and launch_isolation.json.
- [x] **H3 — PASS.** Actual public protocol documents and native method prompts are allowed interface guidance; task measurements arise in agent-authored code after permitted inputs. No operator task-code disclosures were established. See <STUDY_ROOT>/protocol/ and saved initial workspace/history records.
- [x] **H4 — PASS.** All 123 development starts use released training episode/seed records. Test data is host-only and excluded from agent mounts; test execution requires the operator-only frozen capability. See <CASE_ROOT>/study/study.json and <AUDIT_ROOT>/evidence/native_accounting.json.
- [x] **H5 — PASS.** 123 ledger starts reconcile one-to-one with native resets/outcomes and saved code revisions; every task is at or below ten starts, including failures and rehearsals. The shared ledger reserves before construction; no hidden reset or direct simulator path was found. Charged steps: 57,207. See native_accounting.json and reconciliation.json.
- [x] **H6 — PASS.** Released demonstrations are excluded from this setting’s export/context and agent mounts; episode conditioning videos are allowed.
- [x] **H7 — PASS.** All frozen policy/runtime hashes verify, all 800 saved test policy copies match, and each freeze predates its test starts. No test-driven revision/selection was found. The saved Docker image content identity and per-task bundles are also verified.
- [x] **R1 — PASS.** 3,115 hook events and 1,635 referenced task-local history objects verified. Native attempt records and code revisions reconcile. Any hook without a matching post-hook is reconciled to the native CLI rejection result; no development call remains unexplained. See recovered_gaps.json. The original audits provide the broader native-transcript/API review for previously completed cases.
- [x] **R2 — PASS.** Development starts/steps reconcile. Approximate development API cost: $90; frozen evaluation paid API cost: $0. All development API cost records are complete. 4 missing native test step totals are reconstructed as zero policy steps from surviving startup evidence; original rows are unchanged.
- [x] **R3 — PASS.** Human interface, provider-port and scheduling changes remain disclosed with before/after source and patches. No generated-policy operator edits were established for this case. See <STUDY_ROOT>/audit/.

## Cost estimate

Development: **approximately $90**; evaluation: **$0 paid API cost**. Local compute is excluded. The recorded priced subtotal is $87.93.
The estimate uses the archived rate card and saved usage. For interrupted provider requests, an allowance based on the mean complete-request cost is added before rounding to $10; this is an estimate, not reconstructed token usage or an invoice. Frozen execution uses local providers and no paid model calls.

## Warnings and required action

| Warning | Evidence / scope | Action |
| --- | --- | --- |
| Failed test startup records | BinFill/12, ButtonUnmask/12, RouteStick/11, VideoPlaceButton/10: native CUDA OOM killed setup before policy execution. Zero policy control steps are reconstructed from setup timing, empty flushed action/MCP traces and the frozen control flow. | Keep original unfinished ledger rows and the recovery supplement. These episodes remain failures in the 800-test denominator. |
| Infrastructure error summary | Native simulator logs identify 6 CUDA-OOM test startups; the old top-level aggregator reported zero infrastructure errors. | The public metadata now records these startup errors. Keep all failed test outcomes and the original logs; no score changes. See startup_oom.json. |

**Reuse decision:** retain the current frozen policies and all 800 evaluation outcomes. No redevelopment or reevaluation is required by this review. Preserve the supplements, original interruptions/errors and earlier audit reports. Scores are unchanged.

**Evidence index:** `doc/robomme_plus/audits/current/evidence/reaudit_summary.json` records hashes for the full local supplements; the development-log release must include those supplements alongside the original evidence.

</details>

<details>
<summary>Development audit — PASS with warnings</summary>

# RAX · Interaction-only · R10 — development audit

**Conclusion: PASS, with the warnings below.** No benchmark-rule breach was established. Required accounting and freezing checks are supported by the original records and supplemental evidence.

- **Scope:** `r10_20261121`, all 16 tasks; at most ten starts per task across its training seeds. All 800 tests are included.
- **Auditor/date:** Codex (GPT-6 family), 2026-11-22; follow-up self-review, not an independent certification. Earlier reports are retained in the study archive.
- **Versions:** frozen runtime manifest SHA-256 `29eb56530feaa42c022993970687a3d8b9a46cbf6da437ed63092e84a9699fae`; benchmark source manifest SHA-256 `01b96e2565ba99a1ce8e9613f2db1139d66ebb115e51e93bf0a45da7c4a385b7`. The evidence index records the audit-skill hashes and all policy freeze manifests remain in the study.
- **Earlier harness review:** the prepared harness passed on 2026-11-21 (`<STUDY_ROOT>/audit/harness_audit.md`). This follow-up checks the deployed records; it does not substitute that preflight for evidence of actual behavior.
- **Evidence roots:** `<STUDY_ROOT>` = `r10_20261121`; `<CASE_ROOT>` = this method/setting/R10 directory; `<AUDIT_ROOT>` = `reaudit_20261122`. Native archived `demo&interaction` denotes public Hybrid.
- **Coverage:** every development ledger/native outcome pair, 3,115 history events, 1,635 referenced history objects, every frozen manifest and all 800 saved test policies. Structural checks cover all tasks; semantic review targets prompts, boundaries, corrections and flagged events. No claim is made to have manually read every repeated log byte or inspected optional media.
- **Method:** read-only study inspection. No simulator was started, policy executed or development/evaluation API called. Native logs, weights, policies, ledgers and results were not changed.
- **Agent-visible documents:** the saved public protocol bundle, native method instructions, SDK/interface material and setting-allowed observations/demos. No other Plus task implementation or test results are allowed.

## Checklist

- [x] **H1 — PASS.** The frozen trusted observation translator exposes RGB-D, calibration, robot state, goals and public terminal feedback; no object poses, oracle actions or subgoals. See <STUDY_ROOT>/runtime_source/benchmark_io/episode.py:52 and the method adapter.
- [x] **H2 — PASS.** All 16 saved launches use the verified immutable image, network=none, a read-only root and explicit mounts. The complete seven-layer/package inventory and launch review resolve the former image-inspection gap. No forbidden read was observed in the inspected logs. See <AUDIT_ROOT>/evidence/image_inventory.json and launch_isolation.json.
- [x] **H3 — PASS.** Actual public protocol documents and native method prompts are allowed interface guidance; task measurements arise in agent-authored code after permitted inputs. No operator task-code disclosures were established. See <STUDY_ROOT>/protocol/ and saved initial workspace/history records.
- [x] **H4 — PASS.** All 123 development starts use released training episode/seed records. Test data is host-only and excluded from agent mounts; test execution requires the operator-only frozen capability. See <CASE_ROOT>/study/study.json and <AUDIT_ROOT>/evidence/native_accounting.json.
- [x] **H5 — PASS.** 123 ledger starts reconcile one-to-one with native resets/outcomes and saved code revisions; every task is at or below ten starts, including failures and rehearsals. The shared ledger reserves before construction; no hidden reset or direct simulator path was found. Charged steps: 57,207. See native_accounting.json and reconciliation.json.
- [x] **H6 — PASS.** Released demonstrations are excluded from this setting’s export/context and agent mounts; episode conditioning videos are allowed.
- [x] **H7 — PASS.** All frozen policy/runtime hashes verify, all 800 saved test policy copies match, and each freeze predates its test starts. No test-driven revision/selection was found. The saved Docker image content identity and per-task bundles are also verified.
- [x] **R1 — PASS.** 3,115 hook events and 1,635 referenced task-local history objects verified. Native attempt records and code revisions reconcile. Any hook without a matching post-hook is reconciled to the native CLI rejection result; no development call remains unexplained. See recovered_gaps.json. The original audits provide the broader native-transcript/API review for previously completed cases.
- [x] **R2 — PASS.** Development starts/steps reconcile. Approximate development API cost: $90; frozen evaluation paid API cost: $0. All development API cost records are complete. 4 missing native test step totals are reconstructed as zero policy steps from surviving startup evidence; original rows are unchanged.
- [x] **R3 — PASS.** Human interface, provider-port and scheduling changes remain disclosed with before/after source and patches. No generated-policy operator edits were established for this case. See <STUDY_ROOT>/audit/.

## Per-task accounting

Failed starts remain charged. All seed checks pass and all development starts have native outcomes. Steps are the benchmark’s charged control-step totals, including error penalties.

| Task | Budget | Starts | Failed / unclosed | Charged steps | Test policy matches |
| --- | ---: | ---: | ---: | ---: | ---: |
| BinFill | 10 | 8 | 2 / 0 | 4,962 | 50/50 |
| ButtonUnmask | 10 | 6 | 2 / 0 | 1,929 | 50/50 |
| ButtonUnmaskSwap | 10 | 7 | 3 / 0 | 3,194 | 50/50 |
| InsertPeg | 10 | 5 | 3 / 0 | 3,132 | 50/50 |
| MoveCube | 10 | 5 | 5 / 0 | 1,345 | 50/50 |
| PatternLock | 10 | 8 | 6 / 0 | 2,733 | 50/50 |
| PickHighlight | 10 | 10 | 2 / 0 | 11,487 | 50/50 |
| PickXtimes | 10 | 9 | 4 / 0 | 6,490 | 50/50 |
| RouteStick | 10 | 8 | 6 / 0 | 3,655 | 50/50 |
| StopCube | 10 | 9 | 2 / 0 | 1,564 | 50/50 |
| SwingXtimes | 10 | 9 | 3 / 0 | 4,180 | 50/50 |
| VideoPlaceButton | 10 | 10 | 8 / 0 | 1,982 | 50/50 |
| VideoPlaceOrder | 10 | 9 | 6 / 0 | 992 | 50/50 |
| VideoRepick | 10 | 6 | 1 / 0 | 1,412 | 50/50 |
| VideoUnmask | 10 | 7 | 0 / 0 | 3,865 | 50/50 |
| VideoUnmaskSwap | 10 | 7 | 1 / 0 | 4,285 | 50/50 |
| **Total** | **160 maximum** | **123** | **54 / 0** | **57,207** | **800/800** |

## Cost estimate

Development: **approximately $90**; evaluation: **$0 paid API cost**. Local compute is excluded. The recorded priced subtotal is $87.93.
The estimate uses the archived rate card and saved usage. For interrupted provider requests, an allowance based on the mean complete-request cost is added before rounding to $10; this is an estimate, not reconstructed token usage or an invoice. Frozen execution uses local providers and no paid model calls.

## Warnings and required action

| Warning | Evidence / scope | Action |
| --- | --- | --- |
| Failed test startup records | BinFill/12, ButtonUnmask/12, RouteStick/11, VideoPlaceButton/10: native CUDA OOM killed setup before policy execution. Zero policy control steps are reconstructed from setup timing, empty flushed action/MCP traces and the frozen control flow. | Keep original unfinished ledger rows and the recovery supplement. These episodes remain failures in the 800-test denominator. |
| Infrastructure error summary | Native simulator logs identify 6 CUDA-OOM test startups; the old top-level aggregator reported zero infrastructure errors. | The public metadata now records these startup errors. Keep all failed test outcomes and the original logs; no score changes. See startup_oom.json. |

**Reuse decision:** retain the current frozen policies and all 800 evaluation outcomes. No redevelopment or reevaluation is required by this review. Preserve the supplements, original interruptions/errors and earlier audit reports. Scores are unchanged.

**Evidence index:** `doc/robomme_plus/audits/current/evidence/reaudit_summary.json` records hashes for the full local supplements; the development-log release must include those supplements alongside the original evidence.

</details>

### `Hybrid` · R10

<details>
<summary>Harness audit — PASS with warnings</summary>

# RAX · Hybrid · R10 — harness audit

**Conclusion: PASS, with the warnings below.** No benchmark-rule breach was established. Required accounting and freezing checks are supported by the original records and supplemental evidence.

- **Scope:** `r10_20261121`, all 16 tasks; at most ten starts per task across its training seeds. All 800 tests are included.
- **Auditor/date:** Codex (GPT-6 family), 2026-11-22; follow-up self-review, not an independent certification. Earlier reports are retained in the study archive.
- **Versions:** frozen runtime manifest SHA-256 `29eb56530feaa42c022993970687a3d8b9a46cbf6da437ed63092e84a9699fae`; benchmark source manifest SHA-256 `01b96e2565ba99a1ce8e9613f2db1139d66ebb115e51e93bf0a45da7c4a385b7`. The evidence index records the audit-skill hashes and all policy freeze manifests remain in the study.
- **Earlier harness review:** the prepared harness passed on 2026-11-21 (`<STUDY_ROOT>/audit/harness_audit.md`). This follow-up checks the deployed records; it does not substitute that preflight for evidence of actual behavior.
- **Evidence roots:** `<STUDY_ROOT>` = `r10_20261121`; `<CASE_ROOT>` = this method/setting/R10 directory; `<AUDIT_ROOT>` = `reaudit_20261122`. Native archived `demo&interaction` denotes public Hybrid.
- **Coverage:** every development ledger/native outcome pair, 3,799 history events, 1,662 referenced history objects, every frozen manifest and all 800 saved test policies. Structural checks cover all tasks; semantic review targets prompts, boundaries, corrections and flagged events. No claim is made to have manually read every repeated log byte or inspected optional media.
- **Method:** read-only study inspection. No simulator was started, policy executed or development/evaluation API called. Native logs, weights, policies, ledgers and results were not changed.
- **Agent-visible documents:** the saved public protocol bundle, native method instructions, SDK/interface material and setting-allowed observations/demos. No other Plus task implementation or test results are allowed.

## Checklist

- [x] **H1 — PASS.** The frozen trusted observation translator exposes RGB-D, calibration, robot state, goals and public terminal feedback; no object poses, oracle actions or subgoals. See <STUDY_ROOT>/runtime_source/benchmark_io/episode.py:52 and the method adapter.
- [x] **H2 — PASS.** All 16 saved launches use the verified immutable image, network=none, a read-only root and explicit mounts. The complete seven-layer/package inventory and launch review resolve the former image-inspection gap. No forbidden read was observed in the inspected logs. See <AUDIT_ROOT>/evidence/image_inventory.json and launch_isolation.json.
- [x] **H3 — PASS.** Actual public protocol documents and native method prompts are allowed interface guidance; task measurements arise in agent-authored code after permitted inputs. No operator task-code disclosures were established. See <STUDY_ROOT>/protocol/ and saved initial workspace/history records.
- [x] **H4 — PASS.** All 121 development starts use released training episode/seed records. Test data is host-only and excluded from agent mounts; test execution requires the operator-only frozen capability. See <CASE_ROOT>/study/study.json and <AUDIT_ROOT>/evidence/native_accounting.json.
- [x] **H5 — PASS.** 121 ledger starts reconcile one-to-one with native resets/outcomes and saved code revisions; every task is at or below ten starts, including failures and rehearsals. The shared ledger reserves before construction; no hidden reset or direct simulator path was found. Charged steps: 70,547. See native_accounting.json and reconciliation.json.
- [x] **H6 — PASS.** Released offline demonstrations are provided in this setting. Simulator verification and rehearsal still use the shared per-task ledger.
- [x] **H7 — PASS.** All frozen policy/runtime hashes verify, all 800 saved test policy copies match, and each freeze predates its test starts. No test-driven revision/selection was found. The saved Docker image content identity and per-task bundles are also verified.
- [x] **R1 — PASS.** 3,799 hook events and 1,662 referenced task-local history objects verified. Native attempt records and code revisions reconcile. Any hook without a matching post-hook is reconciled to the native CLI rejection result; no development call remains unexplained. See recovered_gaps.json. The original audits provide the broader native-transcript/API review for previously completed cases.
- [x] **R2 — PASS.** Development starts/steps reconcile. Approximate development API cost: $100; frozen evaluation paid API cost: $0. All development API cost records are complete. 5 missing native test step totals are reconstructed as zero policy steps from surviving startup evidence; original rows are unchanged.
- [x] **R3 — PASS.** Human interface, provider-port and scheduling changes remain disclosed with before/after source and patches. No generated-policy operator edits were established for this case. See <STUDY_ROOT>/audit/.

## Cost estimate

Development: **approximately $100**; evaluation: **$0 paid API cost**. Local compute is excluded. The recorded priced subtotal is $103.73.
The estimate uses the archived rate card and saved usage. For interrupted provider requests, an allowance based on the mean complete-request cost is added before rounding to $10; this is an estimate, not reconstructed token usage or an invoice. Frozen execution uses local providers and no paid model calls.

## Warnings and required action

| Warning | Evidence / scope | Action |
| --- | --- | --- |
| Interrupted/rejected tools | Two commands were rejected by the native CLI safety check before execution; their terminal rejection records survive in agent_stdout.jsonl. | Retain inputs, code revisions and interruption/rejection evidence. Do not invent outputs or charge rejected commands as rollouts. |
| Failed test startup records | BinFill/11, ButtonUnmaskSwap/11, PickXtimes/10, PickXtimes/11, VideoRepick/10: native CUDA OOM killed setup before policy execution. Zero policy control steps are reconstructed from setup timing, empty flushed action/MCP traces and the frozen control flow. | Keep original unfinished ledger rows and the recovery supplement. These episodes remain failures in the 800-test denominator. |
| Charged-step totals | 6 development attempts crossed their step ceiling through the benchmark’s 100-step action-error penalty, without extra motion or further policy actions. | Report charged steps as recorded; these are not additional rollouts. |
| Infrastructure error summary | Native simulator logs identify 6 CUDA-OOM test startups; the old top-level aggregator reported zero infrastructure errors. | The public metadata now records these startup errors. Keep all failed test outcomes and the original logs; no score changes. See startup_oom.json. |

**Reuse decision:** retain the current frozen policies and all 800 evaluation outcomes. No redevelopment or reevaluation is required by this review. Preserve the supplements, original interruptions/errors and earlier audit reports. Scores are unchanged.

**Evidence index:** `doc/robomme_plus/audits/current/evidence/reaudit_summary.json` records hashes for the full local supplements; the development-log release must include those supplements alongside the original evidence.

</details>

<details>
<summary>Development audit — PASS with warnings</summary>

# RAX · Hybrid · R10 — development audit

**Conclusion: PASS, with the warnings below.** No benchmark-rule breach was established. Required accounting and freezing checks are supported by the original records and supplemental evidence.

- **Scope:** `r10_20261121`, all 16 tasks; at most ten starts per task across its training seeds. All 800 tests are included.
- **Auditor/date:** Codex (GPT-6 family), 2026-11-22; follow-up self-review, not an independent certification. Earlier reports are retained in the study archive.
- **Versions:** frozen runtime manifest SHA-256 `29eb56530feaa42c022993970687a3d8b9a46cbf6da437ed63092e84a9699fae`; benchmark source manifest SHA-256 `01b96e2565ba99a1ce8e9613f2db1139d66ebb115e51e93bf0a45da7c4a385b7`. The evidence index records the audit-skill hashes and all policy freeze manifests remain in the study.
- **Earlier harness review:** the prepared harness passed on 2026-11-21 (`<STUDY_ROOT>/audit/harness_audit.md`). This follow-up checks the deployed records; it does not substitute that preflight for evidence of actual behavior.
- **Evidence roots:** `<STUDY_ROOT>` = `r10_20261121`; `<CASE_ROOT>` = this method/setting/R10 directory; `<AUDIT_ROOT>` = `reaudit_20261122`. Native archived `demo&interaction` denotes public Hybrid.
- **Coverage:** every development ledger/native outcome pair, 3,799 history events, 1,662 referenced history objects, every frozen manifest and all 800 saved test policies. Structural checks cover all tasks; semantic review targets prompts, boundaries, corrections and flagged events. No claim is made to have manually read every repeated log byte or inspected optional media.
- **Method:** read-only study inspection. No simulator was started, policy executed or development/evaluation API called. Native logs, weights, policies, ledgers and results were not changed.
- **Agent-visible documents:** the saved public protocol bundle, native method instructions, SDK/interface material and setting-allowed observations/demos. No other Plus task implementation or test results are allowed.

## Checklist

- [x] **H1 — PASS.** The frozen trusted observation translator exposes RGB-D, calibration, robot state, goals and public terminal feedback; no object poses, oracle actions or subgoals. See <STUDY_ROOT>/runtime_source/benchmark_io/episode.py:52 and the method adapter.
- [x] **H2 — PASS.** All 16 saved launches use the verified immutable image, network=none, a read-only root and explicit mounts. The complete seven-layer/package inventory and launch review resolve the former image-inspection gap. No forbidden read was observed in the inspected logs. See <AUDIT_ROOT>/evidence/image_inventory.json and launch_isolation.json.
- [x] **H3 — PASS.** Actual public protocol documents and native method prompts are allowed interface guidance; task measurements arise in agent-authored code after permitted inputs. No operator task-code disclosures were established. See <STUDY_ROOT>/protocol/ and saved initial workspace/history records.
- [x] **H4 — PASS.** All 121 development starts use released training episode/seed records. Test data is host-only and excluded from agent mounts; test execution requires the operator-only frozen capability. See <CASE_ROOT>/study/study.json and <AUDIT_ROOT>/evidence/native_accounting.json.
- [x] **H5 — PASS.** 121 ledger starts reconcile one-to-one with native resets/outcomes and saved code revisions; every task is at or below ten starts, including failures and rehearsals. The shared ledger reserves before construction; no hidden reset or direct simulator path was found. Charged steps: 70,547. See native_accounting.json and reconciliation.json.
- [x] **H6 — PASS.** Released offline demonstrations are provided in this setting. Simulator verification and rehearsal still use the shared per-task ledger.
- [x] **H7 — PASS.** All frozen policy/runtime hashes verify, all 800 saved test policy copies match, and each freeze predates its test starts. No test-driven revision/selection was found. The saved Docker image content identity and per-task bundles are also verified.
- [x] **R1 — PASS.** 3,799 hook events and 1,662 referenced task-local history objects verified. Native attempt records and code revisions reconcile. Any hook without a matching post-hook is reconciled to the native CLI rejection result; no development call remains unexplained. See recovered_gaps.json. The original audits provide the broader native-transcript/API review for previously completed cases.
- [x] **R2 — PASS.** Development starts/steps reconcile. Approximate development API cost: $100; frozen evaluation paid API cost: $0. All development API cost records are complete. 5 missing native test step totals are reconstructed as zero policy steps from surviving startup evidence; original rows are unchanged.
- [x] **R3 — PASS.** Human interface, provider-port and scheduling changes remain disclosed with before/after source and patches. No generated-policy operator edits were established for this case. See <STUDY_ROOT>/audit/.

## Per-task accounting

Failed starts remain charged. All seed checks pass and all development starts have native outcomes. Steps are the benchmark’s charged control-step totals, including error penalties.

| Task | Budget | Starts | Failed / unclosed | Charged steps | Test policy matches |
| --- | ---: | ---: | ---: | ---: | ---: |
| BinFill | 10 | 5 | 1 / 0 | 3,840 | 50/50 |
| ButtonUnmask | 10 | 9 | 2 / 0 | 3,806 | 50/50 |
| ButtonUnmaskSwap | 10 | 10 | 6 / 0 | 4,924 | 50/50 |
| InsertPeg | 10 | 10 | 10 / 0 | 8,107 | 50/50 |
| MoveCube | 10 | 10 | 10 / 0 | 9,044 | 50/50 |
| PatternLock | 10 | 10 | 7 / 0 | 3,381 | 50/50 |
| PickHighlight | 10 | 8 | 6 / 0 | 6,351 | 50/50 |
| PickXtimes | 10 | 8 | 4 / 0 | 12,758 | 50/50 |
| RouteStick | 10 | 10 | 7 / 0 | 6,563 | 50/50 |
| StopCube | 10 | 7 | 1 / 0 | 1,403 | 50/50 |
| SwingXtimes | 10 | 6 | 1 / 0 | 2,914 | 50/50 |
| VideoPlaceButton | 10 | 7 | 1 / 0 | 1,302 | 50/50 |
| VideoPlaceOrder | 10 | 4 | 1 / 0 | 339 | 50/50 |
| VideoRepick | 10 | 5 | 0 / 0 | 1,120 | 50/50 |
| VideoUnmask | 10 | 5 | 0 / 0 | 1,915 | 50/50 |
| VideoUnmaskSwap | 10 | 7 | 1 / 0 | 2,780 | 50/50 |
| **Total** | **160 maximum** | **121** | **58 / 0** | **70,547** | **800/800** |

## Cost estimate

Development: **approximately $100**; evaluation: **$0 paid API cost**. Local compute is excluded. The recorded priced subtotal is $103.73.
The estimate uses the archived rate card and saved usage. For interrupted provider requests, an allowance based on the mean complete-request cost is added before rounding to $10; this is an estimate, not reconstructed token usage or an invoice. Frozen execution uses local providers and no paid model calls.

## Warnings and required action

| Warning | Evidence / scope | Action |
| --- | --- | --- |
| Interrupted/rejected tools | Two commands were rejected by the native CLI safety check before execution; their terminal rejection records survive in agent_stdout.jsonl. | Retain inputs, code revisions and interruption/rejection evidence. Do not invent outputs or charge rejected commands as rollouts. |
| Failed test startup records | BinFill/11, ButtonUnmaskSwap/11, PickXtimes/10, PickXtimes/11, VideoRepick/10: native CUDA OOM killed setup before policy execution. Zero policy control steps are reconstructed from setup timing, empty flushed action/MCP traces and the frozen control flow. | Keep original unfinished ledger rows and the recovery supplement. These episodes remain failures in the 800-test denominator. |
| Charged-step totals | 6 development attempts crossed their step ceiling through the benchmark’s 100-step action-error penalty, without extra motion or further policy actions. | Report charged steps as recorded; these are not additional rollouts. |
| Infrastructure error summary | Native simulator logs identify 6 CUDA-OOM test startups; the old top-level aggregator reported zero infrastructure errors. | The public metadata now records these startup errors. Keep all failed test outcomes and the original logs; no score changes. See startup_oom.json. |

**Reuse decision:** retain the current frozen policies and all 800 evaluation outcomes. No redevelopment or reevaluation is required by this review. Preserve the supplements, original interruptions/errors and earlier audit reports. Scores are unchanged.

**Evidence index:** `doc/robomme_plus/audits/current/evidence/reaudit_summary.json` records hashes for the full local supplements; the development-log release must include those supplements alongside the original evidence.

</details>
