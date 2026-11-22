# ASPIRE on RoboMME-Plus (eval_by_RoboMME_team)

- **Track:** Agentic.
- **Code:** [Adapter and run instructions](../../../scripts/plus_eval/agentic_methods/aspire/README.md); [frozen R10 solutions](../../../scripts/plus_eval/agentic_methods/aspire/solutions/README.md).
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
| `Interaction-only` | 10 | 302/800 (37.75%) | 157 | ≈$160 | $0 |
| `Hybrid` | 10 | 342/800 (42.75%) | 134 | ≈$160 | $0 |

API estimates are rounded to the nearest $10 and exclude local compute. Recorded usage is retained in the logs; missing provider-error usage remains unknown.
Requests with incomplete usage records: Interaction-only: 1, Hybrid: 2. Estimates use recorded usage plus a small allowance for interrupted requests; exact missing usage is unknown.

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

Both settings passed the follow-up harness and development reviews. The reports below disclose the self-review, evidence recovery and warnings. Original audit reports and native records are retained.

### `Interaction-only` · R10

<details>
<summary>Harness audit — PASS with warnings</summary>

# ASPIRE · Interaction-only · R10 — harness audit

**Conclusion: PASS, with the warnings below.** No benchmark-rule breach was established. Required accounting and freezing checks are supported by the original records and supplemental evidence.

- **Scope:** `r10_20261121`, all 16 tasks; at most ten starts per task across its training seeds. All 800 tests are included.
- **Auditor/date:** Codex (GPT-6 family), 2026-11-22; follow-up self-review, not an independent certification. Earlier reports are retained in the study archive.
- **Versions:** frozen runtime manifest SHA-256 `29eb56530feaa42c022993970687a3d8b9a46cbf6da437ed63092e84a9699fae`; benchmark source manifest SHA-256 `01b96e2565ba99a1ce8e9613f2db1139d66ebb115e51e93bf0a45da7c4a385b7`. The evidence index records the audit-skill hashes and all policy freeze manifests remain in the study.
- **Earlier harness review:** the prepared harness passed on 2026-11-21 (`<STUDY_ROOT>/audit/harness_audit.md`). This follow-up checks the deployed records; it does not substitute that preflight for evidence of actual behavior.
- **Evidence roots:** `<STUDY_ROOT>` = `r10_20261121`; `<CASE_ROOT>` = this method/setting/R10 directory; `<AUDIT_ROOT>` = `reaudit_20261122`. Native archived `demo&interaction` denotes public Hybrid.
- **Coverage:** every development ledger/native outcome pair, 4,191 history events, 1,296 referenced history objects, every frozen manifest and all 800 saved test policies. Structural checks cover all tasks; semantic review targets prompts, boundaries, corrections and flagged events. No claim is made to have manually read every repeated log byte or inspected optional media.
- **Method:** read-only study inspection. No simulator was started, policy executed or development/evaluation API called. Native logs, weights, policies, ledgers and results were not changed.
- **Agent-visible documents:** the saved public protocol bundle, native method instructions, SDK/interface material and setting-allowed observations/demos. No other Plus task implementation or test results are allowed.

## Checklist

- [x] **H1 — PASS.** The frozen trusted observation translator exposes RGB-D, calibration, robot state, goals and public terminal feedback; no object poses, oracle actions or subgoals. See <STUDY_ROOT>/runtime_source/benchmark_io/episode.py:52 and the method adapter.
- [x] **H2 — PASS.** All 3 saved launches use the verified immutable image, network=none, a read-only root and explicit mounts. The complete seven-layer/package inventory and launch review resolve the former image-inspection gap. No forbidden read was observed in the inspected logs. See <AUDIT_ROOT>/evidence/image_inventory.json and launch_isolation.json.
- [x] **H3 — PASS.** Actual public protocol documents and native method prompts are allowed interface guidance; task measurements arise in agent-authored code after permitted inputs. No operator task-code disclosures were established. See <STUDY_ROOT>/protocol/ and saved initial workspace/history records.
- [x] **H4 — PASS.** All 157 development starts use released training episode/seed records. Test data is host-only and excluded from agent mounts; test execution requires the operator-only frozen capability. See <CASE_ROOT>/study/study.json and <AUDIT_ROOT>/evidence/native_accounting.json.
- [x] **H5 — PASS.** 157 ledger starts reconcile one-to-one with native resets/outcomes and saved code revisions; every task is at or below ten starts, including failures and rehearsals. The shared ledger reserves before construction; no hidden reset or direct simulator path was found. Charged steps: 55,389. See native_accounting.json and reconciliation.json.
- [x] **H6 — PASS.** Released demonstrations are excluded from this setting’s export/context and agent mounts; episode conditioning videos are allowed.
- [x] **H7 — PASS.** All frozen policy/runtime hashes verify, all 800 saved test policy copies match, and each freeze predates its test starts. No test-driven revision/selection was found. Retained provider process identities, the frozen PYTHONPATH, clean pinned vendor commits, checkpoint hashes and 39,188 runtime-file change-time checks establish continuity of the external inference stack; see provider_continuity.json and its retrospective-inspection limitation.
- [x] **R1 — PASS.** 4,191 hook events and 1,296 referenced task-local history objects verified. Native attempt records and code revisions reconcile. Missing terminal hooks are accounted for by saved deadline events and campaign.py:327–359, which terminates and resumes the shared container; interrupted commands are not represented as completed. See recovered_gaps.json. The original audits provide the broader native-transcript/API review for previously completed cases.
- [ ] **R2 — WARN.** Development starts/steps reconcile. Approximate development API cost: $160; frozen evaluation paid API cost: $0. 1 logged provider-error requests have unknown final token usage; this is a warning under the revised cost rule. Saved test step/outcome records reconcile.
- [x] **R3 — PASS.** Human interface, provider-port and scheduling changes remain disclosed with before/after source and patches. No generated-policy operator edits were established for this case. See <STUDY_ROOT>/audit/.

## Cost estimate

Development: **approximately $160**; evaluation: **$0 paid API cost**. Local compute is excluded. The recorded priced subtotal is $158.55.
The estimate uses the archived rate card and saved usage. For interrupted provider requests, an allowance based on the mean complete-request cost is added before rounding to $10; this is an estimate, not reconstructed token usage or an invoice. Frozen execution uses local providers and no paid model calls.

## Warnings and required action

| Warning | Evidence / scope | Action |
| --- | --- | --- |
| Provider-error API usage | 1 recorded failed requests have missing/partial usage: 1810_bffa0bb5. | Report approximately $160; retain unknown token counts and the estimation basis. No rerun is needed. |
| Provider continuity evidence | External provider identities were reconstructed from retained processes, immutable dependency revisions and pre-start filesystem change times; their hashes were not added to the original pre-test manifest. | Keep the supplemental evidence with the original freeze manifest. For future studies, capture the provider inventory at freeze time. |
| Interrupted/rejected tools | Commands interrupted by the logged campaign-container deadline restart; no completed tool response is claimed. | Retain inputs, code revisions and interruption/rejection evidence. Do not invent outputs or charge rejected commands as rollouts. |

**Reuse decision:** retain the current frozen policies and all 800 evaluation outcomes. No redevelopment or reevaluation is required by this review. Preserve the supplements, original interruptions/errors and earlier audit reports. Scores are unchanged.

**Evidence index:** `doc/robomme_plus/audits/current/evidence/reaudit_summary.json` records hashes for the full local supplements; the development-log release must include those supplements alongside the original evidence.

</details>

<details>
<summary>Development audit — PASS with warnings</summary>

# ASPIRE · Interaction-only · R10 — development audit

**Conclusion: PASS, with the warnings below.** No benchmark-rule breach was established. Required accounting and freezing checks are supported by the original records and supplemental evidence.

- **Scope:** `r10_20261121`, all 16 tasks; at most ten starts per task across its training seeds. All 800 tests are included.
- **Auditor/date:** Codex (GPT-6 family), 2026-11-22; follow-up self-review, not an independent certification. Earlier reports are retained in the study archive.
- **Versions:** frozen runtime manifest SHA-256 `29eb56530feaa42c022993970687a3d8b9a46cbf6da437ed63092e84a9699fae`; benchmark source manifest SHA-256 `01b96e2565ba99a1ce8e9613f2db1139d66ebb115e51e93bf0a45da7c4a385b7`. The evidence index records the audit-skill hashes and all policy freeze manifests remain in the study.
- **Earlier harness review:** the prepared harness passed on 2026-11-21 (`<STUDY_ROOT>/audit/harness_audit.md`). This follow-up checks the deployed records; it does not substitute that preflight for evidence of actual behavior.
- **Evidence roots:** `<STUDY_ROOT>` = `r10_20261121`; `<CASE_ROOT>` = this method/setting/R10 directory; `<AUDIT_ROOT>` = `reaudit_20261122`. Native archived `demo&interaction` denotes public Hybrid.
- **Coverage:** every development ledger/native outcome pair, 4,191 history events, 1,296 referenced history objects, every frozen manifest and all 800 saved test policies. Structural checks cover all tasks; semantic review targets prompts, boundaries, corrections and flagged events. No claim is made to have manually read every repeated log byte or inspected optional media.
- **Method:** read-only study inspection. No simulator was started, policy executed or development/evaluation API called. Native logs, weights, policies, ledgers and results were not changed.
- **Agent-visible documents:** the saved public protocol bundle, native method instructions, SDK/interface material and setting-allowed observations/demos. No other Plus task implementation or test results are allowed.

## Checklist

- [x] **H1 — PASS.** The frozen trusted observation translator exposes RGB-D, calibration, robot state, goals and public terminal feedback; no object poses, oracle actions or subgoals. See <STUDY_ROOT>/runtime_source/benchmark_io/episode.py:52 and the method adapter.
- [x] **H2 — PASS.** All 3 saved launches use the verified immutable image, network=none, a read-only root and explicit mounts. The complete seven-layer/package inventory and launch review resolve the former image-inspection gap. No forbidden read was observed in the inspected logs. See <AUDIT_ROOT>/evidence/image_inventory.json and launch_isolation.json.
- [x] **H3 — PASS.** Actual public protocol documents and native method prompts are allowed interface guidance; task measurements arise in agent-authored code after permitted inputs. No operator task-code disclosures were established. See <STUDY_ROOT>/protocol/ and saved initial workspace/history records.
- [x] **H4 — PASS.** All 157 development starts use released training episode/seed records. Test data is host-only and excluded from agent mounts; test execution requires the operator-only frozen capability. See <CASE_ROOT>/study/study.json and <AUDIT_ROOT>/evidence/native_accounting.json.
- [x] **H5 — PASS.** 157 ledger starts reconcile one-to-one with native resets/outcomes and saved code revisions; every task is at or below ten starts, including failures and rehearsals. The shared ledger reserves before construction; no hidden reset or direct simulator path was found. Charged steps: 55,389. See native_accounting.json and reconciliation.json.
- [x] **H6 — PASS.** Released demonstrations are excluded from this setting’s export/context and agent mounts; episode conditioning videos are allowed.
- [x] **H7 — PASS.** All frozen policy/runtime hashes verify, all 800 saved test policy copies match, and each freeze predates its test starts. No test-driven revision/selection was found. Retained provider process identities, the frozen PYTHONPATH, clean pinned vendor commits, checkpoint hashes and 39,188 runtime-file change-time checks establish continuity of the external inference stack; see provider_continuity.json and its retrospective-inspection limitation.
- [x] **R1 — PASS.** 4,191 hook events and 1,296 referenced task-local history objects verified. Native attempt records and code revisions reconcile. Missing terminal hooks are accounted for by saved deadline events and campaign.py:327–359, which terminates and resumes the shared container; interrupted commands are not represented as completed. See recovered_gaps.json. The original audits provide the broader native-transcript/API review for previously completed cases.
- [ ] **R2 — WARN.** Development starts/steps reconcile. Approximate development API cost: $160; frozen evaluation paid API cost: $0. 1 logged provider-error requests have unknown final token usage; this is a warning under the revised cost rule. Saved test step/outcome records reconcile.
- [x] **R3 — PASS.** Human interface, provider-port and scheduling changes remain disclosed with before/after source and patches. No generated-policy operator edits were established for this case. See <STUDY_ROOT>/audit/.

## Per-task accounting

Failed starts remain charged. All seed checks pass and all development starts have native outcomes. Steps are the benchmark’s charged control-step totals, including error penalties.

| Task | Budget | Starts | Failed / unclosed | Charged steps | Test policy matches |
| --- | ---: | ---: | ---: | ---: | ---: |
| BinFill | 10 | 10 | 6 / 0 | 4,313 | 50/50 |
| ButtonUnmask | 10 | 10 | 5 / 0 | 5,770 | 50/50 |
| ButtonUnmaskSwap | 10 | 10 | 8 / 0 | 4,953 | 50/50 |
| InsertPeg | 10 | 10 | 8 / 0 | 5,243 | 50/50 |
| MoveCube | 10 | 10 | 9 / 0 | 1,475 | 50/50 |
| PatternLock | 10 | 10 | 9 / 0 | 3,015 | 50/50 |
| PickHighlight | 10 | 10 | 5 / 0 | 7,447 | 50/50 |
| PickXtimes | 10 | 10 | 5 / 0 | 7,225 | 50/50 |
| RouteStick | 10 | 10 | 9 / 0 | 2,832 | 50/50 |
| StopCube | 10 | 10 | 2 / 0 | 1,865 | 50/50 |
| SwingXtimes | 10 | 10 | 6 / 0 | 4,188 | 50/50 |
| VideoPlaceButton | 10 | 10 | 4 / 0 | 776 | 50/50 |
| VideoPlaceOrder | 10 | 10 | 9 / 0 | 772 | 50/50 |
| VideoRepick | 10 | 7 | 3 / 0 | 791 | 50/50 |
| VideoUnmask | 10 | 10 | 3 / 0 | 2,733 | 50/50 |
| VideoUnmaskSwap | 10 | 10 | 7 / 0 | 1,991 | 50/50 |
| **Total** | **160 maximum** | **157** | **98 / 0** | **55,389** | **800/800** |

## Cost estimate

Development: **approximately $160**; evaluation: **$0 paid API cost**. Local compute is excluded. The recorded priced subtotal is $158.55.
The estimate uses the archived rate card and saved usage. For interrupted provider requests, an allowance based on the mean complete-request cost is added before rounding to $10; this is an estimate, not reconstructed token usage or an invoice. Frozen execution uses local providers and no paid model calls.

## Warnings and required action

| Warning | Evidence / scope | Action |
| --- | --- | --- |
| Provider-error API usage | 1 recorded failed requests have missing/partial usage: 1810_bffa0bb5. | Report approximately $160; retain unknown token counts and the estimation basis. No rerun is needed. |
| Provider continuity evidence | External provider identities were reconstructed from retained processes, immutable dependency revisions and pre-start filesystem change times; their hashes were not added to the original pre-test manifest. | Keep the supplemental evidence with the original freeze manifest. For future studies, capture the provider inventory at freeze time. |
| Interrupted/rejected tools | Commands interrupted by the logged campaign-container deadline restart; no completed tool response is claimed. | Retain inputs, code revisions and interruption/rejection evidence. Do not invent outputs or charge rejected commands as rollouts. |

**Reuse decision:** retain the current frozen policies and all 800 evaluation outcomes. No redevelopment or reevaluation is required by this review. Preserve the supplements, original interruptions/errors and earlier audit reports. Scores are unchanged.

**Evidence index:** `doc/robomme_plus/audits/current/evidence/reaudit_summary.json` records hashes for the full local supplements; the development-log release must include those supplements alongside the original evidence.

</details>

### `Hybrid` · R10

<details>
<summary>Harness audit — PASS with warnings</summary>

# ASPIRE · Hybrid · R10 — harness audit

**Conclusion: PASS, with the warnings below.** No benchmark-rule breach was established. Required accounting and freezing checks are supported by the original records and supplemental evidence.

- **Scope:** `r10_20261121`, all 16 tasks; at most ten starts per task across its training seeds. All 800 tests are included.
- **Auditor/date:** Codex (GPT-6 family), 2026-11-22; follow-up self-review, not an independent certification. Earlier reports are retained in the study archive.
- **Versions:** frozen runtime manifest SHA-256 `29eb56530feaa42c022993970687a3d8b9a46cbf6da437ed63092e84a9699fae`; benchmark source manifest SHA-256 `01b96e2565ba99a1ce8e9613f2db1139d66ebb115e51e93bf0a45da7c4a385b7`. The evidence index records the audit-skill hashes and all policy freeze manifests remain in the study.
- **Earlier harness review:** the prepared harness passed on 2026-11-21 (`<STUDY_ROOT>/audit/harness_audit.md`). This follow-up checks the deployed records; it does not substitute that preflight for evidence of actual behavior.
- **Evidence roots:** `<STUDY_ROOT>` = `r10_20261121`; `<CASE_ROOT>` = this method/setting/R10 directory; `<AUDIT_ROOT>` = `reaudit_20261122`. Native archived `demo&interaction` denotes public Hybrid.
- **Coverage:** every development ledger/native outcome pair, 4,446 history events, 1,440 referenced history objects, every frozen manifest and all 800 saved test policies. Structural checks cover all tasks; semantic review targets prompts, boundaries, corrections and flagged events. No claim is made to have manually read every repeated log byte or inspected optional media.
- **Method:** read-only study inspection. No simulator was started, policy executed or development/evaluation API called. Native logs, weights, policies, ledgers and results were not changed.
- **Agent-visible documents:** the saved public protocol bundle, native method instructions, SDK/interface material and setting-allowed observations/demos. No other Plus task implementation or test results are allowed.

## Checklist

- [x] **H1 — PASS.** The frozen trusted observation translator exposes RGB-D, calibration, robot state, goals and public terminal feedback; no object poses, oracle actions or subgoals. See <STUDY_ROOT>/runtime_source/benchmark_io/episode.py:52 and the method adapter.
- [x] **H2 — PASS.** All 2 saved launches use the verified immutable image, network=none, a read-only root and explicit mounts. The complete seven-layer/package inventory and launch review resolve the former image-inspection gap. No forbidden read was observed in the inspected logs. See <AUDIT_ROOT>/evidence/image_inventory.json and launch_isolation.json.
- [x] **H3 — PASS.** Actual public protocol documents and native method prompts are allowed interface guidance; task measurements arise in agent-authored code after permitted inputs. No operator task-code disclosures were established. See <STUDY_ROOT>/protocol/ and saved initial workspace/history records.
- [x] **H4 — PASS.** All 134 development starts use released training episode/seed records. Test data is host-only and excluded from agent mounts; test execution requires the operator-only frozen capability. See <CASE_ROOT>/study/study.json and <AUDIT_ROOT>/evidence/native_accounting.json.
- [x] **H5 — PASS.** 134 ledger starts reconcile one-to-one with native resets/outcomes and saved code revisions; every task is at or below ten starts, including failures and rehearsals. The shared ledger reserves before construction; no hidden reset or direct simulator path was found. Charged steps: 61,702. See native_accounting.json and reconciliation.json.
- [x] **H6 — PASS.** Released offline demonstrations are provided in this setting. Simulator verification and rehearsal still use the shared per-task ledger.
- [x] **H7 — PASS.** All frozen policy/runtime hashes verify, all 800 saved test policy copies match, and each freeze predates its test starts. No test-driven revision/selection was found. Retained provider process identities, the frozen PYTHONPATH, clean pinned vendor commits, checkpoint hashes and 39,188 runtime-file change-time checks establish continuity of the external inference stack; see provider_continuity.json and its retrospective-inspection limitation.
- [x] **R1 — PASS.** 4,446 hook events and 1,440 referenced task-local history objects verified. Native attempt records and code revisions reconcile. Missing terminal hooks are accounted for by saved deadline events and campaign.py:327–359, which terminates and resumes the shared container; interrupted commands are not represented as completed. See recovered_gaps.json. The original audits provide the broader native-transcript/API review for previously completed cases.
- [ ] **R2 — WARN.** Development starts/steps reconcile. Approximate development API cost: $160; frozen evaluation paid API cost: $0. 2 logged provider-error requests have unknown final token usage; this is a warning under the revised cost rule. Saved test step/outcome records reconcile.
- [x] **R3 — PASS.** Human interface, provider-port and scheduling changes remain disclosed with before/after source and patches. The Hybrid MoveCube submission-only continuation used the existing conversation with no additional rollout; retain its handoff evidence. See <STUDY_ROOT>/audit/.

## Cost estimate

Development: **approximately $160**; evaluation: **$0 paid API cost**. Local compute is excluded. The recorded priced subtotal is $163.89.
The estimate uses the archived rate card and saved usage. For interrupted provider requests, an allowance based on the mean complete-request cost is added before rounding to $10; this is an estimate, not reconstructed token usage or an invoice. Frozen execution uses local providers and no paid model calls.

## Warnings and required action

| Warning | Evidence / scope | Action |
| --- | --- | --- |
| Provider-error API usage | 2 recorded failed requests have missing/partial usage: 0813_c0e69565, 1543_1736d9b4. | Report approximately $160; retain unknown token counts and the estimation basis. No rerun is needed. |
| Provider continuity evidence | External provider identities were reconstructed from retained processes, immutable dependency revisions and pre-start filesystem change times; their hashes were not added to the original pre-test manifest. | Keep the supplemental evidence with the original freeze manifest. For future studies, capture the provider inventory at freeze time. |
| Interrupted/rejected tools | Commands interrupted by the logged campaign-container deadline restart; no completed tool response is claimed. | Retain inputs, code revisions and interruption/rejection evidence. Do not invent outputs or charge rejected commands as rollouts. |

**Reuse decision:** retain the current frozen policies and all 800 evaluation outcomes. No redevelopment or reevaluation is required by this review. Preserve the supplements, original interruptions/errors and earlier audit reports. Scores are unchanged.

**Evidence index:** `doc/robomme_plus/audits/current/evidence/reaudit_summary.json` records hashes for the full local supplements; the development-log release must include those supplements alongside the original evidence.

</details>

<details>
<summary>Development audit — PASS with warnings</summary>

# ASPIRE · Hybrid · R10 — development audit

**Conclusion: PASS, with the warnings below.** No benchmark-rule breach was established. Required accounting and freezing checks are supported by the original records and supplemental evidence.

- **Scope:** `r10_20261121`, all 16 tasks; at most ten starts per task across its training seeds. All 800 tests are included.
- **Auditor/date:** Codex (GPT-6 family), 2026-11-22; follow-up self-review, not an independent certification. Earlier reports are retained in the study archive.
- **Versions:** frozen runtime manifest SHA-256 `29eb56530feaa42c022993970687a3d8b9a46cbf6da437ed63092e84a9699fae`; benchmark source manifest SHA-256 `01b96e2565ba99a1ce8e9613f2db1139d66ebb115e51e93bf0a45da7c4a385b7`. The evidence index records the audit-skill hashes and all policy freeze manifests remain in the study.
- **Earlier harness review:** the prepared harness passed on 2026-11-21 (`<STUDY_ROOT>/audit/harness_audit.md`). This follow-up checks the deployed records; it does not substitute that preflight for evidence of actual behavior.
- **Evidence roots:** `<STUDY_ROOT>` = `r10_20261121`; `<CASE_ROOT>` = this method/setting/R10 directory; `<AUDIT_ROOT>` = `reaudit_20261122`. Native archived `demo&interaction` denotes public Hybrid.
- **Coverage:** every development ledger/native outcome pair, 4,446 history events, 1,440 referenced history objects, every frozen manifest and all 800 saved test policies. Structural checks cover all tasks; semantic review targets prompts, boundaries, corrections and flagged events. No claim is made to have manually read every repeated log byte or inspected optional media.
- **Method:** read-only study inspection. No simulator was started, policy executed or development/evaluation API called. Native logs, weights, policies, ledgers and results were not changed.
- **Agent-visible documents:** the saved public protocol bundle, native method instructions, SDK/interface material and setting-allowed observations/demos. No other Plus task implementation or test results are allowed.

## Checklist

- [x] **H1 — PASS.** The frozen trusted observation translator exposes RGB-D, calibration, robot state, goals and public terminal feedback; no object poses, oracle actions or subgoals. See <STUDY_ROOT>/runtime_source/benchmark_io/episode.py:52 and the method adapter.
- [x] **H2 — PASS.** All 2 saved launches use the verified immutable image, network=none, a read-only root and explicit mounts. The complete seven-layer/package inventory and launch review resolve the former image-inspection gap. No forbidden read was observed in the inspected logs. See <AUDIT_ROOT>/evidence/image_inventory.json and launch_isolation.json.
- [x] **H3 — PASS.** Actual public protocol documents and native method prompts are allowed interface guidance; task measurements arise in agent-authored code after permitted inputs. No operator task-code disclosures were established. See <STUDY_ROOT>/protocol/ and saved initial workspace/history records.
- [x] **H4 — PASS.** All 134 development starts use released training episode/seed records. Test data is host-only and excluded from agent mounts; test execution requires the operator-only frozen capability. See <CASE_ROOT>/study/study.json and <AUDIT_ROOT>/evidence/native_accounting.json.
- [x] **H5 — PASS.** 134 ledger starts reconcile one-to-one with native resets/outcomes and saved code revisions; every task is at or below ten starts, including failures and rehearsals. The shared ledger reserves before construction; no hidden reset or direct simulator path was found. Charged steps: 61,702. See native_accounting.json and reconciliation.json.
- [x] **H6 — PASS.** Released offline demonstrations are provided in this setting. Simulator verification and rehearsal still use the shared per-task ledger.
- [x] **H7 — PASS.** All frozen policy/runtime hashes verify, all 800 saved test policy copies match, and each freeze predates its test starts. No test-driven revision/selection was found. Retained provider process identities, the frozen PYTHONPATH, clean pinned vendor commits, checkpoint hashes and 39,188 runtime-file change-time checks establish continuity of the external inference stack; see provider_continuity.json and its retrospective-inspection limitation.
- [x] **R1 — PASS.** 4,446 hook events and 1,440 referenced task-local history objects verified. Native attempt records and code revisions reconcile. Missing terminal hooks are accounted for by saved deadline events and campaign.py:327–359, which terminates and resumes the shared container; interrupted commands are not represented as completed. See recovered_gaps.json. The original audits provide the broader native-transcript/API review for previously completed cases.
- [ ] **R2 — WARN.** Development starts/steps reconcile. Approximate development API cost: $160; frozen evaluation paid API cost: $0. 2 logged provider-error requests have unknown final token usage; this is a warning under the revised cost rule. Saved test step/outcome records reconcile.
- [x] **R3 — PASS.** Human interface, provider-port and scheduling changes remain disclosed with before/after source and patches. The Hybrid MoveCube submission-only continuation used the existing conversation with no additional rollout; retain its handoff evidence. See <STUDY_ROOT>/audit/.

## Per-task accounting

Failed starts remain charged. All seed checks pass and all development starts have native outcomes. Steps are the benchmark’s charged control-step totals, including error penalties.

| Task | Budget | Starts | Failed / unclosed | Charged steps | Test policy matches |
| --- | ---: | ---: | ---: | ---: | ---: |
| BinFill | 10 | 10 | 4 / 0 | 6,434 | 50/50 |
| ButtonUnmask | 10 | 5 | 0 / 0 | 1,947 | 50/50 |
| ButtonUnmaskSwap | 10 | 10 | 4 / 0 | 3,708 | 50/50 |
| InsertPeg | 10 | 10 | 9 / 0 | 6,809 | 50/50 |
| MoveCube | 10 | 5 | 5 / 0 | 2,489 | 50/50 |
| PatternLock | 10 | 10 | 9 / 0 | 2,811 | 50/50 |
| PickHighlight | 10 | 8 | 4 / 0 | 5,840 | 50/50 |
| PickXtimes | 10 | 8 | 3 / 0 | 9,319 | 50/50 |
| RouteStick | 10 | 10 | 4 / 0 | 9,060 | 50/50 |
| StopCube | 10 | 9 | 3 / 0 | 1,645 | 50/50 |
| SwingXtimes | 10 | 7 | 2 / 0 | 3,426 | 50/50 |
| VideoPlaceButton | 10 | 10 | 3 / 0 | 666 | 50/50 |
| VideoPlaceOrder | 10 | 10 | 4 / 0 | 1,079 | 50/50 |
| VideoRepick | 10 | 9 | 3 / 0 | 1,772 | 50/50 |
| VideoUnmask | 10 | 8 | 1 / 0 | 3,098 | 50/50 |
| VideoUnmaskSwap | 10 | 5 | 1 / 0 | 1,599 | 50/50 |
| **Total** | **160 maximum** | **134** | **59 / 0** | **61,702** | **800/800** |

## Cost estimate

Development: **approximately $160**; evaluation: **$0 paid API cost**. Local compute is excluded. The recorded priced subtotal is $163.89.
The estimate uses the archived rate card and saved usage. For interrupted provider requests, an allowance based on the mean complete-request cost is added before rounding to $10; this is an estimate, not reconstructed token usage or an invoice. Frozen execution uses local providers and no paid model calls.

## Warnings and required action

| Warning | Evidence / scope | Action |
| --- | --- | --- |
| Provider-error API usage | 2 recorded failed requests have missing/partial usage: 0813_c0e69565, 1543_1736d9b4. | Report approximately $160; retain unknown token counts and the estimation basis. No rerun is needed. |
| Provider continuity evidence | External provider identities were reconstructed from retained processes, immutable dependency revisions and pre-start filesystem change times; their hashes were not added to the original pre-test manifest. | Keep the supplemental evidence with the original freeze manifest. For future studies, capture the provider inventory at freeze time. |
| Interrupted/rejected tools | Commands interrupted by the logged campaign-container deadline restart; no completed tool response is claimed. | Retain inputs, code revisions and interruption/rejection evidence. Do not invent outputs or charge rejected commands as rollouts. |

**Reuse decision:** retain the current frozen policies and all 800 evaluation outcomes. No redevelopment or reevaluation is required by this review. Preserve the supplements, original interruptions/errors and earlier audit reports. Scores are unchanged.

**Evidence index:** `doc/robomme_plus/audits/current/evidence/reaudit_summary.json` records hashes for the full local supplements; the development-log release must include those supplements alongside the original evidence.

</details>
