# ASPIRE on RoboMME-Plus

Evaluated by the RoboMME team. [Adapter and reproduction instructions](../../../scripts/plus_eval/agentic_methods/aspire/README.md) · [Frozen solutions](../../../scripts/plus_eval/agentic_methods/aspire/solutions/README.md) · [Logs and rollouts](../../../scripts/plus_eval/results/README.md).

- **Track:** Agentic.
- **Development model:** Claude Opus 5.5, high reasoning effort.
- **Evaluation:** frozen Python policies and local providers; no frontier-model API calls.
- **Demonstrations:** all 50 recordings available in demo settings; front/wrist RGB, goal, phase and timestamps. Interaction-only receives no expert recordings.
- **Interaction inputs:** public RGB-D, camera calibration, robot state, goal, conditioning video and terminal/error feedback.
- **Development limits:** 150 minutes per task; each reported budget was an independent study. All starts, including verification and rehearsals, share the task budget.
- **Study:** `full_20261116_protocol_v2`, with pinned v2/v3 interface prompts. These historical results use its original training snapshot; see the release-version qualification below.

## Main results

Each setting evaluates all 800 tests; failures, timeouts and errors remain in the denominator.

| Setting | Budget per task | Success | Actual starts, all tasks | Known development API cost (USD) | Evaluation API cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Demo-only | 0 | 136/800 (17.00%) | 0 | ≥$164.34 | $0 |
| Interaction-only | 10 | 341/800 (42.63%) | 150 | ≥$167.09 | $0 |
| Demo+interaction | 10 | 279/800 (34.88%) | 135 | ≥$178.99 | $0 |

Development costs use saved token usage and the pinned historical rate card; incomplete requests make the shown value a known subtotal. Local compute is excluded.

| Task | Demo-only | Interaction-only, R10 | Demo+interaction, R10 |
| --- | ---: | ---: | ---: |
| BinFill | 33/50 | 25/50 | 24/50 |
| ButtonUnmask | 2/50 | 29/50 | 26/50 |
| ButtonUnmaskSwap | 9/50 | 10/50 | 8/50 |
| InsertPeg | 0/50 | 0/50 | 0/50 |
| MoveCube | 2/50 | 6/50 | 1/50 |
| PatternLock | 0/50 | 10/50 | 10/50 |
| PickHighlight | 0/50 | 24/50 | 11/50 |
| PickXtimes | 10/50 | 26/50 | 27/50 |
| RouteStick | 10/50 | 10/50 | 10/50 |
| StopCube | 0/50 | 29/50 | 36/50 |
| SwingXtimes | 36/50 | 32/50 | 35/50 |
| VideoPlaceButton | 11/50 | 29/50 | 10/50 |
| VideoPlaceOrder | 15/50 | 10/50 | 18/50 |
| VideoRepick | 8/50 | 35/50 | 28/50 |
| VideoUnmask | 0/50 | 28/50 | 30/50 |
| VideoUnmaskSwap | 0/50 | 38/50 | 5/50 |

## Optional results: R1, R3, R5

These budgets are supplementary and do not enter the leaderboard.

| Budget per task | Interaction-only | Demo+interaction |
| --- | ---: | ---: |
| R1 | 117/800 (14.63%) | 155/800 (19.38%) |
| R3 | 134/800 (16.75%) | 261/800 (32.63%) |
| R5 | 215/800 (26.88%) | 252/800 (31.50%) |

<details>
<summary>Optional per-task successes (each out of 50)</summary>

| Task | Interaction R1 | R3 | R5 | Demo+interaction R1 | R3 | R5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BinFill | 1 | 5 | 10 | 13 | 42 | 26 |
| ButtonUnmask | 3 | 5 | 8 | 24 | 18 | 28 |
| ButtonUnmaskSwap | 1 | 5 | 3 | 0 | 2 | 8 |
| InsertPeg | 0 | 0 | 4 | 3 | 1 | 4 |
| MoveCube | 0 | 4 | 0 | 0 | 3 | 1 |
| PatternLock | 10 | 10 | 10 | 0 | 10 | 10 |
| PickHighlight | 0 | 3 | 7 | 8 | 10 | 3 |
| PickXtimes | 7 | 7 | 17 | 13 | 31 | 12 |
| RouteStick | 10 | 10 | 10 | 0 | 10 | 10 |
| StopCube | 32 | 44 | 35 | 14 | 29 | 22 |
| SwingXtimes | 10 | 7 | 19 | 2 | 20 | 18 |
| VideoPlaceButton | 8 | 5 | 18 | 5 | 10 | 11 |
| VideoPlaceOrder | 9 | 13 | 17 | 16 | 17 | 23 |
| VideoRepick | 16 | 15 | 14 | 37 | 30 | 30 |
| VideoUnmask | 2 | 1 | 33 | 19 | 5 | 30 |
| VideoUnmaskSwap | 8 | 0 | 10 | 1 | 23 | 16 |

</details>

## Development audit

**Harness: PASS WITHIN AUDITED SCOPE. Development records: PASS WITHIN AUDITED SCOPE for the historical main configurations, with the qualifications below.**

Audit date: 2026-11-21. Auditor: Codex, using automated inventories and manual source/evidence review; this is not independent external certification. The scope covers attempt accounting, exposed input boundaries, recorded tool access and frozen-artifact integrity. It does not establish absence of every unlogged OS operation.

[Machine-readable evidence](../audits/agentic_development_20261121.json) contains per-task counts, hashes, source locations and warnings. Raw evidence paths are relative to the audited configuration's `<CASE_ROOT>`. Derived indexes are identified by their content hashes under `<AUDIT_ROOT>`.

| Main setting | Transcript tool calls | Saved API requests checked | Starts reconciled |
| --- | ---: | ---: | ---: |
| Demo-only | 2071 | 2062 | 0 |
| Interaction-only | 2093 | 2087 | 150 |
| Demo+interaction | 2640 | 2452 | 135 |

| Rule / check | Finding | Evidence |
| --- | --- | --- |
| 1. Keep the full development history | Native commands, code history and attempts are retained, with gaps disclosed below. Attempt IDs match the ledger; failed/unfinished starts count and no task exceeds its allowance. | Native logs, `study/attempts.sqlite3`, `attempt.json` and outcomes; evidence A1. |
| 2. Use only permitted resources | Public sensor/calibration/robot packets and setting-gated demo exports; no prohibited Plus task-source read observed in the reviewed records. Recorded launches use network isolation, read-only roots and no host simulator-source mount. | Pinned `runtime_source/benchmark_io/episode.py`, method adapters, launch records, tool indexes and API request inventories; A3. |
| 3. Freeze everything before evaluation | Evaluated files and freeze manifests matched their hashes at audit time. Published annotation edits retain evaluated hashes and preserve executable logic. No inspected development API completion occurs after the first associated test. | `solutions/*/manifest.json`, `evaluation*/frozen_evaluation.json`, API completion records; A2. |
| 4. Report resource usage | Per-task attempts/steps and development/evaluation API costs are reported. Missing usage remains incomplete rather than zero. | Per-task table and saved `api_cost_summary.json`. |

The model-input review covers the main configurations. Optional R1/R3/R5 received attempt, seed, transcript-index and frozen-artifact checks; this does not claim a complete API-input audit for every optional run.

<details>
<summary>Main development attempts and recorded control steps per task</summary>

| Task | Interaction R10 starts | Control steps | Demo+interaction R10 starts | Control steps |
| --- | ---: | ---: | ---: | ---: |
| BinFill | 10/10 | 5138 | 10/10 | 5409 |
| ButtonUnmask | 10/10 | 2682 | 5/10 | 2419 |
| ButtonUnmaskSwap | 10/10 | 5252 | 10/10 | 4879 |
| InsertPeg | 8/10 | 5269 | 7/10 | 3966 |
| MoveCube | 10/10 | 1555 | 10/10 | 3959 |
| PatternLock | 10/10 | 3526 | 10/10 | 3822 |
| PickHighlight | 10/10 | 8650 | 10/10 | 11749 |
| PickXtimes | 10/10 | 3876 | 10/10 | 10268 |
| RouteStick | 9/10 | 3380 | 8/10 | 3239 |
| StopCube | 7/10 | 1414 | 10/10 | 1994 |
| SwingXtimes | 10/10 | 2582 | 9/10 | 3606 |
| VideoPlaceButton | 9/10 | 696 | 6/10 | 548 |
| VideoPlaceOrder | 9/10 | 746 | 9/10 | 904 |
| VideoRepick | 10/10 | 986 | 9/10 | 1096 |
| VideoUnmask | 9/10 | 2936 | 8/10 | 2364 |
| VideoUnmaskSwap | 9/10 | 2053 | 4/10 | 599 |

</details>

Demo-only has zero simulator starts and zero simulator control steps for every task.

### Warnings and release-version qualification

- Demo-only has one missing tool-response transcript entry for an offline mock sweep. The API tool call and five redirected output files survive (`development_api/0969_2657a508/events.jsonl`; `development/workspace/campaign/robomme_plus/robomme_plus/task_15/debug/logs/v0*/sweep_final_mock.txt`). No simulator was involved; the missing transcript entry remains disclosed.
- Optional interaction R5 and demo+interaction R3 each have one missing tool-response entry. Their simulator starts are still fully reconciled. Optional demo+interaction R1 includes a disclosed terminal-handling correction.
- The pinned training snapshot and current v2 differ in **10 recordings and nine execution limits** across ButtonUnmaskSwap, MoveCube, PickHighlight, RouteStick and VideoRepick. Seed identities, difficulty, variants and variation profiles match. The inspected fixed test metadata checks match the current release.
- These results may be retained under their original pinned training version. They must not be relabeled as development on current v2. Renamed settings and the new accounting wrapper alone do not require redevelopment.

For a strict current-v2 comparison, redevelop the three main campaigns and reevaluate their policies. ASPIRE shares learned skills across tasks, so replacing only the five changed tasks would leave uncertain dependencies.

Follow the [audit procedure and prompts](../auditing.md), using the [harness audit](../../skills/audit_robomme_plus_harness.md) and [development-log audit](../../skills/audit_robomme_plus_development.md). No simulator or model API was called during this audit.
