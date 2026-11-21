# RAX on RoboMME-Plus

Evaluated by the RoboMME team. [Adapter and reproduction instructions](../../../scripts/plus_eval/agentic_methods/rax/README.md) · [Frozen solutions](../../../scripts/plus_eval/agentic_methods/rax/solutions/README.md) · [Logs and rollouts](../../../scripts/plus_eval/results/README.md).

- **Track:** Agentic.
- **Development model:** Claude Opus 5.5, high reasoning effort.
- **Evaluation:** frozen Python policies and local providers; no frontier-model API calls.
- **Demonstrations:** all 50 recordings available in demo settings; front/wrist RGB, goal, phase and timestamps. `interaction-only` receives no expert recordings.
- **Interaction inputs:** public RGB-D, camera calibration, robot state, goal, conditioning video and terminal/error feedback.
- **Development limits:** 150 minutes per task; each reported budget was an independent study. All starts, including verification and rehearsals, share the task budget.
- **Study:** `full_20261116_protocol_v2`, with pinned v2/v3 interface prompts. These historical results use its original training snapshot; see the release-version qualification below.

## Main results

Each setting evaluates all 800 tests; failures, timeouts and errors remain in the denominator.

| Setting | Budget per task | Success | Actual starts, all tasks | Known development API cost (USD) | Evaluation API cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `demo-only` | 0 | 152/800 (19.00%) | 0 | ≥$133.34 | $0 |
| `interaction-only` | 10 | 312/800 (39.00%) | 134 | ≈$109.63 | $0 |
| `demo&interaction` | 10 | 337/800 (42.13%) | 114 | ≈$89.03 | $0 |

Development costs use saved token usage and the pinned historical rate card; incomplete requests make the shown value a known subtotal. Local compute is excluded.

| Task | `demo-only` | `interaction-only`, R10 | `demo&interaction`, R10 |
| --- | ---: | ---: | ---: |
| BinFill | 10/50 | 25/50 | 37/50 |
| ButtonUnmask | 0/50 | 29/50 | 27/50 |
| ButtonUnmaskSwap | 0/50 | 10/50 | 6/50 |
| InsertPeg | 0/50 | 2/50 | 10/50 |
| MoveCube | 0/50 | 2/50 | 3/50 |
| PatternLock | 0/50 | 10/50 | 6/50 |
| PickHighlight | 0/50 | 33/50 | 10/50 |
| PickXtimes | 4/50 | 27/50 | 29/50 |
| RouteStick | 7/50 | 10/50 | 10/50 |
| StopCube | 27/50 | 31/50 | 33/50 |
| SwingXtimes | 10/50 | 6/50 | 37/50 |
| VideoPlaceButton | 15/50 | 20/50 | 17/50 |
| VideoPlaceOrder | 19/50 | 16/50 | 16/50 |
| VideoRepick | 29/50 | 28/50 | 29/50 |
| VideoUnmask | 31/50 | 32/50 | 30/50 |
| VideoUnmaskSwap | 0/50 | 31/50 | 37/50 |

## Optional results: R1, R3, R5

These budgets are supplementary and do not enter the leaderboard.

| Budget per task | `interaction-only` | `demo&interaction` |
| --- | ---: | ---: |
| R1 | 26/800 (3.25%) | 156/800 (19.50%) |
| R3 | 351/800 (43.88%) | 315/800 (39.38%) |
| R5 | 265/800 (33.13%) | 283/800 (35.38%) |

<details>
<summary>Optional per-task successes (each out of 50)</summary>

| Task | `interaction-only` R1 | R3 | R5 | `demo&interaction` R1 | R3 | R5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BinFill | 6 | 37 | 35 | 36 | 27 | 37 |
| ButtonUnmask | 6 | 12 | 37 | 0 | 29 | 28 |
| ButtonUnmaskSwap | 0 | 11 | 10 | 1 | 8 | 12 |
| InsertPeg | 0 | 16 | 0 | 1 | 21 | 9 |
| MoveCube | 0 | 1 | 2 | 2 | 3 | 1 |
| PatternLock | 0 | 10 | 9 | 10 | 10 | 8 |
| PickHighlight | 0 | 21 | 36 | 7 | 16 | 28 |
| PickXtimes | 0 | 27 | 4 | 10 | 24 | 27 |
| RouteStick | 10 | 10 | 10 | 7 | 8 | 10 |
| StopCube | 1 | 32 | 26 | 15 | 31 | 33 |
| SwingXtimes | 0 | 29 | 3 | 10 | 38 | 15 |
| VideoPlaceButton | 0 | 35 | 0 | 13 | 10 | 12 |
| VideoPlaceOrder | 1 | 24 | 2 | 27 | 11 | 6 |
| VideoRepick | 0 | 24 | 28 | 17 | 19 | 29 |
| VideoUnmask | 2 | 33 | 28 | 0 | 26 | 28 |
| VideoUnmaskSwap | 0 | 29 | 35 | 0 | 34 | 0 |

</details>

## Development audit

**Harness: PASS WITHIN AUDITED SCOPE. Development records: PASS WITHIN AUDITED SCOPE for the historical main configurations, with the qualifications below.**

Audit date: 2026-11-21. Auditor: Codex, using automated inventories and manual source/evidence review; this is not independent external certification. The scope covers attempt accounting, exposed input boundaries, recorded tool access and frozen-artifact integrity. It does not establish absence of every unlogged OS operation.

[Machine-readable evidence](../audits/agentic_development_20261121.json) contains per-task counts, hashes, source locations and warnings. Raw evidence paths are relative to the audited configuration's `<CASE_ROOT>`. Derived indexes are identified by their content hashes under `<AUDIT_ROOT>`.

| Main setting | Transcript tool calls | Saved API requests checked | Starts reconciled |
| --- | ---: | ---: | ---: |
| `demo-only` | 1863 | 1719 | 0 |
| `interaction-only` | 1818 | 1627 | 134 |
| `demo&interaction` | 1567 | 1437 | 114 |

| Rule / check | Finding | Evidence |
| --- | --- | --- |
| 1. Keep the full development history | Native commands, code history and attempts are retained, with gaps disclosed below. Attempt IDs match the ledger; failed/unfinished starts count and no task exceeds its allowance. | Native logs, `study/attempts.sqlite3`, `attempt.json` and outcomes; evidence A1. |
| 2. Use only permitted resources | Public sensor/calibration/robot packets and setting-gated demo exports; no prohibited Plus task-source read observed in the reviewed records. Recorded launches use network isolation, read-only roots and no host simulator-source mount. | Pinned `runtime_source/benchmark_io/episode.py`, method adapters, launch records, tool indexes and API request inventories; A3. |
| 3. Freeze everything before evaluation | Evaluated files and freeze manifests matched their hashes at audit time. Published annotation edits retain evaluated hashes and preserve executable logic. No inspected development API completion occurs after the first associated test. | `solutions/*/manifest.json`, `evaluation*/frozen_evaluation.json`, API completion records; A2. |
| 4. Report resource usage | Per-task attempts/steps and development/evaluation API costs are reported. Missing usage remains incomplete rather than zero. | Per-task table and saved `api_cost_summary.json`. |

The model-input review covers the main configurations. Optional R1/R3/R5 received attempt, seed, transcript-index and frozen-artifact checks; this does not claim a complete API-input audit for every optional run.

<details>
<summary>Main development attempts and recorded control steps per task</summary>

| Task | `interaction-only` R10 starts | Control steps | `demo&interaction` R10 starts | Control steps |
| --- | ---: | ---: | ---: | ---: |
| BinFill | 10/10 | 6332 | 7/10 | 4699 |
| ButtonUnmask | 8/10 | 4229 | 8/10 | 4560 |
| ButtonUnmaskSwap | 10/10 | 7761 | 4/10 | 1940 |
| InsertPeg | 10/10 | 7301 | 10/10 | 5501 |
| MoveCube | 10/10 | 10420 | 10/10 | 4967 |
| PatternLock | 9/10 | 2427 | 4/10 | 1212 |
| PickHighlight | 6/10 | 3789 | 10/10 | 9165 |
| PickXtimes | 9/10 | 10628 | 9/10 | 7690 |
| RouteStick | 8/10 | 3206 | 6/10 | 2614 |
| StopCube | 6/10 | 1129 | 6/10 | 1152 |
| SwingXtimes | 10/10 | 5908 | 7/10 | 3080 |
| VideoPlaceButton | 9/10 | 2061 | 6/10 | 890 |
| VideoPlaceOrder | 4/10 | 603 | 6/10 | 955 |
| VideoRepick | 9/10 | 2465 | 10/10 | 1931 |
| VideoUnmask | 8/10 | 2826 | 6/10 | 2213 |
| VideoUnmaskSwap | 8/10 | 6134 | 5/10 | 2478 |

</details>

`demo-only` has zero simulator starts and zero simulator control steps for every task.

### Warnings and release-version qualification

- Ten `demo-only` task policies received operator corrections accepting the documented `camera_poses.position_m` field. Original/corrected hashes and reasons are in their solution manifests. Report these as interface-corrected results, not unmodified native submissions. Optional R1 configurations also contain disclosed corrections.
- `demo&interaction` R10 / PickHighlight has one truncated `api-request` metadata line. Complete saved request files parse and all native tool IDs reconcile. Optional `interaction-only` R5 / PickXtimes has two unfinalized reservations; both are counted, but their final control-step totals are incomplete.
- The pinned training snapshot and current v2 differ in **10 recordings and nine execution limits** across ButtonUnmaskSwap, MoveCube, PickHighlight, RouteStick and VideoRepick. Seed identities, difficulty, variants and variation profiles match. The inspected fixed test metadata checks match the current release.
- These results may be retained under their original pinned training version. They must not be relabeled as development on current v2. Renamed settings and the new accounting wrapper alone do not require redevelopment.

For a strict current-v2 comparison, redevelop the five affected tasks in each main setting and reevaluate those policies. RAX uses separate task workspaces; the other tasks can be retained after checking their pinned inputs and policy hashes.

Follow the [audit procedure and prompts](../auditing.md), using the [harness audit](../../skills/audit_robomme_plus_harness.md) and [development-log audit](../../skills/audit_robomme_plus_development.md). No simulator or model API was called during this audit.
