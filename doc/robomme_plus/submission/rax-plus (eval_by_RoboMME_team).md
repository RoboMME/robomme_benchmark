# RAX on RoboMME-Plus (eval_by_RoboMME_team)

RAX is an agentic framework that explores each task through rollouts, rehearses candidate controllers and submits a final program bundle per task. We run the unchanged upstream RAX core.

- **Track:** Agentic.
- **Code:**
  - Original: [robot-agent-exploration/rax](https://github.com/robot-agent-exploration/rax) at [`d49c8ad`](https://github.com/robot-agent-exploration/rax/tree/d49c8ad605008bb5a87e909d0473cc3f5e3600a6).
  - Our adapter: [`scripts/plus_eval/agentic_methods/rax/`](../../../scripts/plus_eval/agentic_methods/rax/README.md).
- **Final evaluation solutions:** [Final solutions](../../../scripts/plus_eval/agentic_methods/rax/solutions/README.md).
- **Development logs:** [RoboMME-plus-agentic-development_logs](https://huggingface.co/datasets/Yinpei/RoboMME-plus-agentic-development_logs).
- **Paper / website / blog:** `<PAPER_URL>` · `<WEBSITE_URL>` · `<BLOG_URL>`
- **Model / checkpoint:** Claude Opus 5.5 develops the policies; the final programs make no model API calls at test time.
- **Parameters:** N/A (API model).
- **Evaluated by:** RoboMME Team.

## Results

### Overall results

| Setting | Rollouts per task | Success | Development starts | Development API cost | Evaluation API cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Demo-only` | 0 | 139/800 (17.38%) | 0 | ≈$80 | $0 |
| `Interaction-only` | 10 | 372/800 (46.50%) | 123 | ≈$90 | $0 |
| `Hybrid` | 10 | 327/800 (40.88%) | 121 | ≈$100 | $0 |

### Per-task success

| Suite | Task | `Demo-only` | `Interaction-only` | `Hybrid` |
| --- | --- | ---: | ---: | ---: |
| Counting | BinFill | 15/50 (30.00%) | 31/50 (62.00%) | 25/50 (50.00%) |
| Counting | PickXtimes | 0/50 (0.00%) | 35/50 (70.00%) | 20/50 (40.00%) |
| Counting | SwingXtimes | 3/50 (6.00%) | 33/50 (66.00%) | 33/50 (66.00%) |
| Counting | StopCube | 23/50 (46.00%) | 26/50 (52.00%) | 33/50 (66.00%) |
| Permanence | VideoUnmask | 0/50 (0.00%) | 28/50 (56.00%) | 29/50 (58.00%) |
| Permanence | VideoUnmaskSwap | 0/50 (0.00%) | 38/50 (76.00%) | 32/50 (64.00%) |
| Permanence | ButtonUnmask | 26/50 (52.00%) | 32/50 (64.00%) | 34/50 (68.00%) |
| Permanence | ButtonUnmaskSwap | 2/50 (4.00%) | 15/50 (30.00%) | 10/50 (20.00%) |
| Reference | PickHighlight | 1/50 (2.00%) | 18/50 (36.00%) | 19/50 (38.00%) |
| Reference | VideoRepick | 31/50 (62.00%) | 33/50 (66.00%) | 32/50 (64.00%) |
| Reference | VideoPlaceButton | 11/50 (22.00%) | 31/50 (62.00%) | 19/50 (38.00%) |
| Reference | VideoPlaceOrder | 12/50 (24.00%) | 16/50 (32.00%) | 18/50 (36.00%) |
| Imitation | MoveCube | 2/50 (4.00%) | 0/50 (0.00%) | 1/50 (2.00%) |
| Imitation | InsertPeg | 0/50 (0.00%) | 16/50 (32.00%) | 0/50 (0.00%) |
| Imitation | PatternLock | 10/50 (20.00%) | 10/50 (20.00%) | 10/50 (20.00%) |
| Imitation | RouteStick | 3/50 (6.00%) | 10/50 (20.00%) | 12/50 (24.00%) |
| **Overall** | **All tasks** | **139/800 (17.38%)** | **372/800 (46.50%)** | **327/800 (40.88%)** |

### Reference results with fewer rollouts

| Setting | 1 rollout | 3 rollouts | 5 rollouts |
| --- | ---: | ---: | ---: |
| `Interaction-only` | 26/800 (3.25%) | 282/800 (35.25%) | 295/800 (36.88%) |
| `Hybrid` | 156/800 (19.50%) | 302/800 (37.75%) | 298/800 (37.25%) |

<details>
<summary>Per-task success with 1 rollout</summary>

| Suite | Task | `Interaction-only` | `Hybrid` |
| --- | --- | ---: | ---: |
| Counting | BinFill | 6/50 (12.00%) | 36/50 (72.00%) |
| Counting | PickXtimes | 0/50 (0.00%) | 10/50 (20.00%) |
| Counting | SwingXtimes | 0/50 (0.00%) | 10/50 (20.00%) |
| Counting | StopCube | 1/50 (2.00%) | 15/50 (30.00%) |
| Permanence | VideoUnmask | 2/50 (4.00%) | 0/50 (0.00%) |
| Permanence | VideoUnmaskSwap | 0/50 (0.00%) | 0/50 (0.00%) |
| Permanence | ButtonUnmask | 6/50 (12.00%) | 0/50 (0.00%) |
| Permanence | ButtonUnmaskSwap | 0/50 (0.00%) | 1/50 (2.00%) |
| Reference | PickHighlight | 0/50 (0.00%) | 7/50 (14.00%) |
| Reference | VideoRepick | 0/50 (0.00%) | 17/50 (34.00%) |
| Reference | VideoPlaceButton | 0/50 (0.00%) | 13/50 (26.00%) |
| Reference | VideoPlaceOrder | 1/50 (2.00%) | 27/50 (54.00%) |
| Imitation | MoveCube | 0/50 (0.00%) | 2/50 (4.00%) |
| Imitation | InsertPeg | 0/50 (0.00%) | 1/50 (2.00%) |
| Imitation | PatternLock | 0/50 (0.00%) | 10/50 (20.00%) |
| Imitation | RouteStick | 10/50 (20.00%) | 7/50 (14.00%) |
| **Overall** | **All tasks** | **26/800 (3.25%)** | **156/800 (19.50%)** |

</details>

<!-- optional-r35-tasks:start -->
R3 and R5 use independent development runs capped at the reported budget.

<details>
<summary><code>Interaction-only</code> · R3/R5 per task</summary>

| Task | 3 rollouts | 5 rollouts |
| --- | ---: | ---: |
| BinFill | 18/50 (36.00%) | 35/50 (70.00%) |
| PickXtimes | 21/50 (42.00%) | 19/50 (38.00%) |
| SwingXtimes | 8/50 (16.00%) | 30/50 (60.00%) |
| StopCube | 26/50 (52.00%) | 27/50 (54.00%) |
| VideoUnmask | 29/50 (58.00%) | 33/50 (66.00%) |
| VideoUnmaskSwap | 25/50 (50.00%) | 25/50 (50.00%) |
| ButtonUnmask | 31/50 (62.00%) | 27/50 (54.00%) |
| ButtonUnmaskSwap | 11/50 (22.00%) | 0/50 (0.00%) |
| PickHighlight | 30/50 (60.00%) | 17/50 (34.00%) |
| VideoRepick | 22/50 (44.00%) | 33/50 (66.00%) |
| VideoPlaceButton | 21/50 (42.00%) | 1/50 (2.00%) |
| VideoPlaceOrder | 18/50 (36.00%) | 25/50 (50.00%) |
| MoveCube | 3/50 (6.00%) | 4/50 (8.00%) |
| InsertPeg | 0/50 (0.00%) | 0/50 (0.00%) |
| PatternLock | 9/50 (18.00%) | 9/50 (18.00%) |
| RouteStick | 10/50 (20.00%) | 10/50 (20.00%) |
| Overall | 282/800 (35.25%) | 295/800 (36.88%) |

</details>

<details>
<summary><code>Hybrid</code> · R3/R5 per task</summary>

| Task | 3 rollouts | 5 rollouts |
| --- | ---: | ---: |
| BinFill | 26/50 (52.00%) | 29/50 (58.00%) |
| PickXtimes | 36/50 (72.00%) | 12/50 (24.00%) |
| SwingXtimes | 44/50 (88.00%) | 37/50 (74.00%) |
| StopCube | 36/50 (72.00%) | 30/50 (60.00%) |
| VideoUnmask | 32/50 (64.00%) | 23/50 (46.00%) |
| VideoUnmaskSwap | 31/50 (62.00%) | 26/50 (52.00%) |
| ButtonUnmask | 30/50 (60.00%) | 31/50 (62.00%) |
| ButtonUnmaskSwap | 2/50 (4.00%) | 9/50 (18.00%) |
| PickHighlight | 1/50 (2.00%) | 17/50 (34.00%) |
| VideoRepick | 27/50 (54.00%) | 36/50 (72.00%) |
| VideoPlaceButton | 8/50 (16.00%) | 5/50 (10.00%) |
| VideoPlaceOrder | 12/50 (24.00%) | 21/50 (42.00%) |
| MoveCube | 1/50 (2.00%) | 3/50 (6.00%) |
| InsertPeg | 0/50 (0.00%) | 9/50 (18.00%) |
| PatternLock | 10/50 (20.00%) | 0/50 (0.00%) |
| RouteStick | 6/50 (12.00%) | 10/50 (20.00%) |
| Overall | 302/800 (37.75%) | 298/800 (37.25%) |

</details>

<!-- optional-r35-tasks:end -->

## Method details

- **Development:** Claude Opus 5.5 with high reasoning effort; 150-minute deadline per task.
- **Demo use:** `Demo-only` and `Hybrid` use all 50 released demonstrations (front/wrist RGB frames, goals and phase labels); `Interaction-only` uses none.
- **Test time:** each task runs its final program; no model API calls.

## Agentic development audit record

Auditor: Claude Opus 5.5, 2026-11-22 (self-review by the RoboMME team's assistant).

### Audit summary

| Setting | Budget per task | Development starts | 1. No forbidden information | 2. Rollout budget | 3. No test-time optimization | Conclusion |
| --- | ---: | ---: | --- | --- | --- | --- |
| `Demo-only` | 0 | 0 | PASS | PASS | PASS | **PASS** |
| `Interaction-only` | 10 | 123 | PASS | PASS | PASS | **PASS** |
| `Hybrid` | 10 | 121 | PASS | PASS | PASS | **PASS** |

### Development starts per task

| Suite | Task | `Demo-only` starts | `Interaction-only` starts | `Hybrid` starts |
| --- | --- | ---: | ---: | ---: |
| Counting | BinFill | 0 | 8 | 5 |
| Counting | PickXtimes | 0 | 9 | 8 |
| Counting | SwingXtimes | 0 | 9 | 6 |
| Counting | StopCube | 0 | 9 | 7 |
| Permanence | VideoUnmask | 0 | 7 | 5 |
| Permanence | VideoUnmaskSwap | 0 | 7 | 7 |
| Permanence | ButtonUnmask | 0 | 6 | 9 |
| Permanence | ButtonUnmaskSwap | 0 | 7 | 10 |
| Reference | PickHighlight | 0 | 10 | 8 |
| Reference | VideoRepick | 0 | 6 | 5 |
| Reference | VideoPlaceButton | 0 | 10 | 7 |
| Reference | VideoPlaceOrder | 0 | 9 | 4 |
| Imitation | MoveCube | 0 | 5 | 10 |
| Imitation | InsertPeg | 0 | 5 | 10 |
| Imitation | PatternLock | 0 | 8 | 10 |
| Imitation | RouteStick | 0 | 8 | 10 |
| **Total** | **All tasks** | **0** | **123** | **121** |

All starts used released training seeds.

### Full audit reports

<details>
<summary><code>Demo-only</code> · R0 — PASS</summary>

### Audit: RAX / Demo-only / R0

This audit follows the skill `doc/skills/audit_robomme_plus.md`. It replaces the earlier `harness_audit.md` and `development_audit.md` in this directory, which were judged under the previous rules.

#### Conclusion: **PASS**

Both checks pass, and the evidence covers every agent session and every simulator start.

- **Check 1:** no forbidden read observed in the inspected logs. The harness kept every forbidden resource out of the agent's reach.
- **Check 2:** 0 Plus simulator starts in all 16 tasks (budget 0).

#### Scope

- **Method / setting / budget:** RAX, `Demo-only` (stored identifier `demo-only`), R0 per task. This is an independent run, not a checkpoint taken from a longer run.
- **Tasks:** all 16.
- **Run ID:** `rax/demo-only/R0` under `demo_only_20261121`.
  - Development ran from 2026-11-21 21:21:41 to 23:03:16 -0500.
  - The harness snapshot is `runtime_source/`.
  - The agent image is `sha256:d143d08a…7f0b` (`study_plan.json`).
  - The model was `us.anthropic.claude-opus-5-5` at high effort.
  - The dataset was `Yinpei/robomme_plus_data_h5@63cb9b5b…`. I re-hashed all 16 H5 files and they match the pin.
- **Auditor / date:** Claude Opus 5.5 (`us.anthropic.claude-opus-5-5`), 2026-11-22. This is a self-review by the RoboMME team's assistant.
- **Documents the agent received:**
  - **Public prompt bundle v9** in `workspace/docs/robomme-protocol-v9/`: README, BENCHMARK, DEMOS, RAX, PUBLIC_RULES, RUNTIME, plus the synthetic fixtures. These are byte-identical to `protocol/`. They match `scripts/plus_eval/benchmark_io/prompts/current/` at commit `f2f32df`. The current `PUBLIC_RULES.md` (sha256 `04c0fbc5…`) is identical to the copy the agents received.
  - **RAX native files:** `AGENTS.md`, `TASK.md`, `CLAUDE.md`, `MCP_SESSION.md`, `MCP_CONTRACT.json`, `api/runtime.py` and 20 generic skills. These are identical across all 16 tasks (tree hash `99d5a533…`) and were unchanged at the end of the run.
  - **The task's own demonstration export.**
  - **The `status`, `register` and `submit` tool replies.**
  - **One operator prompt** (`development_prompt.txt`). There were no follow-up or resume messages.
- **Evidence inspected (per task):**
  - `agent_stdout.jsonl`: 16 sessions, 1,247 tool uses, all with results.
  - The Claude session files.
  - `audit/history/events.jsonl` and the content-addressed `objects/`.
  - `state.json` and 44 registered bundles.
  - `launch_command.json`, `claude-home/settings.json` and `mcp.json`.
- **Evidence inspected (run-wide):**
  - `development_api/`: 1,156 records, matching the 1,156 calls counted in `model-call-count.json`.
  - `study/attempts.sqlite3`, opened with `mode=ro`, and `study.json`.
  - The harness source under `runtime_source/`.
  - The agent image metadata, read with `docker image inspect` and `docker history` only. No container was started.

#### Check 1: no forbidden information — PASS

##### Plus task or simulator source, and other methods' solutions

The harness kept them out of reach:

- **Container restrictions.** The development container runs with `--network=none --read-only --cap-drop=ALL` (`development/<Task>/launch_command.json:8-9`).
- **Mounts.** Only these are mounted (`launch_command.json:70-86`):
  - the task's own workspace and claude-home;
  - the read-only published-runs directory;
  - the audit socket and hook;
  - the model/MCP relay socket, the bridge and the CLI binary.
- **What is not mounted:** `benchmark/src`, `runtime_source`, `/home`, the ASPIRE directories and other tasks' workspaces.
- **Image contents.** The image layers are python:3.12-slim plus git, ripgrep and hash-locked wheels (torch, numpy, scipy, pillow, h5py, opencv; `runtime_source/rax/services/controller/agent_requirements.txt`). There is no robomme package, and `PIP_NO_INDEX=1` and `HF_HUB_OFFLINE=1` are set.
- **Relay.** It forwards only model API calls, through the logging proxy, and the MCP tools (`container_bridge.py`).
- **Bundle registration.** `register` paths are confined to the workspace and do not follow symlinks (`rax/services/controller/frontend.py:155-161`).

The logs show no use:

- No command or file path touches `robomme_plus`, `env_metadata`, `site-packages`, `/home`, `/data` or ASPIRE.
- The only `find /` (VideoPlaceButton) returned just `/workspace/api/runtime.py`.
- All Read/Write/Glob paths are under `/workspace`.
- No tool output contains a forbidden path.

##### Hidden simulator state

- The demo export contains only `rgb_images, goal, phase, recording_time` (`runtime_source/benchmark_io/demonstrations.py:12,55,60`).
- The `status` replies contain only contract, budget, bundle and storage fields.
- No observation or oracle tool was called. There were 0 calls to `start_episode`, `robomme_*`, `rehearse` and `evaluate`.

##### Test seeds, test metadata and test results

- Test metadata, the frozen manifests and `evaluation/` stay on the host and are never mounted.
- `rax/services/robomme_plus/session.py:47-51` refuses test episodes without the operator-only `ROBOMME_PLUS_FROZEN_EVALUATION` setting.
- Development ended at 23:03:16. The first test episode started at 01:28:10 the next day.
- Each task's evaluated policy is the agent's own `submit` choice (`state.json` → `submission`). `robomme_full.py:evaluate` refuses any operator selection. No test result informed any choice.
- All 50 study records equal the released training metadata and are disjoint from the test seeds.

##### Demonstrations

The setting is Demo-only, so the released recordings are allowed. The exports hold 50 episodes (3 per task, 5 for MoveCube). Each manifest's `source_sha256` equals its pinned H5 hash.

##### Prompts, skills and notes

- The operator prompt is generic (`runtime_source/rax/robomme_full.py:121-144`).
- `TASK.md` is generated from `services/robomme_plus/benchmark.py:task_context`, which states only the setting, the public rules and a pointer to the demos.
- The skills are generic. The only task-specific line is `robot-interface/SKILL.md:30` (the drawing-stick note), which is consistent with the public README.
- The ButtonUnmask/Swap press deadlines come from the allowed `RUNTIME.md`.
- Constants the agents wrote themselves trace to the demonstrations or to `RUNTIME.md`. For example, ButtonUnmask `controller.py:689-692` uses the step-90 deadline, and StopCube's `memory/development_report.md` derives its timing from demo frames.

#### Check 2: rollout budget — PASS

##### Ledger

`study/attempts.sqlite3` has 0 rows in `attempts` and in `attempt_provenance`.

##### Enforcement

- `Study` forces R0 for demo-only (`runtime_source/benchmark_io/protocol.py:39-40`).
- `reserve` raises `BudgetExhausted` before any simulator is constructed (`protocol.py:103-118`, `episode.py:95-108`).
- RAX also passes `exploration_seeds=()` (`robomme_full.py:107`).

##### Independent records agree

- 0 `SimulatorStart` events in any `audit/history/events.jsonl`.
- 0 simulator-start, rehearsal or evaluation tool calls.
- `state.json` shows `results=[]` and `exploration_started=0`.
- No recording, `operator_state.json` or `evaluation_outcome.json` files under `development/`.
- `development_result.json` shows `attempts: []` for every task.
- The agents' mock simulators (`experiments/mock_sim.py`, `run_mock.py`) are synthetic offline code. They are not Plus physics and not replays of the demonstrations.

##### Evidence coverage

- There is one native session per task, with no resumes. Five context compactions are captured in `development_api/`.
- Hook Pre/Post events match every tool use that ran. The 3 tool uses without hook events were `sleep` commands that Claude Code blocked before they ran.
- Every file of the 44 registered bundles is in the code-history objects, and every object hash verifies.

#### Per-task rollouts

Every task had a budget of 0 and used 0 starts. Seed validity is N/A because no seed was started; all training records are valid.

- BinFill
- ButtonUnmask
- ButtonUnmaskSwap
- InsertPeg
- MoveCube
- PatternLock
- PickHighlight
- PickXtimes
- RouteStick
- StopCube
- SwingXtimes
- VideoPlaceButton
- VideoPlaceOrder
- VideoRepick
- VideoUnmask
- VideoUnmaskSwap

**Total: 0 Plus simulator starts.**

#### Notes

- **Not judged under the current skill:** the mid-test patch to the freeze checker (`audit/corrections/empty_storage_lock_20261122/`) and the host-environment pinning. Both are test-time and freezing matters. Neither affects check 1 or 2: the policies are unchanged, and development had finished before testing.
- **Lost scratch output:** some offline mock-run output was written to the container's tmpfs `/tmp` (ButtonUnmask) and was not kept. Everything the agent read is in the transcripts and API records.
- **Compaction transcripts:** they are not saved as separate files, but their API requests and responses are in `development_api/`.
- **Cost (reported, accuracy not judged):** development API used 1,156 requests, estimated at $84.17. Evaluation used 0 requests.

#### Check 3 (added 2026-11-23 under the updated skill)

**Check 3: no test-time optimization — PASS**

Scope: RAX · Demo-only · budget 0 · 16 tasks × 50 = 800 test episodes. `<CASE>` = `demo_only_20261121/rax/demo-only/R0`, `<ROOT>` = its study root `demo_only_20261121/`. Auditor: Claude Opus 5.5, 2026-11-22. Read-only inspection. Times are host local (EST, UTC−5).

Evidence
- **Same policy in every episode.** For all 16 tasks, `<CASE>/evaluation/<Task>/policy/` (42 files) recomputes to `policy_sha256` in `frozen_evaluation.json`. That equals the development submission bundle in `<CASE>/development/<Task>/state.json` (`submission.bundle`), and `development/<Task>/bundles/<id>/` recomputes to the same files. Each of the 800 `evaluation/episodes/<Task>/episode_NNN/state.json` files registers exactly one bundle, and its copied files recompute to the frozen hashes. Each task's 50 `result.json` files carry a single `native_result.sha256`. All 800 episodes use the same SDK snapshot (`rax-controller-*/runtime.py`, sha256 `33e2e976…`) and the image digest `sha256:d143d08a…` pinned in `<ROOT>/study_plan.json`.
- **Checks run per episode.** `<ROOT>/runtime_source/benchmark_io/evaluation.py:87-98` (`verify_frozen`: policy hashes, 756 execution-file hashes, development attempt count) runs on every `reserve` (`:100-101`) and at simulator reset (`runtime_source/rax/services/robomme_plus/session.py:52`). `benchmark_io/unified_study.py:200,210` also re-verifies the bundle against the submission. Today the current `runtime_source/`, `protocol/`, `runtime_policy.json` and `study_plan.json` hash to every manifest's `execution_sha256` with 0 differences.
- **The mid-test checker patch did not change the policy.** In `<ROOT>/audit/corrections/empty_storage_lock_20261122/`, `evaluation.patch` adds only an exclusion for the empty file `runtime_source/rax/runtime/workspace-storage.lock`. That file was created at 01:28:26, as testing began. I compared `before/rax/demo-only/R0/evaluation/<Task>/frozen_evaluation.json` with the current manifest for all 16 tasks:
  - `policy_sha256` is identical;
  - `execution_sha256` differs only for `runtime_source/benchmark_io/evaluation.py` (`e0a17366…` → `5e0d34c6…`);
  - the only other change is the appended `operator_harness_corrections` entry.

  `<ROOT>/operator/fix_storage_lock.py:39-49,61-69,92-96` required evaluation to be held with no running jobs, asserted the policies were unchanged, and allowed only the lock filter to change.
- **Episodes before and after the patch ran the same bundle.** The 10 RAX episodes that ran before the patch (episode 0 of BinFill through StopCube, from 2026-11-22 01:28) use the same bundle sha256 as the 790 after it. Their `result.json` hashes still equal those in `recovery_plan.json`. The 790 rejected jobs never reserved an episode: each ledger `evaluation/<Task>/evaluation_attempts.sqlite3` has 50 rows with contiguous ids 1–50, purpose `frozen_full_evaluation_v2`, and seeds equal to the manifest records.
- **Development ended before testing.**
  - The last development API call ended 2026-11-21 23:03:09. Of 1,156 records in `<CASE>/development_api/*/transport.json`, none started at or after the first test.
  - The last submission was at 23:02:55. Manifests were created 23:03:13–16, and `development_complete.json` is stamped 23:03:16 with 0 training starts. `study/attempts.sqlite3` has 0 rows, matching `development_attempt_count` 0.
  - Testing ran from 2026-11-22 01:28:10 (first reservation) to 11:41:57 (last end).
  - In `<ROOT>/queue.sqlite3`, the only RAX `develop` job finished at 23:03:20. No file under `<CASE>/development/` was modified at or after 01:28:10.
- **No test-time model calls.** `api_budget.calls_used` is 0 and `gemini_api_seconds` is 0 in all 800 results, as is `evaluation/api_usage.json`.
- **No state carried between episodes.**
  - Each episode is its own worker, whose supervisor root is its own `episode_NNN/` (`unified_study.py:205`). Each simulator process runs one episode (`session.py:46`).
  - Each episode starts a fresh container (`rax/services/controller/sandbox.py:110-119,155-168`): `--network=none --read-only`, with the bundle, SDK and workspace scaffold mounted read-only. Only that episode's new `writable/` is mounted for output, and `/tmp` is a fresh tmpfs.
  - The policies write only to `/workspace/output` and read only their own run's frames. Every episode `state.json` holds exactly one `formal` result.
- **No test results reached development or selection.** Manifests record `test_feedback_to_agent: false`. Development had already finished, and every submission predates the freeze.

Notes
- The host-environment gap is not a policy change. The manifests do not pin the host `.venv`, but nothing in it changed during testing. The post-test `freeze_supplement_20261122/host_env_demo_only.json` shows no change, and my own stat of 56,870 venv entries found the newest mtime or ctime at 2026-11-21 17:47:24, before the first test.
- `audit/corrections/provider_restart_20261122/` (11:03) concerns ASPIRE's SAM3, graspnet, pyroki and curobo services and a scheduler overlay. `retry_candidates.json` lists no RAX job, and the RAX runtime hashes are unchanged.
- The patch is evaluation tooling, not policy. The manifests can be rewritten in place, so the `before/` copies, `evaluation.patch` and `correction.json` are the audit trail.
- This run had no OOM retries.

</details>

<details>
<summary><code>Interaction-only</code> · R10 — PASS</summary>

### RAX · Interaction-only · R10: development audit

**Conclusion: PASS.** Both checks pass, and there is no breach or missing evidence:

- **Check 1:** no forbidden information was reachable from the agent containers, and no forbidden read was observed in the inspected logs.
- **Check 2:** all 123 Plus simulator starts are in the ledger and reconcile with the native records. Every task stays at or below 10 starts, and every start uses a released training seed.

This report supersedes `harness_audit.md` and `development_audit.md` in this directory, which were written under the earlier two-skill rules. They are kept unchanged.

#### Scope

- **Method / setting / budget:** RAX; `Interaction-only` (stored as `interaction-only`); 10 starts per task across all released training seeds. This is an independent 10-start run, not an extracted R checkpoint.
- **Tasks:** all 16.
- **Run ID:** `r10_20261121`, case `rax/interaction-only/R10`.
- **Paths:** `<ROOT>` = `r10_20261121`; `<CASE>` = `<ROOT>/rax/interaction-only/R10`.
- **Harness configuration:**
  - harness source `<ROOT>/runtime_source/` (manifest sha256 `29eb5653…9fae`);
  - `study_plan.json` (agent image `sha256:d143d08aeca8a1791548cbdf2a3a0bd252222ea067868fc58c4cc09116c77f0b`, model `us.anthropic.claude-opus-5-5`);
  - benchmark fingerprint `b34d910c…1a72`;
  - data `Yinpei/robomme_plus_data_h5@63cb9b5b…`.
- **Auditor / date:** Claude Opus 5.5 (`us.anthropic.claude-opus-5-5`), 2026-11-22. This is a self-review by the RoboMME team's assistant, not an independent certification.
- **Method of review:** read-only. SQLite was opened with `mode=ro`. No simulator, container, generated code or model API was run.
- **Documents the agent received:** identical in all 16 workspaces.
  - **Protocol bundle:** `docs/robomme-protocol-v9/{README,BENCHMARK,DEMOS,RAX,PUBLIC_RULES,RUNTIME}.md` plus `bundle.json`, `manifest.json` and synthetic `fixtures/`, byte-identical to `<ROOT>/protocol/`.
    - `PUBLIC_RULES.md` (sha256 `04c0fbc5…4755`) is byte-identical to the current public bundle.
    - BENCHMARK, DEMOS and README differ from the current bundle only in setting names; RAX.md and RUNTIME.md are identical.
  - **RAX method instructions:** `AGENTS.md`, `CLAUDE.md`, `TASK.md`, `MCP_SESSION.md`, `MCP_CONTRACT.json`, `START_PROMPT.md`, `.agents/skills/*` and the SDK `api/runtime.py`.
  - **Operator prompt:** `development_prompt.txt`, identical for every task apart from the task name. There were no later operator messages.

#### Check 1: no forbidden information · PASS

**Plus source and other methods' solutions were unreachable, and none was read.**
- Each of the 16 `<CASE>/development/<Task>/launch_command.json` files runs the pinned image with `--network=none --read-only --cap-drop=ALL`. Each has exactly 10 explicit mounts: workspace, claude-home, read-only settings/mcp, read-only `published`, the audit socket and hook, the relay socket, the bridge and the CLI binary. No benchmark source, `env_metadata`, H5 data, evaluation directory or other method's directory is mounted.
- **Image contents:** I re-verified the digest and 7 layers with `docker image inspect`. The layer listing (`reaudit_20261122/evidence/image_files.jsonl.gz`, 34,820 entries) has no `robomme`, `robomme_plus`, `mani_skill`, `sapien`, `env_metadata` or ASPIRE paths. The preflight package probe is negative (`<ROOT>/audit/preflight.json`).
- **Controller sandboxes** run with `--network=none --read-only` and read-only bundle mounts (`runtime_source/rax/services/controller/sandbox.py:109-134`).
- **Indirect routes:** the model relay connects only to the logging Bedrock proxy, and MCP connects only to the host gateway (`container_bridge.py`). `WebFetch`/`WebSearch`/`Agent`/`Task` are disallowed.
- **Logs:** I scanned 1,452 tool calls in the 16 `agent_stdout.jsonl` files.
  - Every Read is under `/workspace/` or `/tmp/`.
  - Every Bash match for `robomme` refers to `docs/robomme-protocol-v9`.
  - There are no accesses to `/home`, `/data`, `/opt`, `/proc` or `site-packages`, and no network commands.
  - No tool output contains Plus source paths or tracebacks.

  No forbidden read observed in the inspected logs.

**Hidden simulator state was never exposed.**
- `runtime_source/benchmark_io/episode.py:52-74` and `rax/services/robomme_plus/session.py:77-128` pass on only RGB-D, intrinsics, camera poses, eef/gripper state, instruction, phase, step, limits and the terminal success flag.
- The MCP reply key inventory confirms this: no object poses, oracle actions, subgoals or `failure_reason`. The simulator logs stay on the host.

**Test data and results were unreachable and unused.**
- Test episodes need the operator-only `ROBOMME_PLUS_FROZEN_EVALUATION` capability (`session.py:47-56`), which is set only in the evaluation worker.
- `env_metadata/test/` and `<CASE>/evaluation/` were never mounted.
- No test seed appears in any workspace file or selected policy.
- Each policy was chosen by the agent's own `submit` during development. The last one was at 2026-11-22 00:14:41 UTC; development ended at 00:13:56 and the first test started at 00:59:44, so no test result could inform selection.

**Released demonstrations were not available.**
- `demonstration_context: none` (`runtime_source/rax/robomme_full.py:137`).
- All 16 `demonstration-context.json` files show `kind: none, image_count: 0`, `runtime/demonstrations/` is empty, and no H5 file was mounted or read.
- The conditioning video arrives as reset observations, which is allowed.

**Prompts and skills restate nothing beyond the allowed documents.**
- The only exact task values are in the bundle the run received. That includes the button press deadlines and the ButtonUnmaskSwap "480-step motion limit" sentence (`PUBLIC_RULES.md:9-17`), which is allowed documentation under rule 2.
- `TASK.md`, the RAX skills and the operator prompt contain no task parameters.
- Constants in agent-written code show no task-code values without provenance. A cross-check against Plus task constants matched only generic numbers and the public deadlines.

#### Check 2: rollout budget · PASS

- **Ledger:** `<CASE>/study/attempts.sqlite3` (sha256 `2785566b…f886`) holds 123 reservations. All use `selection=episode` with a policy revision.
- **Native records reconcile 1:1 for every task, deduplicated by attempt ID:**
  - `native/results/*/sim/recording/attempt.json` ids equal the ledger ids;
  - `outcome.json` equals the ledger outcome;
  - `events.jsonl` starts with `reset`, and its last step equals `control_steps`;
  - RAX `state.json` `exploration_started` equals the result count (67 interactive + 56 rehearsal);
  - each `SimulatorStart` history revision equals `attempt_provenance.policy_revision`.
- **Outcomes:** 0 unfinished starts and 0 setup errors. Two `rehearse` calls were rejected before reservation (unknown bundle id) and are not starts.
- **Enforcement:** `Study.reserve` takes an exclusive lock and caps each task at 10 across all seeds (`runtime_source/benchmark_io/protocol.py:103-133`). The reservation is made before the environment is constructed, and training resets are limited to one try (`episode.py:95`, `:120-128`). The agent containers have no simulator of their own.
- **Seeds:** every start's seed and full episode record equal the released `env_metadata/train` (all 50 records match). Training and test seeds are disjoint.
- **Sessions:** every agent session is covered, with no child or resumed sessions. There are 16 sessions with 0 subagents and no resume prompts. The four native context compactions stay inside the same session.
- **Hooks and code history:** 1,452 tool uses = 1,452 PreToolUse events, and 1,440 PostToolUse + 12 PostToolUseFailure. Code history holds 3,115 events and 1,635 objects, with none missing.

#### Per-task rollouts

Starts are given as total (interactive + rehearsal). The seed column lists starts per released training episode index.

| Task | Budget | Starts | Seed validity (starts per index) |
| --- | --- | --- | --- |
| BinFill | 10 | 8 (4+4) | valid, 3/3/2 |
| ButtonUnmask | 10 | 6 (4+2) | valid, 2/2/2 |
| ButtonUnmaskSwap | 10 | 7 (5+2) | valid, 3/2/2 |
| InsertPeg | 10 | 5 (3+2) | valid, 2/2/1 |
| MoveCube | 10 | 5 (2+3) | valid, 1/1/1/1/1 |
| PatternLock | 10 | 8 (7+1) | valid, 3/3/2 |
| PickHighlight | 10 | 10 (4+6) | valid, 4/3/3 |
| PickXtimes | 10 | 9 (5+4) | valid, 3/3/3 |
| RouteStick | 10 | 8 (3+5) | valid, 3/3/2 |
| StopCube | 10 | 9 (2+7) | valid, 3/3/3 |
| SwingXtimes | 10 | 9 (6+3) | valid, 3/3/3 |
| VideoPlaceButton | 10 | 10 (9+1) | valid, 4/3/3 |
| VideoPlaceOrder | 10 | 9 (6+3) | valid, 3/3/3 |
| VideoRepick | 10 | 6 (1+5) | valid, 2/2/2 |
| VideoUnmask | 10 | 7 (4+3) | valid, 3/2/2 |
| VideoUnmaskSwap | 10 | 7 (2+5) | valid, 3/2/2 |
| **Total** | **160 max** | **123 (67+56)** | all valid |

#### Notes

- The earlier H3 finding about the 480-step sentence is resolved. The current public `PUBLIC_RULES.md` is byte-identical to the received v9 file, which the rule treats as allowed documentation.
- The earlier H7 finding (host environment not pinned) and the six CUDA-OOM test reruns concern freezing and test-time evaluation, which are not judged here. Those reruns did not touch the development ledger.
- An ASPIRE-only provider-port fix edited the shared harness file `unified_study.py` during RAX development. It is disclosed in `<ROOT>/audit/corrections/provider_ports/` and has no effect on RAX inputs or the start count.
- Development API usage: 1,307 requests, all logged through the proxy, with an estimated cost of $87.93. This is reported for completeness; cost accuracy is not judged.

#### Check 3 (added 2026-11-23 under the updated skill)

**Check 3: no test-time optimization — PASS**

Scope: RAX · Interaction-only · budget 10 · 16 tasks × 50 = 800 test episodes. `<CASE>` = `r10_20261121/rax/interaction-only/R10`, `<ROOT>` = `r10_20261121/`. Auditor: Claude Opus 5.5, 2026-11-22. Read-only inspection. Times are host local (EST, UTC−5).

Evidence
- **Same policy in every episode.** For all 16 tasks, `<CASE>/evaluation/<Task>/policy/` (58 files) recomputes to `policy_sha256` in `frozen_evaluation.json`. That equals the submission bundle in `<CASE>/development/<Task>/state.json`, and `development/<Task>/bundles/<id>/` recomputes to the same files. All 800 `evaluation/episodes/<Task>/episode_NNN/state.json` files hold one bundle whose copied files recompute to the frozen hashes. Each task has a single `native_result.sha256` across its 50 results. All 800 use the SDK snapshot `runtime.py` `33e2e976…` and the image `sha256:d143d08a…` (`<ROOT>/study_plan.json`).
- **The manifests were never rewritten and the harness is unchanged.** Every manifest's mtime equals its `created` time (2026-11-21 19:15:04–07), and none has `operator_harness_corrections`. The current `<ROOT>/runtime_source/`, `protocol/`, `runtime_policy.json` and `study_plan.json` reproduce all 757 `execution_sha256` entries exactly. That count includes the empty `workspace-storage.lock` (`e3b0c442…`), which was created at 17:58, during development. `verify_frozen` (`runtime_source/benchmark_io/evaluation.py:79-90`) runs on every `reserve` (`:92-93`) and at simulator reset (`rax/services/robomme_plus/session.py:52`).
- **Development ended before testing.**
  - The last development API call ended 19:15:02. Of 1,307 records in `<CASE>/development_api/`, none started at or after the first test.
  - `study/attempts.sqlite3` has 123 rows; the last ended 19:13:56, and the count equals `development_attempt_count` 123. The last submission was at 19:14:41, and `development_complete.json` is stamped 19:15:07.
  - In `<ROOT>/queue.sqlite3`, the RAX `develop` job finished at 19:15:13.
  - Testing started at 19:59:44, and the main pass ended 2026-11-21 22:56:30.
  - No file under `<CASE>/development/` was modified at or after 19:59:44. Evaluation made 0 API calls (`calls_used` 0 in all 800 results).
- **The six OOM retries used the same frozen policy.** Retried episodes: BinFill 12, ButtonUnmask 12, RouteStick 11, SwingXtimes 13, VideoPlaceButton 10 and VideoPlaceOrder 10. They ran 2026-11-22 19:40:18–19:45:50. Each `oom_retry_20261122/records/r10_20261121/rax/interaction-only/R10/<Task>/episode_NNN.json` shows:
  - frozen policy and execution hashes equal to the manifest;
  - the same seed as the original run;
  - original and retry `native_result.sha256` identical;
  - `policy_changes: 0`.

  Against `oom_retry_20261122/database_backups/.../evaluation_attempts.sqlite3`, each affected ledger lost only the original OOM reservation and gained one new reservation (id 51) for the same episode and seed. No other row changed. The six were every RAX episode in this run with a simulator CUDA OOM (`runs/agentic_oom_audit_20261122/episodes.csv`). Each retry outcome replaces the original whether or not it succeeds (`oom_retry_20261122/operator/retry.py:126-132`). These are reruns of an unchanged policy, not optimization.
- **No state carried between episodes.**
  - Each episode is a separate worker with its own supervisor root (`unified_study.py:205`), and each simulator process runs one episode (`session.py:46`).
  - Each episode starts a fresh container (`rax/services/controller/sandbox.py:110-119,155-168`): `--network=none --read-only`, with the bundle, SDK and workspace mounted read-only, plus a new per-episode output mount and tmpfs.
  - Policy file I/O goes only to `/workspace/output` and to the current run's frames. Each episode state holds one `formal` result.
- **No test results reached development or selection.** `test_feedback_to_agent` is false, all RAX development preceded testing, and the submissions predate the freeze.

Notes
- The host `.venv` is not pinned in the manifests. The post-test `freeze_supplement_20261122/host_env_r10.json` and my own stat (newest venv change at 2026-11-21 17:47:24) show no change during testing.
- `audit/rax_evaluation_start/operator_scheduler_changes.patch` only raised the scheduler's concurrency limits, in a separate snapshot (`runtime/operator_scheduler_20`). The frozen `runtime_source/` hashes still match.
- ASPIRE development continued in the same study root during RAX testing, including `audit/movecube_submission_20261121/` (ASPIRE, at 21:22). That is another method's activity, and no RAX record changed.
- The released copies in `release_r10_20261122/staging/rax/solutions/interaction-only/R10/` match the frozen policy hashes for 16 of 16 tasks.

</details>

<details>
<summary><code>Hybrid</code> · R10 — PASS</summary>

### RAX · Hybrid (`demo&interaction`) · R10 — development audit

**Conclusion: PASS.** Both checks pass. The harness kept forbidden information out of
the agents' reach, and no forbidden read was observed in the inspected logs. All
121 Plus simulator starts are in the ledger and reconcile with the native records.
No task exceeds 10 starts, and every start uses a released training seed. The logs
cover every agent session and every start.

#### Scope

- **Method / setting / budget / tasks:** RAX; public `Hybrid` (stored as `demo&interaction`); R10, an independent run capped at 10 starts per task; all 16 tasks.
- **Run:** `<STUDY_ROOT>` = `r10_20261121`. `<CASE>` = `<STUDY_ROOT>/rax/demo&interaction/R10`, with `study/`, `development/<task>/`, `development_api/`.
- **Versions:** benchmark `f8d5ec6b4c53ec57ae8f7204a9d2765b93c0b55a` plus the archived interface patch. Runtime manifest sha256 `29eb5653…9fae`. Agent image `sha256:d143d08a…7f0b`. Dataset `Yinpei/robomme_plus_data_h5@63cb9b5b…`.
- **Auditor / date:** Claude Opus 5.5 (`us.anthropic.claude-opus-5-5`), 2026-11-22. Self-review by the RoboMME team's assistant.
- **Coverage:** harness source (`<STUDY_ROOT>/runtime_source/`) and the 16 agent launch commands and mounts. Ledger `study/attempts.sqlite3`, opened `mode=ro`, and `study.json`. Native rollout records. Hook history (3,799 events and their objects). Transcripts and CLI output (1,792 tool uses). 1,638 API request records. The image file list (34,820 entries) was checked against the live image's layer IDs. Earlier re-audit outputs were used only as leads; all counts were recomputed. Media were counted but not viewed. No simulator, container, policy or API was run.
- **Documents the agent received:**
  - **Protocol v9 public prompt bundle** (`docs/robomme-protocol-v9/`): `BENCHMARK.md`, `DEMOS.md`, `PUBLIC_RULES.md`, `RAX.md`, `RUNTIME.md`, `README.md`, `bundle.json`, synthetic `fixtures/`. All are byte-identical to `<STUDY_ROOT>/protocol/` and to `scripts/plus_eval/benchmark_io/prompts/current/` at `f8d5ec6`. The current `PUBLIC_RULES.md` (`04c0fbc5…`) is again identical.
  - **Native RAX operator material:** `AGENTS.md`, `CLAUDE.md`, `TASK.md`, `START_PROMPT.md`, `MCP_SESSION.md`, `MCP_CONTRACT.json`, `api/runtime.py` and 19 `.agents/skills`. All are identical across tasks; `TASK.md` differs only in the task name and the demo counts.
  - **Released training demonstration frames:** `runtime/demonstrations/`.

#### Check 1: no forbidden information — PASS

- **Plus source and other solutions out of reach.**
  - Agent containers run with `--network=none --read-only --cap-drop=ALL`, and their mounts are listed explicitly (`development/<task>/launch_command.json`; `runtime_source/rax/harness/claude_cli/auto_research.py:205`).
  - Mounts are the task's own workspace and claude-home, its own published runs (read-only), the audit and model/MCP relay sockets, the hook, the bridge and the CLI binary. No benchmark source, study or ledger directory, test data or other method is mounted.
  - The image has no `robomme`, `robomme_plus`, `mani_skill`, `sapien` or ASPIRE files. Rehearsal containers mount only the bundle and the runtime SDK (`rax/services/controller/sandbox.py:109-126`).
  - The bridge relays only the MCP and model sockets (`development/<task>/container_bridge.py:11-29`). WebFetch, WebSearch and Agent are disallowed.
- **Logs:** the inputs of all 1,792 tool uses were checked. Read paths were all under `/workspace` or `/tmp`. Bash touched only the workspace, the agent's own transcript and `/proc/uptime`. Tool outputs contain no `robomme_plus` source paths. No forbidden read was observed in the inspected logs.
- **No hidden state.** The translator passes only RGB-D, calibration, robot state, phase and goal (`runtime_source/benchmark_io/episode.py:52-75`). Logged MCP replies hold only public keys: images, depth, camera intrinsics and poses, state, instruction, phase, limits and transition. `evaluate` and rehearsal results give only binary `task_complete`.
- **No test data.** Training starts go through the training `Study` (`benchmark_io/protocol.py:103-138`), which checks metadata (`episode.py:95-112`). Test records are reachable only through the operator-only `EvaluationStudy`. All development ended at 2026-11-22 00:24:28 UTC, and the first test started at 00:59:44. Each task's policy is the agent's own native `submit` (`development/<task>/development_result.json`), so test results played no part in choosing a policy.
- **Demonstrations allowed in Hybrid.** All 16 exports are released training episodes, image frames only; each source sha256 equals the pinned H5 (`runtime/demonstrations/manifest.json`).
- **No task-code details in prompts.**
  - The operator prompt (`development_prompt.txt`; no resume prompts) and `TASK.md` give no task parameters, thresholds or success criteria.
  - The native skills are generic. Their one task-named line ("PatternLock and RouteStick use a fixed drawing stick…") matches public `RAX.md:108`.
  - Values in the agents' notes and code come from their own demo and rollout analysis.

#### Check 2: rollout budget — PASS

- **Reserved before startup.** A `BEGIN IMMEDIATE` reservation against one per-task count across all seeds is made before environment construction (`protocol.py:111-131`, `episode.py:95`). Training reset retries are limited to 1 (`episode.py:113-121`), and rehearsals use the same ledger.
- **Reconciliation:**
  - The 121 ledger rows match the native `sim/recording/attempt.json` and `outcome.json` records one-to-one by attempt ID, with the same seed, episode and outcome; nothing is extra or missing.
  - The rows also match the `SimulatorStart` hook events and the 121 published runs (interactive plus rehearsal).
  - No start is unfinished, and every `policy_revision` is in the code history.
- **Seeds:** every start's `(episode, seed)` equals `study.json`. Each task's seed set equals the training metadata and is disjoint from its test seeds.
- **Evidence coverage:** each task has one main session plus 8 compaction child sessions overall. The 1,792 transcript tool uses equal the 1,792 PreToolUse hooks, and all have results. The 1,638 API records are all complete. Every start has native records.

#### Per-task rollouts

| Task | Budget | Starts | Seed validity |
| --- | --- | --- | --- |
| BinFill | 10 | 5 | 3 training seeds, valid |
| ButtonUnmask | 10 | 9 | valid |
| ButtonUnmaskSwap | 10 | 10 | valid |
| InsertPeg | 10 | 10 | valid |
| MoveCube | 10 | 10 | 5 training seeds, valid |
| PatternLock | 10 | 10 | valid |
| PickHighlight | 10 | 8 | valid |
| PickXtimes | 10 | 8 | valid |
| RouteStick | 10 | 10 | valid |
| StopCube | 10 | 7 | valid |
| SwingXtimes | 10 | 6 | valid |
| VideoPlaceButton | 10 | 7 | valid |
| VideoPlaceOrder | 10 | 4 | valid |
| VideoRepick | 10 | 5 | valid |
| VideoUnmask | 10 | 5 | valid |
| VideoUnmaskSwap | 10 | 7 | valid |
| **Total** | 160 | **121** | all valid |

#### Notes

- The native CLI blocked two `rm output/*` commands, in PickHighlight and PickXtimes. These are method configuration, not findings.
- The compaction child sessions have no native transcript files. Their prompts and responses are in `development_api/` (`0632`, `0667`, `0680`, `0796`, `0870`, `1138`, `1534`, `1603`) and in the SubagentStop hook records.
- The ASPIRE-only provider-port fix changed `runtime_source/benchmark_io/unified_study.py` at 22:53 UTC on 2026-11-21, during RAX development. It is disclosed in `<STUDY_ROOT>/audit/corrections/provider_ports/` and does not affect access or accounting.
- This report supersedes my earlier `harness_audit.md` and `development_audit.md` under the new rules. Their H7 finding (the unpinned host environment) and the OOM reruns concern freezing and test-time evaluation, which are no longer judged.

#### Check 3 (added 2026-11-23 under the updated skill)

**Check 3: no test-time optimization — PASS**

Scope: RAX · Hybrid (`demo&interaction`) · budget 10 · 16 tasks × 50 = 800 test episodes. `<CASE>` = `r10_20261121/rax/demo&interaction/R10`, `<ROOT>` = `r10_20261121/`. Auditor: Claude Opus 5.5, 2026-11-22. Read-only inspection. Times are host local (EST, UTC−5).

Evidence
- **Same policy in every episode.** For all 16 tasks, `<CASE>/evaluation/<Task>/policy/` (74 files) recomputes to `policy_sha256` in `frozen_evaluation.json`. That equals the submission bundle in `<CASE>/development/<Task>/state.json`, and `development/<Task>/bundles/<id>/` recomputes to the same files. All 800 `evaluation/episodes/<Task>/episode_NNN/state.json` files hold one bundle whose copied files recompute to the frozen hashes. Each task has a single `native_result.sha256` across its 50 results. All 800 use the SDK `runtime.py` `33e2e976…` and the image `sha256:d143d08a…` (`<ROOT>/study_plan.json`).
- **The manifests were never rewritten and the harness is unchanged.** Every manifest's mtime equals its `created` time (2026-11-21 19:25:23–25), and none has `operator_harness_corrections`. The current `<ROOT>/runtime_source/`, `protocol/`, `runtime_policy.json` and `study_plan.json` reproduce all 757 `execution_sha256` entries exactly. `verify_frozen` (`runtime_source/benchmark_io/evaluation.py:79-90`) runs on every `reserve` (`:92-93`) and at simulator reset (`rax/services/robomme_plus/session.py:52`).
- **Development ended before testing.**
  - The last development API call ended 19:25:19. Of 1,638 records in `<CASE>/development_api/`, none started at or after the first test.
  - `study/attempts.sqlite3` has 121 rows; the last ended 19:24:28, and the count equals `development_attempt_count` 121. The last submission was at 19:24:49, and `development_complete.json` is stamped 19:25:25.
  - In `<ROOT>/queue.sqlite3`, the RAX `develop` job finished at 19:25:30.
  - Testing started at 19:59:44, and the main pass ended 2026-11-21 22:55:58.
  - No file under `<CASE>/development/` was modified at or after 19:59:44. Evaluation made 0 API calls (`calls_used` 0 in all 800 results).
- **The six OOM retries used the same frozen policy.** Retried episodes: BinFill 11, ButtonUnmaskSwap 11, PickXtimes 10 and 11, VideoPlaceOrder 10 and VideoRepick 10. They ran 2026-11-22 19:34:45–19:41:10. Each `oom_retry_20261122/records/r10_20261121/rax/demo&interaction/R10/<Task>/episode_NNN.json` shows:
  - frozen policy and execution hashes equal to the manifest;
  - the same seed as the original run;
  - original and retry `native_result.sha256` identical;
  - `policy_changes: 0`.

  Against `oom_retry_20261122/database_backups/.../evaluation_attempts.sqlite3`, each affected ledger lost only the original OOM reservations and gained one new reservation per retried episode with the same episode and seed (PickXtimes: ids 11 and 12 replaced by 51 and 52). No other row changed. The six were every RAX episode in this run with a simulator CUDA OOM (`runs/agentic_oom_audit_20261122/episodes.csv`). Each retry outcome replaces the original whether or not it succeeds (`oom_retry_20261122/operator/retry.py:126-132`). These are reruns of an unchanged policy, not optimization.
- **No state carried between episodes.**
  - Each episode is a separate worker with its own supervisor root (`unified_study.py:205`), and each simulator process runs one episode (`session.py:46`).
  - Each episode starts a fresh container (`rax/services/controller/sandbox.py:110-119,155-168`): `--network=none --read-only`, with the bundle, SDK and workspace mounted read-only, plus a new per-episode output mount and tmpfs.
  - Policy file I/O goes only to `/workspace/output` and to the current run's frames; for example, `ButtonUnmaskSwap/policy/controller.py:151` writes `track_history.json` there. Each episode state holds one `formal` result.
- **No test results reached development or selection.** `test_feedback_to_agent` is false, all RAX development preceded testing, and the submissions predate the freeze.

Notes
- The host `.venv` is not pinned in the manifests. The post-test `freeze_supplement_20261122/host_env_r10.json` and my own stat (newest venv change at 2026-11-21 17:47:24) show no change during testing.
- `audit/rax_evaluation_start/operator_scheduler_changes.patch` only raised the scheduler's concurrency limits, in a separate snapshot (`runtime/operator_scheduler_20`). The frozen `runtime_source/` hashes still match.
- ASPIRE development continued in the same study root during RAX testing, including the ASPIRE Hybrid MoveCube submission continuation at 21:22 (`audit/movecube_submission_20261121/`). That is another method's activity, and no RAX record changed.
- The released copies in `release_r10_20261122/staging/rax/solutions/hybrid/R10/` match the frozen policy hashes for 16 of 16 tasks.

</details>
