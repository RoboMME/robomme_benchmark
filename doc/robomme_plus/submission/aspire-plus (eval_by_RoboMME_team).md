# ASPIRE on RoboMME-Plus (eval_by_RoboMME_team)

ASPIRE is an agentic coding framework: a coordinator and task workers write task programs from demonstrations and rollouts, verify them, and share skills across tasks. We run the unchanged NVlabs/ASPIRE system with its SAM3, Contact-GraspNet, PyRoki and cuRobo providers.

- **Track:** Agentic.
- **Code:**
  - Original: [NVlabs/ASPIRE](https://github.com/NVlabs/ASPIRE) at [`fa00f0e`](https://github.com/NVlabs/ASPIRE/tree/fa00f0ef2bc1d91d594b9aa502c22133c9f538f7).
  - Our adapter: [`scripts/plus_eval/agentic_methods/aspire/`](../../../scripts/plus_eval/agentic_methods/aspire/README.md).
- **Final evaluation solutions:** [Final solutions](../../../scripts/plus_eval/agentic_methods/aspire/solutions/README.md).
- **Development logs:** `<DEVELOPMENT_LOGS_URL>` (public upload pending).
- **Paper / website / blog:** `<PAPER_URL>` · `<WEBSITE_URL>` · `<BLOG_URL>`
- **Model / checkpoint:** Claude Opus 5.5 develops the policies; the final programs make no model API calls at test time.
- **Parameters:** N/A (API model).
- **Evaluated by:** RoboMME Team.

## Results

### Overall results

| Setting | Rollouts per task | Success | Development starts | Development API cost | Evaluation API cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Demo-only` | 0 | 140/800 (17.50%) | 0 | ≈$210 | $0 |
| `Interaction-only` | 10 | 302/800 (37.75%) | 157 | ≈$160 | $0 |
| `Hybrid` | 10 | 342/800 (42.75%) | 134 | ≈$160 | $0 |

### Per-task success

| Suite | Task | `Demo-only` | `Interaction-only` | `Hybrid` |
| --- | --- | ---: | ---: | ---: |
| Counting | BinFill | 18/50 (36.00%) | 25/50 (50.00%) | 41/50 (82.00%) |
| Counting | PickXtimes | 0/50 (0.00%) | 24/50 (48.00%) | 11/50 (22.00%) |
| Counting | SwingXtimes | 17/50 (34.00%) | 27/50 (54.00%) | 33/50 (66.00%) |
| Counting | StopCube | 17/50 (34.00%) | 40/50 (80.00%) | 29/50 (58.00%) |
| Permanence | VideoUnmask | 31/50 (62.00%) | 27/50 (54.00%) | 34/50 (68.00%) |
| Permanence | VideoUnmaskSwap | 0/50 (0.00%) | 27/50 (54.00%) | 38/50 (76.00%) |
| Permanence | ButtonUnmask | 23/50 (46.00%) | 35/50 (70.00%) | 28/50 (56.00%) |
| Permanence | ButtonUnmaskSwap | 2/50 (4.00%) | 4/50 (8.00%) | 14/50 (28.00%) |
| Reference | PickHighlight | 0/50 (0.00%) | 17/50 (34.00%) | 11/50 (22.00%) |
| Reference | VideoRepick | 0/50 (0.00%) | 24/50 (48.00%) | 36/50 (72.00%) |
| Reference | VideoPlaceButton | 11/50 (22.00%) | 9/50 (18.00%) | 9/50 (18.00%) |
| Reference | VideoPlaceOrder | 10/50 (20.00%) | 15/50 (30.00%) | 20/50 (40.00%) |
| Imitation | MoveCube | 1/50 (2.00%) | 3/50 (6.00%) | 1/50 (2.00%) |
| Imitation | InsertPeg | 0/50 (0.00%) | 12/50 (24.00%) | 4/50 (8.00%) |
| Imitation | PatternLock | 0/50 (0.00%) | 3/50 (6.00%) | 10/50 (20.00%) |
| Imitation | RouteStick | 10/50 (20.00%) | 10/50 (20.00%) | 23/50 (46.00%) |
| **Overall** | **All tasks** | **140/800 (17.50%)** | **302/800 (37.75%)** | **342/800 (42.75%)** |

### Reference results with fewer rollouts

| Setting | 1 rollout | 3 rollouts | 5 rollouts |
| --- | ---: | ---: | ---: |
| `Interaction-only` | 117/800 (14.63%) | 142/800 (17.75%) | 193/800 (24.12%) |
| `Hybrid` | 155/800 (19.38%) | 274/800 (34.25%) | 304/800 (38.00%) |

<details>
<summary>Per-task success with 1 rollout</summary>

| Suite | Task | `Interaction-only` | `Hybrid` |
| --- | --- | ---: | ---: |
| Counting | BinFill | 1/50 (2.00%) | 13/50 (26.00%) |
| Counting | PickXtimes | 7/50 (14.00%) | 13/50 (26.00%) |
| Counting | SwingXtimes | 10/50 (20.00%) | 2/50 (4.00%) |
| Counting | StopCube | 32/50 (64.00%) | 14/50 (28.00%) |
| Permanence | VideoUnmask | 2/50 (4.00%) | 19/50 (38.00%) |
| Permanence | VideoUnmaskSwap | 8/50 (16.00%) | 1/50 (2.00%) |
| Permanence | ButtonUnmask | 3/50 (6.00%) | 24/50 (48.00%) |
| Permanence | ButtonUnmaskSwap | 1/50 (2.00%) | 0/50 (0.00%) |
| Reference | PickHighlight | 0/50 (0.00%) | 8/50 (16.00%) |
| Reference | VideoRepick | 16/50 (32.00%) | 37/50 (74.00%) |
| Reference | VideoPlaceButton | 8/50 (16.00%) | 5/50 (10.00%) |
| Reference | VideoPlaceOrder | 9/50 (18.00%) | 16/50 (32.00%) |
| Imitation | MoveCube | 0/50 (0.00%) | 0/50 (0.00%) |
| Imitation | InsertPeg | 0/50 (0.00%) | 3/50 (6.00%) |
| Imitation | PatternLock | 10/50 (20.00%) | 0/50 (0.00%) |
| Imitation | RouteStick | 10/50 (20.00%) | 0/50 (0.00%) |
| **Overall** | **All tasks** | **117/800 (14.63%)** | **155/800 (19.38%)** |

</details>

<!-- optional-r35-tasks:start -->
R3 and R5 use independent development runs capped at the reported budget.

<details>
<summary><code>Interaction-only</code> · R3/R5 per task</summary>

| Task | 3 rollouts | 5 rollouts |
| --- | ---: | ---: |
| BinFill | 10/50 (20.00%) | 23/50 (46.00%) |
| PickXtimes | 4/50 (8.00%) | 8/50 (16.00%) |
| SwingXtimes | 9/50 (18.00%) | 9/50 (18.00%) |
| StopCube | 30/50 (60.00%) | 21/50 (42.00%) |
| VideoUnmask | 24/50 (48.00%) | 2/50 (4.00%) |
| VideoUnmaskSwap | 8/50 (16.00%) | 26/50 (52.00%) |
| ButtonUnmask | 4/50 (8.00%) | 19/50 (38.00%) |
| ButtonUnmaskSwap | 0/50 (0.00%) | 12/50 (24.00%) |
| PickHighlight | 2/50 (4.00%) | 9/50 (18.00%) |
| VideoRepick | 13/50 (26.00%) | 8/50 (16.00%) |
| VideoPlaceButton | 0/50 (0.00%) | 6/50 (12.00%) |
| VideoPlaceOrder | 16/50 (32.00%) | 19/50 (38.00%) |
| MoveCube | 2/50 (4.00%) | 0/50 (0.00%) |
| InsertPeg | 0/50 (0.00%) | 11/50 (22.00%) |
| PatternLock | 10/50 (20.00%) | 10/50 (20.00%) |
| RouteStick | 10/50 (20.00%) | 10/50 (20.00%) |
| Overall | 142/800 (17.75%) | 193/800 (24.12%) |

</details>

<details>
<summary><code>Hybrid</code> · R3/R5 per task</summary>

| Task | 3 rollouts | 5 rollouts |
| --- | ---: | ---: |
| BinFill | 35/50 (70.00%) | 18/50 (36.00%) |
| PickXtimes | 7/50 (14.00%) | 10/50 (20.00%) |
| SwingXtimes | 13/50 (26.00%) | 36/50 (72.00%) |
| StopCube | 30/50 (60.00%) | 36/50 (72.00%) |
| VideoUnmask | 10/50 (20.00%) | 31/50 (62.00%) |
| VideoUnmaskSwap | 30/50 (60.00%) | 25/50 (50.00%) |
| ButtonUnmask | 27/50 (54.00%) | 30/50 (60.00%) |
| ButtonUnmaskSwap | 10/50 (20.00%) | 11/50 (22.00%) |
| PickHighlight | 13/50 (26.00%) | 35/50 (70.00%) |
| VideoRepick | 26/50 (52.00%) | 29/50 (58.00%) |
| VideoPlaceButton | 8/50 (16.00%) | 12/50 (24.00%) |
| VideoPlaceOrder | 23/50 (46.00%) | 10/50 (20.00%) |
| MoveCube | 1/50 (2.00%) | 0/50 (0.00%) |
| InsertPeg | 21/50 (42.00%) | 8/50 (16.00%) |
| PatternLock | 10/50 (20.00%) | 3/50 (6.00%) |
| RouteStick | 10/50 (20.00%) | 10/50 (20.00%) |
| Overall | 274/800 (34.25%) | 304/800 (38.00%) |

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
| `Interaction-only` | 10 | 157 | PASS | PASS | PASS | **PASS** |
| `Hybrid` | 10 | 134 | PASS | PASS | PASS | **PASS** |

### Development starts per task

| Suite | Task | `Demo-only` starts | `Interaction-only` starts | `Hybrid` starts |
| --- | --- | ---: | ---: | ---: |
| Counting | BinFill | 0 | 10 | 10 |
| Counting | PickXtimes | 0 | 10 | 8 |
| Counting | SwingXtimes | 0 | 10 | 7 |
| Counting | StopCube | 0 | 10 | 9 |
| Permanence | VideoUnmask | 0 | 10 | 8 |
| Permanence | VideoUnmaskSwap | 0 | 10 | 5 |
| Permanence | ButtonUnmask | 0 | 10 | 5 |
| Permanence | ButtonUnmaskSwap | 0 | 10 | 10 |
| Reference | PickHighlight | 0 | 10 | 8 |
| Reference | VideoRepick | 0 | 7 | 9 |
| Reference | VideoPlaceButton | 0 | 10 | 10 |
| Reference | VideoPlaceOrder | 0 | 10 | 10 |
| Imitation | MoveCube | 0 | 10 | 5 |
| Imitation | InsertPeg | 0 | 10 | 10 |
| Imitation | PatternLock | 0 | 10 | 10 |
| Imitation | RouteStick | 0 | 10 | 10 |
| **Total** | **All tasks** | **0** | **157** | **134** |

All starts used released training seeds.

### Full audit reports

<details>
<summary><code>Demo-only</code> · R0 — PASS</summary>

### Audit: ASPIRE / Demo-only / R0

Skill: `doc/skills/audit_robomme_plus.md`. `<root>` = `demo_only_20261121`,
`<case>` = `<root>/aspire/demo-only/R0`, `<dev>` = `<case>/development`. Times are UTC.

#### 1. Conclusion

**PASS.** Both checks pass. The agent had no forbidden information: the harness kept it out of reach, and no forbidden read was observed in the inspected logs. The run made 0 Plus simulator starts against an R0 budget. The logs cover every agent session.

#### 2. Scope

- **Method / setting / budget:** ASPIRE; `Demo-only` (stored as `demo-only`); 0 Plus starts per task.
- **Tasks:** all 16.
- **Run ID:** `aspire/demo-only/R0` in study `demo_only_20261121`. There was a single development job, `develop/aspire/demo-only/R0` (`<root>/queue.sqlite3`), from 2026-11-22 02:21:41Z to 06:24:54Z.
- **Model and versions:** Opus 5.5, high effort. ASPIRE `fa00f0ef…` plus two patches (`<root>/runtime_source_manifest.json`). Agent image `sha256:d143d08a…7f0b` (`<root>/study_plan.json`). Dataset `Yinpei/robomme_plus_data_h5@63cb9b5b…`.
- **Auditor / date:** Claude Opus 5.5 (`us.anthropic.claude-opus-5-5`), 2026-11-22. This is a self-review by the RoboMME team's assistant.
- **Documents the agent received** (`<dev>/workspace`, initial snapshot in `<dev>/history/snapshots.jsonl` line 1):
  - **Operator text:** the launch prompt (`<dev>/development_prompt.txt`) and the campaign brief `CLAUDE.md`.
  - **Public prompt bundle v9:** `docs/robomme-protocol-v9/` (`README.md`, `BENCHMARK.md`, `DEMOS.md`, `ASPIRE.md`, `PUBLIC_RULES.md`, `RUNTIME.md`, `bundle.json`, `manifest.json`, synthetic `fixtures/`). Their hashes equal `<root>/protocol/`. `PUBLIC_RULES.md` is byte-identical to the current public bundle (sha256 `04c0fbc5…`). `docs/robomme-plus.md` is the same file as `ASPIRE.md`.
  - **ASPIRE's own method material:** upstream `docs/api-reference.md`, `docs/run-artifacts.md` and `.claude/skills/iterative-debugging/` (prompts, skill library, scripts).
  - **Campaign files:** `campaign/tasks.json` and `campaign/campaign.json` (task names and training indices only).
  - **Demonstrations:** the 50 released demos, read-only.
- **Evidence inspected:**
  - **Agent sessions:** the main transcript, 31 subagent transcripts and the hook log `<dev>/audit/history/events.jsonl` (5,632 events).
  - **Code history:** both content-addressed object stores.
  - **API archive:** all 2,687 records in `<case>/development_api/`.
  - **Rollout evidence:** `<dev>/bridge.log`, the ledger `<case>/study/attempts.sqlite3`, the demo export and the harness source in `<root>/runtime_source/`.

#### 3. Checks

##### Check 1: no forbidden information — PASS

**What the harness made reachable**

- **Container isolation.** The agent container runs with `--network=none --read-only --cap-drop=ALL` and only explicit bind mounts (`<dev>/launch_command.json`):
  - workspace, `claude-home`, bridge;
  - the RPC socket, `published/`, demonstrations;
  - the evidence-hook socket and the `claude` binary.

  It has no mount of the benchmark source, `runtime_source`, test metadata, the evaluation directory or other cases. WebFetch and WebSearch are denied and MCP is empty (`<dev>/claude-home/settings.json`, `mcp.json`).
- **Image contents.** The image history shows only base Python, `apt install git ripgrep` and a hash-locked `pip install`, with no source COPY. The same image's probe found `robomme`, `robomme_plus`, `sapien` and `mani_skill` all absent (`r10_20261121/audit/preflight.json`, `container_forbidden_package_probe`).
- **Host RPC.** It exposes only three endpoints:
  - `GET /api/status`, which returns the budget, clocks and training indices (`runtime_source/aspire/campaign.py:66-77`);
  - `POST /api/run`, which is refused before reservation when the budget is 0 (`campaign.py:103-111`);
  - the `/v1/messages` relay.

  Every other path returns 404 (`runtime_source/aspire/development_runtime.py:27-45`). Program rollouts would see only the public tool registry (`runtime_source/aspire/isolated.py:116-203`).
- **Demonstrations.** `Demo-only` allows them. The export carries only RGB, goal, phase and recording time (`runtime_source/benchmark_io/demonstrations.py:42-60`). All 16 manifests list `contains: [rgb_images, goal, phase, recording_time]`, and their `source_sha256` matches the dataset pin (16/16).
- **Hidden state and test data.** No hidden state or oracle reached the agent. The study records hold only released training seeds: 16/16 match `benchmark/src/robomme_plus/env_metadata/train` and are disjoint from the test seeds. Test metadata stays on the host.
- **Development ended before testing.** The last development API call was at 06:24:29Z and the session ended at 06:24:42Z (`SessionEnd` event `57b45f34…`). The first test reservation was at 06:28:10Z. No test result could have reached development or policy selection.

**What the logs show was used**

- **Commands and outputs.** All 2,768 unique tool calls and 2,768 tool results contain no reference to `/home`, `/data`, `env_metadata`, `robomme_env`, installed `robomme`, `mani_skill`, the network or other methods' runs. Every `rax` hit is a public synthetic fixture.
- **Probes came back empty.** `find / -name contracts.py -path '*cap_harness*'` returned nothing, and `import cap_harness` raised `ModuleNotFoundError` (subagents `agent-a3bb79794…`, `agent-a920c7f6f0e8ce739`).
- **Operator messages.** The only operator message is the launch prompt. Every other user turn is a task notification or a compaction continuation.
- **No task-code details in prompts or policy.**
  - The operator text, `CLAUDE.md` and the upstream ASPIRE skills contain no Plus task names or parameters (grep).
  - The two exact deadlines in the frozen code (`policy/ButtonUnmask.py:24` = 89, `ButtonUnmaskSwap.py:22` = 179) cite `PUBLIC_RULES.md`, which is allowed documentation.
  - A cross-check of the policies' numeric constants against `benchmark/src/robomme_plus/tasks` found only generic overlaps, such as colour thresholds and a commented "generic Panda workspace" reach. Nothing came from task code without provenance.

##### Check 2: rollout budget — PASS

- **Ledger.** `<case>/study/attempts.sqlite3` (opened with `mode=ro`) has `attempts` 0, `attempt_provenance` 0 and `auto_cursor` 0. Its config is `demo-only`, budget 0, accounting v2.
- **Independent records agree:**
  - `<dev>/bridge.log` has 0 `POST /api/run`; it has only 95 `GET /api/status`, 2,687 `/v1/messages` and 84 locally rejected `count_tokens`.
  - There is no `<dev>/runs/`, and `<dev>/published/` is empty.
  - `<dev>/development_result.json` shows `starts_used: 0` for all 16 tasks.
  - The queue holds exactly one development job.
- **Enforcement in code.**
  - `Study` refuses any non-zero budget for `demo-only` (`runtime_source/benchmark_io/protocol.py:39-40`).
  - `reserve()` checks the count inside `BEGIN IMMEDIATE` before any simulator construction (`protocol.py:103-118`).
- **No demonstration replay.** There was no physics replay of demonstrations (0 starts).

##### Evidence coverage — complete

- **Sessions.** One main session and 31 subagents. Their `SubagentStart` events, `SubagentStop` events and transcript files reconcile one to one.
- **Compaction subagents.** There are 20 further `SubagentStop` events with an empty agent type. These are Claude Code context-compaction summaries: no transcript file exists, but their prompts and outputs are in the API archive.
- **Tool calls.** All 2,748 `PreToolUse` events have a matching Post or Failure event and a matching transcript call.
- **Code history.** The hook store has 2,635 objects and the snapshot store 1,715 objects, none missing.
- **Simulator starts.** There were none to log.

#### 4. Per-task rollouts

| Task | Budget | Starts | Seed validity |
| --- | --- | --- | --- |
| BinFill | 0 | 0 | No starts; records = released train seeds 0–2 |
| ButtonUnmask | 0 | 0 | No starts; records = released train seeds 0–2 |
| ButtonUnmaskSwap | 0 | 0 | No starts; records = released train seeds 0–2 |
| InsertPeg | 0 | 0 | No starts; records = released train seeds 0–2 |
| MoveCube | 0 | 0 | No starts; records = released train seeds 0–4 |
| PatternLock | 0 | 0 | No starts; records = released train seeds 0–2 |
| PickHighlight | 0 | 0 | No starts; records = released train seeds 0–2 |
| PickXtimes | 0 | 0 | No starts; records = released train seeds 0–2 |
| RouteStick | 0 | 0 | No starts; records = released train seeds 0–2 |
| StopCube | 0 | 0 | No starts; records = released train seeds 0–2 |
| SwingXtimes | 0 | 0 | No starts; records = released train seeds 0–2 |
| VideoPlaceButton | 0 | 0 | No starts; records = released train seeds 0–2 |
| VideoPlaceOrder | 0 | 0 | No starts; records = released train seeds 0–2 |
| VideoRepick | 0 | 0 | No starts; records = released train seeds 0–2 |
| VideoUnmask | 0 | 0 | No starts; records = released train seeds 0–2 |
| VideoUnmaskSwap | 0 | 0 | No starts; records = released train seeds 0–2 |

**Total Plus simulator starts: 0.**

#### 5. Notes

- This report supersedes `harness_audit.md` and `development_audit.md` in this directory under the simplified rules. Their H7 findings concern test-time evaluation: 40 post-action outage reruns, and the mid-test `evaluation.py` checker patch with its manifest rewrite. Under the new skill these are not judged, and they do not affect checks 1 or 2.
- 20 agent `Write` calls were refused by the client before the hooks ran. They are visible in the transcripts and are a safety block, not a failure.
- The public bundle's `fixtures/` holds non-Markdown synthetic files (blank images, invented values) with no task scene. They are part of the received bundle.
- The bundle README's line "Fresh studies pin v7" is stale wording, and v9 uses lowercase setting names. Neither affects what was allowed.
- The evidence-hook socket mount (`/run/robomme-audit`, read-only) was never accessed by the agent.
- Development API use was 2,687 requests, about $206.65 (`<case>/api_cost_summary.json`). One request (`development_api/2518_57058693`) failed with Bedrock ServiceUnavailable, leaving its usage unknown. Evaluation made no API calls. Cost accuracy is not judged.

#### Check 3 (added 2026-11-23 under the updated skill)

**Check 3: no test-time optimization — PASS**

ASPIRE · Demo-only · budget 0. `<root>` = `demo_only_20261121`, `<case>` = `<root>/aspire/demo-only/R0`. Times are UTC. Inspection was read-only; no code, simulator or API was run.

- **The frozen policy ran every test episode.** All 800 `submitted_program.py` files under `<case>/evaluation/episodes/*/episode_*/artifacts/**/` hash to `policy_sha256` in `<case>/evaluation/frozen_evaluation.json`, with 0 mismatches. So do all 82 archived pre-rerun originals in `<root>/audit/corrections/provider_restart_20261122/outage_episodes/`. The `source/dependency-lock.json` is identical in all 800 runs and equals the hashed `runtime_source/aspire/ASPIRE/configs/dependency-lock.json`. All 800 runs used image `sha256:d143d08a…`.
- **The manifest rewrite did not change the policy.** The original manifest (`<root>/audit/corrections/empty_storage_lock_20261122/before/aspire/demo-only/R0/evaluation/frozen_evaluation.json`, sha `917f41d3…`) was compared key by key with the current one (sha `50225a1b…`). Only two things differ: the `operator_harness_corrections` entry and one execution hash, `runtime_source/benchmark_io/evaluation.py` (`e0a17366…` → `5e0d34c6…`). `policy_sha256`, the episodes, the config and the other 755 execution hashes are identical. `evaluation.patch` only drops the empty `runtime_source/rax/runtime/workspace-storage.lock` (0 bytes now) from the execution hash. That is a checker change, not a policy change.
- **Integrity was re-checked during testing.** `verify_frozen` checks the policy hashes, the execution hashes and the development attempt count on every reservation (`runtime_source/benchmark_io/evaluation.py:87-101`). The check rejected 790 jobs with `Frozen harness/protocol changed` before any reservation (06:28:54Z–07:34:06Z; `blocked_episodes/`). The 10 earlier reservations are unchanged in the current ledger. Today, every policy, `selected_policies/` and execution file matches the manifest. The only hashed file with an mtime or ctime after the first test is `evaluation.py` (14:27:56Z).
- **There was no development during testing.** Freeze and `development_complete.json` are at 06:24:47Z. The last API record (`development_api/2687_244d99f4`) ended at 06:24:41Z. Testing ran from 06:28:10Z to 16:51:07Z. No file under `development/`, `development_job/`, `study/`, `selected_policies/` or `development_api/` was modified after 06:28:10Z. The study ledger has 0 attempts, matching `development_attempt_count` = 0. `evaluation/api_usage.json` reports 0 requests.
- **The reruns used the unchanged policy.** All 82 outage reruns (`retry_records/*/*.json`) kept the same seed and got a new reservation id, so the ledger ids span 1–882 for 800 rows. `operator/retry_provider_outage.py:45` calls `verify_frozen()` before each requeue. The rerun programs match the frozen hashes.
- **No state crossed test episodes.** Each episode ran in a new worker process (800 distinct pids) and a new `docker run` (`aspire/isolated.py:177-203`, `worker_command.json` in every run). Each run had `--network=none` and `--read-only`, a tmpfs `/tmp`, and read-only mounts of a per-episode temporary runtime and the program. All host outputs go to per-episode run directories. The shared providers (`ASPIRE/src/cap_harness/providers/*/service.py`) keep only model and planner instances. cuRobo purges and reloads the scene on every planning request (`curobo/service.py:509-526, 565-569`), and IK requests carry no scene (`curobo/client.py:135`).
- **No test results were fed back.** `selection.json` and all 16 policies predate the freeze and are byte-identical in `selected_policies/` and `evaluation/policy/`. `development_complete.json` has `test_feedback_to_development: false`.

Notes
- Episodes re-run after a provider outage used the same frozen policy. The reported score keeps the original outcome of every episode in which the policy had already acted.
- The scheduler and monitor overlay (`<root>/operator/provider_scheduler/`, `full_study.py.patch`, `unified_study.py.patch`) changes only dispatch and provider restarts. Workers import the frozen `runtime_source`.
- The GPU0 SAM3 provider restarted at 15:59:27Z (`<root>/runtime/providers/gpu0_sam3.json`) with the same venv and checkpoint.
- The provider venvs, vendor trees and weights were not in the pre-test manifest. A post-test supplement covers them (`freeze_supplement_20261122/aspire_demo_only_providers.json`, with 0 files changed after the first test). A spot `find -newerct` also found 0 changes.
- The development container mounted only this case's paths and ran with `--network=none` (`development/launch_command.json`). The ASPIRE R10 tests ran at the same time but were not reachable from it.

</details>

<details>
<summary><code>Interaction-only</code> · R10 — PASS</summary>

### ASPIRE · Interaction-only · R10 — development audit

**Conclusion: PASS.** Both checks pass, and the logs cover every agent session and every simulator start.

This replaces my earlier H3 FAIL, which no longer applies. The 480-step ButtonUnmaskSwap sentence came from the prompt bundle the run received. That bundle is allowed documentation under protocol rule 2. The current public `PUBLIC_RULES.md` (SHA-256 `04c0fbc53bdfbd1d…`) is byte-identical to the run's copy (`<ROOT>/protocol/PUBLIC_RULES.md`) and to the workspace copy (`docs/robomme-protocol-v9/PUBLIC_RULES.md`).

#### Scope

- **Method:** ASPIRE.
- **Setting:** `Interaction-only`, stored as `interaction-only`.
- **Budget:** R10, meaning at most 10 Plus simulator starts per task across all seeds. This is an independent run, not an extracted checkpoint.
- **Tasks:** all 16.
- **Run ID:** `r10_20261121`, case `aspire/interaction-only/R10`.
  - `<ROOT>` = `r10_20261121`.
  - `<CASE>` = `<ROOT>/aspire/interaction-only/R10`.
- **Auditor / date:** Claude Opus 5.5 (`us.anthropic.claude-opus-5-5`), 2026-11-22. This is a self-review by the RoboMME team's assistant, done read-only. No simulator, generated code or model API was run.
- **Documents the agent received** (workspace snapshot at `OperatorStart`, first event of `<CASE>/development/audit/history/events.jsonl`, hashes verified):
  - Operator `CLAUDE.md`, generated by `runtime_source/aspire/campaign.py:182-233`.
  - Public prompt bundle v9 under `docs/robomme-protocol-v9/`:
    - `README.md`, `BENCHMARK.md`, `DEMOS.md`, `ASPIRE.md`, `PUBLIC_RULES.md`, `RUNTIME.md`.
    - `bundle.json`, `manifest.json` and the synthetic `fixtures/`.
  - `docs/robomme-plus.md`, which is the same file as the bundle's `ASPIRE.md`.
  - ASPIRE's own upstream method material, which contains no RoboMME-Plus content:
    - `docs/api-reference.md` and `docs/run-artifacts.md`.
    - `.claude/skills/iterative-debugging/{main-agent,subagent,verifier}-prompt.md`, `skills/*.md` and `scripts/*.py`.
  - `campaign/tasks.json` and `campaign/campaign.json`, which hold task names and training episode indices only.

#### Check 1: no forbidden information — PASS

No forbidden read was observed in the inspected logs. The harness also kept every forbidden item out of reach.

**Plus source and other methods' solutions.**
- The agent container is launched with `--network=none --read-only --cap-drop=ALL` (`campaign.py:271-293`).
- The deployed mounts in all 3 launches (`<CASE>/development/launch_commands.jsonl`) are limited to:
  - this case's workspace, `claude-home` and `published/`;
  - the bridge and RPC socket;
  - the CLI binary and the audit hook.
- The program worker container is also offline and read-only, and receives only the ASPIRE contract files (`isolated.py:71-80,193-203`).
- The image file inventory has no `robomme`, `mani_skill` or `sapien` paths (`reaudit_20261122/evidence/image_files.jsonl.gz`). The preflight probe agrees (`<ROOT>/audit/preflight.json`).
- I scanned all 2,046 PreToolUse inputs. The only hits are the agent's own task directories and code comments. `find / -name runtime.py` returned nothing, and `import cap_harness` failed.

**Hidden simulator state.**
- The observation translator passes only RGB-D, calibration, robot state, the goal and public success/failure/error feedback (`benchmark_io/episode.py:52-74,173-178`).
- Programs can only call the public tool registry plus three `robomme_plus.*` metadata/video tools (`cap_harness/registry.py:78-81`, `isolated.py:175-185`).
- RPC replies and published runs contain only public fields (`campaign.py:142-151`; `run_replies.json` keys).

**Test data.**
- The study holds only the released training records (`study/study.json`).
- Test metadata lives only in the operator's evaluation manifest, which was written after development ended (`evaluation.py:40-77`) and never mounted.
- No test seed, metadata or result appears in the workspace, transcripts or published runs.
- Policy selection used training rollouts only (`selected_policies/selection.json`, `development_result.json`).

**Demonstrations.**
- In `interaction-only`, demonstrations are neither exported nor mounted (`campaign.py:172-174,283-284`).
- No `demonstrations/` directory exists in the development tree, and no tool accessed recordings.

**Prompts, skills and notes.**
- The operator wrote three messages: the launch prompt and two deadline wrap-up prompts (`deadline_events.jsonl`). None contains task parameters, and neither does the operator `CLAUDE.md`.
- `PUBLIC_RULES.md` (button deadlines 90/180; the ButtonUnmaskSwap 480-step motion-limit sentence) is part of the received, allowed bundle.
- So the agent's use of 480 in `task_2/task_analysis.md:28` and `task_2/code/v02_wrist_fusion.py:621` / `v03_freeze_pick.py:648` traces to allowed documentation.
- Other task values in the agent's notes come from its own rollouts.

#### Check 2: rollout budget — PASS

- **Ledger:** `study/attempts.sqlite3`, opened `mode=ro`, has 157 reservations.
  - Each is reserved atomically before the simulator is built (`protocol.py:103-138`, `episode.py:95`).
  - Failed setup stays charged, and reset retries are capped at 1 (`episode.py:120-128`).
  - The budget is fixed at 10 per task.
- **Reconciliation:** the 157 ledger rows match, one-to-one by attempt ID, 157 run directories with native `attempt.json`, `outcome.json` and `events.jsonl` (exactly one reset each).
  - Outcomes and step counts are identical.
  - `bridge.log` shows 157 `POST /api/run` calls.
  - No attempt is left unclosed or duplicated.
  - Each `policy_revision` is the SHA-256 of the submitted code, archived before its run.
- **Coverage:** starts include probes, debugging, verification and ablations. Two VideoUnmaskSwap starts whose recorder crashed after the episode are counted.
- **No bypass:** I found no start path outside the ledger. The program worker has no simulator and no reset tool.
- **Seeds:** every start used a released training seed. Ledger seeds equal `study.json` and `benchmark/src/robomme_plus/env_metadata/train`, and are disjoint from the test seeds. Full episode settings are enforced at each start (`episode.py:110-115`).

#### Evidence coverage

- **Agent sessions:** all present and reconciled with the hook records.
  - 1 main transcript (`d81e53e5…`, launched once and resumed twice) and 16 subagent transcripts (17 files).
  - 2,059 unique tool uses reconcile with 4,191 hook events:
    - 14 subagent `Write` calls were rejected inside the CLI and appear only in the transcripts.
    - 1 Bash call at a deadline stop appears only in the hooks.
    - 2 commands were stopped by the deadline, so they have no completion record.
- **Simulator starts:** all 157 have native records.
- **Model API:** 2,047 request records in `development_api/` match the bridge log.

#### Per-task rollouts

| Task | Budget | Starts | Seed validity (released train episodes used) |
| --- | --- | --- | --- |
| BinFill | 10 | 10 | valid; ep 0/1/2 = 3/5/2 |
| ButtonUnmask | 10 | 10 | valid; 5/3/2 |
| ButtonUnmaskSwap | 10 | 10 | valid; 4/3/3 |
| InsertPeg | 10 | 10 | valid; 5/3/2 |
| MoveCube | 10 | 10 | valid; ep 0–4 = 3/3/2/1/1 |
| PatternLock | 10 | 10 | valid; 3/4/3 |
| PickHighlight | 10 | 10 | valid; 5/2/3 |
| PickXtimes | 10 | 10 | valid; 4/3/3 |
| RouteStick | 10 | 10 | valid; 2/2/6 |
| StopCube | 10 | 10 | valid; 5/3/2 |
| SwingXtimes | 10 | 10 | valid; 4/3/3 |
| VideoPlaceButton | 10 | 10 | valid; 5/3/2 |
| VideoPlaceOrder | 10 | 10 | valid; 4/3/3 |
| VideoRepick | 10 | 7 | valid; 3/2/2 |
| VideoUnmask | 10 | 10 | valid; 4/3/3 |
| VideoUnmaskSwap | 10 | 10 | valid; 5/3/2 |
| **Total** | **160 max** | **157** | all valid |

#### Notes

- This report supersedes my earlier H3 FAIL in `harness_audit.md` and `development_audit.md`, which are kept unchanged. That finding rested on the bundle copy that was edited at the time and has since been restored.
- Two VideoUnmaskSwap starts (`attempt_002_924ec2dd`, `attempt_003_9d66c3a9`) crashed in the recorder (`KeyError: 'episode_terminated'`) after the episode had ended. They are charged and recorded, but no published directory was produced.
- The campaign receives a `--demos` path even in `interaction-only`. It is unused and not mounted.
- Freezing, provider pinning, test-time infrastructure errors and API cost accuracy are not judged under this skill. Earlier observations on them remain in the prior reports.

#### Check 3 (added 2026-11-23 under the updated skill)

**Check 3: no test-time optimization — PASS**

ASPIRE · Interaction-only · budget 10. `<root>` = `r10_20261121`, `<case>` = `<root>/aspire/interaction-only/R10`. Times are UTC. Inspection was read-only; no code, simulator or API was run.

- **The frozen policy ran every test episode.** All 800 `submitted_program.py` files under `<case>/evaluation/episodes/*/episode_*/artifacts/**/` hash to `policy_sha256` in `<case>/evaluation/frozen_evaluation.json` (sha `af1fd98d…`, created 02:15:31Z), with 0 mismatches. The per-run `source/provenance.json` gives 16 program hashes × 50 episodes and one dependency-lock hash (`05b40506…`, equal to the hashed `configs/dependency-lock.json`). All 800 runs used image `sha256:d143d08a…`, the one pinned in `<root>/study_plan.json`.
- **Nothing hashed changed during testing.** `verify_frozen` checks the policy, all 757 execution files and the development attempt count on every reservation (`runtime_source/benchmark_io/evaluation.py:79-93`). Today, all policy, `selected_policies/` and execution files match the manifest. None has an mtime or ctime at or after the first test (04:05:13Z). The manifest itself was not rewritten (no `operator_harness_corrections`).
- **There was no development during testing.** Development completed and the policy was frozen at 02:15:31Z (`development_complete.json`). The last API record ended at 02:15:22Z (`development_api/2047_42438980`). Testing ran from 04:05:13Z to 06:27:39Z. No file under `development/`, `development_job/`, `study/`, `selected_policies/` or `development_api/` was modified after 04:05:13Z. The study ledger has 157 attempts, the last ending at 02:11:13Z, which equals `development_attempt_count` = 157. The queue (`<root>/queue.sqlite3`) holds one develop job, finished at 02:15:36Z.
- **There were no reruns.** `<case>/evaluation/evaluation_attempts.sqlite3` has 800 rows with contiguous ids 1–800. All are closed, all have purpose `frozen_full_evaluation_v2`, and all seeds equal the frozen records. The 6 provider-infrastructure failures in `results.json` stay as recorded. `oom_retry_20261122/plan.json` does not target this case.
- **No state crossed test episodes.** Each episode ran in a new worker process (800 distinct pids) and a new `docker run` (`runtime_source/aspire/isolated.py:177-203`). Each run had `--network=none` and `--read-only`, a tmpfs `/tmp`, and only read-only mounts (`worker_command.json` in all 800 runs). Host outputs are per episode. The shared providers keep only model and planner instances: cuRobo reloads the scene on every planning request (`curobo/service.py:509-526, 565-569`), and IK requests send no scene (`curobo/client.py:135`).
- **No test results were fed back.** The frozen `selection.json` and policies predate the freeze and are byte-identical to `selected_policies/`. The ASPIRE R10 cases were excluded from evaluation until 04:05:12Z (`orchestration_20261121/transitions.jsonl`). `test_feedback_to_development` is `false`.

Notes
- The GPU1 SAM3 provider restarted during testing, at 06:11:22Z (`<root>/runtime/providers/gpu1_sam3.json`). It used the same interpreter, module and checkpoint symlink as before, and its code loads from the hashed `runtime_source/aspire/ASPIRE/src`. This is runtime continuity, not a policy change.
- Three `.pyc` files in `runtime_source/benchmark_io/__pycache__/` (evaluation, `__init__`, protocol) were written at 04:35Z. They are not hashed, but their headers match the sources and their code objects are structurally equal to the compiled hashed sources. This is evaluation tooling, not policy code.
- The provider venvs, vendor trees and weights were not in the pre-test manifest. A post-test supplement covers them (`freeze_supplement_20261122/aspire_providers.json`, with 0 changes after the first test), and a spot `find -newerct` also found 0 changes.
- The R10 scheduler overlay `<root>/runtime/operator_scheduler_20/benchmark_io/unified_study.py` is identical to the frozen copy.
- The R3/R5 extractions (`extracted_r35_20261121`) are out of scope.

</details>

<details>
<summary><code>Hybrid</code> · R10 — PASS</summary>

### ASPIRE · Hybrid · R10: development audit

**Conclusion: PASS.** Both checks pass. Check 1: the harness kept forbidden information out of the agent's reach, and no forbidden read was observed in the inspected logs. Check 2: all 134 Plus simulator starts are in the ledger and reconcile with the native records. Every task is at or below 10 starts, and every start uses a released training seed. The logs cover every agent session and every start.

This report applies the simplified skill `doc/skills/audit_robomme_plus.md` and replaces neither of my earlier reports in this directory (`harness_audit.md`, `development_audit.md`). Those reports failed H3 under the earlier rules. They treated the 480-step sentence in `PUBLIC_RULES.md` as beyond the allowed documentation because the then-current bundle omitted it. The current bundle file (`scripts/plus_eval/benchmark_io/prompts/current/PUBLIC_RULES.md`) has since been restored and is byte-identical to the file the run received (SHA-256 `04c0fbc53bdfbd1d70009e830b5baf095d39073e6bda592a743ed1ae85e94755`). The new skill also allows the bundle the run received.

#### Scope

- **Method / setting / budget:** ASPIRE, **Hybrid** (stored identifier `demo&interaction`), at most 10 Plus starts per task across all training seeds, workers and sessions. The run is independent; no checkpoint was extracted.
- **Tasks:** all 16.
- **Run ID:** `r10_20261121/aspire/demo&interaction/R10`, with study root `r10_20261121`.
- **Versions:**
  - Runtime manifest SHA-256 `29eb5653…`; 722/722 files verified.
  - `robomme_plus` source fingerprint `b34d910c…`.
  - ASPIRE `fa00f0ef…` plus hashed patches.
  - Agent image `sha256:d143d08a…`.
  - Model `us.anthropic.claude-opus-5-5`, effort high.
  - Protocol bundle v9.
- **Auditor / date:** Opus 5.5 (`us.anthropic.claude-opus-5-5`), 2026-11-22. This is a **self-review by the RoboMME team's assistant**. Inspection was read-only: SQLite opened with `mode=ro`; no code was executed and no simulator or API was called.
- **Documents the agent received:**
  - The v9 protocol bundle in `/workspace/docs/robomme-protocol-v9/`: `README.md`, `BENCHMARK.md`, `DEMOS.md`, `ASPIRE.md`, `PUBLIC_RULES.md`, `RUNTIME.md`, `bundle.json`, `manifest.json`, synthetic `fixtures/`. It is byte-identical to `<root>/protocol/`. Against the current bundle, `PUBLIC_RULES.md` and `RUNTIME.md` are identical; the other files differ only in setting-name spelling.
  - `docs/robomme-plus.md`, a copy of `ASPIRE.md`.
  - ASPIRE's native `docs/api-reference.md` and `docs/run-artifacts.md`, and the upstream `.claude/skills/iterative-debugging/` prompts and library. These contain no Plus task names or values.
  - The generated `CLAUDE.md`, `campaign/tasks.json` and `campaign/campaign.json`.
  - Three operator prompts: the initial coordinator prompt, the deadline wrap-up prompt (`development/deadline_events.jsonl`) and the MoveCube finalization prompt (`<root>/audit/movecube_submission_20261121/prompt.txt`).

#### Check 1: no forbidden information — PASS

- **Plus source and other methods' solutions.**
  - *Reachable:* nothing.
    - The coding-agent container ran `--network=none --read-only --cap-drop=ALL` (`runtime_source/aspire/campaign.py:271`).
    - Its only mounts were this case's workspace, claude-home, bridge, RPC socket, published runs, Hybrid demos, the CLI and the audit hook. See `development/launch_commands.jsonl` (2 launches) and `audit/movecube_submission_20261121/launch_command.json`.
    - The image inventory (`reaudit_20261122/evidence/image_inventory.json`; layer digests match the live `docker image inspect`) lists 34,820 files with no `robomme`, `robomme_plus`, `mani_skill` or `sapien` paths, and the preflight package probe was negative (`audit/preflight.json`).
    - Generated programs ran in a separate offline container that held only the public `cap_harness` contracts. The host serves only the public tool registry (`runtime_source/aspire/isolated.py:89`, `:194`).
    - The 43 files of the first workspace snapshot all trace to the upstream ASPIRE files and the protocol bundle, apart from the four generated files listed above. The Interaction-only or RAX workspaces were never mounted.
  - *Used:* no forbidden read was observed in the inspected logs. I searched all 2,174 tool inputs, the native transcripts and `agent_stdout.jsonl` for Plus source paths, `env_metadata`, task constants, `BenchmarkEnvBuilder`, simulator imports, ledger and study files, and package fetches. The hits were benign: variable names and a host path echoed in a published run manifest.
- **Hidden simulator state.**
  - The trusted translator returns only RGB-D, intrinsics, camera poses, joint/EEF/gripper state, goal, step count and public success/error feedback (`runtime_source/benchmark_io/episode.py:52`, `:161`, `:173`).
  - The adapter's metadata is limited to task, language, control frequency, `max_steps` and robot calibration (`ASPIRE/src/cap_harness/robomme_plus/adapter.py:154`, `:158`).
  - No published run JSON/JSONL contains object-pose, oracle, subgoal or simulator-state keys.
- **Test seeds, metadata and results.**
  - The run endpoint accepts only released training indices (`campaign.py:117`), and the ledger accepts only released training seeds (`benchmark_io/protocol.py:103`). Test metadata and results are host-only.
  - Development activity ended at 2026-11-22T02:23:45Z (last hook event), before the policy freeze at 02:24:38Z and the first test start at 04:05:13Z. No development step or policy choice could use test results.
- **Demonstrations.** Hybrid permits released demonstrations.
  - The export is gated by `study.require_demos()` (`benchmark_io/demonstrations.py:12`).
  - It holds RGB JPEGs and manifests only.
  - All 16 source hashes match the pinned H5 data in `requested_dataset_pin.json`.
- **Prompts, skills and notes.**
  - The button deadlines (90/180) and the ButtonUnmaskSwap 480-step motion-limit sentence come from the bundle's `PUBLIC_RULES.md:16`, which is allowed documentation under protocol rule 2.
  - The coordinator's copy in `workspace/campaign/prompts/task_2_block.md:153` restates that allowed text.
  - The other operator prompts carry no task parameters.
  - The MoveCube colour roles and other agent-written values were measured from released demonstrations and live training rollouts.

#### Check 2: rollout budget — PASS

- **Ledger.** `study/attempts.sqlite3` has 134 attempts and 0 unended, each with a provenance row. Reservations are made atomically before the simulator is built (`episode.py:95`; `protocol.py:112–118`). Training resets are limited to one internal attempt (`episode.py:128`).
- **Native records.** The 134 attempt IDs reconcile one-to-one with the 134 runs under `development/runs/*/attempt_*`: task, seed, episode, success and control steps all match, and each `events.jsonl` begins with `reset`. No run lacks a ledger entry.
- **Requests that returned errors.** Of 135 `/api/run` POSTs in `development/bridge.log`, 3 errors are charged attempts 39, 45 and 47. The fourth was rejected before reservation at 01:50:24Z.
- **No other simulator path.** The agent containers held no simulator, so the ledgered `/api/run` path was the only way to start the simulator.
- **Budget and seeds.** Every task is at or below 10 starts. Every start's seed and full episode record equal the released training metadata (0 mismatches) and none is a test seed.
- **Code before starts.** All 134 submitted programs were archived in the code history before their start, as recorded by `policy_revision`.

#### Evidence coverage

- **Sessions:** one coordinator session, `b087d7db-…`, launched three times (initial, deadline resume, operator finalization), plus 16 subagent transcripts.
- **Model requests:** 2,159 relay records in `development_api/`, reconciled with the 2,160 bridge log entries. The extra entry is one request logged twice at the deadline restart.
- **Tool events:** 4,446 hook events.
- **Code history:** 2,295 snapshots, with no missing objects.
- **Starts:** all 134 have ledger and native records.

#### Per-task rollouts

| Task | Budget | Starts | Seed validity |
| --- | --- | --- | --- |
| BinFill | 10 | 10 | Released training seeds |
| ButtonUnmask | 10 | 5 | Released training seeds |
| ButtonUnmaskSwap | 10 | 10 | Released training seeds |
| InsertPeg | 10 | 10 | Released training seeds |
| MoveCube | 10 | 5 | Released training seeds |
| PatternLock | 10 | 10 | Released training seeds |
| PickHighlight | 10 | 8 | Released training seeds |
| PickXtimes | 10 | 8 | Released training seeds |
| RouteStick | 10 | 10 | Released training seeds |
| StopCube | 10 | 9 | Released training seeds |
| SwingXtimes | 10 | 7 | Released training seeds |
| VideoPlaceButton | 10 | 10 | Released training seeds |
| VideoPlaceOrder | 10 | 10 | Released training seeds |
| VideoRepick | 10 | 9 | Released training seeds |
| VideoUnmask | 10 | 8 | Released training seeds |
| VideoUnmaskSwap | 10 | 5 | Released training seeds |
| **Total** | — | **134** | All valid |

#### Notes

- A recorder `KeyError: 'episode_terminated'` dropped the agent replies for attempts 39, 45 and 47 (ButtonUnmaskSwap). Their ledger and native records are intact, and all three are counted.
- One tool call was interrupted by the 01:50:16Z deadline restart and has no completion event. It is recorded as interrupted, not completed.
- An operator-requested MoveCube submission-only session (02:22–02:23Z) copied the existing `code/v04_wrist_unwind.py` byte-for-byte to `fix_code.py` and added `final_submission.md`. It made no starts and was logged.
- The first job attempt failed on a provider-port conflict before any agent session started (0 API calls, 0 starts). The fix is disclosed in `audit/corrections/provider_ports/`.
- The bundle also includes non-Markdown synthetic fixtures. They contain blank images and invented values, with no task data.

#### Check 3 (added 2026-11-23 under the updated skill)

**Check 3: no test-time optimization — PASS**

ASPIRE · Hybrid (`demo&interaction`) · budget 10. `<root>` = `r10_20261121`, `<case>` = `<root>/aspire/demo&interaction/R10`. Times are UTC. Inspection was read-only; no code, simulator or API was run.

- **The frozen policy ran every test episode.** All 800 `submitted_program.py` files under `<case>/evaluation/episodes/*/episode_*/artifacts/**/` hash to `policy_sha256` in `<case>/evaluation/frozen_evaluation.json` (sha `7fcaf767…`, created 02:24:38Z), with 0 mismatches. The per-run `source/provenance.json` gives 16 program hashes × 50 episodes and one dependency-lock hash (`05b40506…`, equal to the hashed `configs/dependency-lock.json`). All 800 runs used image `sha256:d143d08a…`, the one pinned in `<root>/study_plan.json`.
- **Nothing hashed changed during testing.** `verify_frozen` checks the policy, all 757 execution files and the development attempt count on every reservation (`runtime_source/benchmark_io/evaluation.py:79-93`). Today, all policy, `selected_policies/` and execution files match the manifest. None has an mtime or ctime at or after the first test (04:05:13Z). The manifest was not rewritten.
- **There was no development during testing.** The last API record ended at 02:23:43Z (`development_api/2159_eae87a1e`). The freeze and `development_complete.json` are at 02:24:38Z. Testing ran from 04:05:13Z to 06:07:31Z. No file under `development/`, `development_job/`, `study/`, `selected_policies/` or `development_api/` was modified after 04:05:13Z. The study ledger has 134 attempts, the last ending at 02:02:17Z, which equals `development_attempt_count` = 134.
- **The MoveCube submission handoff happened before the freeze and testing.** It ran from 02:22:40Z to 02:23:45Z (`<root>/audit/movecube_submission_20261121/{before,completed,freeze_handoff}.json`). It copied the existing `code/v04_wrist_unwind.py` (`77b60c2d…`), which is the frozen `MoveCube.py` hash. The other 15 frozen hashes equal `before.json:other_policies`, and attempts stayed at 134.
- **There were no reruns.** `<case>/evaluation/evaluation_attempts.sqlite3` has 800 rows with contiguous ids 1–800. All are closed, all have purpose `frozen_full_evaluation_v2`, and all seeds equal the frozen records. `results.json` has 0 infrastructure failures. `oom_retry_20261122/plan.json` does not target this case.
- **No state crossed test episodes.** Each episode ran in a new worker process (800 distinct pids) and a new `docker run` (`runtime_source/aspire/isolated.py:177-203`). Each run had `--network=none` and `--read-only`, a tmpfs `/tmp`, and only read-only mounts (`worker_command.json` in all 800 runs). The frozen study refuses demonstrations at test time (`runtime_source/benchmark_io/evaluation.py:106`). Host outputs are per episode. The shared providers keep only model and planner instances, and cuRobo reloads the scene on every planning request (`curobo/service.py:509-526, 565-569`).
- **No test results were fed back.** `evaluation/policy/selection.json` and all policies predate the freeze and match `selected_policies/`. The ButtonUnmaskSwap entry points to `../deadline_frozen/` and was also set before the freeze. Evaluation for this case opened only at 04:05:12Z (`orchestration_20261121/transitions.jsonl`). `test_feedback_to_development` is `false`.

Notes
- The GPU1 SAM3 restart at 06:11:22Z (`<root>/runtime/providers/gpu1_sam3.json`) came after this case's last test ended (06:07:31Z). It used the same argv and checkpoint.
- Three unhashed `.pyc` files in `runtime_source/benchmark_io/__pycache__/` were written at 04:35Z. They are structurally equal to the compiled hashed sources, and they are evaluation tooling, not policy code.
- The provider venvs, vendor trees and weights were not in the pre-test manifest. A post-test supplement covers them (`freeze_supplement_20261122/aspire_providers.json`, with 0 changes after the first test), and a spot `find -newerct` also found 0 changes.
- Whether the operator's MoveCube prompt was appropriate is a development-phase question for Checks 1 and 2, not Check 3.

</details>
