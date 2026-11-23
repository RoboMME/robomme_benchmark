# ASPIRE on RoboMME-Plus (eval_by_RoboMME_team)

- **Track:** Agentic.
- **Code:** [Adapter and run instructions](../../../scripts/plus_eval/agentic_methods/aspire/README.md); [frozen R10 solutions](../../../scripts/plus_eval/agentic_methods/aspire/solutions/README.md).
- **Development logs:** Public upload pending (`<DEVELOPMENT_LOGS_URL>`). Complete code history, commands, responses, attempts and usage records are retained.
- **Paper:** `<PAPER_URL>`
- **Website:** `<WEBSITE_URL>`
- **Blog:** `<BLOG_URL>`
- **Development model:** Claude Opus 5.5 via AWS Bedrock, high reasoning effort.
- **Runs:** `demo_only_20261121` for Demo-only (released demonstrations, zero simulator starts); `r10_20261121` for Interaction-only and Hybrid (at most 10 development starts per task, shared across all training seeds). 150-minute task deadline.
- **Evaluation:** Frozen policies on the 800 released tests, with default variations. No paid model API calls during frozen execution.

## Results

| Setting | Budget per task | Success | Training starts | Development API estimate | Evaluation API cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Demo-only` | 0 | 140/800 (17.50%) | 0 | ≈$210 | $0 |
| `Interaction-only` | 10 | 302/800 (37.75%) | 157 | ≈$160 | $0 |
| `Hybrid` | 10 | 342/800 (42.75%) | 134 | ≈$160 | $0 |

API estimates are rounded to the nearest $10 and exclude local compute. Recorded usage is retained in the logs; missing provider-error usage remains unknown.
Requests with incomplete usage records: Demo-only: 1, Interaction-only: 1, Hybrid: 2. Estimates use recorded usage plus a small allowance for interrupted requests; exact missing usage is unknown.

Failures, timeouts and infrastructure errors remain in the denominator. In Demo-only, 42 test episodes that failed before the policy's first action during a SAM3 outage were rerun under the infrastructure-error rule; 40 that failed after the policy acted keep their original outcomes.

| Task | Demo-only | Interaction-only | Hybrid |
| --- | ---: | ---: | ---: |
| BinFill | 18/50 (36.00%) | 25/50 (50.00%) | 41/50 (82.00%) |
| ButtonUnmask | 23/50 (46.00%) | 35/50 (70.00%) | 28/50 (56.00%) |
| ButtonUnmaskSwap | 2/50 (4.00%) | 4/50 (8.00%) | 14/50 (28.00%) |
| InsertPeg | 0/50 (0.00%) | 12/50 (24.00%) | 4/50 (8.00%) |
| MoveCube | 1/50 (2.00%) | 3/50 (6.00%) | 1/50 (2.00%) |
| PatternLock | 0/50 (0.00%) | 3/50 (6.00%) | 10/50 (20.00%) |
| PickHighlight | 0/50 (0.00%) | 17/50 (34.00%) | 11/50 (22.00%) |
| PickXtimes | 0/50 (0.00%) | 24/50 (48.00%) | 11/50 (22.00%) |
| RouteStick | 10/50 (20.00%) | 10/50 (20.00%) | 23/50 (46.00%) |
| StopCube | 17/50 (34.00%) | 40/50 (80.00%) | 29/50 (58.00%) |
| SwingXtimes | 17/50 (34.00%) | 27/50 (54.00%) | 33/50 (66.00%) |
| VideoPlaceButton | 11/50 (22.00%) | 9/50 (18.00%) | 9/50 (18.00%) |
| VideoPlaceOrder | 10/50 (20.00%) | 15/50 (30.00%) | 20/50 (40.00%) |
| VideoRepick | 0/50 (0.00%) | 24/50 (48.00%) | 36/50 (72.00%) |
| VideoUnmask | 31/50 (62.00%) | 27/50 (54.00%) | 34/50 (68.00%) |
| VideoUnmaskSwap | 0/50 (0.00%) | 27/50 (54.00%) | 38/50 (76.00%) |

## Optional results: smaller rollout budgets

These budgets are supplementary and do not enter the leaderboard; they are not part of the audits below.
Each result evaluates all 800 tests.

- **R1:** independent one-attempt development runs from the earlier full study (2026-11-16 to 11-18). That study used an earlier training-data snapshot, which differs from the current release in 10 recordings and nine execution limits across ButtonUnmaskSwap, MoveCube, PickHighlight, RouteStick and VideoRepick.
- **R3 and R5:** results pending. Budget construction: [SAVED CHECKPOINTS / SEPARATE DEVELOPMENT RUNS].

<details>
<summary><code>Interaction-only</code> · per task</summary>

| Task | 1 rollout | 3 rollouts | 5 rollouts |
| --- | ---: | ---: | ---: |
| BinFill | 1/50 (2.00%) | PENDING | PENDING |
| ButtonUnmask | 3/50 (6.00%) | PENDING | PENDING |
| ButtonUnmaskSwap | 1/50 (2.00%) | PENDING | PENDING |
| InsertPeg | 0/50 (0.00%) | PENDING | PENDING |
| MoveCube | 0/50 (0.00%) | PENDING | PENDING |
| PatternLock | 10/50 (20.00%) | PENDING | PENDING |
| PickHighlight | 0/50 (0.00%) | PENDING | PENDING |
| PickXtimes | 7/50 (14.00%) | PENDING | PENDING |
| RouteStick | 10/50 (20.00%) | PENDING | PENDING |
| StopCube | 32/50 (64.00%) | PENDING | PENDING |
| SwingXtimes | 10/50 (20.00%) | PENDING | PENDING |
| VideoPlaceButton | 8/50 (16.00%) | PENDING | PENDING |
| VideoPlaceOrder | 9/50 (18.00%) | PENDING | PENDING |
| VideoRepick | 16/50 (32.00%) | PENDING | PENDING |
| VideoUnmask | 2/50 (4.00%) | PENDING | PENDING |
| VideoUnmaskSwap | 8/50 (16.00%) | PENDING | PENDING |
| **Overall** | **117/800 (14.63%)** | **PENDING** | **PENDING** |

</details>

<details>
<summary><code>Hybrid</code> · per task</summary>

| Task | 1 rollout | 3 rollouts | 5 rollouts |
| --- | ---: | ---: | ---: |
| BinFill | 13/50 (26.00%) | PENDING | PENDING |
| ButtonUnmask | 24/50 (48.00%) | PENDING | PENDING |
| ButtonUnmaskSwap | 0/50 (0.00%) | PENDING | PENDING |
| InsertPeg | 3/50 (6.00%) | PENDING | PENDING |
| MoveCube | 0/50 (0.00%) | PENDING | PENDING |
| PatternLock | 0/50 (0.00%) | PENDING | PENDING |
| PickHighlight | 8/50 (16.00%) | PENDING | PENDING |
| PickXtimes | 13/50 (26.00%) | PENDING | PENDING |
| RouteStick | 0/50 (0.00%) | PENDING | PENDING |
| StopCube | 14/50 (28.00%) | PENDING | PENDING |
| SwingXtimes | 2/50 (4.00%) | PENDING | PENDING |
| VideoPlaceButton | 5/50 (10.00%) | PENDING | PENDING |
| VideoPlaceOrder | 16/50 (32.00%) | PENDING | PENDING |
| VideoRepick | 37/50 (74.00%) | PENDING | PENDING |
| VideoUnmask | 19/50 (38.00%) | PENDING | PENDING |
| VideoUnmaskSwap | 1/50 (2.00%) | PENDING | PENDING |
| **Overall** | **155/800 (19.38%)** | **PENDING** | **PENDING** |

</details>

| Setting | Budget per task | Success | Development API estimate | Evaluation API cost |
| --- | ---: | ---: | ---: | ---: |
| `Interaction-only` | 1 | 117/800 (14.63%) | ≈$130 | $0 |
| `Interaction-only` | 3 | PENDING | PENDING | PENDING |
| `Interaction-only` | 5 | PENDING | PENDING | PENDING |
| `Hybrid` | 1 | 155/800 (19.38%) | ≈$170 | $0 |
| `Hybrid` | 3 | PENDING | PENDING | PENDING |
| `Hybrid` | 5 | PENDING | PENDING | PENDING |

API estimates are rounded to the nearest $10 and exclude local compute. R1 requests with incomplete usage records: Interaction-only: 2, Hybrid: 3.

## Agentic development audit record

Each run was audited after development with the [development audit](../agentic_development.md#step-8-audit),
which checks two things: the agent had no access to forbidden information, and it used
at most the allowed simulator starts. Auditor: Claude Opus 5.5, 2026-11-22; a self-review
by the RoboMME team's assistant. Machine-readable copies are in
[`audits/current/aspire/`](../audits/current/aspire/).

| Setting | Budget per task | Development starts | 1. No forbidden information | 2. Rollout budget | Conclusion |
| --- | ---: | ---: | --- | --- | --- |
| `Demo-only` | 0 | 0 | PASS | PASS | **PASS** |
| `Interaction-only` | 10 | 157 | PASS | PASS | **PASS** |
| `Hybrid` | 10 | 134 | PASS | PASS | **PASS** |

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
- **Model and versions:** Opus 5.5 via Bedrock, high effort. ASPIRE `fa00f0ef…` plus two patches (`<root>/runtime_source_manifest.json`). Agent image `sha256:d143d08a…7f0b` (`<root>/study_plan.json`). Dataset `Yinpei/robomme_plus_data_h5@63cb9b5b…`.
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

</details>
