# RAX on RoboMME-Plus (eval_by_RoboMME_team)

- **Track:** Agentic.
- **Code:** [Adapter and run instructions](../../../scripts/plus_eval/agentic_methods/rax/README.md); [final solutions](../../../scripts/plus_eval/agentic_methods/rax/solutions/README.md).
- **Development logs:** `<DEVELOPMENT_LOGS_URL>` (public upload pending).
- **Paper / website / blog:** `<PAPER_URL>` · `<WEBSITE_URL>` · `<BLOG_URL>`
- **Development model:** Claude Opus 5.5, high reasoning effort.
- **Settings:** `Demo-only` (0 simulator starts); `Interaction-only` and `Hybrid` (at most 10 starts per task).
- **Evaluation:** final policies on all 800 test episodes (50 per task); no model API calls at test time.

## Results

API costs are rough estimates, rounded to the nearest $10, and exclude local compute.

| Setting | Rollouts per task | Success | Development starts | Development API cost | Evaluation API cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Demo-only` | 0 | 139/800 (17.38%) | 0 | ≈$80 | $0 |
| `Interaction-only` | 10 | 372/800 (46.50%) | 123 | ≈$90 | $0 |
| `Hybrid` | 10 | 327/800 (40.88%) | 121 | ≈$100 | $0 |

| Task | `Demo-only` | `Interaction-only` | `Hybrid` |
| --- | ---: | ---: | ---: |
| BinFill | 15/50 (30.00%) | 31/50 (62.00%) | 25/50 (50.00%) |
| PickXtimes | 0/50 (0.00%) | 35/50 (70.00%) | 20/50 (40.00%) |
| SwingXtimes | 3/50 (6.00%) | 33/50 (66.00%) | 33/50 (66.00%) |
| StopCube | 23/50 (46.00%) | 26/50 (52.00%) | 33/50 (66.00%) |
| VideoUnmask | 0/50 (0.00%) | 28/50 (56.00%) | 29/50 (58.00%) |
| VideoUnmaskSwap | 0/50 (0.00%) | 38/50 (76.00%) | 32/50 (64.00%) |
| ButtonUnmask | 26/50 (52.00%) | 32/50 (64.00%) | 34/50 (68.00%) |
| ButtonUnmaskSwap | 2/50 (4.00%) | 15/50 (30.00%) | 10/50 (20.00%) |
| PickHighlight | 1/50 (2.00%) | 18/50 (36.00%) | 19/50 (38.00%) |
| VideoRepick | 31/50 (62.00%) | 33/50 (66.00%) | 32/50 (64.00%) |
| VideoPlaceButton | 11/50 (22.00%) | 31/50 (62.00%) | 19/50 (38.00%) |
| VideoPlaceOrder | 12/50 (24.00%) | 16/50 (32.00%) | 18/50 (36.00%) |
| MoveCube | 2/50 (4.00%) | 0/50 (0.00%) | 1/50 (2.00%) |
| InsertPeg | 0/50 (0.00%) | 16/50 (32.00%) | 0/50 (0.00%) |
| PatternLock | 10/50 (20.00%) | 10/50 (20.00%) | 10/50 (20.00%) |
| RouteStick | 3/50 (6.00%) | 10/50 (20.00%) | 12/50 (24.00%) |
| **Overall** | **139/800 (17.38%)** | **372/800 (46.50%)** | **327/800 (40.88%)** |

### Reference results with fewer rollouts

Optional; not part of the leaderboard or the audit record below.

| Setting | 1 rollout | 3 rollouts | 5 rollouts |
| --- | ---: | ---: | ---: |
| `Interaction-only` | 26/800 (3.25%) | pending | pending |
| `Hybrid` | 156/800 (19.50%) | pending | pending |

<details>
<summary>Per-task success with 1 rollout</summary>

| Task | `Interaction-only` | `Hybrid` |
| --- | ---: | ---: |
| BinFill | 6/50 (12.00%) | 36/50 (72.00%) |
| PickXtimes | 0/50 (0.00%) | 10/50 (20.00%) |
| SwingXtimes | 0/50 (0.00%) | 10/50 (20.00%) |
| StopCube | 1/50 (2.00%) | 15/50 (30.00%) |
| VideoUnmask | 2/50 (4.00%) | 0/50 (0.00%) |
| VideoUnmaskSwap | 0/50 (0.00%) | 0/50 (0.00%) |
| ButtonUnmask | 6/50 (12.00%) | 0/50 (0.00%) |
| ButtonUnmaskSwap | 0/50 (0.00%) | 1/50 (2.00%) |
| PickHighlight | 0/50 (0.00%) | 7/50 (14.00%) |
| VideoRepick | 0/50 (0.00%) | 17/50 (34.00%) |
| VideoPlaceButton | 0/50 (0.00%) | 13/50 (26.00%) |
| VideoPlaceOrder | 1/50 (2.00%) | 27/50 (54.00%) |
| MoveCube | 0/50 (0.00%) | 2/50 (4.00%) |
| InsertPeg | 0/50 (0.00%) | 1/50 (2.00%) |
| PatternLock | 0/50 (0.00%) | 10/50 (20.00%) |
| RouteStick | 10/50 (20.00%) | 7/50 (14.00%) |
| **Overall** | **26/800 (3.25%)** | **156/800 (19.50%)** |

</details>

## Agentic development audit record

Audited with the [development audit](../agentic_development.md#step-8-audit): no forbidden information, and at most
the allowed simulator starts. Auditor: Claude Opus 5.5, 2026-11-22 (self-review by the RoboMME team's assistant).

| Setting | Budget per task | Development starts | 1. No forbidden information | 2. Rollout budget | Conclusion | Full report |
| --- | ---: | ---: | --- | --- | --- | --- |
| `Demo-only` | 0 | 0 | PASS | PASS | **PASS** | [audit.json](../audits/current/rax/demo-only/R0/audit.json) |
| `Interaction-only` | 10 | 123 | PASS | PASS | **PASS** | [audit.json](../audits/current/rax/interaction-only/R10/audit.json) |
| `Hybrid` | 10 | 121 | PASS | PASS | **PASS** | [audit.json](../audits/current/rax/hybrid/R10/audit.json) |

| Task | `Demo-only` starts | `Interaction-only` starts | `Hybrid` starts |
| --- | ---: | ---: | ---: |
| BinFill | 0 | 8 | 5 |
| PickXtimes | 0 | 9 | 8 |
| SwingXtimes | 0 | 9 | 6 |
| StopCube | 0 | 9 | 7 |
| VideoUnmask | 0 | 7 | 5 |
| VideoUnmaskSwap | 0 | 7 | 7 |
| ButtonUnmask | 0 | 6 | 9 |
| ButtonUnmaskSwap | 0 | 7 | 10 |
| PickHighlight | 0 | 10 | 8 |
| VideoRepick | 0 | 6 | 5 |
| VideoPlaceButton | 0 | 10 | 7 |
| VideoPlaceOrder | 0 | 9 | 4 |
| MoveCube | 0 | 5 | 10 |
| InsertPeg | 0 | 5 | 10 |
| PatternLock | 0 | 8 | 10 |
| RouteStick | 0 | 8 | 10 |
| **Total** | **0** | **123** | **121** |

All starts used released training seeds.
