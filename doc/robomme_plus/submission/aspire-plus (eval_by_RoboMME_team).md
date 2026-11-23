# ASPIRE on RoboMME-Plus (eval_by_RoboMME_team)

- **Track:** Agentic.
- **Code:** [Adapter and run instructions](../../../scripts/plus_eval/agentic_methods/aspire/README.md); [final solutions](../../../scripts/plus_eval/agentic_methods/aspire/solutions/README.md).
- **Development logs:** `<DEVELOPMENT_LOGS_URL>` (public upload pending).
- **Paper / website / blog:** `<PAPER_URL>` · `<WEBSITE_URL>` · `<BLOG_URL>`
- **Development model:** Claude Opus 5.5, high reasoning effort.
- **Settings:** `Demo-only` (0 simulator starts); `Interaction-only` and `Hybrid` (at most 10 starts per task).
- **Evaluation:** final policies on all 800 test episodes (50 per task); no model API calls at test time.

## Results

API costs are rough estimates, rounded to the nearest $10, and exclude local compute.

| Setting | Rollouts per task | Success | Development starts | Development API cost | Evaluation API cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Demo-only` | 0 | 140/800 (17.50%) | 0 | ≈$210 | $0 |
| `Interaction-only` | 10 | 302/800 (37.75%) | 157 | ≈$160 | $0 |
| `Hybrid` | 10 | 342/800 (42.75%) | 134 | ≈$160 | $0 |

| Task | `Demo-only` | `Interaction-only` | `Hybrid` |
| --- | ---: | ---: | ---: |
| BinFill | 18/50 (36.00%) | 25/50 (50.00%) | 41/50 (82.00%) |
| PickXtimes | 0/50 (0.00%) | 24/50 (48.00%) | 11/50 (22.00%) |
| SwingXtimes | 17/50 (34.00%) | 27/50 (54.00%) | 33/50 (66.00%) |
| StopCube | 17/50 (34.00%) | 40/50 (80.00%) | 29/50 (58.00%) |
| VideoUnmask | 31/50 (62.00%) | 27/50 (54.00%) | 34/50 (68.00%) |
| VideoUnmaskSwap | 0/50 (0.00%) | 27/50 (54.00%) | 38/50 (76.00%) |
| ButtonUnmask | 23/50 (46.00%) | 35/50 (70.00%) | 28/50 (56.00%) |
| ButtonUnmaskSwap | 2/50 (4.00%) | 4/50 (8.00%) | 14/50 (28.00%) |
| PickHighlight | 0/50 (0.00%) | 17/50 (34.00%) | 11/50 (22.00%) |
| VideoRepick | 0/50 (0.00%) | 24/50 (48.00%) | 36/50 (72.00%) |
| VideoPlaceButton | 11/50 (22.00%) | 9/50 (18.00%) | 9/50 (18.00%) |
| VideoPlaceOrder | 10/50 (20.00%) | 15/50 (30.00%) | 20/50 (40.00%) |
| MoveCube | 1/50 (2.00%) | 3/50 (6.00%) | 1/50 (2.00%) |
| InsertPeg | 0/50 (0.00%) | 12/50 (24.00%) | 4/50 (8.00%) |
| PatternLock | 0/50 (0.00%) | 3/50 (6.00%) | 10/50 (20.00%) |
| RouteStick | 10/50 (20.00%) | 10/50 (20.00%) | 23/50 (46.00%) |
| **Overall** | **140/800 (17.50%)** | **302/800 (37.75%)** | **342/800 (42.75%)** |

### Reference results with fewer rollouts

Optional; not part of the leaderboard or the audit record below.

| Setting | 1 rollout | 3 rollouts | 5 rollouts |
| --- | ---: | ---: | ---: |
| `Interaction-only` | 117/800 (14.63%) | pending | pending |
| `Hybrid` | 155/800 (19.38%) | pending | pending |

<details>
<summary>Per-task success with 1 rollout</summary>

| Task | `Interaction-only` | `Hybrid` |
| --- | ---: | ---: |
| BinFill | 1/50 (2.00%) | 13/50 (26.00%) |
| PickXtimes | 7/50 (14.00%) | 13/50 (26.00%) |
| SwingXtimes | 10/50 (20.00%) | 2/50 (4.00%) |
| StopCube | 32/50 (64.00%) | 14/50 (28.00%) |
| VideoUnmask | 2/50 (4.00%) | 19/50 (38.00%) |
| VideoUnmaskSwap | 8/50 (16.00%) | 1/50 (2.00%) |
| ButtonUnmask | 3/50 (6.00%) | 24/50 (48.00%) |
| ButtonUnmaskSwap | 1/50 (2.00%) | 0/50 (0.00%) |
| PickHighlight | 0/50 (0.00%) | 8/50 (16.00%) |
| VideoRepick | 16/50 (32.00%) | 37/50 (74.00%) |
| VideoPlaceButton | 8/50 (16.00%) | 5/50 (10.00%) |
| VideoPlaceOrder | 9/50 (18.00%) | 16/50 (32.00%) |
| MoveCube | 0/50 (0.00%) | 0/50 (0.00%) |
| InsertPeg | 0/50 (0.00%) | 3/50 (6.00%) |
| PatternLock | 10/50 (20.00%) | 0/50 (0.00%) |
| RouteStick | 10/50 (20.00%) | 0/50 (0.00%) |
| **Overall** | **117/800 (14.63%)** | **155/800 (19.38%)** |

</details>

## Agentic development audit record

Audited with the [development audit](../agentic_development.md#step-8-audit): no forbidden information, and at most
the allowed simulator starts. Auditor: Claude Opus 5.5, 2026-11-22 (self-review by the RoboMME team's assistant).

| Setting | Budget per task | Development starts | 1. No forbidden information | 2. Rollout budget | Conclusion | Full report |
| --- | ---: | ---: | --- | --- | --- | --- |
| `Demo-only` | 0 | 0 | PASS | PASS | **PASS** | [audit.json](../audits/current/aspire/demo-only/R0/audit.json) |
| `Interaction-only` | 10 | 157 | PASS | PASS | **PASS** | [audit.json](../audits/current/aspire/interaction-only/R10/audit.json) |
| `Hybrid` | 10 | 134 | PASS | PASS | **PASS** | [audit.json](../audits/current/aspire/hybrid/R10/audit.json) |

| Task | `Demo-only` starts | `Interaction-only` starts | `Hybrid` starts |
| --- | ---: | ---: | ---: |
| BinFill | 0 | 10 | 10 |
| PickXtimes | 0 | 10 | 8 |
| SwingXtimes | 0 | 10 | 7 |
| StopCube | 0 | 10 | 9 |
| VideoUnmask | 0 | 10 | 8 |
| VideoUnmaskSwap | 0 | 10 | 5 |
| ButtonUnmask | 0 | 10 | 5 |
| ButtonUnmaskSwap | 0 | 10 | 10 |
| PickHighlight | 0 | 10 | 8 |
| VideoRepick | 0 | 7 | 9 |
| VideoPlaceButton | 0 | 10 | 10 |
| VideoPlaceOrder | 0 | 10 | 10 |
| MoveCube | 0 | 10 | 5 |
| InsertPeg | 0 | 10 | 10 |
| PatternLock | 0 | 10 | 10 |
| RouteStick | 0 | 10 | 10 |
| **Total** | **0** | **157** | **134** |

All starts used released training seeds.
