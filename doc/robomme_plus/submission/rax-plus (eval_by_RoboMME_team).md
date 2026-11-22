# RAX on RoboMME-Plus (eval_by_RoboMME_team)

- **Track:** Agentic.
- **Code:** [Adapter and run instructions](../../../scripts/plus_eval/agentic_methods/rax/README.md); [frozen R10 solutions](../../../scripts/plus_eval/agentic_methods/rax/solutions/README.md).
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
| `interaction-only` | 10 | 370/800 (46.25%) | 123 | ≈$90 | $0 |
| `Hybrid` | 10 | 326/800 (40.75%) | 121 | ≈$100 | $0 |

API estimates are rounded to the nearest $10 and exclude local compute. Exact usage is retained in the logs.

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

The reports below review each setting separately. They are retrospective reviews of the saved harness and complete development record; numerical scores alone do not establish compliance.

### `interaction-only` · R10

**Harness audit: PENDING.** The independent review is running or queued; its returned report will be inserted here.

**Development audit: PENDING.** The independent review is running or queued; its returned report will be inserted here.

### `Hybrid` · R10

**Harness audit: PENDING.** The independent review is running or queued; its returned report will be inserted here.

**Development audit: PENDING.** The independent review is running or queued; its returned report will be inserted here.
