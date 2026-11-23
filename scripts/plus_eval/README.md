# RoboMME-Plus baselines

Final results on **800 test episodes** with the default benchmark variations.
See the [evaluation guide](../../src/robomme_plus/README.md) for rules and
[per-task results](../../doc/robomme_plus/robomme_plus_baseline_results.md) for the full breakdown.
Each method link below provides setup and running instructions.

## Model baselines

| Method | Setting | Successes (rate) |
|---|---|---:|
| [Embodied Astra](agentic_methods/astra/embodied/README.md) | `Zero-shot` | 194/800 (24.25%) |
| [One-shot Astra](agentic_methods/astra/embodied/README.md#one-shot) | `Demo-only` | 281/800 (35.13%) |
| [3-tier Astra](agentic_methods/astra/three_tier/README.md) | `Zero-shot` | 70/800 (8.75%) |
| [FrameSamp+Modul](non_agentic_methods/framesamp_modul/README.md) | `Zero-shot` | 19/800 (2.38%) |
| [SimpleMemVLA](non_agentic_methods/simplememvla/README.md) | `Zero-shot` | 14/800 (1.75%) |
| [PonderPounce](non_agentic_methods/ponderpounce/README.md) | `Zero-shot` | 20/800 (2.50%) |

## Agentic adaptation

`Demo-only` uses the 50 released demonstrations and no simulator rollouts. The
interaction settings use at most **10 development rollouts per task**, shared across
training seeds. The packaged solutions cover the R10 runs.

| Setting | Budget | [ASPIRE](agentic_methods/aspire/README.md) | [RAX](agentic_methods/rax/README.md) |
|---|---|---:|---:|
| `Demo-only` | 0 | 140/800 (17.50%) | 139/800 (17.38%) |
| `Interaction-only` | 10 | 302/800 (37.75%) | 372/800 (46.50%) |
| `Hybrid` | 10 | 342/800 (42.75%) | 327/800 (40.88%) |

The submissions include the [ASPIRE](<../../doc/robomme_plus/submission/aspire-plus (eval_by_RoboMME_team).md#agentic-development-audit-record>)
and [RAX](<../../doc/robomme_plus/submission/rax-plus (eval_by_RoboMME_team).md#agentic-development-audit-record>) audit records.

Final summaries: [model baselines](results/model_baselines.json) ·
[ASPIRE/RAX](results/agentic_current.json).
