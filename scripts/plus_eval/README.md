# RoboMME-Plus baselines

Final results on **800 test episodes** with the default benchmark variations.
See the [evaluation guide](../../src/robomme_plus/README.md) for rules and
[per-task results](../../doc/robomme_plus/robomme_plus_baseline_results.md) for the full breakdown.
Each method link below provides setup and running instructions.

## Model baselines

| Method | Setting | Successes (rate) |
|---|---|---:|
| [Embodied Astra](agentic_methods/astra/embodied/README.md) | Zero-shot | 194/800 (24.25%) |
| [One-shot Astra](agentic_methods/astra/embodied/README.md#one-shot) | Few-shot demonstrations | 281/800 (35.13%) |
| [3-tier Astra](agentic_methods/astra/three_tier/README.md) | Zero-shot | 70/800 (8.75%) |
| [FrameSamp+Modul](non_agentic_methods/framesamp_modul/README.md) | Zero-shot | 19/800 (2.38%) |
| [SimpleMemVLA](non_agentic_methods/simplememvla/README.md) | Zero-shot | 14/800 (1.75%) |
| [PonderPounce](non_agentic_methods/ponderpounce/README.md) | Zero-shot | 20/800 (2.50%) |

## Agentic adaptation

The interaction settings allow at most **5 or 10 development rollouts per task**.
ASPIRE and RAX develop each budget independently. All results use 800 test episodes.

| Setting | Budget | [ASPIRE](agentic_methods/aspire/README.md) | [RAX](agentic_methods/rax/README.md) |
|---|---|---:|---:|
| Few-shot demonstrations | 0 | 136/800 (17.00%) | 152/800 (19.00%) |
| Few-shot interactions | 5 | 215/800 (26.88%) | 265/800 (33.13%) |
| Few-shot interactions | 10 | 341/800 (42.63%) | 312/800 (39.00%) |
| Few-shot demonstrations+interactions | 5 | 252/800 (31.50%) | 283/800 (35.38%) |
| Few-shot demonstrations+interactions | 10 | 279/800 (34.88%) | 337/800 (42.13%) |

Archived R1/R3 results remain in the JSON summaries and frozen solution bundles.

Final summaries: [model baselines](results/model_baselines.json) ·
[ASPIRE/RAX](results/agentic_20261118.json).
