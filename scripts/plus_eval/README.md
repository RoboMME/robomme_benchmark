# RoboMME-Plus baselines

Final results on **800 test episodes** with the default benchmark variations.
See the [evaluation guide](../../src/robomme_plus/README.md) for rules and
[per-task results](../../doc/robomme_plus/robomme_plus_baseline_results.md) for the full breakdown.
Each method link below provides setup and running instructions.

## Model baselines

| Method | Setting | Successes (rate) |
|---|---|---:|
| [Embodied Astra](agentic_methods/astra/embodied/README.md) | ZS | 194/800 (24.25%) |
| [One-shot Astra](agentic_methods/astra/embodied/README.md#one-shot) | AfD | 281/800 (35.13%) |
| [3-tier Astra](agentic_methods/astra/three_tier/README.md) | ZS | 70/800 (8.75%) |
| [FrameSamp+Modul](non_agentic_methods/framesamp_modul/README.md) | ZS | 19/800 (2.38%) |
| [SimpleMemVLA](non_agentic_methods/simplememvla/README.md) | ZS | 14/800 (1.75%) |
| [PonderPounce](non_agentic_methods/ponderpounce/README.md) | ZS | 20/800 (2.50%) |

## Agentic adaptation

AfD adapts from demos; AfI from interactions on training seeds; AfD+I from both.
R caps simulator attempts per task. ASPIRE and RAX develop each budget independently.

| Setting | Budget | [ASPIRE](agentic_methods/aspire/README.md) | [RAX](agentic_methods/rax/README.md) |
|---|---|---:|---:|
| AfD | R0 | 136/800 (17.00%) | 152/800 (19.00%) |
| AfI | R1 | 117/800 (14.63%) | 26/800 (3.25%) |
| AfI | R3 | 134/800 (16.75%) | 351/800 (43.88%) |
| AfI | R5 | 215/800 (26.88%) | 265/800 (33.13%) |
| AfI | R10 | 341/800 (42.63%) | 312/800 (39.00%) |
| AfD+I | R1 | 155/800 (19.38%) | 156/800 (19.50%) |
| AfD+I | R3 | 261/800 (32.63%) | 315/800 (39.38%) |
| AfD+I | R5 | 252/800 (31.50%) | 283/800 (35.38%) |
| AfD+I | R10 | 279/800 (34.88%) | 337/800 (42.13%) |

Final summaries: [model baselines](results/model_baselines.json) ·
[ASPIRE/RAX](results/agentic_20261118.json).
